"""Experiment 16: SAE on spike trains (Steinmetz).

Train a sparse autoencoder on population activity vectors to test:
do neural populations encode task variables in superposition?

If the SAE finds more monosemantic features than there are neurons,
the population uses superposition (like transformers).
If SAE features ≈ neurons, there's no superposition.

Compare SAE feature directions to LDA choice direction:
  - If a SAE feature aligns with LDA → it found a causal feature
  - If SAE features are orthogonal to LDA → SAE finds structure but not choice

This is the direct bridge between MI (SAE/superposition) and neuroscience.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all
from geometry.subspace import fit_lda_subspace

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp16"
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
SAE_EXPANSION = 4
SAE_SPARSITY = 0.05
SAE_EPOCHS = 200
SAE_LR = 1e-3


class SimpleSAE:
    """Minimal sparse autoencoder for population activity."""

    def __init__(self, n_input, n_hidden, sparsity_target=0.05, lr=1e-3):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.sparsity_target = sparsity_target
        self.lr = lr

        scale = np.sqrt(2.0 / n_input)
        self.W_enc = np.random.randn(n_input, n_hidden) * scale
        self.b_enc = np.zeros(n_hidden)
        self.W_dec = np.random.randn(n_hidden, n_input) * scale
        self.b_dec = np.zeros(n_input)

    def encode(self, x):
        return np.maximum(0, x @ self.W_enc + self.b_enc)

    def decode(self, h):
        return h @ self.W_dec + self.b_dec

    def forward(self, x):
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h

    def train(self, X, epochs=200, l1_weight=0.1):
        n = X.shape[0]
        losses = []

        for epoch in range(epochs):
            x_hat, h = self.forward(X)

            recon_loss = np.mean((X - x_hat) ** 2)
            l1_loss = np.mean(np.abs(h))
            total_loss = recon_loss + l1_weight * l1_loss

            recon_grad = -2 * (X - x_hat) / n

            dW_dec = h.T @ recon_grad / n
            db_dec = recon_grad.mean(axis=0)

            d_h = recon_grad @ self.W_dec.T + l1_weight * np.sign(h) / n
            relu_mask = (h > 0).astype(float)
            d_pre = d_h * relu_mask

            dW_enc = X.T @ d_pre / n
            db_enc = d_pre.mean(axis=0)

            self.W_enc -= self.lr * dW_enc
            self.b_enc -= self.lr * db_enc
            self.W_dec -= self.lr * dW_dec
            self.b_dec -= self.lr * db_dec

            if epoch % 50 == 0:
                losses.append({"epoch": epoch, "recon": float(recon_loss), "l1": float(l1_loss)})

        return losses

    def get_feature_directions(self):
        """Decoder columns = feature directions in input space."""
        norms = np.linalg.norm(self.W_dec, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return self.W_dec / norms


def run(max_sessions: int | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    all_results = []

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Sessions")):
        labels = get_choice_labels(sess)
        if len(np.unique(labels)) < 2:
            continue

        mouse = str(sess.get("mouse_name", f"mouse_{sess_idx}"))
        regions = list_regions(sess, min_neurons=MIN_NEURONS)

        for region in regions:
            act = get_region_activity(sess, region)
            if act is None or act.shape[1] < MIN_NEURONS:
                continue

            n = min(act.shape[0], len(labels))
            activity = act[:n, :, TIME_WINDOW].mean(axis=2)
            n_neurons = activity.shape[1]

            activity_norm = (activity - activity.mean(axis=0)) / (activity.std(axis=0) + 1e-8)

            n_hidden = n_neurons * SAE_EXPANSION
            sae = SimpleSAE(n_neurons, n_hidden, sparsity_target=SAE_SPARSITY, lr=SAE_LR)
            train_losses = sae.train(activity_norm, epochs=SAE_EPOCHS)

            _, h = sae.forward(activity_norm)
            active_mask = (h > 0).mean(axis=0) > 0.01
            n_active_features = int(active_mask.sum())
            avg_active_per_trial = float((h > 0).sum(axis=1).mean())
            sparsity = float((h > 0).mean())

            try:
                k = min(5, n_neurons - 1)
                U = fit_lda_subspace(activity, labels[:n], k=k)
                lda_direction = U[:, 0]

                feature_dirs = sae.get_feature_directions()
                cosines = np.abs(feature_dirs @ lda_direction)
                max_alignment = float(cosines.max())
                mean_alignment = float(cosines[active_mask].mean()) if active_mask.any() else 0.0
                best_feature_idx = int(cosines.argmax())
            except Exception:
                max_alignment = None
                mean_alignment = None
                best_feature_idx = None

            recon_error = float(np.mean((activity_norm - sae.forward(activity_norm)[0]) ** 2))

            all_results.append({
                "session_idx": sess_idx,
                "mouse": mouse,
                "region": region,
                "n_neurons": n_neurons,
                "n_hidden": n_hidden,
                "n_active_features": n_active_features,
                "expansion_ratio": n_active_features / n_neurons if n_neurons > 0 else 0,
                "avg_active_per_trial": avg_active_per_trial,
                "sparsity": sparsity,
                "recon_error": recon_error,
                "max_lda_alignment": max_alignment,
                "mean_lda_alignment": mean_alignment,
                "best_feature_idx": best_feature_idx,
                "superposition": n_active_features > n_neurons * 1.5,
            })

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_regions_analyzed": len(all_results),
        "sae_config": {
            "expansion": SAE_EXPANSION,
            "sparsity_target": SAE_SPARSITY,
            "epochs": SAE_EPOCHS,
        },
        "regions": all_results,
    }

    if all_results:
        n_super = sum(1 for r in all_results if r["superposition"])
        results["summary"] = {
            "n_superposition": n_super,
            "n_not_superposition": len(all_results) - n_super,
            "mean_expansion_ratio": float(np.mean([r["expansion_ratio"] for r in all_results])),
            "mean_max_lda_alignment": float(np.mean([r["max_lda_alignment"] for r in all_results if r["max_lda_alignment"] is not None])),
        }

    out_path = RESULTS_DIR / "sae_spike_trains.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved to {out_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sessions", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(max_sessions=args.max_sessions)
