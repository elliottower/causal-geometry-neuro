"""Experiment 58: Mixture-of-Experts VAE for multi-region neural data.

Each brain region is a separate "modality" with its own encoder/decoder
(handling different neuron counts naturally). A product-of-experts (PoE)
posterior combination learns a shared latent space across all simultaneously
recorded regions.

Based on: Shi et al., "Variational Mixture-of-Experts Autoencoders for
Multi-Modal Deep Generative Models," NeurIPS 2019.

Replaces the broken sheaf restriction maps (exp7/exp15) for cross-region
comparison: instead of noisy correlation-based projections between different-
dimensional spaces, we learn a shared generative latent space where each
region's encoder maps into a common coordinate system.

After training, cross-region alignment is read off by comparing encoder
weight matrices in the shared latent space (no ad-hoc projection needed).

Key outputs:
  - Learned latent representations per trial per session
  - Cross-region alignment matrices (cosine similarity of encoder projections)
  - Per-region reconstruction loss
  - Choice-decoding accuracy from shared latent space
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp58"
MIN_NEURONS = 10
TIME_WINDOW = slice(15, 35)
LATENT_DIM = 16
HIDDEN_DIM = 64
N_EPOCHS = 200
BATCH_SIZE = 64
LR = 1e-3
KL_WEIGHT = 1.0
MIN_REGIONS_PER_SESSION = 3
PCA_DIM = 50


class RegionEncoder(nn.Module):
    """Per-region encoder: x_r (n_neurons_r,) -> q_r(z | x_r) = N(mu_r, diag(sigma_r^2))."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class RegionDecoder(nn.Module):
    """Per-region decoder: z (latent_dim,) -> reconstructed x_r (n_neurons_r,)."""

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MultiRegionMoEVAE(nn.Module):
    """Mixture-of-Experts VAE for multi-region neural data.

    Product-of-experts posterior: given K regions observed on a trial,
    q(z | x_1, ..., x_K) = PoE(q_1(z|x_1), ..., q_K(z|x_K), p(z))

    For Gaussians, PoE has closed form:
        precision_combined = sum(1/sigma_r^2) + 1  (prior precision)
        mu_combined = precision_combined^{-1} * sum(mu_r / sigma_r^2)
    """

    def __init__(self, region_dims: dict[str, int], hidden_dim: int, latent_dim: int):
        super().__init__()
        self.region_names = sorted(region_dims.keys())
        self.latent_dim = latent_dim

        self.encoders = nn.ModuleDict({
            name: RegionEncoder(dim, hidden_dim, latent_dim)
            for name, dim in region_dims.items()
        })
        self.decoders = nn.ModuleDict({
            name: RegionDecoder(latent_dim, hidden_dim, dim)
            for name, dim in region_dims.items()
        })

    def _product_of_experts(
        self,
        mus: list[torch.Tensor],
        logvars: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """PoE combination of Gaussian experts, including unit-variance prior as an expert."""
        # Prior: N(0, I) has precision 1 everywhere
        # precision = 1/var, so log_precision = -logvar
        prior_precision = torch.ones_like(mus[0])  # (batch, latent_dim)
        precision_sum = prior_precision.clone()
        weighted_mu_sum = torch.zeros_like(mus[0])

        for mu, logvar in zip(mus, logvars):
            precision = torch.exp(-logvar)
            precision_sum = precision_sum + precision
            weighted_mu_sum = weighted_mu_sum + mu * precision

        combined_var = 1.0 / precision_sum
        combined_mu = weighted_mu_sum * combined_var
        combined_logvar = torch.log(combined_var)
        return combined_mu, combined_logvar

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        region_data: dict[str, torch.Tensor],
    ) -> dict:
        """Forward pass with arbitrary subset of regions.

        Args:
            region_data: {region_name: (batch, n_neurons_r)} for observed regions

        Returns dict with: z, mu, logvar, recons, per_region_mu, per_region_logvar
        """
        mus, logvars = [], []
        per_region_mu = {}
        per_region_logvar = {}

        for name, x in region_data.items():
            mu, logvar = self.encoders[name](x)
            mus.append(mu)
            logvars.append(logvar)
            per_region_mu[name] = mu
            per_region_logvar[name] = logvar

        combined_mu, combined_logvar = self._product_of_experts(mus, logvars)
        z = self._reparameterize(combined_mu, combined_logvar)

        recons = {}
        for name in region_data:
            recons[name] = self.decoders[name](z)

        return {
            "z": z,
            "mu": combined_mu,
            "logvar": combined_logvar,
            "recons": recons,
            "per_region_mu": per_region_mu,
            "per_region_logvar": per_region_logvar,
        }


def _vae_loss(
    region_data: dict[str, torch.Tensor],
    output: dict,
    kl_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """ELBO loss: sum of per-region reconstruction + KL divergence.

    Reconstruction: MSE per region (normalized by region dim).
    KL: closed-form KL(q(z|x) || p(z)) for diagonal Gaussians.
    """
    recon_losses = {}
    total_recon = torch.tensor(0.0, device=output["mu"].device)

    for name, x in region_data.items():
        x_recon = output["recons"][name]
        recon = F.mse_loss(x_recon, x, reduction="mean")
        recon_losses[name] = recon.item()
        total_recon = total_recon + recon

    mu, logvar = output["mu"], output["logvar"]
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    loss = total_recon + kl_weight * kl
    diagnostics = {
        "loss": loss.item(),
        "recon": total_recon.item(),
        "kl": kl.item(),
        **{f"recon_{name}": v for name, v in recon_losses.items()},
    }
    return loss, diagnostics


def _prepare_session_data(
    sess: dict,
    sess_idx: int,
    min_neurons: int = MIN_NEURONS,
    pca_dim: int = PCA_DIM,
) -> tuple[dict[str, np.ndarray], np.ndarray] | None:
    """Extract multi-region trial-averaged activity for one session.

    Returns:
        region_activities: {region_name: (n_trials, n_dims)} where n_dims = min(n_neurons, pca_dim)
        labels: (n_trials,) binary choice labels
    """
    labels = get_choice_labels(sess)
    if len(np.unique(labels)) < 2:
        return None

    regions = list_regions(sess, min_neurons=min_neurons)
    region_activities = {}

    for region in regions:
        act = get_region_activity(sess, region)
        if act is None or act.shape[1] < min_neurons:
            continue
        n = min(act.shape[0], len(labels))
        # Trial-averaged over time window: (n_trials, n_neurons)
        activity = act[:n, :, TIME_WINDOW].mean(axis=2)

        # PCA reduce if too many neurons (keeps things tractable)
        if activity.shape[1] > pca_dim:
            pca = PCA(n_components=pca_dim)
            activity = pca.fit_transform(activity)

        # z-score normalize per neuron
        mu = activity.mean(axis=0, keepdims=True)
        std = activity.std(axis=0, keepdims=True) + 1e-8
        activity = (activity - mu) / std

        region_activities[region] = activity

    if len(region_activities) < MIN_REGIONS_PER_SESSION:
        return None

    # Align trial counts across regions
    min_trials = min(a.shape[0] for a in region_activities.values())
    min_trials = min(min_trials, len(labels))
    region_activities = {k: v[:min_trials] for k, v in region_activities.items()}
    labels = labels[:min_trials]

    return region_activities, labels


def _train_vae(
    region_activities: dict[str, np.ndarray],
    latent_dim: int = LATENT_DIM,
    hidden_dim: int = HIDDEN_DIM,
    n_epochs: int = N_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    kl_weight: float = KL_WEIGHT,
    device: str = "cpu",
) -> tuple[MultiRegionMoEVAE, dict]:
    """Train MoE-VAE on one session's multi-region data.

    Returns: (trained model, training_log dict)
    """
    region_dims = {name: act.shape[1] for name, act in region_activities.items()}
    n_trials = next(iter(region_activities.values())).shape[0]

    model = MultiRegionMoEVAE(region_dims, hidden_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Convert to tensors
    tensors = {
        name: torch.tensor(act, dtype=torch.float32, device=device)
        for name, act in region_activities.items()
    }

    log = {"epoch_losses": [], "final_per_region_recon": {}}

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_trials, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_trials, batch_size):
            idx = perm[start:start + batch_size]
            batch = {name: t[idx] for name, t in tensors.items()}

            output = model(batch)
            loss, diagnostics = _vae_loss(batch, output, kl_weight=kl_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += diagnostics["loss"]
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        log["epoch_losses"].append(avg_loss)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        output = model(tensors)
        _, diagnostics = _vae_loss(tensors, output, kl_weight=kl_weight)
        log["final_loss"] = diagnostics["loss"]
        log["final_kl"] = diagnostics["kl"]
        for name in region_activities:
            key = f"recon_{name}"
            if key in diagnostics:
                log["final_per_region_recon"][name] = diagnostics[key]

    return model, log


def _extract_latents(
    model: MultiRegionMoEVAE,
    region_activities: dict[str, np.ndarray],
    device: str = "cpu",
) -> np.ndarray:
    """Extract PoE latent means for all trials."""
    model.eval()
    tensors = {
        name: torch.tensor(act, dtype=torch.float32, device=device)
        for name, act in region_activities.items()
    }
    with torch.no_grad():
        output = model(tensors)
    return output["mu"].cpu().numpy()


def _cross_region_alignment(model: MultiRegionMoEVAE) -> dict:
    """Compare regions by cosine similarity of their encoder projections into latent space.

    For each region, the "encoding direction" is the first-layer weight matrix
    of the encoder (hidden_dim x input_dim) composed with the mu projection
    (latent_dim x hidden_dim), giving an effective (latent_dim x input_dim) map.

    Since input dims differ, we compare the latent-space representations by
    looking at how each encoder's mu-head maps the hidden features. The
    mu projection W_mu (latent_dim x hidden_dim) is shared-dimensioned, so
    we compute cosine similarity between pairs of W_mu matrices.
    """
    alignments = {}
    region_names = sorted(model.encoders.keys())

    # Extract the mu projection weight for each region: (latent_dim, hidden_dim)
    mu_weights = {}
    for name in region_names:
        w = model.encoders[name].fc_mu.weight.detach().cpu().numpy()  # (latent_dim, hidden_dim)
        mu_weights[name] = w

    for i, r1 in enumerate(region_names):
        for j, r2 in enumerate(region_names):
            if j <= i:
                continue
            w1 = mu_weights[r1].flatten()
            w2 = mu_weights[r2].flatten()
            cos_sim = float(np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2) + 1e-10))
            alignments[f"{r1}___{r2}"] = cos_sim

    return alignments


def _choice_decoding_from_latents(latents: np.ndarray, labels: np.ndarray) -> float:
    """Logistic regression accuracy on choice from shared latent space."""
    if len(np.unique(labels)) < 2 or latents.shape[0] < 20:
        return float("nan")
    n = min(latents.shape[0], len(labels))
    X, y = latents[:n], labels[:n]
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    return float(np.mean(scores))


def _per_region_decoding(
    model: MultiRegionMoEVAE,
    region_activities: dict[str, np.ndarray],
    labels: np.ndarray,
    device: str = "cpu",
) -> dict[str, float]:
    """Choice decoding from each individual region's latent (not PoE combined)."""
    model.eval()
    results = {}
    for name, act in region_activities.items():
        x = torch.tensor(act, dtype=torch.float32, device=device)
        with torch.no_grad():
            mu, _ = model.encoders[name](x)
        latent = mu.cpu().numpy()
        acc = _choice_decoding_from_latents(latent, labels)
        results[name] = acc
    return results


def run(max_sessions: int | None = None) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"[{datetime.now().isoformat()}] Starting exp58_moe_vae, device={device}")
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]
    logger.info(f"[{datetime.now().isoformat()}] Loaded {len(sessions)} sessions")

    session_results = []

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Sessions")):
        prepared = _prepare_session_data(sess, sess_idx)
        if prepared is None:
            continue

        region_activities, labels = prepared
        mouse = str(sess.get("mouse_name", f"mouse_{sess_idx}"))
        n_regions = len(region_activities)
        region_dims = {k: v.shape[1] for k, v in region_activities.items()}
        n_trials = next(iter(region_activities.values())).shape[0]

        logger.info(
            f"[{datetime.now().isoformat()}] Session {sess_idx}: mouse={mouse}, "
            f"{n_regions} regions, {n_trials} trials, dims={region_dims}"
        )

        model, train_log = _train_vae(
            region_activities,
            latent_dim=LATENT_DIM,
            hidden_dim=HIDDEN_DIM,
            n_epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            kl_weight=KL_WEIGHT,
            device=device,
        )

        latents = _extract_latents(model, region_activities, device=device)
        alignments = _cross_region_alignment(model)
        poe_decoding_acc = _choice_decoding_from_latents(latents, labels)
        per_region_decoding = _per_region_decoding(model, region_activities, labels, device=device)

        # Save latents for this session
        npz_path = RESULTS_DIR / f"latents_sess{sess_idx}.npz"
        np.savez_compressed(
            npz_path,
            latents=latents,
            labels=labels,
            regions=np.array(sorted(region_activities.keys())),
        )

        sess_result = {
            "session_idx": sess_idx,
            "mouse": mouse,
            "n_trials": n_trials,
            "n_regions": n_regions,
            "regions": sorted(region_activities.keys()),
            "region_dims": region_dims,
            "final_loss": train_log["final_loss"],
            "final_kl": train_log["final_kl"],
            "per_region_recon": train_log["final_per_region_recon"],
            "cross_region_alignment": alignments,
            "poe_choice_decoding_acc": poe_decoding_acc,
            "per_region_choice_decoding_acc": per_region_decoding,
            "convergence": {
                "loss_epoch_0": train_log["epoch_losses"][0] if train_log["epoch_losses"] else None,
                "loss_epoch_final": train_log["epoch_losses"][-1] if train_log["epoch_losses"] else None,
            },
        }
        session_results.append(sess_result)

        logger.info(
            f"[{datetime.now().isoformat()}] Session {sess_idx} done: "
            f"loss={train_log['final_loss']:.4f}, "
            f"PoE decoding={poe_decoding_acc:.3f}"
        )

    # Aggregate results across sessions
    all_alignments = {}
    for sr in session_results:
        for pair_key, sim in sr["cross_region_alignment"].items():
            if pair_key not in all_alignments:
                all_alignments[pair_key] = []
            all_alignments[pair_key].append(sim)

    alignment_summary = {
        pair: {
            "mean_cosine_sim": float(np.mean(vals)),
            "std_cosine_sim": float(np.std(vals)),
            "n_sessions": len(vals),
        }
        for pair, vals in all_alignments.items()
        if len(vals) >= 2
    }

    poe_accs = [sr["poe_choice_decoding_acc"] for sr in session_results
                if not np.isnan(sr["poe_choice_decoding_acc"])]

    per_region_acc_agg = {}
    for sr in session_results:
        for region, acc in sr["per_region_choice_decoding_acc"].items():
            if not np.isnan(acc):
                if region not in per_region_acc_agg:
                    per_region_acc_agg[region] = []
                per_region_acc_agg[region].append(acc)

    region_decoding_summary = {
        region: {
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs)),
            "n_sessions": len(accs),
        }
        for region, accs in per_region_acc_agg.items()
    }

    # Does PoE (multi-region) beat best single region?
    best_single_per_session = []
    poe_per_session = []
    for sr in session_results:
        per_reg = sr["per_region_choice_decoding_acc"]
        valid = [v for v in per_reg.values() if not np.isnan(v)]
        poe = sr["poe_choice_decoding_acc"]
        if valid and not np.isnan(poe):
            best_single_per_session.append(max(valid))
            poe_per_session.append(poe)

    poe_vs_single = None
    if best_single_per_session:
        poe_advantage = [p - s for p, s in zip(poe_per_session, best_single_per_session)]
        poe_vs_single = {
            "mean_poe_acc": float(np.mean(poe_per_session)),
            "mean_best_single_acc": float(np.mean(best_single_per_session)),
            "mean_poe_advantage": float(np.mean(poe_advantage)),
            "n_sessions_poe_wins": int(sum(1 for a in poe_advantage if a > 0)),
            "n_sessions_total": len(poe_advantage),
        }

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "latent_dim": LATENT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "n_epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "kl_weight": KL_WEIGHT,
            "min_neurons": MIN_NEURONS,
            "min_regions_per_session": MIN_REGIONS_PER_SESSION,
            "pca_dim": PCA_DIM,
            "time_window": f"slice({TIME_WINDOW.start}, {TIME_WINDOW.stop})",
        },
        "n_sessions_total": len(sessions),
        "n_sessions_used": len(session_results),
        "session_results": session_results,
        "alignment_summary": alignment_summary,
        "decoding_summary": {
            "poe_mean_acc": float(np.mean(poe_accs)) if poe_accs else None,
            "poe_std_acc": float(np.std(poe_accs)) if poe_accs else None,
            "per_region": region_decoding_summary,
            "poe_vs_best_single_region": poe_vs_single,
        },
    }

    out_path = RESULTS_DIR / "moe_vae_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"[{datetime.now().isoformat()}] Saved results to {out_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sessions", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(max_sessions=args.max_sessions)
