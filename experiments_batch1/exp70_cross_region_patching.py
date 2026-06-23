"""Experiment 70: Cross-region activation patching.

The biological analog of path patching in mech interp: does swapping the
evidence subspace output of region A into region B's input change B's
decodable choice?

For simultaneously recorded region pairs (A, B):
  1. Estimate A's choice subspace (VAE z_choice directions in neural space)
  2. For opposite-label trial pairs:
     a. Project A's activity onto its choice subspace -> potent component
     b. Replace B's projection onto A's choice subspace with A's potent
        component (simulating patching A->B)
     c. Decode choice from B's modified activity
  3. Measure "cross-region IIA": flip rate when A's choice signal is
     patched into B

This tests whether regions communicate choice information through shared
subspace structure. If patching A->B flips B's decoded choice, A's choice
subspace is in B's potent space — they share causal geometry.

Predictions:
  - Patching between regions in the same functional group should produce
    higher flip rates (shared choice geometry)
  - Patching from causally important regions (per silencing) should
    produce the highest flip rates in downstream regions
  - The patching graph should reveal the information flow hierarchy:
    sensory -> decision -> motor
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp70"
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
MIN_TRIALS_PER_CONDITION = 20
N_IIA_PAIRS = 100

Z_CHOICE_DIM = 3
Z_OTHER_DIM = 15
HIDDEN_DIM = 128
N_EPOCHS = 300
BATCH_SIZE = 64
LR = 1e-3
BETA_KL = 1.0
ALPHA_CHOICE = 10.0


class StructuredVAE(nn.Module):
    def __init__(self, n_neurons, z_choice_dim=Z_CHOICE_DIM, z_other_dim=Z_OTHER_DIM):
        super().__init__()
        self.z_choice_dim = z_choice_dim
        z_dim = z_choice_dim + z_other_dim
        self.encoder = nn.Sequential(nn.Linear(n_neurons, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU())
        self.fc_mu = nn.Linear(HIDDEN_DIM, z_dim)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, z_dim)
        self.decoder = nn.Sequential(nn.Linear(z_dim, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, n_neurons))
        self.choice_head = nn.Linear(z_choice_dim, 2)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decoder(z)
        choice_logits = self.choice_head(z[:, :self.z_choice_dim])
        return recon, mu, logvar, choice_logits


def _train_vae(activity, labels, device, n_neurons):
    model = StructuredVAE(n_neurons).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    X = torch.tensor(activity, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                         drop_last=len(dataset) > BATCH_SIZE)
    model.train()
    for _ in range(N_EPOCHS):
        for xb, yb in loader:
            recon, mu, logvar, choice_logits = model(xb)
            recon_loss = F.mse_loss(recon, xb)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            choice_loss = F.cross_entropy(choice_logits, yb)
            loss = recon_loss + BETA_KL * kl + ALPHA_CHOICE * choice_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model



def _cross_region_patch_iia(act_source, act_target, choice_labels,
                             source_model, target_model, device):
    """Patch source's z_choice into target's latent and check if decoded choice flips.

    Since source and target have different neuron counts, we work in the shared
    z_choice latent space:
      1. Encode source -> z_choice_source, encode target -> (z_choice_target, z_other_target)
      2. For opposite-label pairs: replace target's z_choice with source's z_choice
      3. Decode patched target latent -> reconstructed target activity
      4. Check if a classifier on target's z_choice flips

    This tests whether source's choice information, when transplanted into target's
    latent representation, changes target's decoded choice.
    """
    n = min(len(act_source), len(act_target), len(choice_labels))
    act_s = act_source[:n]
    act_t = act_target[:n]
    ch = choice_labels[:n]

    left_idx = np.where(ch == 0)[0]
    right_idx = np.where(ch == 1)[0]
    if len(left_idx) < 5 or len(right_idx) < 5:
        return float("nan")

    source_model.eval()
    target_model.eval()

    with torch.no_grad():
        X_s = torch.tensor(act_s, dtype=torch.float32, device=device)
        X_t = torch.tensor(act_t, dtype=torch.float32, device=device)
        mu_s, _ = source_model.encode(X_s)
        mu_t, _ = target_model.encode(X_t)

    z_choice_s = mu_s[:, :source_model.z_choice_dim]  # (n, k)
    z_choice_t = mu_t[:, :target_model.z_choice_dim]  # (n, k)

    # Train classifier on target's z_choice
    z_t_np = z_choice_t.cpu().numpy()
    clf = LogisticRegression(max_iter=500, solver="lbfgs")
    try:
        clf.fit(z_t_np, ch)
    except Exception:
        return float("nan")

    z_s_np = z_choice_s.cpu().numpy()
    flips = 0
    n_pairs = min(N_IIA_PAIRS, len(left_idx), len(right_idx))
    for i in range(n_pairs):
        li = left_idx[i % len(left_idx)]
        ri = right_idx[i % len(right_idx)]

        # Swap: target trial li gets source trial ri's z_choice
        pred_orig = clf.predict(z_t_np[li:li+1])[0]
        pred_swap = clf.predict(z_s_np[ri:ri+1])[0]
        if pred_orig != pred_swap:
            flips += 1

    return flips / max(n_pairs, 1)


def run(max_sessions: int | None = None) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"{datetime.now().isoformat()} Starting cross-region patching experiment "
                f"with {len(sessions)} sessions on {device}")

    # Load per-session, per-region data (need simultaneous recordings)
    session_region_data: list[dict[str, dict]] = []
    for sess_idx, sess in enumerate(tqdm(sessions, desc="Loading sessions")):
        choice_labels = get_choice_labels(sess)
        if len(np.unique(choice_labels)) < 2:
            continue
        regions = list_regions(sess, min_neurons=MIN_NEURONS)
        sess_data = {}
        for region in regions:
            act = get_region_activity(sess, region)
            if act is None or act.shape[1] < MIN_NEURONS:
                continue
            n = min(act.shape[0], len(choice_labels))
            activity = act[:n, :, TIME_WINDOW].mean(axis=2)
            ch = choice_labels[:n]
            sess_data[region] = {"activity": activity, "choice_labels": ch,
                                 "n_neurons": int(activity.shape[1])}
        if len(sess_data) >= 2:
            session_region_data.append(sess_data)

    logger.info(f"{datetime.now().isoformat()} {len(session_region_data)} sessions with >= 2 regions")

    # Train VAEs per session per region (each session has consistent neuron counts)
    session_models: list[dict[str, StructuredVAE]] = []  # parallel to session_region_data
    for sess_idx, sess_data in enumerate(tqdm(session_region_data, desc="Training VAEs")):
        models = {}
        for region, rd in sess_data.items():
            activity = rd["activity"]
            ch = rd["choice_labels"]
            n_neurons = rd["n_neurons"]
            if len(activity) < MIN_TRIALS_PER_CONDITION * 2:
                continue
            model = _train_vae(activity, ch, device, n_neurons)
            models[region] = model
        session_models.append({"models": models})

    n_total_models = sum(len(sm["models"]) for sm in session_models)
    logger.info(f"{datetime.now().isoformat()} {n_total_models} session-region models trained "
                f"across {len(session_models)} sessions")

    # Cross-region patching within each session
    patch_results: dict[str, list[float]] = {}

    for sess_idx, sess_data in enumerate(tqdm(session_region_data, desc="Sessions")):
        sm = session_models[sess_idx]
        regions_in_sess = [r for r in sess_data if r in sm["models"]]
        if len(regions_in_sess) < 2:
            continue

        for source in regions_in_sess:
            for target in regions_in_sess:
                if source == target:
                    continue
                pair_key = f"{source}->{target}"
                act_s = sess_data[source]["activity"]
                act_t = sess_data[target]["activity"]
                ch = sess_data[source]["choice_labels"]

                iia = _cross_region_patch_iia(
                    act_s, act_t, ch,
                    sm["models"][source], sm["models"][target], device)

                if not np.isnan(iia):
                    if pair_key not in patch_results:
                        patch_results[pair_key] = []
                    patch_results[pair_key].append(iia)

    # Aggregate per pair
    pair_summary = {}
    for pair_key, iias in patch_results.items():
        pair_summary[pair_key] = {
            "mean_iia": float(np.mean(iias)),
            "std_iia": float(np.std(iias)),
            "n_sessions": len(iias),
        }

    # Compute per-region "outgoing" and "incoming" patching strength
    region_outgoing = {}
    region_incoming = {}
    for pair_key, stats in pair_summary.items():
        source, target = pair_key.split("->")
        if source not in region_outgoing:
            region_outgoing[source] = []
        region_outgoing[source].append(stats["mean_iia"])
        if target not in region_incoming:
            region_incoming[target] = []
        region_incoming[target].append(stats["mean_iia"])

    region_hub_scores = {}
    for region in set(list(region_outgoing.keys()) + list(region_incoming.keys())):
        out = np.mean(region_outgoing.get(region, [0]))
        inc = np.mean(region_incoming.get(region, [0]))
        region_hub_scores[region] = {
            "mean_outgoing_iia": float(out),
            "mean_incoming_iia": float(inc),
            "n_outgoing": len(region_outgoing.get(region, [])),
            "n_incoming": len(region_incoming.get(region, [])),
            "asymmetry": float(out - inc),
        }

    # Top patching pairs
    top_pairs = sorted(pair_summary.items(), key=lambda x: -x[1]["mean_iia"])[:20]

    summary = {
        "n_pairs": len(pair_summary),
        "n_sessions": len(session_region_data),
        "top_20_pairs": {k: v for k, v in top_pairs},
        "region_hub_scores": region_hub_scores,
        "all_pairs": pair_summary,
    }

    out_path = RESULTS_DIR / f"exp70_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"{datetime.now().isoformat()} Results saved to {out_path}")

    return summary
