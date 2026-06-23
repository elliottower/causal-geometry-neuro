"""Experiment 73: Structured pi-SAE ablation for causal subspace discovery.

Tests whether combining ALL three inductive biases improves IIA on neural data:
  1. Structured split (z_choice / z_other)
  2. Label-conditional prior on z_choice
  3. Overcomplete z_choice with L1 sparsity (SAE-style)

6-model ablation (all combinations of the three biases):
  A. structured_vae:     split + Gaussian prior + classification head (exp69 baseline)
  B. pi_vae:             label prior + NO split (exp69 baseline)
  C. pi_structured_vae:  split + label prior + classification head (exp69 baseline)
  D. sae_structured:     split + Gaussian prior + overcomplete z_choice + L1
  E. pi_sae_plain:       label prior + overcomplete + L1 + NO split
  F. pi_sae_structured:  split + label prior + overcomplete z_choice + L1 (THE NEW ONE)

The grokking repo showed pi-SAE (E) works on Fourier-basis representations.
exp69 showed pi-VAE (B) and pi-structured-VAE (C) fail on neural data.
This tests whether adding sparsity (D, F) rescues either approach.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "exp73"
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
MIN_TRIALS_PER_CONDITION = 20

Z_CHOICE_DIM = 3
Z_OTHER_DIM = 15
HIDDEN_DIM = 128
N_EPOCHS = 300
BATCH_SIZE = 64
LR = 1e-3
BETA_KL = 1.0
ALPHA_CHOICE = 10.0
L1_COEFF = 1e-3
EXPANSION_FACTOR = 8
N_IIA_PAIRS = 100


class StructuredVAE(nn.Module):
    """Model A: structured split + Gaussian prior + classification head."""
    def __init__(self, n_neurons, z_choice_dim=Z_CHOICE_DIM, z_other_dim=Z_OTHER_DIM):
        super().__init__()
        self.z_choice_dim = z_choice_dim
        self.z_other_dim = z_other_dim
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
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
        recon = self.decoder(z)
        choice_logits = self.choice_head(z[:, :self.z_choice_dim])
        return recon, mu, logvar, choice_logits

    def loss(self, x, labels):
        recon, mu, logvar, choice_logits = self.forward(x)
        recon_loss = F.mse_loss(recon, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        choice_loss = F.cross_entropy(choice_logits, labels)
        return recon_loss + BETA_KL * kl + ALPHA_CHOICE * choice_loss


class PiVAE(nn.Module):
    """Model B: label-conditional prior, no structured split."""
    def __init__(self, n_neurons, z_dim=Z_CHOICE_DIM + Z_OTHER_DIM):
        super().__init__()
        self.z_choice_dim = Z_CHOICE_DIM
        self.z_dim = z_dim
        self.encoder = nn.Sequential(nn.Linear(n_neurons, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU())
        self.fc_mu = nn.Linear(HIDDEN_DIM, z_dim)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, z_dim)
        self.decoder = nn.Sequential(nn.Linear(z_dim, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, n_neurons))
        self.prior_mu = nn.Embedding(2, z_dim)
        self.prior_logvar = nn.Embedding(2, z_dim)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def loss(self, x, labels):
        recon, mu, logvar = self.forward(x)
        recon_loss = F.mse_loss(recon, x)
        prior_mu = self.prior_mu(labels)
        prior_lv = self.prior_logvar(labels)
        kl = -0.5 * (1 + logvar - prior_lv
                      - ((mu - prior_mu).pow(2) + logvar.exp()) / prior_lv.exp()).mean()
        return recon_loss + BETA_KL * kl


class PiStructuredVAE(nn.Module):
    """Model C: structured split + label-conditional prior on z_choice."""
    def __init__(self, n_neurons, z_choice_dim=Z_CHOICE_DIM, z_other_dim=Z_OTHER_DIM):
        super().__init__()
        self.z_choice_dim = z_choice_dim
        self.z_other_dim = z_other_dim
        z_dim = z_choice_dim + z_other_dim
        self.encoder = nn.Sequential(nn.Linear(n_neurons, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU())
        self.fc_mu = nn.Linear(HIDDEN_DIM, z_dim)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, z_dim)
        self.decoder = nn.Sequential(nn.Linear(z_dim, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, n_neurons))
        self.choice_head = nn.Linear(z_choice_dim, 2)
        self.prior_mu = nn.Embedding(2, z_choice_dim)
        self.prior_logvar = nn.Embedding(2, z_choice_dim)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
        recon = self.decoder(z)
        choice_logits = self.choice_head(z[:, :self.z_choice_dim])
        return recon, mu, logvar, choice_logits

    def loss(self, x, labels):
        recon, mu, logvar, choice_logits = self.forward(x)
        recon_loss = F.mse_loss(recon, x)
        mu_c, logvar_c = mu[:, :self.z_choice_dim], logvar[:, :self.z_choice_dim]
        mu_o, logvar_o = mu[:, self.z_choice_dim:], logvar[:, self.z_choice_dim:]
        prior_mu = self.prior_mu(labels)
        prior_lv = self.prior_logvar(labels)
        kl_c = -0.5 * (1 + logvar_c - prior_lv
                        - ((mu_c - prior_mu).pow(2) + logvar_c.exp()) / prior_lv.exp()).mean()
        kl_o = -0.5 * torch.mean(1 + logvar_o - mu_o.pow(2) - logvar_o.exp())
        choice_loss = F.cross_entropy(choice_logits, labels)
        return recon_loss + BETA_KL * (kl_c + kl_o) + ALPHA_CHOICE * choice_loss


class SAEStructured(nn.Module):
    """Model D: structured split + Gaussian prior + overcomplete z_choice + L1."""
    def __init__(self, n_neurons, z_choice_dim=Z_CHOICE_DIM, z_other_dim=Z_OTHER_DIM,
                 expansion=EXPANSION_FACTOR):
        super().__init__()
        self.z_choice_dim = z_choice_dim * expansion
        self.z_other_dim = z_other_dim
        z_dim = self.z_choice_dim + z_other_dim
        self.encoder = nn.Sequential(nn.Linear(n_neurons, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU())
        self.fc_mu = nn.Linear(HIDDEN_DIM, z_dim)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, z_dim)
        self.decoder = nn.Sequential(nn.Linear(z_dim, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, n_neurons))
        self.choice_head = nn.Linear(self.z_choice_dim, 2)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
        recon = self.decoder(z)
        choice_logits = self.choice_head(z[:, :self.z_choice_dim])
        return recon, mu, logvar, choice_logits

    def loss(self, x, labels):
        recon, mu, logvar, choice_logits = self.forward(x)
        recon_loss = F.mse_loss(recon, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        choice_loss = F.cross_entropy(choice_logits, labels)
        sparsity = mu[:, :self.z_choice_dim].abs().mean()
        return recon_loss + BETA_KL * kl + ALPHA_CHOICE * choice_loss + L1_COEFF * sparsity


class PiSAEPlain(nn.Module):
    """Model E: label prior + overcomplete + L1, NO structured split."""
    def __init__(self, n_neurons, z_dim=Z_CHOICE_DIM + Z_OTHER_DIM,
                 expansion=EXPANSION_FACTOR):
        super().__init__()
        self.z_choice_dim = Z_CHOICE_DIM * expansion
        self.z_dim = self.z_choice_dim
        self.encoder = nn.Sequential(nn.Linear(n_neurons, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU())
        self.fc_mu = nn.Linear(HIDDEN_DIM, self.z_dim)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, self.z_dim)
        self.decoder = nn.Sequential(nn.Linear(self.z_dim, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, n_neurons))
        self.classifier = nn.Linear(self.z_dim, 2)
        self.prior_mu = nn.Embedding(2, self.z_dim)
        self.prior_logvar = nn.Embedding(2, self.z_dim)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
        recon = self.decoder(z)
        logits = self.classifier(z)
        return recon, mu, logvar, logits

    def loss(self, x, labels):
        recon, mu, logvar, logits = self.forward(x)
        recon_loss = F.mse_loss(recon, x)
        prior_mu = self.prior_mu(labels)
        prior_lv = self.prior_logvar(labels)
        kl = -0.5 * (1 + logvar - prior_lv
                      - ((mu - prior_mu).pow(2) + logvar.exp()) / prior_lv.exp()).mean()
        ce = F.cross_entropy(logits, labels)
        sparsity = mu.abs().mean()
        return recon_loss + BETA_KL * kl + ALPHA_CHOICE * ce + L1_COEFF * sparsity


class PiSAEStructured(nn.Module):
    """Model F: THE NEW ONE — structured split + label prior on z_choice + overcomplete + L1."""
    def __init__(self, n_neurons, z_choice_dim=Z_CHOICE_DIM, z_other_dim=Z_OTHER_DIM,
                 expansion=EXPANSION_FACTOR):
        super().__init__()
        self.z_choice_dim = z_choice_dim * expansion
        self.z_other_dim = z_other_dim
        z_dim = self.z_choice_dim + z_other_dim

        self.enc_trunk = nn.Sequential(nn.Linear(n_neurons, HIDDEN_DIM), nn.ReLU(),
                                       nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU())
        self.enc_choice_mu = nn.Linear(HIDDEN_DIM, self.z_choice_dim)
        self.enc_choice_logvar = nn.Linear(HIDDEN_DIM, self.z_choice_dim)
        self.enc_other_mu = nn.Linear(HIDDEN_DIM, z_other_dim)
        self.enc_other_logvar = nn.Linear(HIDDEN_DIM, z_other_dim)

        self.decoder = nn.Sequential(nn.Linear(z_dim, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, n_neurons))
        self.choice_head = nn.Linear(self.z_choice_dim, 2)
        self.prior_mu = nn.Embedding(2, self.z_choice_dim)
        self.prior_logvar = nn.Embedding(2, self.z_choice_dim)

    def encode(self, x):
        h = self.enc_trunk(x)
        return (self.enc_choice_mu(h), self.enc_choice_logvar(h),
                self.enc_other_mu(h), self.enc_other_logvar(h))

    def forward(self, x):
        mu_c, lv_c, mu_o, lv_o = self.encode(x)
        z_c = mu_c + torch.exp(0.5 * lv_c) * torch.randn_like(lv_c)
        z_o = mu_o + torch.exp(0.5 * lv_o) * torch.randn_like(lv_o)
        z = torch.cat([z_c, z_o], dim=-1)
        recon = self.decoder(z)
        choice_logits = self.choice_head(z_c)
        return recon, mu_c, lv_c, mu_o, lv_o, choice_logits

    def loss(self, x, labels):
        recon, mu_c, lv_c, mu_o, lv_o, choice_logits = self.forward(x)
        recon_loss = F.mse_loss(recon, x)
        prior_mu = self.prior_mu(labels)
        prior_lv = self.prior_logvar(labels)
        kl_c = -0.5 * (1 + lv_c - prior_lv
                        - ((mu_c - prior_mu).pow(2) + lv_c.exp()) / prior_lv.exp()).mean()
        kl_o = -0.5 * torch.mean(1 + lv_o - mu_o.pow(2) - lv_o.exp())
        choice_loss = F.cross_entropy(choice_logits, labels)
        sparsity = mu_c.abs().mean()
        return recon_loss + BETA_KL * (kl_c + kl_o) + ALPHA_CHOICE * choice_loss + L1_COEFF * sparsity


MODEL_SPECS = {
    "structured_vae": {"cls": StructuredVAE, "structured": True},
    "pi_vae": {"cls": PiVAE, "structured": False},
    "pi_structured_vae": {"cls": PiStructuredVAE, "structured": True},
    "sae_structured": {"cls": SAEStructured, "structured": True},
    "pi_sae_plain": {"cls": PiSAEPlain, "structured": False},
    "pi_sae_structured": {"cls": PiSAEStructured, "structured": True},
}


def _train_model(model, activity, labels, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    X = torch.tensor(activity, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                         drop_last=len(dataset) > BATCH_SIZE)
    model.train()
    for _ in range(N_EPOCHS):
        for xb, yb in loader:
            loss = model.loss(xb, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def _compute_iia(model, activity, choice_labels, device):
    """Compute IIA by swapping z_choice between opposite-label pairs."""
    X = torch.tensor(activity, dtype=torch.float32, device=device)
    z_choice_dim = model.z_choice_dim

    with torch.no_grad():
        if hasattr(model, 'enc_trunk'):
            mu_c, _, mu_o, _ = model.encode(X)
            mu_all = torch.cat([mu_c, mu_o], dim=-1)
        else:
            mu_all, _ = model.encode(X)

    z_choice = mu_all[:, :z_choice_dim].cpu().numpy()
    clf = LogisticRegression(max_iter=500, solver="lbfgs")
    clf.fit(z_choice, choice_labels)

    left_idx = np.where(choice_labels == 0)[0]
    right_idx = np.where(choice_labels == 1)[0]
    if len(left_idx) < 5 or len(right_idx) < 5:
        return float("nan")

    flips = 0
    n_pairs = min(N_IIA_PAIRS, len(left_idx), len(right_idx))
    for i in range(n_pairs):
        li = left_idx[i % len(left_idx)]
        ri = right_idx[i % len(right_idx)]
        z_swapped = mu_all[li].clone().cpu().numpy()
        z_swapped[:z_choice_dim] = mu_all[ri, :z_choice_dim].cpu().numpy()
        pred_orig = clf.predict(z_choice[li:li+1])[0]
        pred_swap = clf.predict(z_swapped[:z_choice_dim].reshape(1, -1))[0]
        if pred_orig != pred_swap:
            flips += 1
    return flips / n_pairs


def run(max_sessions: int | None = None, model_filter: str | None = None) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    if model_filter:
        active_specs = {model_filter: MODEL_SPECS[model_filter]}
    else:
        active_specs = MODEL_SPECS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"{datetime.now().isoformat()} Starting structured pi-SAE ablation "
                f"models={list(active_specs.keys())} "
                f"with {len(sessions)} sessions on {device}")

    region_data: dict[str, list[dict]] = {}
    for sess_idx, sess in enumerate(tqdm(sessions, desc="Loading sessions")):
        choice_labels = get_choice_labels(sess)
        if len(np.unique(choice_labels)) < 2:
            continue
        regions = list_regions(sess, min_neurons=MIN_NEURONS)
        for region in regions:
            act = get_region_activity(sess, region)
            if act is None or act.shape[1] < MIN_NEURONS:
                continue
            n = min(act.shape[0], len(choice_labels))
            activity = act[:n, :, TIME_WINDOW].mean(axis=2)
            ch = choice_labels[:n]
            if region not in region_data:
                region_data[region] = []
            region_data[region].append({"activity": activity, "choice_labels": ch,
                                        "n_neurons": int(activity.shape[1])})

    logger.info(f"{datetime.now().isoformat()} {len(region_data)} regions loaded")

    model_names = list(active_specs.keys())
    results_per_region = {}
    suffix = f"_{model_filter}" if model_filter else ""
    incremental_path = RESULTS_DIR / f"exp73{suffix}_incremental.jsonl"

    for region in tqdm(sorted(region_data.keys()), desc="Regions"):
        sessions_data = region_data[region]
        per_session_results = []

        for sess in sessions_data:
            activity = sess["activity"]
            ch = sess["choice_labels"]
            n_neurons = sess["n_neurons"]

            if len(activity) < MIN_TRIALS_PER_CONDITION * 2:
                continue

            sess_results = {}
            for name, spec in active_specs.items():
                model = spec["cls"](n_neurons).to(device)
                _train_model(model, activity, ch, device)
                iia = _compute_iia(model, activity, ch, device)
                sess_results[name] = {"iia": float(iia)}

            per_session_results.append(sess_results)

        if not per_session_results:
            continue

        region_results = {}
        for name in model_names:
            iias = [sr[name]["iia"] for sr in per_session_results
                    if not np.isnan(sr[name]["iia"])]
            region_results[name] = {"iia": float(np.mean(iias)) if iias else float("nan")}

        results_per_region[region] = region_results

        with open(incremental_path, "a") as f:
            f.write(json.dumps({"region": region, **region_results}, default=str) + "\n")
        logger.info(f"{datetime.now().isoformat()} {region}: " +
                    " | ".join(f"{n}={region_results[n]['iia']:.3f}" for n in model_names))

    regions = sorted(results_per_region.keys())
    summary_stats = {}
    for name in model_names:
        iias = [results_per_region[r][name]["iia"] for r in regions
                if not np.isnan(results_per_region[r][name]["iia"])]
        summary_stats[name] = {
            "mean_iia": float(np.mean(iias)) if iias else float("nan"),
            "std_iia": float(np.std(iias)) if iias else float("nan"),
            "n_regions": len(iias),
        }

    comparisons = {}
    if not model_filter:
        pairs_to_compare = [
            ("pi_sae_structured", "structured_vae"),
            ("pi_sae_structured", "pi_structured_vae"),
            ("pi_sae_structured", "pi_sae_plain"),
            ("sae_structured", "structured_vae"),
            ("pi_sae_plain", "pi_vae"),
            ("sae_structured", "pi_sae_structured"),
        ]
        for m1, m2 in pairs_to_compare:
            iia1 = [results_per_region[r][m1]["iia"] for r in regions]
            iia2 = [results_per_region[r][m2]["iia"] for r in regions]
            valid = [(a, b) for a, b in zip(iia1, iia2)
                     if not np.isnan(a) and not np.isnan(b)]
            if len(valid) >= 5:
                v1, v2 = zip(*valid)
                wins = sum(1 for a, b in valid if a > b)
                stat, p = wilcoxon([a - b for a, b in valid])
                comparisons[f"{m1}_vs_{m2}"] = {
                    "wins": wins, "total": len(valid),
                    "wilcoxon_p": float(p),
                    "mean_diff": float(np.mean([a - b for a, b in valid])),
                }

    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_regions": len(regions),
        "model_stats": summary_stats,
        "comparisons": comparisons,
        "per_region": results_per_region,
        "config": {
            "z_choice_dim": Z_CHOICE_DIM,
            "z_other_dim": Z_OTHER_DIM,
            "expansion_factor": EXPANSION_FACTOR,
            "l1_coeff": L1_COEFF,
            "n_epochs": N_EPOCHS,
            "hidden_dim": HIDDEN_DIM,
            "alpha_choice": ALPHA_CHOICE,
            "beta_kl": BETA_KL,
        },
    }

    out_path = RESULTS_DIR / f"exp73{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"{datetime.now().isoformat()} Saved to {out_path}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--model", type=str, default=None,
                        choices=list(MODEL_SPECS.keys()))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(max_sessions=args.max_sessions, model_filter=args.model)
