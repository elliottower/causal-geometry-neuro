"""Experiment 71: VAE mechanistic interpretability — causal circuit analysis.

Applies the multi-level causal intervention framework (arXiv:2505.03530) to
understand HOW our structured VAE's encoder processes neural activity.

Three metrics from the framework:
  1. CES (Causal Effect Strength): E_x[||D(z) - D(z_tilde_i)||_2]
     How much does intervening on latent dimension i change the reconstruction?
  2. Intervention Specificity: S(i) = 1/(H(p_i) + eps)
     Are changes localized (high specificity) or diffuse (low specificity)?
  3. Circuit Modularity: M = 1 - mean(|rho(Delta_a_i, Delta_a_j)|)
     Do different latent dimensions use separate encoder pathways?

Four manipulation types:
  A. Input manipulation: modify neural activity, track latent changes
  B. Latent perturbation: clamp z_i, measure reconstruction changes
  C. Activation patching: replace encoder hidden activations between trials
  D. Causal mediation: quantify information flow through specific encoder channels

Key questions:
  - Does the structured VAE develop specialized encoder channels for choice vs other?
  - Is there a "choice detector" channel analogous to FactorVAE's shape detectors?
  - Do high-CES latent dimensions correspond to the choice dimensions (z_choice)?
  - Is modularity higher for the structured VAE than an unstructured one?
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
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp71"
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
MIN_TRIALS_PER_CONDITION = 20

Z_CHOICE_DIM = 3
Z_OTHER_DIM = 5
HIDDEN_DIM = 128
N_EPOCHS = 300
BATCH_SIZE = 64
LR = 1e-3
BETA_KL = 1.0
ALPHA_CHOICE = 50.0

N_RANDOM_SPLITS = 10  # number of random channel assignments for baseline

N_CES_SAMPLES = 200
N_PATCH_SAMPLES = 100
INTERVENTION_RANGE = np.linspace(-3, 3, 7)


class StructuredVAE(nn.Module):
    def __init__(self, n_neurons, z_choice_dim=Z_CHOICE_DIM, z_other_dim=Z_OTHER_DIM):
        super().__init__()
        self.z_choice_dim = z_choice_dim
        self.z_other_dim = z_other_dim
        z_dim = z_choice_dim + z_other_dim
        self.enc_layer1 = nn.Linear(n_neurons, HIDDEN_DIM)
        self.enc_layer2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc_mu = nn.Linear(HIDDEN_DIM, z_dim)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, z_dim)
        self.dec_layer1 = nn.Linear(z_dim, HIDDEN_DIM)
        self.dec_layer2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.dec_layer3 = nn.Linear(HIDDEN_DIM, n_neurons)
        self.choice_head = nn.Linear(z_choice_dim, 2)

    def encode_with_intermediates(self, x):
        """Encode and return all intermediate activations for patching."""
        h1 = F.relu(self.enc_layer1(x))
        h2 = F.relu(self.enc_layer2(h1))
        mu = self.fc_mu(h2)
        logvar = self.fc_logvar(h2)
        return mu, logvar, {"h1": h1, "h2": h2}

    def encode(self, x):
        mu, logvar, _ = self.encode_with_intermediates(x)
        return mu, logvar

    def decode(self, z):
        h = F.relu(self.dec_layer1(z))
        h = F.relu(self.dec_layer2(h))
        return self.dec_layer3(h)

    def forward(self, x):
        mu, logvar, intermediates = self.encode_with_intermediates(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decode(z)
        choice_logits = self.choice_head(z[:, :self.z_choice_dim])
        return recon, mu, logvar, choice_logits, intermediates


class UnstructuredVAE(nn.Module):
    """Control: same architecture but no choice/other split, no classification head."""
    def __init__(self, n_neurons, z_dim=Z_CHOICE_DIM + Z_OTHER_DIM):
        super().__init__()
        self.z_dim = z_dim
        self.enc_layer1 = nn.Linear(n_neurons, HIDDEN_DIM)
        self.enc_layer2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc_mu = nn.Linear(HIDDEN_DIM, z_dim)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, z_dim)
        self.dec_layer1 = nn.Linear(z_dim, HIDDEN_DIM)
        self.dec_layer2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.dec_layer3 = nn.Linear(HIDDEN_DIM, n_neurons)

    def encode_with_intermediates(self, x):
        h1 = F.relu(self.enc_layer1(x))
        h2 = F.relu(self.enc_layer2(h1))
        mu = self.fc_mu(h2)
        logvar = self.fc_logvar(h2)
        return mu, logvar, {"h1": h1, "h2": h2}

    def encode(self, x):
        mu, logvar, _ = self.encode_with_intermediates(x)
        return mu, logvar

    def decode(self, z):
        h = F.relu(self.dec_layer1(z))
        h = F.relu(self.dec_layer2(h))
        return self.dec_layer3(h)

    def forward(self, x):
        mu, logvar, intermediates = self.encode_with_intermediates(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decode(z)
        return recon, mu, logvar, intermediates


def _train_structured(model, activity, labels, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    X = torch.tensor(activity, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                         drop_last=len(dataset) > BATCH_SIZE)
    model.train()
    for _ in range(N_EPOCHS):
        for xb, yb in loader:
            recon, mu, logvar, choice_logits, _ = model(xb)
            recon_loss = F.mse_loss(recon, xb)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            choice_loss = F.cross_entropy(choice_logits, yb)
            loss = recon_loss + BETA_KL * kl + ALPHA_CHOICE * choice_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def _train_unstructured(model, activity, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    X = torch.tensor(activity, dtype=torch.float32, device=device)
    dataset = torch.utils.data.TensorDataset(X,)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                         drop_last=len(dataset) > BATCH_SIZE)
    model.train()
    for _ in range(N_EPOCHS):
        for (xb,) in loader:
            mu, logvar, _ = model.encode_with_intermediates(xb)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            recon = model.decode(z)
            recon_loss = F.mse_loss(recon, xb)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + BETA_KL * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def compute_ces(model, X, device):
    """Causal Effect Strength: E[||D(z) - D(z_tilde_i)||] for each latent dim."""
    model.eval()
    X_t = torch.tensor(X[:N_CES_SAMPLES], dtype=torch.float32, device=device)
    with torch.no_grad():
        mu, _ = model.encode(X_t)
        z_dim = mu.shape[1]
        recon_base = model.decode(mu)

        ces = np.zeros(z_dim)
        for dim in range(z_dim):
            effects = []
            for val in INTERVENTION_RANGE:
                z_mod = mu.clone()
                z_mod[:, dim] = val
                recon_mod = model.decode(z_mod)
                diff = (recon_base - recon_mod).pow(2).sum(dim=-1).sqrt()
                effects.append(diff.mean().item())
            ces[dim] = np.mean(effects)

    return ces


def compute_intervention_specificity(model, X, device):
    """How localized are the reconstruction changes when intervening on dim i?"""
    model.eval()
    X_t = torch.tensor(X[:N_CES_SAMPLES], dtype=torch.float32, device=device)
    with torch.no_grad():
        mu, _ = model.encode(X_t)
        z_dim = mu.shape[1]
        recon_base = model.decode(mu)

        specificity = np.zeros(z_dim)
        for dim in range(z_dim):
            z_mod = mu.clone()
            z_mod[:, dim] = 0.0  # zero-out intervention
            recon_mod = model.decode(z_mod)
            diff_sq = (recon_base - recon_mod).pow(2).mean(dim=0)  # per-output-neuron
            # Normalize to distribution
            p = diff_sq / (diff_sq.sum() + 1e-12)
            entropy = -(p * (p + 1e-12).log()).sum().item()
            specificity[dim] = 1.0 / (entropy + 1e-6)

    return specificity


def compute_circuit_modularity(model, X, choice_labels, device):
    """Are activation changes for different latent dims correlated (low modularity)
    or independent (high modularity)?"""
    model.eval()
    X_t = torch.tensor(X[:N_PATCH_SAMPLES], dtype=torch.float32, device=device)
    with torch.no_grad():
        mu, _, intermediates = model.encode_with_intermediates(X_t)
        h2 = intermediates["h2"]  # (batch, hidden_dim)
        z_dim = mu.shape[1]

        # For each latent dim, compute activation change when that dim is clamped
        delta_activations = []
        for dim in range(z_dim):
            z_mod = mu.clone()
            z_mod[:, dim] = 0.0
            # We need the encoder activations that WOULD produce z_mod
            # Approximate: measure which h2 channels correlate most with z_dim
            # Use the mu projection weights as proxy
            w_mu_dim = model.fc_mu.weight[dim]  # (hidden_dim,)
            delta_activations.append(w_mu_dim.cpu().numpy())

        delta_activations = np.array(delta_activations)  # (z_dim, hidden_dim)

    # Modularity: 1 - mean absolute correlation between activation change vectors
    n = delta_activations.shape[0]
    if n < 2:
        return 1.0
    corrs = []
    for i in range(n):
        for j in range(i + 1, n):
            r = np.corrcoef(delta_activations[i], delta_activations[j])[0, 1]
            corrs.append(abs(r) if not np.isnan(r) else 0.0)
    modularity = 1.0 - np.mean(corrs)
    return float(modularity)


def compute_random_split_ces(model, X, device, z_choice_dim, n_splits=N_RANDOM_SPLITS):
    """Compute CES ratio for random assignments of channels as 'choice' vs 'other'.

    Takes the unstructured VAE and randomly partitions its latent dims into
    z_choice_dim 'choice' channels and the rest as 'other', then computes the
    CES ratio (choice/other). This tells us whether the structured VAE's CES
    separation is meaningful or just an artifact of any arbitrary split.
    """
    ces = compute_ces(model, X, device)
    z_dim = len(ces)
    ratios = []
    for _ in range(n_splits):
        perm = np.random.permutation(z_dim)
        choice_idx = perm[:z_choice_dim]
        other_idx = perm[z_choice_dim:]
        choice_mean = np.mean(ces[choice_idx])
        other_mean = np.mean(ces[other_idx])
        ratios.append(choice_mean / max(other_mean, 1e-12))
    return float(np.mean(ratios)), float(np.std(ratios)), ratios


def compute_channel_mediation(model, X, choice_labels, device):
    """Which encoder channels mediate choice information?

    For each hidden channel c in layer 2:
      - Patch channel c's activation from an opposite-label trial
      - Measure how much z_choice changes
    Channels with high mediation are "choice detectors".
    """
    model.eval()
    X_t = torch.tensor(X[:N_PATCH_SAMPLES], dtype=torch.float32, device=device)
    labels = choice_labels[:N_PATCH_SAMPLES]

    with torch.no_grad():
        _, _, intermediates = model.encode_with_intermediates(X_t)
        h2 = intermediates["h2"]  # (batch, hidden_dim)
        mu_base, _ = model.encode(X_t)

    left_idx = np.where(labels == 0)[0]
    right_idx = np.where(labels == 1)[0]
    if len(left_idx) < 5 or len(right_idx) < 5:
        return np.zeros(HIDDEN_DIM)

    hidden_dim = h2.shape[1]
    mediation = np.zeros(hidden_dim)

    with torch.no_grad():
        n_pairs = min(50, len(left_idx), len(right_idx))
        for pair_i in range(n_pairs):
            li = left_idx[pair_i % len(left_idx)]
            ri = right_idx[pair_i % len(right_idx)]

            for ch in range(hidden_dim):
                # Patch channel ch from right trial into left trial's h2
                h2_patched = h2[li].clone()
                h2_patched[ch] = h2[ri, ch]
                # Forward from h2 to mu
                mu_patched = model.fc_mu(h2_patched)
                # Measure z_choice change
                z_choice_dim = model.z_choice_dim if hasattr(model, 'z_choice_dim') else Z_CHOICE_DIM
                delta_z_choice = (mu_patched[:z_choice_dim] - mu_base[li, :z_choice_dim]).pow(2).sum().sqrt()
                mediation[ch] += delta_z_choice.item()

        mediation /= n_pairs

    return mediation


def run(max_sessions: int | None = None) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"{datetime.now().isoformat()} Starting VAE causal circuits experiment "
                f"with {len(sessions)} sessions on {device}")

    # Load data
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
            region_data[region].append({
                "activity": activity, "choice_labels": ch,
                "n_neurons": int(activity.shape[1]),
            })

    logger.info(f"{datetime.now().isoformat()} {len(region_data)} regions loaded")

    results_per_region = {}

    for region in tqdm(sorted(region_data.keys()), desc="Regions"):
        sessions_data = region_data[region]

        # Process each session independently (sessions have different n_neurons)
        per_session_results = []
        for sess in sessions_data:
            activity = sess["activity"]
            ch = sess["choice_labels"]
            n_neurons = sess["n_neurons"]

            if len(activity) < MIN_TRIALS_PER_CONDITION * 2:
                continue

            # Train structured VAE
            model_s = StructuredVAE(n_neurons).to(device)
            _train_structured(model_s, activity, ch, device)

            # Train unstructured VAE (control)
            model_u = UnstructuredVAE(n_neurons).to(device)
            _train_unstructured(model_u, activity, device)

            # Compute metrics for structured VAE
            ces_s = compute_ces(model_s, activity, device)
            spec_s = compute_intervention_specificity(model_s, activity, device)
            mod_s = compute_circuit_modularity(model_s, activity, ch, device)
            mediation_s = compute_channel_mediation(model_s, activity, ch, device)

            # Compute metrics for unstructured VAE
            ces_u = compute_ces(model_u, activity, device)
            spec_u = compute_intervention_specificity(model_u, activity, device)
            mod_u = compute_circuit_modularity(model_u, activity, ch, device)

            # Random-split baseline: randomly partition unstructured VAE channels
            rand_ces_ratio_mean, rand_ces_ratio_std, _ = compute_random_split_ces(
                model_u, activity, device, Z_CHOICE_DIM)

            # Scalar summaries (safe to aggregate across sessions with different n_neurons)
            per_session_results.append({
                "structured": {
                    "ces_choice_mean": float(np.mean(ces_s[:Z_CHOICE_DIM])),
                    "ces_other_mean": float(np.mean(ces_s[Z_CHOICE_DIM:])),
                    "spec_choice_mean": float(np.mean(spec_s[:Z_CHOICE_DIM])),
                    "spec_other_mean": float(np.mean(spec_s[Z_CHOICE_DIM:])),
                    "modularity": mod_s,
                    "mediation_mean": float(np.mean(mediation_s)),
                    "mediation_max": float(np.max(mediation_s)),
                },
                "unstructured": {
                    "ces_mean": float(np.mean(ces_u)),
                    "spec_mean": float(np.mean(spec_u)),
                    "modularity": mod_u,
                },
                "random_split_baseline": {
                    "ces_ratio_mean": rand_ces_ratio_mean,
                    "ces_ratio_std": rand_ces_ratio_std,
                },
                "n_neurons": n_neurons,
                "n_trials": len(activity),
            })

        if not per_session_results:
            continue

        # Aggregate across sessions by averaging scalar summaries
        def _avg(key_path):
            keys = key_path.split(".")
            vals = []
            for r in per_session_results:
                v = r
                for k in keys:
                    v = v[k]
                vals.append(v)
            return float(np.mean(vals))

        results_per_region[region] = {
            "n_sessions": len(per_session_results),
            "structured": {
                "ces_choice_mean": _avg("structured.ces_choice_mean"),
                "ces_other_mean": _avg("structured.ces_other_mean"),
                "spec_choice_mean": _avg("structured.spec_choice_mean"),
                "spec_other_mean": _avg("structured.spec_other_mean"),
                "modularity": _avg("structured.modularity"),
                "mediation_mean": _avg("structured.mediation_mean"),
                "mediation_max": _avg("structured.mediation_max"),
            },
            "unstructured": {
                "ces_mean": _avg("unstructured.ces_mean"),
                "spec_mean": _avg("unstructured.spec_mean"),
                "modularity": _avg("unstructured.modularity"),
            },
            "random_split_baseline": {
                "ces_ratio_mean": _avg("random_split_baseline.ces_ratio_mean"),
                "ces_ratio_std": _avg("random_split_baseline.ces_ratio_std"),
            },
            "per_session": per_session_results,
        }

    # Aggregate
    regions = sorted(results_per_region.keys())
    struct_mods = [results_per_region[r]["structured"]["modularity"] for r in regions]
    unstruct_mods = [results_per_region[r]["unstructured"]["modularity"] for r in regions]
    ces_choice_means = [results_per_region[r]["structured"]["ces_choice_mean"] for r in regions]
    ces_other_means = [results_per_region[r]["structured"]["ces_other_mean"] for r in regions]
    rand_ces_ratios = [results_per_region[r]["random_split_baseline"]["ces_ratio_mean"] for r in regions]

    structured_ces_ratio = float(np.mean(ces_choice_means) / max(np.mean(ces_other_means), 1e-12))

    summary = {
        "n_regions": len(regions),
        "hyperparams": {
            "z_choice_dim": Z_CHOICE_DIM,
            "z_other_dim": Z_OTHER_DIM,
            "alpha_choice": ALPHA_CHOICE,
            "beta_kl": BETA_KL,
            "n_epochs": N_EPOCHS,
            "hidden_dim": HIDDEN_DIM,
        },
        "aggregate": {
            "structured_modularity_mean": float(np.mean(struct_mods)),
            "unstructured_modularity_mean": float(np.mean(unstruct_mods)),
            "ces_choice_mean": float(np.mean(ces_choice_means)),
            "ces_other_mean": float(np.mean(ces_other_means)),
            "ces_choice_vs_other_ratio": structured_ces_ratio,
            "random_split_ces_ratio_mean": float(np.mean(rand_ces_ratios)),
            "random_split_ces_ratio_std": float(np.std(rand_ces_ratios)),
            "structured_vs_random_lift": float(structured_ces_ratio / max(np.mean(rand_ces_ratios), 1e-12)),
            "modularity_structured_wins": sum(1 for s, u in zip(struct_mods, unstruct_mods) if s > u),
        },
        "per_region": results_per_region,
    }

    out_path = RESULTS_DIR / f"exp71_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"{datetime.now().isoformat()} Results saved to {out_path}")

    return summary
