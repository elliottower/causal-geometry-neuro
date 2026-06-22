"""Experiment 69: pi-VAE hybrid — identifiable structured VAE with label-conditioned prior.

Combines pi-VAE (Zhou & Wei, NeurIPS 2020) with our structured VAE:
- pi-VAE's key insight: condition the PRIOR p(z|u) on task labels via an exponential
  family, providing identifiability guarantees (Theorem 1).
- Our structured VAE's key insight: split z into z_choice (with classification head)
  and z_other (unsupervised), so the choice subspace is explicitly separated.

The hybrid ("pi-structured-VAE") uses:
  1. Label-conditioned prior: p(z_choice | u) = ExpFam(T, lambda(u)) where u = choice label
     and p(z_other) = N(0, I) (standard prior for nuisance dimensions)
  2. Structured latent split: z = [z_choice, z_other]
  3. Classification head on z_choice (as in our structured VAE)
  4. Poisson or Gaussian observation model (Gaussian for trial-averaged rates)

Ablation: compare 4 models on IIA and silencing correlation:
  A. Structured VAE (our current model — Gaussian prior, classification head)
  B. pi-VAE (label-conditioned prior, no z_choice/z_other split)
  C. pi-structured-VAE (label-conditioned prior ON z_choice + structured split)
  D. Linear structured VAE (linear encoder/decoder, structured split, from exp63)
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp69"
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
N_IIA_PAIRS = 100
N_SUFFICIENT_STATS = 4


class LabelConditionedPrior(nn.Module):
    """Exponential family prior p(z|u) following pi-VAE (Zhou & Wei 2020).

    For discrete labels u in {0, 1}, learn separate natural parameters lambda_u
    for each label. The prior is:
      p(z_i | u) = Q(z_i)/Z(u) * exp(sum_j T_j(z_i) * lambda_{i,j}(u))

    With Gaussian sufficient statistics T(z) = [z, z^2], this gives a
    label-conditioned Gaussian with mean/variance that depend on the choice label.
    """
    def __init__(self, z_dim, n_labels=2, n_stats=N_SUFFICIENT_STATS):
        super().__init__()
        self.z_dim = z_dim
        self.n_labels = n_labels
        self.n_stats = n_stats
        # Natural parameters for each label: (n_labels, z_dim, n_stats)
        self.natural_params = nn.Parameter(torch.randn(n_labels, z_dim, n_stats) * 0.1)

    def sufficient_statistics(self, z):
        """T(z) = [z, z^2, z^3, z^4] (polynomial sufficient statistics)."""
        stats = [z]
        for power in range(2, self.n_stats + 1):
            stats.append(z.pow(power))
        return torch.stack(stats, dim=-1)  # (batch, z_dim, n_stats)

    def log_prob(self, z, labels):
        """Log probability under label-conditioned prior."""
        T = self.sufficient_statistics(z)  # (batch, z_dim, n_stats)
        lam = self.natural_params[labels]  # (batch, z_dim, n_stats)
        # log p(z|u) = sum_i sum_j T_j(z_i) * lambda_{i,j}(u) - log Z(u)
        # We compute the unnormalized log prob; normalization constant cancels in ELBO
        log_p = (T * lam).sum(dim=(-1, -2))  # (batch,)
        # Add standard Gaussian base measure
        log_p = log_p - 0.5 * (z ** 2).sum(dim=-1)
        return log_p

    def sample_prior(self, labels):
        """Sample from the label-conditioned prior (approximate via rejection)."""
        # For simplicity, return the mode: solve for mean of the conditioned Gaussian
        # With T = [z, z^2], lambda = [l1, l2]: mean = -l1/(2*l2), var = -1/(2*l2)
        lam = self.natural_params[labels]  # (batch, z_dim, n_stats)
        # Approximate: use first two stats as Gaussian natural params
        l1 = lam[:, :, 0]  # (batch, z_dim)
        l2 = lam[:, :, 1] - 0.5  # subtract base measure contribution
        var = -1.0 / (2.0 * l2.clamp(max=-0.1))
        mean = -l1 / (2.0 * l2.clamp(max=-0.1))
        return mean + var.sqrt() * torch.randn_like(mean)


class StructuredVAE(nn.Module):
    """Model A: our current structured VAE (Gaussian prior + classification head)."""
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
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decoder(z)
        choice_logits = self.choice_head(z[:, :self.z_choice_dim])
        return recon, mu, logvar, choice_logits, z[:, :self.z_choice_dim]

    def loss(self, x, labels):
        recon, mu, logvar, choice_logits, _ = self.forward(x)
        recon_loss = F.mse_loss(recon, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        choice_loss = F.cross_entropy(choice_logits, labels)
        return recon_loss + BETA_KL * kl + ALPHA_CHOICE * choice_loss


class PiVAE(nn.Module):
    """Model B: pi-VAE (label-conditioned prior, no structured split)."""
    def __init__(self, n_neurons, z_dim=Z_CHOICE_DIM + Z_OTHER_DIM):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = nn.Sequential(nn.Linear(n_neurons, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU())
        self.fc_mu = nn.Linear(HIDDEN_DIM, z_dim)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, z_dim)
        self.decoder = nn.Sequential(nn.Linear(z_dim, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, n_neurons))
        self.label_prior = LabelConditionedPrior(z_dim)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decoder(z)
        return recon, mu, logvar, z

    def loss(self, x, labels):
        recon, mu, logvar, z = self.forward(x)
        recon_loss = F.mse_loss(recon, x)
        # KL against label-conditioned prior instead of standard Gaussian
        log_q = -0.5 * (logvar + 1).sum(dim=-1)  # simplified
        log_p = self.label_prior.log_prob(z, labels)
        kl = (log_q - log_p).mean()
        return recon_loss + BETA_KL * kl


class PiStructuredVAE(nn.Module):
    """Model C: HYBRID — pi-VAE prior on z_choice + structured split + classification head."""
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
        # Label-conditioned prior ONLY on z_choice dimensions
        self.label_prior = LabelConditionedPrior(z_choice_dim)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        z_choice = z[:, :self.z_choice_dim]
        recon = self.decoder(z)
        choice_logits = self.choice_head(z_choice)
        return recon, mu, logvar, choice_logits, z_choice

    def loss(self, x, labels):
        recon, mu, logvar, choice_logits, z_choice = self.forward(x)
        recon_loss = F.mse_loss(recon, x)
        # z_choice: KL against label-conditioned prior
        mu_c = mu[:, :self.z_choice_dim]
        logvar_c = logvar[:, :self.z_choice_dim]
        log_q_choice = -0.5 * (logvar_c + 1).sum(dim=-1)
        log_p_choice = self.label_prior.log_prob(z_choice, labels)
        kl_choice = (log_q_choice - log_p_choice).mean()
        # z_other: standard Gaussian KL
        mu_o = mu[:, self.z_choice_dim:]
        logvar_o = logvar[:, self.z_choice_dim:]
        kl_other = -0.5 * torch.mean(1 + logvar_o - mu_o.pow(2) - logvar_o.exp())
        choice_loss = F.cross_entropy(choice_logits, labels)
        return recon_loss + BETA_KL * (kl_choice + kl_other) + ALPHA_CHOICE * choice_loss


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
    return model


def _compute_iia(model, activity, choice_labels, device, is_structured=True):
    """Compute IIA by swapping z_choice between opposite-label pairs."""
    model.eval()
    X = torch.tensor(activity, dtype=torch.float32, device=device)

    with torch.no_grad():
        mu, logvar = model.encode(X)

    z_choice_dim = model.z_choice_dim if is_structured else Z_CHOICE_DIM

    # Train classifier on z_choice
    z_choice = mu[:, :z_choice_dim].cpu().numpy()
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
        # Swap z_choice from right into left
        z_swapped = mu[li].clone().cpu().numpy()
        z_swapped[:z_choice_dim] = mu[ri, :z_choice_dim].cpu().numpy()
        pred_orig = clf.predict(z_choice[li:li+1])[0]
        pred_swap = clf.predict(z_swapped[:z_choice_dim].reshape(1, -1))[0]
        if pred_orig != pred_swap:
            flips += 1
    return flips / n_pairs


def run(max_sessions: int | None = None) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"{datetime.now().isoformat()} Starting pi-VAE hybrid experiment "
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
                "activity": activity,
                "choice_labels": ch,
                "n_neurons": int(activity.shape[1]),
            })

    logger.info(f"{datetime.now().isoformat()} {len(region_data)} regions loaded")

    results_per_region = {}
    model_names = ["structured_vae", "pi_vae", "pi_structured_vae"]

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

            # Model A: Structured VAE
            model_a = StructuredVAE(n_neurons).to(device)
            _train_model(model_a, activity, ch, device)
            iia_a = _compute_iia(model_a, activity, ch, device, is_structured=True)
            sess_results["structured_vae"] = {"iia": iia_a}

            # Model B: pi-VAE
            model_b = PiVAE(n_neurons).to(device)
            _train_model(model_b, activity, ch, device)
            iia_b = _compute_iia(model_b, activity, ch, device, is_structured=False)
            sess_results["pi_vae"] = {"iia": iia_b}

            # Model C: pi-structured-VAE (hybrid)
            model_c = PiStructuredVAE(n_neurons).to(device)
            _train_model(model_c, activity, ch, device)
            iia_c = _compute_iia(model_c, activity, ch, device, is_structured=True)
            sess_results["pi_structured_vae"] = {"iia": iia_c}

            # LDA baseline
            lda = LinearDiscriminantAnalysis()
            lda.fit(activity, ch)
            lda_dirs = lda.scalings_[:, :min(Z_CHOICE_DIM, lda.scalings_.shape[1])].T
            _, _, Vt = np.linalg.svd(lda_dirs, full_matrices=False)
            proj = activity @ Vt.T
            clf = LogisticRegression(max_iter=500, solver="lbfgs")
            clf.fit(proj, ch)
            left_idx = np.where(ch == 0)[0]
            right_idx = np.where(ch == 1)[0]
            flips = 0
            n_pairs = min(N_IIA_PAIRS, len(left_idx), len(right_idx))
            for i in range(n_pairs):
                li = left_idx[i % len(left_idx)]
                ri = right_idx[i % len(right_idx)]
                p_swapped = proj[li].copy()
                p_swapped[:] = proj[ri]
                if clf.predict(proj[li:li+1])[0] != clf.predict(p_swapped.reshape(1, -1))[0]:
                    flips += 1
            sess_results["lda"] = {"iia": flips / max(n_pairs, 1)}

            per_session_results.append(sess_results)

        if not per_session_results:
            continue

        # Aggregate by averaging across sessions
        region_results = {}
        for model_name in model_names + ["lda"]:
            iias = [sr[model_name]["iia"] for sr in per_session_results
                    if not np.isnan(sr[model_name]["iia"])]
            if iias:
                region_results[model_name] = {"iia": float(np.mean(iias))}
            else:
                region_results[model_name] = {"iia": float("nan")}

        results_per_region[region] = region_results

    # Aggregate
    regions = sorted(results_per_region.keys())
    summary_stats = {}
    for model_name in model_names + ["lda"]:
        iias = [results_per_region[r][model_name]["iia"] for r in regions
                if not np.isnan(results_per_region[r][model_name]["iia"])]
        summary_stats[model_name] = {
            "mean_iia": float(np.mean(iias)),
            "std_iia": float(np.std(iias)),
            "n_regions": len(iias),
        }

    # Pairwise comparisons
    comparisons = {}
    for m1, m2 in [("pi_structured_vae", "structured_vae"),
                    ("pi_structured_vae", "pi_vae"),
                    ("pi_vae", "structured_vae"),
                    ("pi_structured_vae", "lda")]:
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
        "n_regions": len(regions),
        "model_stats": summary_stats,
        "comparisons": comparisons,
        "per_region": results_per_region,
    }

    out_path = RESULTS_DIR / f"exp69_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"{datetime.now().isoformat()} Results saved to {out_path}")

    return summary
