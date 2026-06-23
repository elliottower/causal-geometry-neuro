"""Experiment 80: iVAE identifiability verification for biological neural data.

Adapted from causal-geometry-grokking/experiments/k1_vae_vs_das.py.

Tests whether the iVAE identifiability conditions (Khemakhem et al. 2020) hold
for our structured VAE / pi-VAE / pi-structured-VAE on Steinmetz choice data.

The key question: WHY does pi-VAE destroy performance (IIA 0.036 vs 0.941)?
Binary choice (n_classes=2) barely meets the rank condition, so we measure:
  1. Spearman correlation between z_causal components and true labels
  2. Empirical rank condition: do per-label means span the causal subspace?
  3. Prior rank condition: do the learned label-conditional prior means differ?
  4. Decoder injectivity: ratio of output distances to latent distances
  5. MCC (mean correlation coefficient): standard disentanglement metric

We test all 4 model variants from exp69:
  A. Structured VAE (our model — Gaussian prior + classification head)
  B. pi-VAE (label-conditioned prior, no split)
  C. pi-structured-VAE hybrid (label-conditioned prior + split)
  D. LDA baseline

Usage:
    modal run modal_run.py --experiment exp80
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

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp80"
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
N_SUFFICIENT_STATS = 4


class LabelConditionedPrior(nn.Module):
    def __init__(self, z_dim, n_labels=2, n_stats=N_SUFFICIENT_STATS):
        super().__init__()
        self.z_dim = z_dim
        self.n_labels = n_labels
        self.n_stats = n_stats
        self.natural_params = nn.Parameter(torch.randn(n_labels, z_dim, n_stats) * 0.1)
        # Expose as prior_mu for identifiability checking
        self.prior_mu = nn.Embedding(n_labels, z_dim)

    def sufficient_statistics(self, z):
        stats = [z]
        for power in range(2, self.n_stats + 1):
            stats.append(z.pow(power))
        return torch.stack(stats, dim=-1)

    def log_prob(self, z, labels):
        T = self.sufficient_statistics(z)
        lam = self.natural_params[labels]
        log_p = (T * lam).sum(dim=(-1, -2))
        log_p = log_p - 0.5 * (z ** 2).sum(dim=-1)
        return log_p


class StructuredVAE(nn.Module):
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
    def __init__(self, n_neurons, z_dim=Z_CHOICE_DIM + Z_OTHER_DIM):
        super().__init__()
        self.z_dim = z_dim
        self.z_choice_dim = z_dim  # for compatibility
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
        log_q = -0.5 * (logvar + 1).sum(dim=-1)
        log_p = self.label_prior.log_prob(z, labels)
        kl = (log_q - log_p).mean()
        return recon_loss + BETA_KL * kl


class PiStructuredVAE(nn.Module):
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
        mu_c = mu[:, :self.z_choice_dim]
        logvar_c = logvar[:, :self.z_choice_dim]
        log_q_choice = -0.5 * (logvar_c + 1).sum(dim=-1)
        log_p_choice = self.label_prior.log_prob(z_choice, labels)
        kl_choice = (log_q_choice - log_p_choice).mean()
        mu_o = mu[:, self.z_choice_dim:]
        logvar_o = logvar[:, self.z_choice_dim:]
        kl_other = -0.5 * torch.mean(1 + logvar_o - mu_o.pow(2) - logvar_o.exp())
        choice_loss = F.cross_entropy(choice_logits, labels)
        return recon_loss + BETA_KL * (kl_choice + kl_other) + ALPHA_CHOICE * choice_loss


def _train_model(model, activity, labels, device="cpu"):
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


def measure_identifiability(model, activity, labels, n_classes, device="cpu"):
    """Measure iVAE identifiability metrics for a trained model.

    Adapted from causal-geometry-grokking measure_identifiability().
    """
    model.eval()
    X = torch.tensor(activity, dtype=torch.float32, device=device)
    y_np = np.asarray(labels)

    with torch.inference_mode():
        mu, logvar = model.encode(X)

    z_choice_dim = model.z_choice_dim
    z = mu[:, :z_choice_dim].cpu().numpy()

    # 1. Spearman correlation between each z component and true label
    spearman_per_dim = []
    for j in range(z.shape[1]):
        rho, _ = spearmanr(z[:, j], y_np)
        spearman_per_dim.append(abs(rho) if not np.isnan(rho) else 0.0)

    # 2. Empirical rank condition: per-label means span the causal subspace
    corr_matrix = np.zeros((z.shape[1], n_classes))
    for c in range(n_classes):
        mask = y_np == c
        if mask.sum() < 2:
            continue
        for j in range(z.shape[1]):
            corr_matrix[j, c] = z[mask, j].mean()

    _, s, _ = np.linalg.svd(corr_matrix, full_matrices=False)
    effective_rank = int((s > s[0] * 1e-5).sum())
    rank_condition_met = effective_rank >= z.shape[1]

    # 3. Prior rank condition (for pi-VAE variants with label_prior)
    has_prior = hasattr(model, 'label_prior')
    rank_condition_prior = False
    prior_rank = 0
    prior_mean_separation = 0.0
    if has_prior:
        prior_mu = model.label_prior.prior_mu.weight.detach().cpu().numpy()
        n_labels = prior_mu.shape[0]
        prior_mean_separation = float(np.linalg.norm(prior_mu[1] - prior_mu[0]))
        if n_labels >= 2:
            diffs = prior_mu[1:] - prior_mu[0:1]
            _, s_prior, _ = np.linalg.svd(diffs, full_matrices=False)
            prior_rank = int((s_prior > s_prior[0] * 1e-5).sum())
            rank_condition_prior = prior_rank >= z.shape[1]

    # 4. Decoder injectivity: ratio of decoder output distances to z distances
    decoder_injectivity = 0.0
    with torch.inference_mode():
        z_sample = mu[:min(200, len(mu))]
        n_pairs = min(500, len(z_sample) * (len(z_sample) - 1) // 2)
        idx_a = torch.randint(0, len(z_sample), (n_pairs,))
        idx_b = torch.randint(0, len(z_sample), (n_pairs,))
        # Use full z (choice + other) for decoder
        za_full = z_sample[idx_a]
        zb_full = z_sample[idx_b]
        dec_a = model.decoder(za_full)
        dec_b = model.decoder(zb_full)
        # Distance in z_choice only
        z_dist = (za_full[:, :z_choice_dim] - zb_full[:, :z_choice_dim]).norm(dim=-1)
        dec_dist = (dec_a - dec_b).norm(dim=-1)
        valid = z_dist > 1e-6
        if valid.sum() > 10:
            ratios = dec_dist[valid] / z_dist[valid]
            decoder_injectivity = float(ratios.mean().item())

    # 5. Reconstruction MSE
    with torch.inference_mode():
        if hasattr(model, 'choice_head'):
            recon, _, _, _, _ = model.forward(X)
        else:
            recon, _, _, _ = model.forward(X)
        recon_mse = float(F.mse_loss(recon, X).item())

    # 6. MCC
    abs_corr = np.abs(corr_matrix)
    row_max = abs_corr.max(axis=1)
    mcc = float(np.mean(row_max))

    # 7. Label separation in z-space
    if n_classes == 2:
        z_left = z[y_np == 0]
        z_right = z[y_np == 1]
        z_separation = float(np.linalg.norm(z_left.mean(axis=0) - z_right.mean(axis=0)))
    else:
        z_separation = 0.0

    return {
        "max_spearman": float(max(spearman_per_dim)),
        "mean_spearman": float(np.mean(spearman_per_dim)),
        "spearman_per_dim": [round(x, 4) for x in spearman_per_dim],
        "rank_condition_empirical": bool(rank_condition_met),
        "effective_rank": effective_rank,
        "n_components": z.shape[1],
        "has_label_conditional_prior": has_prior,
        "rank_condition_prior": bool(rank_condition_prior),
        "prior_rank": prior_rank if has_prior else None,
        "prior_mean_separation": round(prior_mean_separation, 4) if has_prior else None,
        "decoder_injectivity_ratio": round(decoder_injectivity, 4),
        "reconstruction_mse": round(recon_mse, 6),
        "mcc": round(mcc, 4),
        "z_label_separation": round(z_separation, 4),
        "n_classes": n_classes,
    }


def run(max_sessions: int | None = None) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]
    n_classes = 2

    logger.info(f"{datetime.now().isoformat()} Starting iVAE verification with "
                f"{len(sessions)} sessions on {device}")

    region_data: dict[str, list[dict]] = {}
    for sess in tqdm(sessions, desc="Loading sessions"):
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
            left_count = (ch == 0).sum()
            right_count = (ch == 1).sum()
            if left_count < MIN_TRIALS_PER_CONDITION or right_count < MIN_TRIALS_PER_CONDITION:
                continue
            if region not in region_data:
                region_data[region] = []
            region_data[region].append({"activity": activity, "choice_labels": ch,
                                        "n_neurons": int(activity.shape[1])})

    logger.info(f"{datetime.now().isoformat()} {len(region_data)} regions loaded")

    all_results = {}
    for region in tqdm(sorted(region_data.keys()), desc="iVAE verification"):
        sessions_for_region = region_data[region]
        per_session_results = []

        for sess in sessions_for_region:
            activity = sess["activity"]
            ch = sess["choice_labels"]
            n_neurons = sess["n_neurons"]

            sess_result = {
                "n_neurons": n_neurons,
                "n_trials": len(ch),
                "n_left": int((ch == 0).sum()),
                "n_right": int((ch == 1).sum()),
            }

            for name, ModelClass, mkwargs in [
                ("structured_vae", StructuredVAE, {"n_neurons": n_neurons}),
                ("pi_vae", PiVAE, {"n_neurons": n_neurons}),
                ("pi_structured_vae", PiStructuredVAE, {"n_neurons": n_neurons}),
            ]:
                model = ModelClass(**mkwargs).to(device)
                model = _train_model(model, activity, ch, device)
                metrics = measure_identifiability(model, activity, ch, n_classes, device)
                sess_result[name] = metrics
                logger.info(f"  {region} {name}: max_rho={metrics['max_spearman']:.3f} "
                            f"rank={metrics['effective_rank']}/{metrics['n_components']} "
                            f"inj={metrics['decoder_injectivity_ratio']:.3f} "
                            f"sep={metrics['z_label_separation']:.3f} "
                            f"mse={metrics['reconstruction_mse']:.4f}")
                del model

            per_session_results.append(sess_result)

        all_results[region] = per_session_results

    # Aggregate across all (region, session) pairs
    aggregate = {}
    for variant in ["structured_vae", "pi_vae", "pi_structured_vae"]:
        variant_metrics = []
        for region_sessions in all_results.values():
            for sess_result in region_sessions:
                if variant in sess_result:
                    variant_metrics.append(sess_result[variant])
        if not variant_metrics:
            continue
        aggregate[variant] = {
            "mean_max_spearman": round(np.mean([m["max_spearman"] for m in variant_metrics]), 4),
            "mean_effective_rank": round(np.mean([m["effective_rank"] for m in variant_metrics]), 2),
            "rank_condition_met_frac": round(np.mean([m["rank_condition_empirical"] for m in variant_metrics]), 4),
            "mean_decoder_injectivity": round(np.mean([m["decoder_injectivity_ratio"] for m in variant_metrics]), 4),
            "mean_recon_mse": round(np.mean([m["reconstruction_mse"] for m in variant_metrics]), 6),
            "mean_mcc": round(np.mean([m["mcc"] for m in variant_metrics]), 4),
            "mean_z_separation": round(np.mean([m["z_label_separation"] for m in variant_metrics]), 4),
            "n_observations": len(variant_metrics),
        }
        if variant in ("pi_vae", "pi_structured_vae"):
            prior_metrics = [m for m in variant_metrics if m.get("prior_rank") is not None]
            if prior_metrics:
                aggregate[variant]["prior_rank_condition_met_frac"] = round(
                    np.mean([m["rank_condition_prior"] for m in prior_metrics]), 4)
                aggregate[variant]["mean_prior_rank"] = round(
                    np.mean([m["prior_rank"] for m in prior_metrics]), 2)
                aggregate[variant]["mean_prior_mean_separation"] = round(
                    np.mean([m["prior_mean_separation"] for m in prior_metrics]), 4)

    logger.info(f"\n{'=' * 80}")
    logger.info(f"iVAE Identifiability Summary ({len(all_results)} regions, n_classes={n_classes})")
    logger.info(f"{'Variant':>25s} {'max_rho':>8s} {'rank':>6s} {'rank_ok':>8s} "
                f"{'dec_inj':>8s} {'recon':>8s} {'MCC':>6s} {'z_sep':>6s}")
    logger.info(f"{'-' * 80}")
    for variant, agg in aggregate.items():
        logger.info(f"{variant:>25s} {agg['mean_max_spearman']:>8.4f} "
                    f"{agg['mean_effective_rank']:>6.2f} {agg['rank_condition_met_frac']:>8.4f} "
                    f"{agg['mean_decoder_injectivity']:>8.4f} {agg['mean_recon_mse']:>8.4f} "
                    f"{agg['mean_mcc']:>6.4f} {agg['mean_z_separation']:>6.4f}")
    logger.info(f"{'=' * 80}")

    return {
        "n_regions": len(all_results),
        "n_classes": n_classes,
        "z_choice_dim": Z_CHOICE_DIM,
        "aggregate": aggregate,
        "per_region": all_results,
    }
