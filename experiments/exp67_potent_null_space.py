"""Experiment 67: Potent/null space decomposition.

Motivated by Kaufman et al. (2014) and Gallego et al. (2017): the choice subspace
is the "potent space" for decision output. High-dimensional regions may harbor large
null spaces that contain variance but no causal signal --- explaining why linear
methods (which are dominated by variance in the null space) systematically fail
in these regions.

For each region:
  1. Estimate choice subspace via LDA (top k principal discriminant directions)
     and via VAE encoder weights (z_choice columns).
  2. Project activity onto the choice subspace (potent) and its orthogonal
     complement (null).
  3. Compute:
     a. Fraction of total variance in potent vs null space
     b. Choice decodability from potent-only vs null-only projections
     c. IIA when only potent-space activity is swapped
  4. Correlate null-space variance fraction with alpha (power-law exponent)
     and with optogenetic silencing effect.

The manifold hypothesis predicts:
  - High-dimensional (low-alpha) regions have large null spaces (most variance
    is orthogonal to choice).
  - LDA fails because its discriminant axis is pulled toward the null space
    by high-variance irrelevant dimensions.
  - VAE succeeds because the encoder learns to project through the null space.
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
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Ridge
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp67"
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


class StructuredVAE(nn.Module):
    def __init__(self, n_neurons, z_choice_dim=Z_CHOICE_DIM, z_other_dim=Z_OTHER_DIM,
                 hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.z_choice_dim = z_choice_dim
        self.z_other_dim = z_other_dim
        z_dim = z_choice_dim + z_other_dim
        self.encoder = nn.Sequential(nn.Linear(n_neurons, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)
        self.decoder = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, n_neurons))
        self.choice_head = nn.Linear(z_choice_dim, 2)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z_choice = z[:, :self.z_choice_dim]
        recon = self.decoder(z)
        choice_logits = self.choice_head(z_choice)
        return recon, mu, logvar, choice_logits, z_choice


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
            recon, mu, logvar, choice_logits, _ = model(xb)
            recon_loss = F.mse_loss(recon, xb)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            choice_loss = F.cross_entropy(choice_logits, yb)
            loss = recon_loss + BETA_KL * kl + ALPHA_CHOICE * choice_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def _get_vae_choice_directions(model, device):
    """Extract the encoder projection for z_choice dimensions."""
    with torch.no_grad():
        W1 = model.encoder[0].weight  # (hidden, n_neurons)
        b1 = model.encoder[0].bias
        W2 = model.encoder[2].weight  # (hidden, hidden)
        b2 = model.encoder[2].bias
        W_mu = model.fc_mu.weight[:model.z_choice_dim]  # (z_choice, hidden)
    return W_mu.cpu().numpy()


def _compute_potent_null(activity, choice_labels, subspace_dirs):
    """Decompose activity into potent (choice subspace) and null (complement).

    subspace_dirs: (k, n_neurons) matrix whose rows span the choice subspace.
    Returns variance fractions, decodability, and per-space activities.
    """
    _, _, Vt = np.linalg.svd(subspace_dirs, full_matrices=False)
    U = Vt  # (k, n_neurons) orthonormal rows spanning the choice subspace
    X = activity  # (n_trials, n_neurons)
    X_potent = X @ U.T  # (n_trials, k)
    X_null = X - X_potent @ U  # (n_trials, n_neurons) residual

    var_total = np.var(X, axis=0).sum()
    var_potent = np.var(X_potent, axis=0).sum()
    var_null = np.var(X_null, axis=0).sum()
    frac_potent = var_potent / max(var_total, 1e-12)

    # Choice decodability from potent vs null
    acc_potent = _decode_accuracy(X_potent, choice_labels)
    acc_null = _decode_accuracy(X_null, choice_labels)
    acc_full = _decode_accuracy(X, choice_labels)

    return {
        "var_frac_potent": float(frac_potent),
        "var_frac_null": float(1 - frac_potent),
        "decode_acc_potent": float(acc_potent),
        "decode_acc_null": float(acc_null),
        "decode_acc_full": float(acc_full),
    }


def _decode_accuracy(X, labels):
    """Logistic regression leave-half-out accuracy."""
    n = len(labels)
    if X.shape[1] == 0 or n < 10:
        return 0.5
    idx = np.random.permutation(n)
    mid = n // 2
    clf = LogisticRegression(max_iter=500, solver="lbfgs")
    try:
        clf.fit(X[idx[:mid]], labels[idx[:mid]])
        return float(clf.score(X[idx[mid:]], labels[idx[mid:]]))
    except Exception:
        return 0.5


def _compute_iia_potent_only(activity, choice_labels, subspace_dirs):
    """IIA where only the potent-space component is swapped."""
    _, _, Vt = np.linalg.svd(subspace_dirs, full_matrices=False)
    U = Vt  # (k, n_neurons) orthonormal rows spanning the choice subspace
    X = activity
    X_potent = X @ U.T  # (n, k)
    X_null_component = X - X_potent @ U  # (n, n_neurons)

    left_idx = np.where(choice_labels == 0)[0]
    right_idx = np.where(choice_labels == 1)[0]
    if len(left_idx) < 5 or len(right_idx) < 5:
        return float("nan")

    # Train classifier on full data
    clf = LogisticRegression(max_iter=500, solver="lbfgs")
    clf.fit(X, choice_labels)

    flips = 0
    total = 0
    n_pairs = min(N_IIA_PAIRS, len(left_idx), len(right_idx))
    for i in range(n_pairs):
        li = left_idx[i % len(left_idx)]
        ri = right_idx[i % len(right_idx)]

        # Swap potent component: left trial gets right's potent, keeps own null
        x_swapped = X_null_component[li] + X_potent[ri] @ U
        pred_orig = clf.predict(X[li:li+1])[0]
        pred_swap = clf.predict(x_swapped.reshape(1, -1))[0]
        if pred_orig != pred_swap:
            flips += 1
        total += 1

    return flips / max(total, 1)


def _fit_alpha(activity):
    """Power-law exponent from PCA eigenvalue spectrum."""
    n_components = min(50, activity.shape[1], activity.shape[0] - 1)
    if n_components < 15:
        return float("nan")
    pca = PCA(n_components=n_components)
    pca.fit(activity)
    eigenvalues = pca.explained_variance_
    ranks = np.arange(10, min(50, len(eigenvalues)))
    if len(ranks) < 5:
        return float("nan")
    log_ranks = np.log(ranks + 1)
    log_eig = np.log(eigenvalues[ranks] + 1e-12)
    slope, _ = np.polyfit(log_ranks, log_eig, 1)
    return float(-slope)


def run(max_sessions: int | None = None) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"{datetime.now().isoformat()} Starting potent/null space experiment "
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
            region_data[region].append({
                "session_idx": sess_idx,
                "activity": activity,
                "choice_labels": ch,
                "n_neurons": int(activity.shape[1]),
            })

    logger.info(f"{datetime.now().isoformat()} {len(region_data)} regions loaded")

    results_per_region = {}

    for region in tqdm(sorted(region_data.keys()), desc="Regions"):
        sessions_data = region_data[region]

        per_session_results = []
        for sess in sessions_data:
            activity = sess["activity"]
            ch = sess["choice_labels"]
            n_neurons = sess["n_neurons"]

            if len(activity) < MIN_TRIALS_PER_CONDITION * 2:
                continue

            alpha = _fit_alpha(activity)

            # --- LDA subspace ---
            lda = LinearDiscriminantAnalysis()
            lda.fit(activity, ch)
            lda_dirs = lda.scalings_[:, :min(Z_CHOICE_DIM, lda.scalings_.shape[1])].T
            lda_potent_null = _compute_potent_null(activity, ch, lda_dirs)
            lda_iia_potent = _compute_iia_potent_only(activity, ch, lda_dirs)

            # --- VAE subspace ---
            vae_model = _train_vae(activity, ch, device, n_neurons)
            vae_model.eval()
            with torch.no_grad():
                X_t = torch.tensor(activity, dtype=torch.float32, device=device)
                mu, _ = vae_model.encode(X_t)
                z_choice = mu[:, :Z_CHOICE_DIM].cpu().numpy()
            ridge = Ridge(alpha=1.0)
            ridge.fit(activity, z_choice)
            vae_dirs = ridge.coef_
            vae_potent_null = _compute_potent_null(activity, ch, vae_dirs)
            vae_iia_potent = _compute_iia_potent_only(activity, ch, vae_dirs)

            per_session_results.append({
                "n_trials": int(len(activity)),
                "n_neurons": n_neurons,
                "alpha": alpha,
                "lda": {**lda_potent_null, "iia_potent_only": lda_iia_potent},
                "vae": {**vae_potent_null, "iia_potent_only": vae_iia_potent},
            })

        if not per_session_results:
            continue

        def _mean_field(results, *keys):
            vals = []
            for r in results:
                v = r
                for k in keys:
                    v = v[k]
                if not np.isnan(v):
                    vals.append(v)
            return float(np.mean(vals)) if vals else float("nan")

        results_per_region[region] = {
            "n_sessions": len(per_session_results),
            "alpha": _mean_field(per_session_results, "alpha"),
            "lda": {
                "var_frac_potent": _mean_field(per_session_results, "lda", "var_frac_potent"),
                "var_frac_null": _mean_field(per_session_results, "lda", "var_frac_null"),
                "decode_acc_potent": _mean_field(per_session_results, "lda", "decode_acc_potent"),
                "decode_acc_null": _mean_field(per_session_results, "lda", "decode_acc_null"),
                "decode_acc_full": _mean_field(per_session_results, "lda", "decode_acc_full"),
                "iia_potent_only": _mean_field(per_session_results, "lda", "iia_potent_only"),
            },
            "vae": {
                "var_frac_potent": _mean_field(per_session_results, "vae", "var_frac_potent"),
                "var_frac_null": _mean_field(per_session_results, "vae", "var_frac_null"),
                "decode_acc_potent": _mean_field(per_session_results, "vae", "decode_acc_potent"),
                "decode_acc_null": _mean_field(per_session_results, "vae", "decode_acc_null"),
                "decode_acc_full": _mean_field(per_session_results, "vae", "decode_acc_full"),
                "iia_potent_only": _mean_field(per_session_results, "vae", "iia_potent_only"),
            },
            "per_session": per_session_results,
        }

    # --- Cross-region correlations ---
    regions_with_alpha = [r for r, v in results_per_region.items()
                          if not np.isnan(v["alpha"])]
    alphas = [results_per_region[r]["alpha"] for r in regions_with_alpha]
    lda_null_fracs = [results_per_region[r]["lda"]["var_frac_null"] for r in regions_with_alpha]
    vae_null_fracs = [results_per_region[r]["vae"]["var_frac_null"] for r in regions_with_alpha]
    lda_potent_accs = [results_per_region[r]["lda"]["decode_acc_potent"] for r in regions_with_alpha]
    vae_potent_accs = [results_per_region[r]["vae"]["decode_acc_potent"] for r in regions_with_alpha]

    rho_alpha_lda_null, p_alpha_lda_null = spearmanr(alphas, lda_null_fracs)
    rho_alpha_vae_null, p_alpha_vae_null = spearmanr(alphas, vae_null_fracs)
    rho_alpha_lda_potent_acc, p_alpha_lda_potent_acc = spearmanr(alphas, lda_potent_accs)
    rho_alpha_vae_potent_acc, p_alpha_vae_potent_acc = spearmanr(alphas, vae_potent_accs)

    summary = {
        "n_regions": len(results_per_region),
        "cross_region_correlations": {
            "alpha_vs_lda_null_frac": {"rho": float(rho_alpha_lda_null),
                                       "p": float(p_alpha_lda_null)},
            "alpha_vs_vae_null_frac": {"rho": float(rho_alpha_vae_null),
                                       "p": float(p_alpha_vae_null)},
            "alpha_vs_lda_potent_decode": {"rho": float(rho_alpha_lda_potent_acc),
                                            "p": float(p_alpha_lda_potent_acc)},
            "alpha_vs_vae_potent_decode": {"rho": float(rho_alpha_vae_potent_acc),
                                            "p": float(p_alpha_vae_potent_acc)},
        },
        "mean_lda_null_frac": float(np.mean(lda_null_fracs)),
        "mean_vae_null_frac": float(np.mean(vae_null_fracs)),
        "mean_lda_potent_decode": float(np.mean(lda_potent_accs)),
        "mean_vae_potent_decode": float(np.mean(vae_potent_accs)),
        "per_region": results_per_region,
    }

    out_path = RESULTS_DIR / f"exp67_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"{datetime.now().isoformat()} Results saved to {out_path}")

    return summary
