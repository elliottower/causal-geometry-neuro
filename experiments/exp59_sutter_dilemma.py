"""Experiment 59: Sutter et al. nonlinear representation dilemma — killer experiment.

Implements the central test from Sutter et al. (NeurIPS 2025): unconstrained nonlinear
alignment maps make IIA vacuous because they can fit ANY data, including random noise.
We compare three alignment methods on BOTH real neural data AND random Gaussian noise
to show that structured VAE is nonlinear yet non-vacuous.

Three methods:
1. Unconstrained MLP: 3-layer MLP encoder -> z -> classifier. No structural constraints.
   Expected: high IIA on real AND random data (vacuous, replicating Sutter's result).
2. Linear DAS (PCA + LDA): Standard linear baseline.
   Expected: high IIA on real data, chance on random (linearity prevents overfitting).
3. Structured VAE (from exp57): Semi-supervised VAE with z_choice / z_other.
   Expected: high IIA on real data, chance on random (structure prevents vacuousness).

For each method x data condition, IIA is computed by swapping the "choice-relevant"
representation between opposite-evidence trial pairs and measuring classifier flip rate.
If a method achieves high IIA on random Gaussian noise (with real labels), it is
vacuously fitting the labels and provides no evidence of causal structure.
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
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp59_sutter"
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
MIN_TRIALS_PER_CONDITION = 20

# Shared hyperparameters
Z_DIM = 3
HIDDEN_DIM = 128
N_EPOCHS = 300
BATCH_SIZE = 64
LR = 1e-3

# VAE-specific
Z_OTHER_DIM = 15
BETA_KL = 1.0
ALPHA_CHOICE = 10.0

# MLP-specific
MLP_RECON_WEIGHT = 1.0
MLP_CLASS_WEIGHT = 10.0

N_IIA_PAIRS = 100


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class UnconstrainedMLP(nn.Module):
    """3-layer MLP encoder with no structural constraints.

    Encodes activity -> z (low dim), trained with reconstruction + classification.
    This is the "vacuous" method: enough capacity to fit arbitrary label structure.
    """

    def __init__(self, n_neurons: int, z_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_neurons, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, z_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_neurons),
        )
        self.classifier = nn.Linear(z_dim, 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        logits = self.classifier(z)
        return {"z": z, "x_hat": x_hat, "logits": logits}


class StructuredVAE(nn.Module):
    """Semi-supervised VAE with supervised z_choice and unsupervised z_other.

    Same architecture as exp57. The structural constraint (KL divergence, separate
    latent groups) prevents vacuous fitting of arbitrary label structure.
    """

    def __init__(self, n_neurons: int, z_choice_dim: int, z_other_dim: int, hidden_dim: int):
        super().__init__()
        self.z_choice_dim = z_choice_dim
        self.z_other_dim = z_other_dim
        z_dim = z_choice_dim + z_other_dim

        self.enc_trunk = nn.Sequential(
            nn.Linear(n_neurons, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.enc_choice_mu = nn.Linear(hidden_dim, z_choice_dim)
        self.enc_choice_logvar = nn.Linear(hidden_dim, z_choice_dim)
        self.enc_other_mu = nn.Linear(hidden_dim, z_other_dim)
        self.enc_other_logvar = nn.Linear(hidden_dim, z_other_dim)

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_neurons),
        )
        self.choice_classifier = nn.Linear(z_choice_dim, 2)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.enc_trunk(x)
        return (
            self.enc_choice_mu(h),
            self.enc_choice_logvar(h),
            self.enc_other_mu(h),
            self.enc_other_logvar(h),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z_choice: torch.Tensor, z_other: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z_choice, z_other], dim=-1)
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu_c, logvar_c, mu_o, logvar_o = self.encode(x)
        z_choice = self.reparameterize(mu_c, logvar_c)
        z_other = self.reparameterize(mu_o, logvar_o)
        x_hat = self.decode(z_choice, z_other)
        choice_logits = self.choice_classifier(z_choice)
        return {
            "x_hat": x_hat,
            "mu_choice": mu_c,
            "logvar_choice": logvar_c,
            "mu_other": mu_o,
            "logvar_other": logvar_o,
            "z_choice": z_choice,
            "z_other": z_other,
            "choice_logits": choice_logits,
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _zscore(activity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = activity.mean(axis=0, keepdims=True)
    std = activity.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (activity - mu) / std, mu, std


def train_unconstrained_mlp(
    activity: np.ndarray,
    choice_labels: np.ndarray,
    z_dim: int = Z_DIM,
    hidden_dim: int = HIDDEN_DIM,
    n_epochs: int = N_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    device: str = "cpu",
) -> dict:
    n_trials, n_neurons = activity.shape
    activity_norm, _, _ = _zscore(activity)

    X = torch.tensor(activity_norm, dtype=torch.float32, device=device)
    y = torch.tensor(choice_labels, dtype=torch.long, device=device)

    model = UnconstrainedMLP(n_neurons, z_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    effective_batch = min(batch_size, n_trials)

    for _epoch in range(n_epochs):
        perm = torch.randperm(n_trials, device=device)
        for start in range(0, n_trials, effective_batch):
            idx = perm[start:start + effective_batch]
            out = model(X[idx])
            recon = F.mse_loss(out["x_hat"], X[idx])
            cls = F.cross_entropy(out["logits"], y[idx])
            loss = MLP_RECON_WEIGHT * recon + MLP_CLASS_WEIGHT * cls
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        out_full = model(X)
        acc = (out_full["logits"].argmax(dim=1) == y).float().mean().item()
    return {"model": model, "choice_accuracy": acc}


def train_structured_vae(
    activity: np.ndarray,
    choice_labels: np.ndarray,
    z_choice_dim: int = Z_DIM,
    z_other_dim: int = Z_OTHER_DIM,
    hidden_dim: int = HIDDEN_DIM,
    n_epochs: int = N_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    beta_kl: float = BETA_KL,
    alpha_choice: float = ALPHA_CHOICE,
    device: str = "cpu",
) -> dict:
    n_trials, n_neurons = activity.shape
    activity_norm, _, _ = _zscore(activity)

    X = torch.tensor(activity_norm, dtype=torch.float32, device=device)
    y = torch.tensor(choice_labels, dtype=torch.long, device=device)

    model = StructuredVAE(n_neurons, z_choice_dim, z_other_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    effective_batch = min(batch_size, n_trials)

    for _epoch in range(n_epochs):
        perm = torch.randperm(n_trials, device=device)
        for start in range(0, n_trials, effective_batch):
            idx = perm[start:start + effective_batch]
            out = model(X[idx])

            recon = F.mse_loss(out["x_hat"], X[idx])
            kl_choice = -0.5 * torch.mean(
                1 + out["logvar_choice"] - out["mu_choice"].pow(2) - out["logvar_choice"].exp()
            )
            kl_other = -0.5 * torch.mean(
                1 + out["logvar_other"] - out["mu_other"].pow(2) - out["logvar_other"].exp()
            )
            cls = F.cross_entropy(out["choice_logits"], y[idx])
            loss = recon + beta_kl * (kl_choice + kl_other) + alpha_choice * cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        out_full = model(X)
        acc = (out_full["choice_logits"].argmax(dim=1) == y).float().mean().item()
    return {"model": model, "choice_accuracy": acc}


# ---------------------------------------------------------------------------
# IIA computation
# ---------------------------------------------------------------------------

def _compute_iia_latent(
    model: nn.Module,
    activity_norm: torch.Tensor,
    evidence_labels: np.ndarray,
    choice_labels: np.ndarray,
    method: str,
    n_pairs: int = N_IIA_PAIRS,
) -> float | None:
    """IIA via latent-space swap for MLP or VAE.

    For each opposite-evidence trial pair:
    1. Encode both trials to get latent z
    2. Swap the choice-relevant component of z
    3. Decode back (or classify from z) and check if prediction flips

    For unconstrained MLP: swap all of z (there is no "other" component).
    For structured VAE: swap only z_choice, keep z_other fixed.
    """
    left_idx = np.where(evidence_labels == 0)[0]
    right_idx = np.where(evidence_labels == 1)[0]
    if len(left_idx) < MIN_TRIALS_PER_CONDITION or len(right_idx) < MIN_TRIALS_PER_CONDITION:
        return None

    n_pairs_actual = min(n_pairs, len(left_idx), len(right_idx))
    left_sample = left_idx[np.random.choice(len(left_idx), n_pairs_actual, replace=False)]
    right_sample = right_idx[np.random.choice(len(right_idx), n_pairs_actual, replace=False)]

    flips = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for li, ri in zip(left_sample, right_sample):
            x_l = activity_norm[li].unsqueeze(0)
            x_r = activity_norm[ri].unsqueeze(0)

            if method == "mlp":
                out_l = model(x_l)
                out_r = model(x_r)
                z_l = out_l["z"]
                z_r = out_r["z"]

                orig_pred_l = out_l["logits"].argmax(dim=1).item()
                orig_pred_r = out_r["logits"].argmax(dim=1).item()

                # Swap all of z (no structure to preserve)
                swap_logits_l = model.classifier(z_r)
                swap_logits_r = model.classifier(z_l)
                swap_pred_l = swap_logits_l.argmax(dim=1).item()
                swap_pred_r = swap_logits_r.argmax(dim=1).item()

            elif method == "vae":
                mu_c_l, _, mu_o_l, _ = model.encode(x_l)
                mu_c_r, _, mu_o_r, _ = model.encode(x_r)

                orig_logits_l = model.choice_classifier(mu_c_l)
                orig_logits_r = model.choice_classifier(mu_c_r)
                orig_pred_l = orig_logits_l.argmax(dim=1).item()
                orig_pred_r = orig_logits_r.argmax(dim=1).item()

                # Swap z_choice only, keep z_other fixed
                swap_logits_l = model.choice_classifier(mu_c_r)
                swap_logits_r = model.choice_classifier(mu_c_l)
                swap_pred_l = swap_logits_l.argmax(dim=1).item()
                swap_pred_r = swap_logits_r.argmax(dim=1).item()

            else:
                raise ValueError(f"Unknown method: {method}")

            if swap_pred_l != orig_pred_l:
                flips += 1
            if swap_pred_r != orig_pred_r:
                flips += 1
            total += 2

    return float(flips / total) if total > 0 else None


def _compute_iia_linear(
    activity: np.ndarray,
    evidence_labels: np.ndarray,
    choice_labels: np.ndarray,
    V: np.ndarray,
    n_pairs: int = N_IIA_PAIRS,
) -> float | None:
    """IIA using linear subspace projection (DAS baseline).

    Swap the projection onto V between opposite-evidence trial pairs,
    measure classifier flip rate.
    """
    left_idx = np.where(evidence_labels == 0)[0]
    right_idx = np.where(evidence_labels == 1)[0]
    if len(left_idx) < MIN_TRIALS_PER_CONDITION or len(right_idx) < MIN_TRIALS_PER_CONDITION:
        return None

    lda = LinearDiscriminantAnalysis()
    try:
        lda.fit(activity, choice_labels)
    except Exception:
        return None

    n_pairs_actual = min(n_pairs, len(left_idx), len(right_idx))
    left_sample = left_idx[np.random.choice(len(left_idx), n_pairs_actual, replace=False)]
    right_sample = right_idx[np.random.choice(len(right_idx), n_pairs_actual, replace=False)]

    flips = 0
    total = 0
    for li, ri in zip(left_sample, right_sample):
        act_l = activity[li].copy()
        act_r = activity[ri].copy()
        proj_l = V @ (V.T @ act_l)
        proj_r = V @ (V.T @ act_r)
        act_l_swapped = act_l - proj_l + proj_r
        act_r_swapped = act_r - proj_r + proj_l

        orig_pred_l = lda.predict(act_l.reshape(1, -1))[0]
        orig_pred_r = lda.predict(act_r.reshape(1, -1))[0]
        swap_pred_l = lda.predict(act_l_swapped.reshape(1, -1))[0]
        swap_pred_r = lda.predict(act_r_swapped.reshape(1, -1))[0]

        if swap_pred_l != orig_pred_l:
            flips += 1
        if swap_pred_r != orig_pred_r:
            flips += 1
        total += 2

    return float(flips / total) if total > 0 else None


def _estimate_lda_subspace(activity: np.ndarray, labels: np.ndarray,
                           n_dims: int = 5) -> np.ndarray | None:
    """LDA+PCA baseline subspace."""
    n_dims = min(n_dims, activity.shape[1] - 1, activity.shape[0] - 2)
    if n_dims < 1 or len(np.unique(labels)) < 2:
        return None
    pca_dim = min(20, activity.shape[1] - 1, activity.shape[0] - 1)
    pca = PCA(n_components=pca_dim)
    scores = pca.fit_transform(activity)
    lda = LinearDiscriminantAnalysis()
    try:
        lda.fit(scores, labels)
    except Exception:
        return None
    lda_dir = lda.coef_[0]
    lda_dir = lda_dir / (np.linalg.norm(lda_dir) + 1e-10)
    lda_neuron = pca.components_.T @ lda_dir
    pca_components = pca.components_[:n_dims].T
    combined = np.column_stack([lda_neuron.reshape(-1, 1), pca_components])
    Q, _ = np.linalg.qr(combined)
    return Q[:, :n_dims]


# ---------------------------------------------------------------------------
# Contrast-to-evidence helper
# ---------------------------------------------------------------------------

def _contrast_to_evidence_label(sess: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    cl = sess.get("contrast_left", np.array([]))
    cr = sess.get("contrast_right", np.array([]))
    if len(cl) == 0 or len(cr) == 0:
        return None, None
    n = sess["spks"].shape[2]
    n = min(n, len(cl), len(cr))
    cl, cr = cl[:n], cr[:n]
    evidence = cr - cl
    nonzero = evidence != 0
    if nonzero.sum() < MIN_TRIALS_PER_CONDITION:
        return None, None
    labels = np.full(n, -1, dtype=int)
    labels[evidence > 0] = 1
    labels[evidence < 0] = 0
    return labels, evidence


# ---------------------------------------------------------------------------
# Per-region evaluation for one method x data condition
# ---------------------------------------------------------------------------

def _evaluate_method(
    activity: np.ndarray,
    choice_labels: np.ndarray,
    evidence_labels: np.ndarray,
    method: str,
    device: str,
) -> dict:
    """Train and evaluate one method on one dataset. Returns IIA and accuracy."""
    n_neurons = activity.shape[1]
    z_dim = min(Z_DIM, n_neurons // 5, n_neurons - 1)
    if z_dim < 1:
        return {"iia": None, "choice_accuracy": None, "error": "too few neurons for z_dim"}

    hidden = min(HIDDEN_DIM, n_neurons * 2)

    if method == "mlp":
        try:
            result = train_unconstrained_mlp(
                activity, choice_labels, z_dim=z_dim, hidden_dim=hidden, device=device,
            )
        except Exception as e:
            return {"iia": None, "choice_accuracy": None, "error": str(e)}

        activity_norm, _, _ = _zscore(activity)
        X = torch.tensor(activity_norm, dtype=torch.float32, device=device)
        iia = _compute_iia_latent(result["model"], X, evidence_labels, choice_labels, "mlp")
        return {"iia": iia, "choice_accuracy": result["choice_accuracy"]}

    elif method == "linear_das":
        V = _estimate_lda_subspace(activity, evidence_labels, n_dims=z_dim)
        if V is None:
            return {"iia": None, "choice_accuracy": None, "error": "LDA subspace estimation failed"}
        iia = _compute_iia_linear(activity, evidence_labels, choice_labels, V)

        # Choice accuracy for linear DAS: LDA classifier accuracy
        lda = LinearDiscriminantAnalysis()
        try:
            lda.fit(activity, choice_labels)
            preds = lda.predict(activity)
            acc = float(np.mean(preds == choice_labels))
        except Exception:
            acc = None
        return {"iia": iia, "choice_accuracy": acc}

    elif method == "vae":
        z_other = min(Z_OTHER_DIM, n_neurons // 3, n_neurons - z_dim - 1)
        if z_other < 1:
            return {"iia": None, "choice_accuracy": None, "error": "too few neurons for z_other"}

        try:
            result = train_structured_vae(
                activity, choice_labels, z_choice_dim=z_dim, z_other_dim=z_other,
                hidden_dim=hidden, device=device,
            )
        except Exception as e:
            return {"iia": None, "choice_accuracy": None, "error": str(e)}

        activity_norm, _, _ = _zscore(activity)
        X = torch.tensor(activity_norm, dtype=torch.float32, device=device)
        iia = _compute_iia_latent(result["model"], X, evidence_labels, choice_labels, "vae")
        return {"iia": iia, "choice_accuracy": result["choice_accuracy"]}

    else:
        raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

METHODS = ["mlp", "linear_das", "vae"]
DATA_CONDITIONS = ["real", "random"]


def run(max_sessions: int | None = None) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"{datetime.now().isoformat()} Starting Sutter dilemma experiment "
                f"with {len(sessions)} sessions on {device}")

    # --- Load data ---
    region_data: dict[str, list[dict]] = {}
    for sess_idx, sess in enumerate(tqdm(sessions, desc="Loading sessions")):
        choice_labels = get_choice_labels(sess)
        if len(np.unique(choice_labels)) < 2:
            continue
        evidence_labels, _evidence_values = _contrast_to_evidence_label(sess)
        if evidence_labels is None:
            continue
        mouse = str(sess.get("mouse_name", f"mouse_{sess_idx}"))
        regions = list_regions(sess, min_neurons=MIN_NEURONS)

        for region in regions:
            act = get_region_activity(sess, region)
            if act is None or act.shape[1] < MIN_NEURONS:
                continue
            n = min(act.shape[0], len(choice_labels), len(evidence_labels))
            activity = act[:n, :, TIME_WINDOW].mean(axis=2)
            ch = choice_labels[:n]
            ev = evidence_labels[:n]

            valid = ev >= 0
            if valid.sum() < MIN_TRIALS_PER_CONDITION * 2:
                continue

            if region not in region_data:
                region_data[region] = []
            region_data[region].append({
                "session_idx": sess_idx,
                "mouse": mouse,
                "activity": activity[valid],
                "choice_labels": ch[valid],
                "evidence_labels": ev[valid],
                "n_neurons": int(activity.shape[1]),
            })

    logger.info(f"{datetime.now().isoformat()} {len(region_data)} regions loaded")

    # --- Per-region evaluation ---
    jsonl_path = RESULTS_DIR / "sutter_incremental.jsonl"

    # Resume support
    computed_regions: set[str] = set()
    region_results: dict[str, dict] = {}
    if jsonl_path.exists():
        with open(jsonl_path) as jf:
            for line in jf:
                r = json.loads(line)
                computed_regions.add(r["region"])
                region_results[r["region"]] = r
        logger.info(f"Resuming: loaded {len(computed_regions)} pre-computed regions")

    for region, measurements in tqdm(region_data.items(), desc="Regions"):
        if region in computed_regions:
            continue

        # Collect IIA across sessions for each method x condition
        results_by_key: dict[str, list[float]] = {}
        acc_by_key: dict[str, list[float]] = {}
        for method in METHODS:
            for condition in DATA_CONDITIONS:
                results_by_key[f"{method}_{condition}"] = []
                acc_by_key[f"{method}_{condition}"] = []

        for m in measurements:
            activity_real = m["activity"]
            ch = m["choice_labels"]
            ev = m["evidence_labels"]
            n_trials, n_neurons = activity_real.shape

            # Generate random Gaussian noise matching the same shape
            activity_random = np.random.randn(n_trials, n_neurons).astype(np.float32)

            data_map = {"real": activity_real, "random": activity_random}

            for method in METHODS:
                for condition in DATA_CONDITIONS:
                    key = f"{method}_{condition}"
                    result = _evaluate_method(
                        data_map[condition], ch, ev, method, device,
                    )
                    if result["iia"] is not None:
                        results_by_key[key].append(result["iia"])
                    if result["choice_accuracy"] is not None:
                        acc_by_key[key].append(result["choice_accuracy"])

        # Aggregate for this region
        region_result: dict = {"region": region, "n_sessions": len(measurements)}
        for key in results_by_key:
            vals = results_by_key[key]
            accs = acc_by_key[key]
            region_result[f"iia_{key}_mean"] = float(np.mean(vals)) if vals else None
            region_result[f"iia_{key}_std"] = float(np.std(vals)) if len(vals) > 1 else None
            region_result[f"iia_{key}_n"] = len(vals)
            region_result[f"acc_{key}_mean"] = float(np.mean(accs)) if accs else None

        # Compute the critical "vacuousness gap" per method:
        # gap = IIA_random - chance (0.5). Large positive = vacuous.
        for method in METHODS:
            real_vals = results_by_key[f"{method}_real"]
            rand_vals = results_by_key[f"{method}_random"]
            region_result[f"vacuousness_gap_{method}"] = (
                float(np.mean(rand_vals) - 0.5) if rand_vals else None
            )
            region_result[f"real_random_diff_{method}"] = (
                float(np.mean(real_vals) - np.mean(rand_vals))
                if real_vals and rand_vals else None
            )

        region_results[region] = region_result
        with open(jsonl_path, "a") as jf:
            jf.write(json.dumps(region_result, default=str) + "\n")

    logger.info(f"{datetime.now().isoformat()} All regions processed")

    # --- Aggregate statistical tests ---
    prediction_tests = {}

    # Test 1: Unconstrained MLP has high IIA on random data (vacuous)
    mlp_random_iias = [
        v["iia_mlp_random_mean"] for v in region_results.values()
        if v.get("iia_mlp_random_mean") is not None
    ]
    mlp_real_iias = [
        v["iia_mlp_real_mean"] for v in region_results.values()
        if v.get("iia_mlp_real_mean") is not None
    ]
    if mlp_random_iias:
        prediction_tests["mlp_random_iia"] = {
            "mean": float(np.mean(mlp_random_iias)),
            "std": float(np.std(mlp_random_iias)),
            "n": len(mlp_random_iias),
            "interpretation": (
                "If mean >> 0.5, the unconstrained MLP achieves high IIA on random "
                "Gaussian noise, confirming Sutter's result that unconstrained "
                "nonlinear alignment makes IIA vacuous."
            ),
        }

    # Test 2: Structured VAE does NOT have high IIA on random data
    vae_random_iias = [
        v["iia_vae_random_mean"] for v in region_results.values()
        if v.get("iia_vae_random_mean") is not None
    ]
    vae_real_iias = [
        v["iia_vae_real_mean"] for v in region_results.values()
        if v.get("iia_vae_real_mean") is not None
    ]
    if vae_random_iias:
        prediction_tests["vae_random_iia"] = {
            "mean": float(np.mean(vae_random_iias)),
            "std": float(np.std(vae_random_iias)),
            "n": len(vae_random_iias),
            "interpretation": (
                "If mean ~= 0.5, the structured VAE does NOT achieve high IIA on "
                "random noise, proving it is nonlinear but non-vacuous."
            ),
        }

    # Test 3: Linear DAS does NOT have high IIA on random data
    das_random_iias = [
        v["iia_linear_das_random_mean"] for v in region_results.values()
        if v.get("iia_linear_das_random_mean") is not None
    ]
    das_real_iias = [
        v["iia_linear_das_real_mean"] for v in region_results.values()
        if v.get("iia_linear_das_real_mean") is not None
    ]
    if das_random_iias:
        prediction_tests["das_random_iia"] = {
            "mean": float(np.mean(das_random_iias)),
            "std": float(np.std(das_random_iias)),
            "n": len(das_random_iias),
            "interpretation": (
                "If mean ~= 0.5, linear DAS does not fit random noise either "
                "(the linearity constraint prevents vacuous fitting)."
            ),
        }

    # Test 4: Paired comparison — MLP_random vs VAE_random IIA
    paired_mlp_rand = []
    paired_vae_rand = []
    for v in region_results.values():
        if (v.get("iia_mlp_random_mean") is not None
                and v.get("iia_vae_random_mean") is not None):
            paired_mlp_rand.append(v["iia_mlp_random_mean"])
            paired_vae_rand.append(v["iia_vae_random_mean"])

    if len(paired_mlp_rand) >= 5:
        diffs = np.array(paired_mlp_rand) - np.array(paired_vae_rand)
        try:
            w_stat, w_p = wilcoxon(diffs, alternative="greater")
        except Exception:
            w_stat, w_p = None, None
        prediction_tests["mlp_vs_vae_random_iia"] = {
            "mlp_random_mean": float(np.mean(paired_mlp_rand)),
            "vae_random_mean": float(np.mean(paired_vae_rand)),
            "mean_diff": float(np.mean(diffs)),
            "n_regions": len(paired_mlp_rand),
            "n_mlp_higher": int(np.sum(diffs > 0)),
            "wilcoxon_W": float(w_stat) if w_stat is not None else None,
            "wilcoxon_p": float(w_p) if w_p is not None else None,
            "interpretation": (
                "Positive mean_diff = MLP achieves higher IIA on random data than VAE. "
                "Significant Wilcoxon p = MLP is systematically more vacuous."
            ),
        }

    # Test 5: All three methods on real data (should all be high)
    paired_real = {"mlp": [], "linear_das": [], "vae": []}
    for v in region_results.values():
        if all(v.get(f"iia_{m}_real_mean") is not None for m in METHODS):
            for m in METHODS:
                paired_real[m].append(v[f"iia_{m}_real_mean"])

    if len(paired_real["mlp"]) >= 5:
        prediction_tests["all_methods_real_iia"] = {
            "mlp_real_mean": float(np.mean(paired_real["mlp"])),
            "linear_das_real_mean": float(np.mean(paired_real["linear_das"])),
            "vae_real_mean": float(np.mean(paired_real["vae"])),
            "n_regions": len(paired_real["mlp"]),
            "interpretation": (
                "All three methods should achieve high IIA on real data. "
                "The critical distinction is what happens on random data."
            ),
        }

    # Test 6: VAE real vs VAE random (paired, should be significant)
    paired_vae_real = []
    paired_vae_random = []
    for v in region_results.values():
        if (v.get("iia_vae_real_mean") is not None
                and v.get("iia_vae_random_mean") is not None):
            paired_vae_real.append(v["iia_vae_real_mean"])
            paired_vae_random.append(v["iia_vae_random_mean"])

    if len(paired_vae_real) >= 5:
        diffs = np.array(paired_vae_real) - np.array(paired_vae_random)
        try:
            w_stat, w_p = wilcoxon(diffs, alternative="greater")
        except Exception:
            w_stat, w_p = None, None
        prediction_tests["vae_real_vs_random"] = {
            "vae_real_mean": float(np.mean(paired_vae_real)),
            "vae_random_mean": float(np.mean(paired_vae_random)),
            "mean_diff": float(np.mean(diffs)),
            "n_regions": len(paired_vae_real),
            "wilcoxon_W": float(w_stat) if w_stat is not None else None,
            "wilcoxon_p": float(w_p) if w_p is not None else None,
            "interpretation": (
                "Significant p with positive diff = VAE gets meaningfully higher IIA "
                "on real data than random, confirming its IIA reflects genuine structure."
            ),
        }

    # --- Top-level summary ---
    results = {
        "timestamp": datetime.now().isoformat(),
        "n_sessions": len(sessions),
        "n_regions": len(region_data),
        "n_regions_analyzed": len(region_results),
        "device": device,
        "hyperparameters": {
            "z_dim": Z_DIM,
            "z_other_dim": Z_OTHER_DIM,
            "hidden_dim": HIDDEN_DIM,
            "n_epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "beta_kl": BETA_KL,
            "alpha_choice": ALPHA_CHOICE,
            "mlp_recon_weight": MLP_RECON_WEIGHT,
            "mlp_class_weight": MLP_CLASS_WEIGHT,
            "n_iia_pairs": N_IIA_PAIRS,
        },
        "region_results": {r: v for r, v in region_results.items()},
        "prediction_tests": prediction_tests,
    }

    out_path = RESULTS_DIR / "sutter_dilemma.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"{datetime.now().isoformat()} Saved results to {out_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sessions", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    run(max_sessions=args.max_sessions)
