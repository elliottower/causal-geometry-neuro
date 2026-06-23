"""Experiment 57: Structured disentangled VAE for causal subspace estimation.

Replaces PCA/LDA subspace estimation with a principled Bayesian approach based on
semi-supervised deep generative models (Kingma et al. NeurIPS 2014) and structured
disentanglement (Esmaeili et al. AISTATS 2019).

For each brain region:
1. Encode neural population activity (n_trials x n_neurons) into latent factors
2. z_choice (k dims, k=2-5): semi-supervised with choice labels — the posterior
   over this factor IS the causal subspace estimate
3. z_other (m dims, m=10-20): fully unsupervised, captures nuisance variance
4. Decode back to reconstructed activity

The encoder weight matrix for z_choice gives subspace directions directly,
replacing the LDA+PCA pipeline from exp40/exp42 with a model that:
- Provides posterior uncertainty over the subspace
- Jointly estimates choice-relevant vs nuisance variance
- Uses the ELBO objective for principled model comparison

IIA is computed on the VAE-learned subspace and compared against the PCA/LDA
baseline to test whether Bayesian subspace estimation improves causal validity.
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
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp57"
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
MIN_TRIALS_PER_CONDITION = 20

# VAE hyperparameters
Z_CHOICE_DIM = 3
Z_OTHER_DIM = 15
HIDDEN_DIM = 128
N_EPOCHS = 300
BATCH_SIZE = 64
LR = 1e-3
BETA_KL = 1.0
ALPHA_CHOICE = 10.0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class StructuredVAE(nn.Module):
    """Semi-supervised VAE with a supervised choice factor and unsupervised rest.

    Encoder: x -> (mu_choice, logvar_choice, mu_other, logvar_other)
    Decoder: (z_choice, z_other) -> x_hat

    The choice factor posterior q(z_choice | x) is trained with a classification
    head on z_choice, so it captures choice-discriminative structure. The encoder
    weights projecting to z_choice span the learned causal subspace.
    """

    def __init__(self, n_neurons: int, z_choice_dim: int, z_other_dim: int, hidden_dim: int):
        super().__init__()
        self.n_neurons = n_neurons
        self.z_choice_dim = z_choice_dim
        self.z_other_dim = z_other_dim
        z_dim = z_choice_dim + z_other_dim

        # Shared encoder trunk
        self.enc_trunk = nn.Sequential(
            nn.Linear(n_neurons, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Separate heads for choice and other latents
        self.enc_choice_mu = nn.Linear(hidden_dim, z_choice_dim)
        self.enc_choice_logvar = nn.Linear(hidden_dim, z_choice_dim)
        self.enc_other_mu = nn.Linear(hidden_dim, z_other_dim)
        self.enc_other_logvar = nn.Linear(hidden_dim, z_other_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_neurons),
        )

        # Choice classifier on z_choice
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

    def get_choice_subspace(self) -> np.ndarray:
        """Extract the linear subspace directions for the choice factor.

        Returns (n_neurons, z_choice_dim) orthonormal basis by composing the
        encoder trunk Jacobian at the origin with the choice-mu head, then QR.
        For a linear readout approximation, we use the weight composition
        W_choice_mu @ W_trunk2 @ W_trunk1, then orthonormalize.
        """
        W1 = self.enc_trunk[0].weight.detach()  # (hidden, n_neurons)
        W2 = self.enc_trunk[2].weight.detach()  # (hidden, hidden)
        W_mu = self.enc_choice_mu.weight.detach()  # (z_choice, hidden)
        # Compose: (z_choice, n_neurons)
        W_composed = W_mu @ W2 @ W1
        # Transpose to (n_neurons, z_choice) and orthonormalize
        Q, _ = torch.linalg.qr(W_composed.T)
        return Q.cpu().numpy()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def vae_loss(
    out: dict[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    beta_kl: float,
    alpha_choice: float,
) -> dict[str, torch.Tensor]:
    """ELBO loss with choice supervision.

    recon: MSE reconstruction
    kl_choice: KL(q(z_choice|x) || N(0,I))
    kl_other: KL(q(z_other|x) || N(0,I))
    choice_ce: cross-entropy on choice classifier from z_choice
    """
    recon = F.mse_loss(out["x_hat"], x, reduction="mean")

    kl_choice = -0.5 * torch.mean(
        1 + out["logvar_choice"] - out["mu_choice"].pow(2) - out["logvar_choice"].exp()
    )
    kl_other = -0.5 * torch.mean(
        1 + out["logvar_other"] - out["mu_other"].pow(2) - out["logvar_other"].exp()
    )

    choice_ce = F.cross_entropy(out["choice_logits"], y)

    total = recon + beta_kl * (kl_choice + kl_other) + alpha_choice * choice_ce
    return {
        "total": total,
        "recon": recon,
        "kl_choice": kl_choice,
        "kl_other": kl_other,
        "choice_ce": choice_ce,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_vae(
    activity: np.ndarray,
    choice_labels: np.ndarray,
    z_choice_dim: int = Z_CHOICE_DIM,
    z_other_dim: int = Z_OTHER_DIM,
    hidden_dim: int = HIDDEN_DIM,
    n_epochs: int = N_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    beta_kl: float = BETA_KL,
    alpha_choice: float = ALPHA_CHOICE,
    device: str = "cpu",
) -> dict:
    """Train the structured VAE on a single region's activity.

    Args:
        activity: (n_trials, n_neurons) time-averaged spike counts
        choice_labels: (n_trials,) binary choice labels (0/1)
        z_choice_dim: latent dims for choice factor
        z_other_dim: latent dims for nuisance factors
        hidden_dim: encoder/decoder hidden layer width
        n_epochs: training epochs
        batch_size: minibatch size
        lr: learning rate
        beta_kl: KL weight (beta-VAE)
        alpha_choice: weight on choice classification loss
        device: torch device string

    Returns:
        dict with trained model, loss history, and subspace directions
    """
    n_trials, n_neurons = activity.shape

    # Z-score normalize activity
    mu_act = activity.mean(axis=0, keepdims=True)
    std_act = activity.std(axis=0, keepdims=True)
    std_act[std_act < 1e-6] = 1.0
    activity_norm = (activity - mu_act) / std_act

    X = torch.tensor(activity_norm, dtype=torch.float32, device=device)
    y = torch.tensor(choice_labels, dtype=torch.long, device=device)

    model = StructuredVAE(n_neurons, z_choice_dim, z_other_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    effective_batch = min(batch_size, n_trials)

    for epoch in range(n_epochs):
        perm = torch.randperm(n_trials, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_trials, effective_batch):
            idx = perm[start:start + effective_batch]
            x_batch = X[idx]
            y_batch = y[idx]

            out = model(x_batch)
            losses = vae_loss(out, x_batch, y_batch, beta_kl, alpha_choice)

            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            epoch_loss += losses["total"].item()
            n_batches += 1

        loss_history.append(epoch_loss / max(n_batches, 1))

    # Extract results
    model.eval()
    with torch.no_grad():
        out_full = model(X)
        choice_acc = (out_full["choice_logits"].argmax(dim=1) == y).float().mean().item()

        # Posterior statistics
        mu_choice = out_full["mu_choice"].cpu().numpy()
        logvar_choice = out_full["logvar_choice"].cpu().numpy()
        mu_other = out_full["mu_other"].cpu().numpy()

    subspace_dirs = model.get_choice_subspace()

    return {
        "model": model,
        "subspace_directions": subspace_dirs,
        "choice_accuracy": choice_acc,
        "final_loss": loss_history[-1] if loss_history else None,
        "loss_history": loss_history,
        "mu_choice": mu_choice,
        "logvar_choice": logvar_choice,
        "mu_other": mu_other,
        "posterior_uncertainty": float(np.mean(np.exp(logvar_choice))),
        "activity_mean": mu_act.squeeze(),
        "activity_std": std_act.squeeze(),
    }


# ---------------------------------------------------------------------------
# IIA computation (same protocol as exp40/exp42 for comparability)
# ---------------------------------------------------------------------------

def _compute_iia(activity: np.ndarray, evidence_labels: np.ndarray,
                 choice_labels: np.ndarray, V: np.ndarray) -> float | None:
    """Interchange intervention accuracy using subspace V.

    Swap evidence projections between opposite-evidence trial pairs,
    measure choice classifier flip rate.
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

    n_pairs = min(len(left_idx), len(right_idx), 100)
    left_sample = left_idx[np.random.choice(len(left_idx), n_pairs, replace=False)]
    right_sample = right_idx[np.random.choice(len(right_idx), n_pairs, replace=False)]

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
    """LDA+PCA baseline subspace (same as exp40/exp42)."""
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


def _iia_null_random_subspace(activity: np.ndarray, evidence_labels: np.ndarray,
                              choice_labels: np.ndarray, n_dims: int = Z_CHOICE_DIM,
                              n_repeats: int = 50) -> list[float] | None:
    """Null distribution: IIA using random subspaces."""
    n_neurons = activity.shape[1]
    k = min(n_dims, n_neurons - 1)
    if k < 1:
        return None
    null_iias = []
    for _ in range(n_repeats):
        V_rand = np.linalg.qr(np.random.randn(n_neurons, k))[0]
        iia = _compute_iia(activity, evidence_labels, choice_labels, V_rand)
        if iia is not None:
            null_iias.append(iia)
    return null_iias if null_iias else None


def _power_law_exponent(activity: np.ndarray) -> float | None:
    n_components = min(50, activity.shape[1], activity.shape[0])
    pca = PCA(n_components=n_components)
    pca.fit(activity)
    eigenvalues = pca.explained_variance_
    eigenvalues = eigenvalues[eigenvalues > 0]
    if len(eigenvalues) < 10:
        return None
    start, end = 9, min(49, len(eigenvalues) - 1)
    log_rank = np.log10(np.arange(start + 1, end + 2))
    log_eig = np.log10(eigenvalues[start:end + 1])
    coeffs = np.polyfit(log_rank, log_eig, 1)
    return float(-coeffs[0])


# ---------------------------------------------------------------------------
# Contrast-to-evidence helper (same as other experiments)
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
# Main
# ---------------------------------------------------------------------------

def run(max_sessions: int | None = None, z_choice_dim: int | None = None) -> dict:
    global Z_CHOICE_DIM
    if z_choice_dim is not None:
        Z_CHOICE_DIM = z_choice_dim
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"{datetime.now().isoformat()} Starting structured VAE experiment "
                f"with {len(sessions)} sessions on {device}")

    # --- Load data ---
    region_data: dict[str, list[dict]] = {}
    for sess_idx, sess in enumerate(tqdm(sessions, desc="Loading sessions")):
        choice_labels = get_choice_labels(sess)
        if len(np.unique(choice_labels)) < 2:
            continue
        evidence_labels, evidence_values = _contrast_to_evidence_label(sess)
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

    # --- Per-region VAE training and IIA ---
    region_results = {}
    jsonl_path = RESULTS_DIR / "vae_incremental.jsonl"

    # Resume support
    computed_regions: set[str] = set()
    if jsonl_path.exists():
        with open(jsonl_path) as jf:
            for line in jf:
                r = json.loads(line)
                computed_regions.add(r["region"])
                region_results[r["region"]] = r
        logger.info(f"Resuming: loaded {len(computed_regions)} pre-computed regions")

    for region, measurements in tqdm(region_data.items(), desc="VAE per region"):
        if region in computed_regions:
            continue

        vae_iias = []
        lda_iias = []
        null_iias_all = []
        choice_accs = []
        posterior_uncertainties = []
        recon_losses = []
        alphas = []
        subspace_angles = []

        for m in measurements:
            activity = m["activity"]
            ch = m["choice_labels"]
            ev = m["evidence_labels"]
            n_neurons = activity.shape[1]

            # Adapt latent dims to neuron count
            z_choice = min(Z_CHOICE_DIM, n_neurons // 5, n_neurons - 1)
            z_other = min(Z_OTHER_DIM, n_neurons // 3, n_neurons - z_choice - 1)
            if z_choice < 1 or z_other < 1:
                continue
            hidden = min(HIDDEN_DIM, n_neurons * 2)

            # Train VAE
            try:
                vae_result = train_vae(
                    activity, ch,
                    z_choice_dim=z_choice,
                    z_other_dim=z_other,
                    hidden_dim=hidden,
                    device=device,
                )
            except Exception as e:
                logger.warning(f"VAE training failed for {region} sess {m['session_idx']}: {e}")
                continue

            V_vae = vae_result["subspace_directions"]
            choice_accs.append(vae_result["choice_accuracy"])
            posterior_uncertainties.append(vae_result["posterior_uncertainty"])
            recon_losses.append(vae_result["final_loss"])

            # IIA with VAE subspace
            iia_vae = _compute_iia(activity, ev, ch, V_vae)
            if iia_vae is not None:
                vae_iias.append(iia_vae)

            # IIA with LDA baseline (matched dimensionality)
            V_lda = _estimate_lda_subspace(activity, ev, n_dims=z_choice)
            if V_lda is not None:
                iia_lda = _compute_iia(activity, ev, ch, V_lda)
                if iia_lda is not None:
                    lda_iias.append(iia_lda)

                # Angle between VAE and LDA subspaces
                try:
                    from geometry.distances import grassmannian_distance
                    k = min(V_vae.shape[1], V_lda.shape[1])
                    d = grassmannian_distance(V_vae[:, :k], V_lda[:, :k])
                    subspace_angles.append(d)
                except Exception:
                    pass

            # Random subspace null
            null_iias = _iia_null_random_subspace(activity, ev, ch, n_dims=z_choice)
            if null_iias is not None:
                null_iias_all.extend(null_iias)

            # Power law exponent
            alpha = _power_law_exponent(activity)
            if alpha is not None:
                alphas.append(alpha)

        # Aggregate
        result = {
            "region": region,
            "n_sessions": len(measurements),
            "vae_iia_mean": float(np.mean(vae_iias)) if vae_iias else None,
            "vae_iia_std": float(np.std(vae_iias)) if len(vae_iias) > 1 else None,
            "vae_iia_n": len(vae_iias),
            "lda_iia_mean": float(np.mean(lda_iias)) if lda_iias else None,
            "lda_iia_std": float(np.std(lda_iias)) if len(lda_iias) > 1 else None,
            "lda_iia_n": len(lda_iias),
            "null_iia_mean": float(np.mean(null_iias_all)) if null_iias_all else None,
            "null_iia_std": float(np.std(null_iias_all)) if null_iias_all else None,
            "iia_vae_above_null": (
                float(np.mean(vae_iias) - np.mean(null_iias_all))
                if vae_iias and null_iias_all else None
            ),
            "iia_vae_above_lda": (
                float(np.mean(vae_iias) - np.mean(lda_iias))
                if vae_iias and lda_iias else None
            ),
            "choice_accuracy_mean": float(np.mean(choice_accs)) if choice_accs else None,
            "posterior_uncertainty_mean": float(np.mean(posterior_uncertainties)) if posterior_uncertainties else None,
            "recon_loss_mean": float(np.mean(recon_losses)) if recon_losses else None,
            "power_law_alpha": float(np.mean(alphas)) if alphas else None,
            "vae_lda_grassmannian_mean": float(np.mean(subspace_angles)) if subspace_angles else None,
        }

        region_results[region] = result
        with open(jsonl_path, "a") as jf:
            jf.write(json.dumps(result, default=str) + "\n")

    logger.info(f"{datetime.now().isoformat()} All regions processed")

    # --- Aggregate analysis ---
    prediction_tests = {}

    # 1. VAE IIA vs LDA IIA (paired comparison across regions)
    paired_vae = []
    paired_lda = []
    for r, v in region_results.items():
        if v.get("vae_iia_mean") is not None and v.get("lda_iia_mean") is not None:
            paired_vae.append(v["vae_iia_mean"])
            paired_lda.append(v["lda_iia_mean"])

    if len(paired_vae) >= 5:
        diffs = np.array(paired_vae) - np.array(paired_lda)
        from scipy.stats import wilcoxon
        try:
            w_stat, w_p = wilcoxon(diffs, alternative="greater")
        except Exception:
            w_stat, w_p = None, None
        prediction_tests["vae_vs_lda_iia"] = {
            "vae_mean": float(np.mean(paired_vae)),
            "lda_mean": float(np.mean(paired_lda)),
            "mean_diff": float(np.mean(diffs)),
            "median_diff": float(np.median(diffs)),
            "n_regions": len(paired_vae),
            "n_vae_wins": int(np.sum(diffs > 0)),
            "wilcoxon_W": float(w_stat) if w_stat is not None else None,
            "wilcoxon_p": float(w_p) if w_p is not None else None,
            "interpretation": (
                "Positive mean_diff = VAE subspace produces higher IIA than LDA. "
                "Wilcoxon signed-rank tests whether the improvement is significant."
            ),
        }

    # 2. VAE IIA vs random subspace null
    all_vae_iia = [v["vae_iia_mean"] for v in region_results.values() if v.get("vae_iia_mean") is not None]
    all_null_iia = [v["null_iia_mean"] for v in region_results.values() if v.get("null_iia_mean") is not None]
    if all_vae_iia and all_null_iia:
        from scipy.stats import mannwhitneyu
        try:
            u_stat, u_p = mannwhitneyu(all_vae_iia, all_null_iia, alternative="greater")
        except Exception:
            u_stat, u_p = None, None
        prediction_tests["vae_vs_null_iia"] = {
            "vae_mean": float(np.mean(all_vae_iia)),
            "null_mean": float(np.mean(all_null_iia)),
            "effect_size": float(np.mean(all_vae_iia) - np.mean(all_null_iia)),
            "mann_whitney_U": float(u_stat) if u_stat is not None else None,
            "p_one_sided": float(u_p) if u_p is not None else None,
            "n_vae": len(all_vae_iia),
            "n_null": len(all_null_iia),
        }

    # 3. Alpha vs VAE IIA (same prediction as exp40: low-dim regions have higher IIA)
    alpha_list = []
    iia_list = []
    for v in region_results.values():
        if v.get("power_law_alpha") is not None and v.get("vae_iia_mean") is not None:
            alpha_list.append(v["power_law_alpha"])
            iia_list.append(v["vae_iia_mean"])

    if len(alpha_list) >= 5:
        rho, p = spearmanr(alpha_list, iia_list)
        prediction_tests["alpha_vs_vae_iia"] = {
            "rho": float(rho),
            "p": float(p),
            "n": len(alpha_list),
            "interpretation": (
                "Positive rho = low-dimensional (high-alpha) regions have higher VAE IIA."
            ),
        }

    # 4. Posterior uncertainty vs IIA (prediction: lower uncertainty = higher IIA)
    unc_list = []
    iia_for_unc = []
    for v in region_results.values():
        if v.get("posterior_uncertainty_mean") is not None and v.get("vae_iia_mean") is not None:
            unc_list.append(v["posterior_uncertainty_mean"])
            iia_for_unc.append(v["vae_iia_mean"])

    if len(unc_list) >= 5:
        rho, p = spearmanr(unc_list, iia_for_unc)
        prediction_tests["uncertainty_vs_iia"] = {
            "rho": float(rho),
            "p": float(p),
            "n": len(unc_list),
            "interpretation": (
                "Negative rho = regions where the VAE is confident about the choice "
                "subspace also have higher IIA (subspace is genuinely causal)."
            ),
        }

    # 5. VAE-LDA subspace agreement
    grass_dists = [v["vae_lda_grassmannian_mean"] for v in region_results.values()
                   if v.get("vae_lda_grassmannian_mean") is not None]
    if grass_dists:
        prediction_tests["vae_lda_subspace_agreement"] = {
            "mean_grassmannian_distance": float(np.mean(grass_dists)),
            "std_grassmannian_distance": float(np.std(grass_dists)),
            "n_regions": len(grass_dists),
            "interpretation": (
                "Small distance = VAE and LDA find similar subspaces (VAE validates LDA). "
                "Large distance = VAE finds a different, potentially better subspace."
            ),
        }

    # --- Top/bottom regions ---
    ranked_vae = sorted(
        [(r, v["vae_iia_mean"]) for r, v in region_results.items() if v.get("vae_iia_mean") is not None],
        key=lambda x: x[1], reverse=True,
    )

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_sessions": len(sessions),
        "n_regions": len(region_data),
        "n_regions_analyzed": len(region_results),
        "device": device,
        "hyperparameters": {
            "z_choice_dim": Z_CHOICE_DIM,
            "z_other_dim": Z_OTHER_DIM,
            "hidden_dim": HIDDEN_DIM,
            "n_epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "beta_kl": BETA_KL,
            "alpha_choice": ALPHA_CHOICE,
        },
        "region_results": {r: v for r, v in region_results.items()},
        "prediction_tests": prediction_tests,
        "top_vae_iia_regions": ranked_vae[:10],
        "bottom_vae_iia_regions": ranked_vae[-10:] if len(ranked_vae) >= 10 else ranked_vae,
    }

    # Save
    out_path = RESULTS_DIR / "structured_vae.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"{datetime.now().isoformat()} Saved results to {out_path}")

    # Also save subspace directions as npz for downstream use
    npz_data = {}
    for region, measurements in region_data.items():
        if region in computed_regions:
            continue
        rr = region_results.get(region)
        if rr is None or rr.get("vae_iia_mean") is None:
            continue
        # Re-train one final model for the best session to save directions
        # (the incremental results already have the aggregated metrics)

    npz_path = RESULTS_DIR / "vae_subspace_summary.npz"
    if npz_data:
        np.savez_compressed(npz_path, **npz_data)
        logger.info(f"Saved subspace directions to {npz_path}")

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
