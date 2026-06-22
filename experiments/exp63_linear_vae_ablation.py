"""Experiment 63: Linear VAE ablation — isolating structure vs nonlinearity.

Disentangles whether the VAE advantage (from exp57) comes from the STRUCTURED
LATENT SPLIT (z_choice + z_other with supervised classification) or from
NONLINEARITY in the encoder/decoder.

Three models compared per region:
1. Nonlinear structured VAE (baseline from exp57): MLP encoder/decoder with
   z_choice + z_other split and classification loss on z_choice.
2. Linear structured VAE: Linear encoder/decoder (no hidden layers, no
   activations) with the SAME z_choice + z_other split and same training
   procedure (recon + KL + classification).
3. Nonlinear unstructured VAE: MLP encoder/decoder but with a SINGLE latent z
   (no choice/other split, no supervised classification loss). Same total
   latent dimensionality (z_dim = z_choice_dim + z_other_dim). IIA is computed
   by finding the top-k LDA directions in the latent space post-hoc, then
   swapping those.

For each model and region, IIA is computed by swapping the choice-relevant
subspace between opposite-evidence trial pairs.
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
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp63"
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
MIN_TRIALS_PER_CONDITION = 20

# VAE hyperparameters (matching exp57)
Z_CHOICE_DIM = 3
Z_OTHER_DIM = 15
HIDDEN_DIM = 128
N_EPOCHS = 300
BATCH_SIZE = 64
LR = 1e-3
BETA_KL = 1.0
ALPHA_CHOICE = 10.0


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class StructuredVAE(nn.Module):
    """Nonlinear structured VAE (same as exp57). MLP encoder/decoder with
    z_choice + z_other split and classification head on z_choice."""

    def __init__(self, n_neurons: int, z_choice_dim: int, z_other_dim: int, hidden_dim: int):
        super().__init__()
        self.n_neurons = n_neurons
        self.z_choice_dim = z_choice_dim
        self.z_other_dim = z_other_dim

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
            nn.Linear(z_choice_dim + z_other_dim, hidden_dim),
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
        return mu + torch.randn_like(std) * std

    def decode(self, z_choice: torch.Tensor, z_other: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z_choice, z_other], dim=-1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu_c, logvar_c, mu_o, logvar_o = self.encode(x)
        z_choice = self.reparameterize(mu_c, logvar_c)
        z_other = self.reparameterize(mu_o, logvar_o)
        x_hat = self.decode(z_choice, z_other)
        choice_logits = self.choice_classifier(z_choice)
        return {
            "x_hat": x_hat,
            "mu_choice": mu_c, "logvar_choice": logvar_c,
            "mu_other": mu_o, "logvar_other": logvar_o,
            "z_choice": z_choice, "z_other": z_other,
            "choice_logits": choice_logits,
        }

    def get_choice_subspace(self) -> np.ndarray:
        """(n_neurons, z_choice_dim) orthonormal basis via composed encoder weights."""
        W1 = self.enc_trunk[0].weight.detach()  # (hidden, n_neurons)
        W2 = self.enc_trunk[2].weight.detach()  # (hidden, hidden)
        W_mu = self.enc_choice_mu.weight.detach()  # (z_choice, hidden)
        W_composed = W_mu @ W2 @ W1  # (z_choice, n_neurons)
        Q, _ = torch.linalg.qr(W_composed.T)
        return Q.cpu().numpy()


class LinearStructuredVAE(nn.Module):
    """Linear structured VAE: same z_choice + z_other split and classification
    loss, but encoder and decoder are single linear layers with no hidden
    layers or activation functions."""

    def __init__(self, input_dim: int, z_choice_dim: int = 3, z_other_dim: int = 15):
        super().__init__()
        self.input_dim = input_dim
        self.z_choice_dim = z_choice_dim
        self.z_other_dim = z_other_dim

        # Encoder: single linear layer to mu and logvar for both z_choice and z_other
        self.fc_mu_choice = nn.Linear(input_dim, z_choice_dim)
        self.fc_logvar_choice = nn.Linear(input_dim, z_choice_dim)
        self.fc_mu_other = nn.Linear(input_dim, z_other_dim)
        self.fc_logvar_other = nn.Linear(input_dim, z_other_dim)
        # Classifier on z_choice
        self.classifier = nn.Linear(z_choice_dim, 2)
        # Decoder: single linear layer
        self.decoder = nn.Linear(z_choice_dim + z_other_dim, input_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.fc_mu_choice(x),
            self.fc_logvar_choice(x),
            self.fc_mu_other(x),
            self.fc_logvar_other(x),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z_choice: torch.Tensor, z_other: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z_choice, z_other], dim=-1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu_c, logvar_c, mu_o, logvar_o = self.encode(x)
        z_choice = self.reparameterize(mu_c, logvar_c)
        z_other = self.reparameterize(mu_o, logvar_o)
        x_hat = self.decode(z_choice, z_other)
        choice_logits = self.classifier(z_choice)
        return {
            "x_hat": x_hat,
            "mu_choice": mu_c, "logvar_choice": logvar_c,
            "mu_other": mu_o, "logvar_other": logvar_o,
            "z_choice": z_choice, "z_other": z_other,
            "choice_logits": choice_logits,
        }

    def get_choice_subspace(self) -> np.ndarray:
        """(input_dim, z_choice_dim) orthonormal basis. Encoder is linear so
        the mu_choice weight matrix directly spans the choice subspace."""
        W = self.fc_mu_choice.weight.detach()  # (z_choice, input_dim)
        Q, _ = torch.linalg.qr(W.T)
        return Q.cpu().numpy()


class UnstructuredVAE(nn.Module):
    """Nonlinear unstructured VAE: MLP encoder/decoder with a single latent z
    (no choice/other split, no classification loss). Same total latent
    dimensionality as the structured models."""

    def __init__(self, n_neurons: int, z_dim: int, hidden_dim: int):
        super().__init__()
        self.n_neurons = n_neurons
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Linear(n_neurons, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_neurons),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return {"x_hat": x_hat, "mu": mu, "logvar": logvar, "z": z}


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def structured_vae_loss(
    out: dict[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    beta_kl: float,
    alpha_choice: float,
) -> dict[str, torch.Tensor]:
    """ELBO loss with choice supervision (used by both structured models)."""
    recon = F.mse_loss(out["x_hat"], x, reduction="mean")
    kl_choice = -0.5 * torch.mean(
        1 + out["logvar_choice"] - out["mu_choice"].pow(2) - out["logvar_choice"].exp()
    )
    kl_other = -0.5 * torch.mean(
        1 + out["logvar_other"] - out["mu_other"].pow(2) - out["logvar_other"].exp()
    )
    choice_ce = F.cross_entropy(out["choice_logits"], y)
    total = recon + beta_kl * (kl_choice + kl_other) + alpha_choice * choice_ce
    return {"total": total, "recon": recon, "kl_choice": kl_choice,
            "kl_other": kl_other, "choice_ce": choice_ce}


def unstructured_vae_loss(
    out: dict[str, torch.Tensor],
    x: torch.Tensor,
    beta_kl: float,
) -> dict[str, torch.Tensor]:
    """Standard VAE ELBO (no classification term)."""
    recon = F.mse_loss(out["x_hat"], x, reduction="mean")
    kl = -0.5 * torch.mean(
        1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp()
    )
    total = recon + beta_kl * kl
    return {"total": total, "recon": recon, "kl": kl}


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _normalize_activity(activity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = activity.mean(axis=0, keepdims=True)
    std = activity.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (activity - mu) / std, mu.squeeze(), std.squeeze()


def _train_loop(model: nn.Module, X: torch.Tensor, y: torch.Tensor | None,
                loss_fn, n_epochs: int, batch_size: int, lr: float,
                device: str) -> list[float]:
    """Generic training loop. loss_fn(out, x_batch, y_batch) or loss_fn(out, x_batch)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    effective_batch = min(batch_size, n)
    history = []

    for _ in range(n_epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, effective_batch):
            idx = perm[start:start + effective_batch]
            x_batch = X[idx]
            out = model(x_batch)

            if y is not None:
                losses = loss_fn(out, x_batch, y[idx])
            else:
                losses = loss_fn(out, x_batch)

            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()
            epoch_loss += losses["total"].item()
            n_batches += 1

        history.append(epoch_loss / max(n_batches, 1))
    return history


def train_nonlinear_structured(activity: np.ndarray, choice_labels: np.ndarray,
                               z_choice: int, z_other: int, hidden: int,
                               device: str) -> dict:
    activity_norm, _, _ = _normalize_activity(activity)
    X = torch.tensor(activity_norm, dtype=torch.float32, device=device)
    y = torch.tensor(choice_labels, dtype=torch.long, device=device)
    n_neurons = activity.shape[1]

    model = StructuredVAE(n_neurons, z_choice, z_other, hidden).to(device)

    def loss_fn(out, x, yb):
        return structured_vae_loss(out, x, yb, BETA_KL, ALPHA_CHOICE)

    history = _train_loop(model, X, y, loss_fn, N_EPOCHS, BATCH_SIZE, LR, device)

    model.eval()
    with torch.no_grad():
        out = model(X)
        acc = (out["choice_logits"].argmax(dim=1) == y).float().mean().item()

    return {
        "model": model,
        "subspace": model.get_choice_subspace(),
        "choice_acc": acc,
        "final_loss": history[-1] if history else None,
    }


def train_linear_structured(activity: np.ndarray, choice_labels: np.ndarray,
                            z_choice: int, z_other: int,
                            device: str) -> dict:
    activity_norm, _, _ = _normalize_activity(activity)
    X = torch.tensor(activity_norm, dtype=torch.float32, device=device)
    y = torch.tensor(choice_labels, dtype=torch.long, device=device)
    n_neurons = activity.shape[1]

    model = LinearStructuredVAE(n_neurons, z_choice, z_other).to(device)

    def loss_fn(out, x, yb):
        return structured_vae_loss(out, x, yb, BETA_KL, ALPHA_CHOICE)

    history = _train_loop(model, X, y, loss_fn, N_EPOCHS, BATCH_SIZE, LR, device)

    model.eval()
    with torch.no_grad():
        out = model(X)
        acc = (out["choice_logits"].argmax(dim=1) == y).float().mean().item()

    return {
        "model": model,
        "subspace": model.get_choice_subspace(),
        "choice_acc": acc,
        "final_loss": history[-1] if history else None,
    }


def train_nonlinear_unstructured(activity: np.ndarray, choice_labels: np.ndarray,
                                 z_dim: int, hidden: int,
                                 device: str) -> dict:
    """Train unstructured VAE. choice_labels are NOT used during training but are
    needed post-hoc for LDA subspace extraction."""
    activity_norm, _, _ = _normalize_activity(activity)
    X = torch.tensor(activity_norm, dtype=torch.float32, device=device)
    y_np = choice_labels
    n_neurons = activity.shape[1]

    model = UnstructuredVAE(n_neurons, z_dim, hidden).to(device)

    def loss_fn(out, x):
        return unstructured_vae_loss(out, x, BETA_KL)

    history = _train_loop(model, X, None, loss_fn, N_EPOCHS, BATCH_SIZE, LR, device)

    # Post-hoc: find choice-discriminative directions in latent space via LDA
    model.eval()
    with torch.no_grad():
        mu = model.encode(X)[0].cpu().numpy()  # (n_trials, z_dim)

    # LDA on latent means to find choice-relevant directions
    lda = LinearDiscriminantAnalysis()
    try:
        lda.fit(mu, y_np)
        lda_dir = lda.coef_[0]
        lda_dir = lda_dir / (np.linalg.norm(lda_dir) + 1e-10)
    except Exception:
        lda_dir = np.zeros(z_dim)
        lda_dir[0] = 1.0

    # Map latent LDA direction back to neuron space via Jacobian composition
    W1 = model.encoder[0].weight.detach()  # (hidden, n_neurons)
    W2 = model.encoder[2].weight.detach()  # (hidden, hidden)
    W_mu = model.fc_mu.weight.detach()  # (z_dim, hidden)
    W_composed = W_mu @ W2 @ W1  # (z_dim, n_neurons)

    # Project latent LDA direction to neuron space
    lda_dir_t = torch.tensor(lda_dir, dtype=torch.float32)
    # neuron_dir = W_composed^T @ lda_dir gives (n_neurons,) direction
    neuron_dir = (W_composed.T @ lda_dir_t).cpu().numpy()
    neuron_dir = neuron_dir / (np.linalg.norm(neuron_dir) + 1e-10)

    # Build a k-dim subspace: LDA direction + top PCA directions in latent space
    # that are orthogonal to LDA, mapped back to neuron space
    pca = PCA(n_components=min(z_dim, mu.shape[0] - 1))
    pca.fit(mu)
    subspace_cols = [neuron_dir.reshape(-1, 1)]
    for pc in pca.components_:
        pc_neuron = (W_composed.T @ torch.tensor(pc, dtype=torch.float32)).cpu().numpy()
        pc_neuron = pc_neuron / (np.linalg.norm(pc_neuron) + 1e-10)
        subspace_cols.append(pc_neuron.reshape(-1, 1))
        if len(subspace_cols) >= Z_CHOICE_DIM:
            break

    combined = np.hstack(subspace_cols)
    Q, _ = np.linalg.qr(combined)
    subspace = Q[:, :Z_CHOICE_DIM]

    # Post-hoc classification accuracy on latent means
    try:
        acc = float(lda.score(mu, y_np))
    except Exception:
        acc = None

    return {
        "model": model,
        "subspace": subspace,
        "choice_acc": acc,
        "final_loss": history[-1] if history else None,
    }


# ---------------------------------------------------------------------------
# IIA computation (same protocol as exp57/exp40/exp42)
# ---------------------------------------------------------------------------

def _compute_iia(activity: np.ndarray, evidence_labels: np.ndarray,
                 choice_labels: np.ndarray, V: np.ndarray) -> float | None:
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


def _iia_null_random_subspace(activity: np.ndarray, evidence_labels: np.ndarray,
                              choice_labels: np.ndarray, n_dims: int = Z_CHOICE_DIM,
                              n_repeats: int = 50) -> list[float] | None:
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
# Main
# ---------------------------------------------------------------------------

def run(max_sessions: int | None = None) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"{datetime.now().isoformat()} Starting linear VAE ablation "
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

    # --- Per-region training and IIA ---
    region_results = {}
    jsonl_path = RESULTS_DIR / "ablation_incremental.jsonl"

    # Resume support
    computed_regions: set[str] = set()
    if jsonl_path.exists():
        with open(jsonl_path) as jf:
            for line in jf:
                r = json.loads(line)
                computed_regions.add(r["region"])
                region_results[r["region"]] = r
        logger.info(f"Resuming: loaded {len(computed_regions)} pre-computed regions")

    for region, measurements in tqdm(region_data.items(), desc="Ablation per region"):
        if region in computed_regions:
            continue

        # Accumulators per model type
        iia_nonlinear_struct = []
        iia_linear_struct = []
        iia_nonlinear_unstruct = []
        iia_null = []
        acc_nonlinear_struct = []
        acc_linear_struct = []
        acc_nonlinear_unstruct = []
        loss_nonlinear_struct = []
        loss_linear_struct = []
        loss_nonlinear_unstruct = []

        for m in measurements:
            activity = m["activity"]
            ch = m["choice_labels"]
            ev = m["evidence_labels"]
            n_neurons = activity.shape[1]

            z_choice = min(Z_CHOICE_DIM, n_neurons // 5, n_neurons - 1)
            z_other = min(Z_OTHER_DIM, n_neurons // 3, n_neurons - z_choice - 1)
            if z_choice < 1 or z_other < 1:
                continue
            z_total = z_choice + z_other
            hidden = min(HIDDEN_DIM, n_neurons * 2)

            # --- Model 1: Nonlinear structured VAE ---
            try:
                res1 = train_nonlinear_structured(activity, ch, z_choice, z_other, hidden, device)
                iia1 = _compute_iia(activity, ev, ch, res1["subspace"])
                if iia1 is not None:
                    iia_nonlinear_struct.append(iia1)
                if res1["choice_acc"] is not None:
                    acc_nonlinear_struct.append(res1["choice_acc"])
                if res1["final_loss"] is not None:
                    loss_nonlinear_struct.append(res1["final_loss"])
            except Exception as e:
                logger.warning(f"Nonlinear structured failed for {region} sess {m['session_idx']}: {e}")

            # --- Model 2: Linear structured VAE ---
            try:
                res2 = train_linear_structured(activity, ch, z_choice, z_other, device)
                iia2 = _compute_iia(activity, ev, ch, res2["subspace"])
                if iia2 is not None:
                    iia_linear_struct.append(iia2)
                if res2["choice_acc"] is not None:
                    acc_linear_struct.append(res2["choice_acc"])
                if res2["final_loss"] is not None:
                    loss_linear_struct.append(res2["final_loss"])
            except Exception as e:
                logger.warning(f"Linear structured failed for {region} sess {m['session_idx']}: {e}")

            # --- Model 3: Nonlinear unstructured VAE ---
            try:
                res3 = train_nonlinear_unstructured(activity, ch, z_total, hidden, device)
                iia3 = _compute_iia(activity, ev, ch, res3["subspace"])
                if iia3 is not None:
                    iia_nonlinear_unstruct.append(iia3)
                if res3["choice_acc"] is not None:
                    acc_nonlinear_unstruct.append(res3["choice_acc"])
                if res3["final_loss"] is not None:
                    loss_nonlinear_unstruct.append(res3["final_loss"])
            except Exception as e:
                logger.warning(f"Nonlinear unstructured failed for {region} sess {m['session_idx']}: {e}")

            # --- Null baseline ---
            null_iias = _iia_null_random_subspace(activity, ev, ch, n_dims=z_choice)
            if null_iias is not None:
                iia_null.extend(null_iias)

        def _stats(vals: list[float]) -> dict:
            if not vals:
                return {"mean": None, "std": None, "n": 0}
            return {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)) if len(vals) > 1 else None,
                "n": len(vals),
            }

        result = {
            "region": region,
            "n_sessions": len(measurements),
            "nonlinear_structured": {
                "iia": _stats(iia_nonlinear_struct),
                "choice_acc": _stats(acc_nonlinear_struct),
                "final_loss": _stats(loss_nonlinear_struct),
            },
            "linear_structured": {
                "iia": _stats(iia_linear_struct),
                "choice_acc": _stats(acc_linear_struct),
                "final_loss": _stats(loss_linear_struct),
            },
            "nonlinear_unstructured": {
                "iia": _stats(iia_nonlinear_unstruct),
                "choice_acc": _stats(acc_nonlinear_unstruct),
                "final_loss": _stats(loss_nonlinear_unstruct),
            },
            "null": _stats(iia_null),
        }

        region_results[region] = result
        with open(jsonl_path, "a") as jf:
            jf.write(json.dumps(result, default=str) + "\n")

    logger.info(f"{datetime.now().isoformat()} All regions processed")

    # --- Aggregate analysis ---
    prediction_tests = {}

    # Collect paired IIA means across regions
    paired_nl_struct = []
    paired_lin_struct = []
    paired_nl_unstruct = []
    paired_null = []

    for v in region_results.values():
        nl_s = v["nonlinear_structured"]["iia"]["mean"]
        li_s = v["linear_structured"]["iia"]["mean"]
        nl_u = v["nonlinear_unstructured"]["iia"]["mean"]
        nu = v["null"]["mean"]
        if nl_s is not None and li_s is not None and nl_u is not None:
            paired_nl_struct.append(nl_s)
            paired_lin_struct.append(li_s)
            paired_nl_unstruct.append(nl_u)
            if nu is not None:
                paired_null.append(nu)

    n_paired = len(paired_nl_struct)

    # Test 1: Nonlinear structured vs Linear structured (isolates nonlinearity)
    if n_paired >= 5:
        diffs = np.array(paired_nl_struct) - np.array(paired_lin_struct)
        try:
            w_stat, w_p = wilcoxon(diffs, alternative="greater")
        except Exception:
            w_stat, w_p = None, None
        prediction_tests["nonlinearity_effect"] = {
            "description": "Nonlinear structured vs linear structured (isolates nonlinearity contribution)",
            "nonlinear_struct_mean": float(np.mean(paired_nl_struct)),
            "linear_struct_mean": float(np.mean(paired_lin_struct)),
            "mean_diff": float(np.mean(diffs)),
            "median_diff": float(np.median(diffs)),
            "n_nonlinear_wins": int(np.sum(diffs > 0)),
            "n_regions": n_paired,
            "wilcoxon_W": float(w_stat) if w_stat is not None else None,
            "wilcoxon_p": float(w_p) if w_p is not None else None,
        }

    # Test 2: Linear structured vs Nonlinear unstructured (isolates structure)
    if n_paired >= 5:
        diffs = np.array(paired_lin_struct) - np.array(paired_nl_unstruct)
        try:
            w_stat, w_p = wilcoxon(diffs, alternative="greater")
        except Exception:
            w_stat, w_p = None, None
        prediction_tests["structure_effect"] = {
            "description": "Linear structured vs nonlinear unstructured (isolates structure contribution)",
            "linear_struct_mean": float(np.mean(paired_lin_struct)),
            "nonlinear_unstruct_mean": float(np.mean(paired_nl_unstruct)),
            "mean_diff": float(np.mean(diffs)),
            "median_diff": float(np.median(diffs)),
            "n_linear_struct_wins": int(np.sum(diffs > 0)),
            "n_regions": n_paired,
            "wilcoxon_W": float(w_stat) if w_stat is not None else None,
            "wilcoxon_p": float(w_p) if w_p is not None else None,
        }

    # Test 3: Nonlinear structured vs Nonlinear unstructured (full structure advantage)
    if n_paired >= 5:
        diffs = np.array(paired_nl_struct) - np.array(paired_nl_unstruct)
        try:
            w_stat, w_p = wilcoxon(diffs, alternative="greater")
        except Exception:
            w_stat, w_p = None, None
        prediction_tests["full_structure_advantage"] = {
            "description": "Nonlinear structured vs nonlinear unstructured (full structure advantage with nonlinearity held constant)",
            "nonlinear_struct_mean": float(np.mean(paired_nl_struct)),
            "nonlinear_unstruct_mean": float(np.mean(paired_nl_unstruct)),
            "mean_diff": float(np.mean(diffs)),
            "median_diff": float(np.median(diffs)),
            "n_struct_wins": int(np.sum(diffs > 0)),
            "n_regions": n_paired,
            "wilcoxon_W": float(w_stat) if w_stat is not None else None,
            "wilcoxon_p": float(w_p) if w_p is not None else None,
        }

    # Test 4: All three vs null
    if paired_null and n_paired >= 5:
        null_mean = float(np.mean(paired_null))
        prediction_tests["above_null"] = {
            "null_mean": null_mean,
            "nonlinear_struct_above_null": float(np.mean(paired_nl_struct) - null_mean),
            "linear_struct_above_null": float(np.mean(paired_lin_struct) - null_mean),
            "nonlinear_unstruct_above_null": float(np.mean(paired_nl_unstruct) - null_mean),
        }

    # Rank regions by largest nonlinearity effect
    nonlinearity_effects = []
    for r, v in region_results.items():
        nl = v["nonlinear_structured"]["iia"]["mean"]
        li = v["linear_structured"]["iia"]["mean"]
        if nl is not None and li is not None:
            nonlinearity_effects.append((r, nl - li))
    nonlinearity_effects.sort(key=lambda x: x[1], reverse=True)

    # Rank regions by largest structure effect
    structure_effects = []
    for r, v in region_results.items():
        li = v["linear_structured"]["iia"]["mean"]
        nu = v["nonlinear_unstructured"]["iia"]["mean"]
        if li is not None and nu is not None:
            structure_effects.append((r, li - nu))
    structure_effects.sort(key=lambda x: x[1], reverse=True)

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
        "region_results": region_results,
        "prediction_tests": prediction_tests,
        "top_nonlinearity_effect_regions": nonlinearity_effects[:10],
        "top_structure_effect_regions": structure_effects[:10],
    }

    out_path = RESULTS_DIR / "linear_vae_ablation.json"
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
