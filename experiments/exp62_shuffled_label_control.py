"""Experiment 62: Shuffled label control — the decisive VAE vacuousness test.

The Sutter dilemma result (exp59) showed that our structured VAE achieves IIA=0.70
on BOTH real data AND random Gaussian noise, same as an unconstrained MLP. This
means the IIA metric may be vacuous for nonlinear methods.

But exp59 tested random DATA with real labels. This experiment tests real DATA with
SHUFFLED labels. The distinction matters:

- If the VAE achieves high IIA on real neural data with shuffled labels, the VAE is
  truly vacuous — it fits any label structure regardless of whether it is real.
- If IIA drops with shuffled labels, the VAE IS learning real neural structure. The
  high IIA on random data in exp59 means the encoder is flexible enough to also fit
  random data, but the labels (not the data) are what drive the learned subspace.

For each region:
  a. Real activity + real labels -> train VAE -> IIA (baseline, should match exp57)
  b. Real activity + shuffled choice labels -> train VAE -> IIA (x5 shuffles)
  c. Real activity + shuffled evidence labels -> train VAE -> IIA (x5 shuffles)

Additional diagnostic: does the shuffled-label VAE's z_choice still predict the
ORIGINAL (unshuffled) labels? If yes, the encoder found real structure despite
being trained on noise labels.
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp62"
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
MIN_TRIALS_PER_CONDITION = 20

# VAE hyperparameters (match exp57/exp59)
Z_CHOICE_DIM = 3
Z_OTHER_DIM = 15
HIDDEN_DIM = 128
N_EPOCHS = 300
BATCH_SIZE = 64
LR = 1e-3
BETA_KL = 1.0
ALPHA_CHOICE = 10.0

N_IIA_PAIRS = 100
N_SHUFFLES = 5


# ---------------------------------------------------------------------------
# Model (same StructuredVAE as exp57/exp59)
# ---------------------------------------------------------------------------

class StructuredVAE(nn.Module):
    """Semi-supervised VAE with supervised z_choice and unsupervised z_other."""

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

    def get_choice_subspace(self) -> np.ndarray:
        W1 = self.enc_trunk[0].weight.detach()
        W2 = self.enc_trunk[2].weight.detach()
        W_mu = self.enc_choice_mu.weight.detach()
        W_composed = W_mu @ W2 @ W1
        Q, _ = torch.linalg.qr(W_composed.T)
        return Q.cpu().numpy()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _zscore(activity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = activity.mean(axis=0, keepdims=True)
    std = activity.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (activity - mu) / std, mu, std


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
        mu_choice = out_full["mu_choice"]

    subspace_dirs = model.get_choice_subspace()
    return {
        "model": model,
        "choice_accuracy": acc,
        "subspace_directions": subspace_dirs,
        "mu_choice": mu_choice,
        "activity_norm_tensor": X,
    }


# ---------------------------------------------------------------------------
# IIA computation
# ---------------------------------------------------------------------------

def _compute_iia_latent(
    model: StructuredVAE,
    activity_norm: torch.Tensor,
    evidence_labels: np.ndarray,
    choice_labels: np.ndarray,
    n_pairs: int = N_IIA_PAIRS,
) -> float | None:
    """IIA via latent-space swap: swap z_choice between opposite-evidence pairs."""
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

            mu_c_l, _, mu_o_l, _ = model.encode(x_l)
            mu_c_r, _, mu_o_r, _ = model.encode(x_r)

            orig_pred_l = model.choice_classifier(mu_c_l).argmax(dim=1).item()
            orig_pred_r = model.choice_classifier(mu_c_r).argmax(dim=1).item()

            swap_pred_l = model.choice_classifier(mu_c_r).argmax(dim=1).item()
            swap_pred_r = model.choice_classifier(mu_c_l).argmax(dim=1).item()

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
    """IIA using linear subspace projection swap."""
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


# ---------------------------------------------------------------------------
# Helper: check if shuffled-label VAE still predicts real labels
# ---------------------------------------------------------------------------

def _cross_label_accuracy(
    model: StructuredVAE,
    activity_norm: torch.Tensor,
    original_labels: np.ndarray,
) -> float:
    """Accuracy of the shuffled-label VAE's z_choice on the ORIGINAL labels."""
    model.eval()
    with torch.no_grad():
        mu_c, _, _, _ = model.encode(activity_norm)
        logits = model.choice_classifier(mu_c)
        preds = logits.argmax(dim=1).cpu().numpy()

    # The classifier was trained on shuffled labels, so its output classes may
    # be flipped relative to the original labels. Check both orientations.
    acc_direct = float(np.mean(preds == original_labels))
    acc_flipped = float(np.mean(preds != original_labels))
    return max(acc_direct, acc_flipped)


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
# Per-region evaluation
# ---------------------------------------------------------------------------

def _evaluate_one_condition(
    activity: np.ndarray,
    choice_labels: np.ndarray,
    evidence_labels: np.ndarray,
    device: str,
) -> dict | None:
    """Train VAE on given labels, return IIA and accuracy."""
    n_neurons = activity.shape[1]
    z_choice = min(Z_CHOICE_DIM, n_neurons // 5, n_neurons - 1)
    z_other = min(Z_OTHER_DIM, n_neurons // 3, n_neurons - z_choice - 1)
    if z_choice < 1 or z_other < 1:
        return None
    hidden = min(HIDDEN_DIM, n_neurons * 2)

    try:
        result = train_vae(
            activity, choice_labels,
            z_choice_dim=z_choice, z_other_dim=z_other,
            hidden_dim=hidden, device=device,
        )
    except Exception as e:
        logger.warning(f"VAE training failed: {e}")
        return None

    iia = _compute_iia_latent(
        result["model"], result["activity_norm_tensor"],
        evidence_labels, choice_labels,
    )

    return {
        "iia": iia,
        "choice_accuracy": result["choice_accuracy"],
        "model": result["model"],
        "activity_norm_tensor": result["activity_norm_tensor"],
        "subspace_directions": result["subspace_directions"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(max_sessions: int | None = None) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"{datetime.now().isoformat()} Starting shuffled label control experiment "
                f"with {len(sessions)} sessions on {device}, {N_SHUFFLES} shuffles")

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
    jsonl_path = RESULTS_DIR / "shuffled_label_incremental.jsonl"

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

        real_iias: list[float] = []
        real_accs: list[float] = []
        shuffled_choice_iias: list[float] = []
        shuffled_choice_accs: list[float] = []
        shuffled_evidence_iias: list[float] = []
        shuffled_evidence_accs: list[float] = []
        cross_label_accs: list[float] = []

        for m in measurements:
            activity = m["activity"]
            ch_real = m["choice_labels"]
            ev_real = m["evidence_labels"]

            # (a) Real labels baseline
            result_real = _evaluate_one_condition(activity, ch_real, ev_real, device)
            if result_real is not None and result_real["iia"] is not None:
                real_iias.append(result_real["iia"])
                real_accs.append(result_real["choice_accuracy"])

            # (b) Shuffled choice labels
            for _shuf_idx in range(N_SHUFFLES):
                ch_shuffled = np.random.permutation(ch_real)
                result_shuf_ch = _evaluate_one_condition(activity, ch_shuffled, ev_real, device)
                if result_shuf_ch is not None and result_shuf_ch["iia"] is not None:
                    shuffled_choice_iias.append(result_shuf_ch["iia"])
                    shuffled_choice_accs.append(result_shuf_ch["choice_accuracy"])

                    # Diagnostic: does the shuffled-label model still predict real labels?
                    cross_acc = _cross_label_accuracy(
                        result_shuf_ch["model"],
                        result_shuf_ch["activity_norm_tensor"],
                        ch_real,
                    )
                    cross_label_accs.append(cross_acc)

            # (c) Shuffled evidence labels
            for _shuf_idx in range(N_SHUFFLES):
                ev_shuffled = np.random.permutation(ev_real)
                result_shuf_ev = _evaluate_one_condition(activity, ch_real, ev_shuffled, device)
                if result_shuf_ev is not None and result_shuf_ev["iia"] is not None:
                    shuffled_evidence_iias.append(result_shuf_ev["iia"])
                    shuffled_evidence_accs.append(result_shuf_ev["choice_accuracy"])

        # Aggregate for this region
        region_result: dict = {
            "region": region,
            "n_sessions": len(measurements),
            "n_shuffles": N_SHUFFLES,
            # Real labels
            "real_iia_mean": float(np.mean(real_iias)) if real_iias else None,
            "real_iia_std": float(np.std(real_iias)) if len(real_iias) > 1 else None,
            "real_iia_n": len(real_iias),
            "real_acc_mean": float(np.mean(real_accs)) if real_accs else None,
            # Shuffled choice labels
            "shuffled_choice_iia_mean": float(np.mean(shuffled_choice_iias)) if shuffled_choice_iias else None,
            "shuffled_choice_iia_std": float(np.std(shuffled_choice_iias)) if len(shuffled_choice_iias) > 1 else None,
            "shuffled_choice_iia_n": len(shuffled_choice_iias),
            "shuffled_choice_acc_mean": float(np.mean(shuffled_choice_accs)) if shuffled_choice_accs else None,
            # Shuffled evidence labels
            "shuffled_evidence_iia_mean": float(np.mean(shuffled_evidence_iias)) if shuffled_evidence_iias else None,
            "shuffled_evidence_iia_std": float(np.std(shuffled_evidence_iias)) if len(shuffled_evidence_iias) > 1 else None,
            "shuffled_evidence_iia_n": len(shuffled_evidence_iias),
            "shuffled_evidence_acc_mean": float(np.mean(shuffled_evidence_accs)) if shuffled_evidence_accs else None,
            # Cross-label diagnostic
            "cross_label_acc_mean": float(np.mean(cross_label_accs)) if cross_label_accs else None,
            "cross_label_acc_std": float(np.std(cross_label_accs)) if len(cross_label_accs) > 1 else None,
            # Gaps
            "real_vs_shuffled_choice_gap": (
                float(np.mean(real_iias) - np.mean(shuffled_choice_iias))
                if real_iias and shuffled_choice_iias else None
            ),
            "real_vs_shuffled_evidence_gap": (
                float(np.mean(real_iias) - np.mean(shuffled_evidence_iias))
                if real_iias and shuffled_evidence_iias else None
            ),
        }

        region_results[region] = region_result
        with open(jsonl_path, "a") as jf:
            jf.write(json.dumps(region_result, default=str) + "\n")

    logger.info(f"{datetime.now().isoformat()} All regions processed")

    # --- Aggregate statistical tests ---
    prediction_tests = {}

    # Test 1: Real IIA vs shuffled-choice IIA (the main test)
    paired_real: list[float] = []
    paired_shuf_ch: list[float] = []
    for v in region_results.values():
        if v.get("real_iia_mean") is not None and v.get("shuffled_choice_iia_mean") is not None:
            paired_real.append(v["real_iia_mean"])
            paired_shuf_ch.append(v["shuffled_choice_iia_mean"])

    if len(paired_real) >= 5:
        diffs = np.array(paired_real) - np.array(paired_shuf_ch)
        try:
            w_stat, w_p = wilcoxon(diffs, alternative="greater")
        except Exception:
            w_stat, w_p = None, None
        prediction_tests["real_vs_shuffled_choice"] = {
            "real_iia_mean": float(np.mean(paired_real)),
            "shuffled_choice_iia_mean": float(np.mean(paired_shuf_ch)),
            "mean_diff": float(np.mean(diffs)),
            "median_diff": float(np.median(diffs)),
            "n_regions": len(paired_real),
            "n_real_wins": int(np.sum(diffs > 0)),
            "wilcoxon_W": float(w_stat) if w_stat is not None else None,
            "wilcoxon_p": float(w_p) if w_p is not None else None,
            "interpretation": (
                "THE DECISIVE TEST. Positive mean_diff with significant p = "
                "the VAE learns genuine choice structure, not just fitting any labels. "
                "If mean_diff ~= 0, the VAE is truly vacuous."
            ),
        }

    # Test 2: Real IIA vs shuffled-evidence IIA
    paired_real_ev: list[float] = []
    paired_shuf_ev: list[float] = []
    for v in region_results.values():
        if v.get("real_iia_mean") is not None and v.get("shuffled_evidence_iia_mean") is not None:
            paired_real_ev.append(v["real_iia_mean"])
            paired_shuf_ev.append(v["shuffled_evidence_iia_mean"])

    if len(paired_real_ev) >= 5:
        diffs = np.array(paired_real_ev) - np.array(paired_shuf_ev)
        try:
            w_stat, w_p = wilcoxon(diffs, alternative="greater")
        except Exception:
            w_stat, w_p = None, None
        prediction_tests["real_vs_shuffled_evidence"] = {
            "real_iia_mean": float(np.mean(paired_real_ev)),
            "shuffled_evidence_iia_mean": float(np.mean(paired_shuf_ev)),
            "mean_diff": float(np.mean(diffs)),
            "median_diff": float(np.median(diffs)),
            "n_regions": len(paired_real_ev),
            "n_real_wins": int(np.sum(diffs > 0)),
            "wilcoxon_W": float(w_stat) if w_stat is not None else None,
            "wilcoxon_p": float(w_p) if w_p is not None else None,
            "interpretation": (
                "Evidence labels define the swap pairs for IIA. Shuffling them "
                "should destroy IIA structure even with a good encoder, since the "
                "swap pairs are now random (not matched by evidence direction)."
            ),
        }

    # Test 3: Cross-label accuracy — does the shuffled-label VAE find real structure?
    cross_accs = [
        v["cross_label_acc_mean"] for v in region_results.values()
        if v.get("cross_label_acc_mean") is not None
    ]
    if cross_accs:
        prediction_tests["cross_label_accuracy"] = {
            "mean": float(np.mean(cross_accs)),
            "std": float(np.std(cross_accs)),
            "n_regions": len(cross_accs),
            "interpretation": (
                "Accuracy of the shuffled-choice-label VAE's z_choice at predicting "
                "ORIGINAL (unshuffled) choice labels. If >> 0.5, the encoder found "
                "real neural structure despite being trained on noise labels — the "
                "data geometry forces it toward real structure even when labels are wrong."
            ),
        }

    # Test 4: Shuffled-choice accuracy on shuffled labels (sanity: should be high)
    shuf_accs = [
        v["shuffled_choice_acc_mean"] for v in region_results.values()
        if v.get("shuffled_choice_acc_mean") is not None
    ]
    if shuf_accs:
        prediction_tests["shuffled_choice_training_accuracy"] = {
            "mean": float(np.mean(shuf_accs)),
            "std": float(np.std(shuf_accs)),
            "n_regions": len(shuf_accs),
            "interpretation": (
                "The VAE's accuracy on the shuffled labels it was trained on. "
                "If high, the VAE successfully fit the shuffled labels (expected). "
                "The question is whether this translates to high IIA."
            ),
        }

    # Test 5: Shuffled-choice IIA distribution (is it at chance?)
    all_shuf_ch_iias = [
        v["shuffled_choice_iia_mean"] for v in region_results.values()
        if v.get("shuffled_choice_iia_mean") is not None
    ]
    if all_shuf_ch_iias:
        prediction_tests["shuffled_choice_iia_distribution"] = {
            "mean": float(np.mean(all_shuf_ch_iias)),
            "std": float(np.std(all_shuf_ch_iias)),
            "n_regions": len(all_shuf_ch_iias),
            "above_chance_fraction": float(np.mean(np.array(all_shuf_ch_iias) > 0.55)),
            "interpretation": (
                "Distribution of IIA when choice labels are shuffled. "
                "If mean ~= 0.5, the VAE cannot achieve high IIA with wrong labels. "
                "If mean >> 0.5, the VAE is vacuous (fits any label structure)."
            ),
        }

    # --- Summary ---
    results = {
        "timestamp": datetime.now().isoformat(),
        "n_sessions": len(sessions),
        "n_regions": len(region_data),
        "n_regions_analyzed": len(region_results),
        "device": device,
        "n_shuffles": N_SHUFFLES,
        "hyperparameters": {
            "z_choice_dim": Z_CHOICE_DIM,
            "z_other_dim": Z_OTHER_DIM,
            "hidden_dim": HIDDEN_DIM,
            "n_epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "beta_kl": BETA_KL,
            "alpha_choice": ALPHA_CHOICE,
            "n_iia_pairs": N_IIA_PAIRS,
            "n_shuffles": N_SHUFFLES,
        },
        "region_results": {r: v for r, v in region_results.items()},
        "prediction_tests": prediction_tests,
    }

    out_path = RESULTS_DIR / "shuffled_label_control.json"
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
