"""Experiment 72: Sutter vacuity test with continuous metrics.

Re-runs the exp59 Sutter test (real vs random noise) but measures continuous
metrics alongside binary IIA:
  - KL divergence: KL(p_orig || p_swapped) — how much does the classifier's
    full output distribution shift?
  - JS divergence: symmetric version of KL
  - Probability shift: |p(choice=1|orig) - p(choice=1|swapped)| — magnitude
    of confidence change
  - Logit difference: |logit_orig - logit_swapped| — raw pre-softmax shift

Key hypothesis: IIA is vacuous for nonlinear methods because it's BINARY —
any flexible model can push predictions past 50%. But continuous metrics may
show that the VAE produces LARGER distribution shifts on real data than random,
even if the binary flip rate is similar. If so, the vacuity is partly a
measurement artifact of the binary metric, not purely a model problem.
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

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp72"
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
MIN_TRIALS_PER_CONDITION = 20

Z_DIM = 3
Z_OTHER_DIM = 15
HIDDEN_DIM = 128
N_EPOCHS = 300
BATCH_SIZE = 64
LR = 1e-3
BETA_KL = 1.0
ALPHA_CHOICE = 10.0
MLP_RECON_WEIGHT = 1.0
MLP_CLASS_WEIGHT = 10.0
N_IIA_PAIRS = 100


def _zscore(X):
    mu = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    return (X - mu) / std, mu, std


class UnconstrainedMLP(nn.Module):
    def __init__(self, n_neurons, z_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_neurons, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, z_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_neurons),
        )
        self.classifier = nn.Linear(z_dim, 2)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        logits = self.classifier(z)
        return {"z": z, "recon": recon, "logits": logits}


class StructuredVAE(nn.Module):
    def __init__(self, n_neurons, z_choice_dim, z_other_dim, hidden_dim):
        super().__init__()
        self.z_choice_dim = z_choice_dim
        self.z_other_dim = z_other_dim
        self.encoder = nn.Sequential(
            nn.Linear(n_neurons, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        z_dim = z_choice_dim + z_other_dim
        self.fc_mu_choice = nn.Linear(hidden_dim, z_choice_dim)
        self.fc_logvar_choice = nn.Linear(hidden_dim, z_choice_dim)
        self.fc_mu_other = nn.Linear(hidden_dim, z_other_dim)
        self.fc_logvar_other = nn.Linear(hidden_dim, z_other_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_neurons),
        )
        self.choice_classifier = nn.Linear(z_choice_dim, 2)

    def encode(self, x):
        h = self.encoder(x)
        return (self.fc_mu_choice(h), self.fc_logvar_choice(h),
                self.fc_mu_other(h), self.fc_logvar_other(h))

    def forward(self, x):
        mu_c, lv_c, mu_o, lv_o = self.encode(x)
        z_c = mu_c + torch.exp(0.5 * lv_c) * torch.randn_like(mu_c)
        z_o = mu_o + torch.exp(0.5 * lv_o) * torch.randn_like(mu_o)
        z = torch.cat([z_c, z_o], dim=-1)
        recon = self.decoder(z)
        logits = self.choice_classifier(z_c)
        return {"recon": recon, "logits": logits, "mu_c": mu_c, "lv_c": lv_c,
                "mu_o": mu_o, "lv_o": lv_o, "z_c": z_c, "z_o": z_o}


def _train_mlp(activity, choice_labels, z_dim, hidden_dim, device):
    activity_norm, _, _ = _zscore(activity)
    X = torch.tensor(activity_norm, dtype=torch.float32, device=device)
    y = torch.tensor(choice_labels, dtype=torch.long, device=device)
    model = UnconstrainedMLP(activity.shape[1], z_dim, hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ds = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                                         drop_last=len(ds) > BATCH_SIZE)
    model.train()
    for _ in range(N_EPOCHS):
        for xb, yb in loader:
            out = model(xb)
            loss = (MLP_RECON_WEIGHT * F.mse_loss(out["recon"], xb)
                    + MLP_CLASS_WEIGHT * F.cross_entropy(out["logits"], yb))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model, activity_norm


def _train_vae(activity, choice_labels, z_choice_dim, z_other_dim, hidden_dim, device):
    activity_norm, _, _ = _zscore(activity)
    X = torch.tensor(activity_norm, dtype=torch.float32, device=device)
    y = torch.tensor(choice_labels, dtype=torch.long, device=device)
    model = StructuredVAE(activity.shape[1], z_choice_dim, z_other_dim, hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ds = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                                         drop_last=len(ds) > BATCH_SIZE)
    model.train()
    for _ in range(N_EPOCHS):
        for xb, yb in loader:
            out = model(xb)
            recon_loss = F.mse_loss(out["recon"], xb)
            kl_c = -0.5 * torch.mean(1 + out["lv_c"] - out["mu_c"].pow(2) - out["lv_c"].exp())
            kl_o = -0.5 * torch.mean(1 + out["lv_o"] - out["mu_o"].pow(2) - out["lv_o"].exp())
            cls_loss = F.cross_entropy(out["logits"], yb)
            loss = recon_loss + BETA_KL * (kl_c + kl_o) + ALPHA_CHOICE * cls_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model, activity_norm


def _kl_divergence(p, q):
    """KL(p || q) for probability vectors, with epsilon for stability."""
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    return float(np.sum(p * np.log(p / q)))


def _js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def _compute_continuous_metrics_latent(model, activity_norm, evidence_labels,
                                       choice_labels, method, device):
    """Compute IIA + continuous metrics for MLP or VAE."""
    X = torch.tensor(activity_norm, dtype=torch.float32, device=device)
    left_idx = np.where(evidence_labels == 0)[0]
    right_idx = np.where(evidence_labels == 1)[0]
    if len(left_idx) < MIN_TRIALS_PER_CONDITION or len(right_idx) < MIN_TRIALS_PER_CONDITION:
        return None

    n_pairs = min(N_IIA_PAIRS, len(left_idx), len(right_idx))
    left_sample = left_idx[np.random.choice(len(left_idx), n_pairs, replace=False)]
    right_sample = right_idx[np.random.choice(len(right_idx), n_pairs, replace=False)]

    flips = 0
    total = 0
    kl_values = []
    js_values = []
    prob_shift_values = []
    logit_diff_values = []

    model.eval()
    with torch.no_grad():
        for li, ri in zip(left_sample, right_sample):
            x_l = X[li].unsqueeze(0)
            x_r = X[ri].unsqueeze(0)

            if method == "mlp":
                out_l = model(x_l)
                out_r = model(x_r)
                orig_logits_l = out_l["logits"]
                orig_logits_r = out_r["logits"]
                swap_logits_l = model.classifier(out_r["z"])
                swap_logits_r = model.classifier(out_l["z"])
            elif method == "vae":
                mu_c_l, _, mu_o_l, _ = model.encode(x_l)
                mu_c_r, _, mu_o_r, _ = model.encode(x_r)
                orig_logits_l = model.choice_classifier(mu_c_l)
                orig_logits_r = model.choice_classifier(mu_c_r)
                swap_logits_l = model.choice_classifier(mu_c_r)
                swap_logits_r = model.choice_classifier(mu_c_l)
            else:
                raise ValueError(method)

            for orig_logits, swap_logits in [(orig_logits_l, swap_logits_l),
                                              (orig_logits_r, swap_logits_r)]:
                orig_pred = orig_logits.argmax(dim=1).item()
                swap_pred = swap_logits.argmax(dim=1).item()
                if orig_pred != swap_pred:
                    flips += 1
                total += 1

                p_orig = F.softmax(orig_logits, dim=-1).cpu().numpy().flatten()
                p_swap = F.softmax(swap_logits, dim=-1).cpu().numpy().flatten()

                kl_values.append(_kl_divergence(p_orig, p_swap))
                js_values.append(_js_divergence(p_orig, p_swap))
                prob_shift_values.append(float(abs(p_orig[1] - p_swap[1])))

                lo = orig_logits.cpu().numpy().flatten()
                ls = swap_logits.cpu().numpy().flatten()
                logit_diff_values.append(float(np.mean(np.abs(lo - ls))))

    return {
        "iia": float(flips / total) if total > 0 else None,
        "kl_mean": float(np.mean(kl_values)),
        "kl_std": float(np.std(kl_values)),
        "js_mean": float(np.mean(js_values)),
        "js_std": float(np.std(js_values)),
        "prob_shift_mean": float(np.mean(prob_shift_values)),
        "prob_shift_std": float(np.std(prob_shift_values)),
        "logit_diff_mean": float(np.mean(logit_diff_values)),
        "logit_diff_std": float(np.std(logit_diff_values)),
    }


def _compute_continuous_metrics_linear(activity, evidence_labels, choice_labels, V):
    """Compute IIA + continuous metrics for linear DAS."""
    left_idx = np.where(evidence_labels == 0)[0]
    right_idx = np.where(evidence_labels == 1)[0]
    if len(left_idx) < MIN_TRIALS_PER_CONDITION or len(right_idx) < MIN_TRIALS_PER_CONDITION:
        return None

    lda = LinearDiscriminantAnalysis()
    try:
        lda.fit(activity, choice_labels)
    except Exception:
        return None

    n_pairs = min(N_IIA_PAIRS, len(left_idx), len(right_idx))
    left_sample = left_idx[np.random.choice(len(left_idx), n_pairs, replace=False)]
    right_sample = right_idx[np.random.choice(len(right_idx), n_pairs, replace=False)]

    flips = 0
    total = 0
    prob_shift_values = []

    for li, ri in zip(left_sample, right_sample):
        act_l = activity[li].copy()
        act_r = activity[ri].copy()
        proj_l = V @ (V.T @ act_l)
        proj_r = V @ (V.T @ act_r)
        act_l_swapped = act_l - proj_l + proj_r
        act_r_swapped = act_r - proj_r + proj_l

        for act_orig, act_swap in [(act_l, act_l_swapped), (act_r, act_r_swapped)]:
            orig_pred = lda.predict(act_orig.reshape(1, -1))[0]
            swap_pred = lda.predict(act_swap.reshape(1, -1))[0]
            if orig_pred != swap_pred:
                flips += 1
            total += 1

            # LDA doesn't have softmax, use decision function as proxy
            orig_dec = lda.decision_function(act_orig.reshape(1, -1)).flatten()
            swap_dec = lda.decision_function(act_swap.reshape(1, -1)).flatten()
            prob_shift_values.append(float(abs(orig_dec[0] - swap_dec[0])))

    return {
        "iia": float(flips / total) if total > 0 else None,
        "decision_shift_mean": float(np.mean(prob_shift_values)),
        "decision_shift_std": float(np.std(prob_shift_values)),
    }


def _estimate_lda_subspace(activity, labels, n_dims=5):
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


def _contrast_to_evidence_label(sess):
    cl = sess.get("contrast_left", np.array([]))
    cr = sess.get("contrast_right", np.array([]))
    if len(cl) == 0 or len(cr) == 0:
        return None
    n = sess["spks"].shape[2]
    n = min(n, len(cl), len(cr))
    cl, cr = cl[:n], cr[:n]
    evidence = cr - cl
    nonzero = evidence != 0
    if nonzero.sum() < MIN_TRIALS_PER_CONDITION:
        return None
    labels = np.full(n, -1, dtype=int)
    labels[evidence > 0] = 1
    labels[evidence < 0] = 0
    return labels


def run(max_sessions: int | None = None) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"{datetime.now().isoformat()} Starting Sutter continuous metrics experiment "
                f"with {len(sessions)} sessions on {device}")

    region_data: dict[str, list[dict]] = {}
    for sess_idx, sess in enumerate(tqdm(sessions, desc="Loading sessions")):
        choice_labels = get_choice_labels(sess)
        if len(np.unique(choice_labels)) < 2:
            continue
        evidence_labels = _contrast_to_evidence_label(sess)
        if evidence_labels is None:
            continue
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
                "activity": activity[valid],
                "choice_labels": ch[valid],
                "evidence_labels": ev[valid],
                "n_neurons": int(activity.shape[1]),
            })

    logger.info(f"{datetime.now().isoformat()} {len(region_data)} regions loaded")

    METHODS = ["mlp", "vae", "linear_das"]
    CONDITIONS = ["real", "random"]

    region_results = {}
    for region in tqdm(sorted(region_data.keys()), desc="Regions"):
        measurements = region_data[region]
        results_by_key: dict[str, list[dict]] = {
            f"{m}_{c}": [] for m in METHODS for c in CONDITIONS
        }

        for meas in measurements:
            act_real = meas["activity"]
            ch = meas["choice_labels"]
            ev = meas["evidence_labels"]
            n_trials, n_neurons = act_real.shape
            act_random = np.random.randn(n_trials, n_neurons).astype(np.float32)

            z_dim = min(Z_DIM, n_neurons // 5, n_neurons - 1)
            if z_dim < 1:
                continue
            hidden = min(HIDDEN_DIM, n_neurons * 2)
            z_other = min(Z_OTHER_DIM, n_neurons // 3, n_neurons - z_dim - 1)
            if z_other < 1:
                continue

            data_map = {"real": act_real, "random": act_random}

            for condition in CONDITIONS:
                act = data_map[condition]

                # MLP
                try:
                    mlp_model, mlp_norm = _train_mlp(act, ch, z_dim, hidden, device)
                    r = _compute_continuous_metrics_latent(
                        mlp_model, mlp_norm, ev, ch, "mlp", device)
                    if r:
                        results_by_key[f"mlp_{condition}"].append(r)
                except Exception:
                    pass

                # VAE
                try:
                    vae_model, vae_norm = _train_vae(act, ch, z_dim, z_other, hidden, device)
                    r = _compute_continuous_metrics_latent(
                        vae_model, vae_norm, ev, ch, "vae", device)
                    if r:
                        results_by_key[f"vae_{condition}"].append(r)
                except Exception:
                    pass

                # Linear DAS
                try:
                    V = _estimate_lda_subspace(act, ev, n_dims=z_dim)
                    if V is not None:
                        r = _compute_continuous_metrics_linear(act, ev, ch, V)
                        if r:
                            results_by_key[f"linear_das_{condition}"].append(r)
                except Exception:
                    pass

        # Aggregate per region
        agg = {"region": region, "n_sessions": len(measurements)}
        for key, results_list in results_by_key.items():
            if not results_list:
                continue
            for metric in ["iia", "kl_mean", "js_mean", "prob_shift_mean",
                           "logit_diff_mean", "decision_shift_mean"]:
                vals = [r[metric] for r in results_list if metric in r and r[metric] is not None]
                if vals:
                    agg[f"{key}_{metric}"] = float(np.mean(vals))

        region_results[region] = agg

    # Summary comparisons: real vs random for each metric
    summary = {"n_regions": len(region_results)}
    for method in ["mlp", "vae"]:
        for metric in ["iia", "kl_mean", "js_mean", "prob_shift_mean", "logit_diff_mean"]:
            real_key = f"{method}_real_{metric}"
            rand_key = f"{method}_random_{metric}"
            reals = [v[real_key] for v in region_results.values() if real_key in v]
            rands = [v[rand_key] for v in region_results.values() if rand_key in v]
            if reals and rands:
                # Paired comparison where both exist
                paired_r, paired_d = [], []
                for v in region_results.values():
                    if real_key in v and rand_key in v:
                        paired_r.append(v[real_key])
                        paired_d.append(v[rand_key])
                if len(paired_r) >= 5:
                    diffs = np.array(paired_r) - np.array(paired_d)
                    try:
                        w_stat, w_p = wilcoxon(diffs, alternative="greater")
                    except Exception:
                        w_stat, w_p = None, None
                    summary[f"{method}_{metric}"] = {
                        "real_mean": float(np.mean(paired_r)),
                        "random_mean": float(np.mean(paired_d)),
                        "diff_mean": float(np.mean(diffs)),
                        "n": len(paired_r),
                        "wilcoxon_p": float(w_p) if w_p is not None else None,
                    }

    # Linear DAS summary
    for metric in ["iia", "decision_shift_mean"]:
        real_key = f"linear_das_real_{metric}"
        rand_key = f"linear_das_random_{metric}"
        paired_r, paired_d = [], []
        for v in region_results.values():
            if real_key in v and rand_key in v:
                paired_r.append(v[real_key])
                paired_d.append(v[rand_key])
        if len(paired_r) >= 5:
            diffs = np.array(paired_r) - np.array(paired_d)
            try:
                w_stat, w_p = wilcoxon(diffs, alternative="greater")
            except Exception:
                w_stat, w_p = None, None
            summary[f"linear_das_{metric}"] = {
                "real_mean": float(np.mean(paired_r)),
                "random_mean": float(np.mean(paired_d)),
                "diff_mean": float(np.mean(diffs)),
                "n": len(paired_r),
                "wilcoxon_p": float(w_p) if w_p is not None else None,
            }

    results = {
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "per_region": region_results,
    }

    out_path = RESULTS_DIR / f"exp72_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"{datetime.now().isoformat()} Results saved to {out_path}")

    return results
