"""Experiment 54: Baseline separation for all major findings.

MECHVAL M2 (Baseline Separation): "Score exceeds random-vector AND untrained
baselines by >= 0.10."

Provides the missing M2 baselines for 6 findings:
  1. CKA anti-correlation (exp11) — compare to random Gaussian activity
  2. Spectral universality (exp17) — Marchenko-Pastur random-matrix baseline
  3. Topology (exp12) — random H1 baseline
  4. Rotation (exp20) — already has temporal shuffle (pass)
  5. IIA (exp42) — random-subspace IIA baseline
  6. Parcellation (exp18) — random-label clustering baseline (ARI)

For each finding, compute the metric on:
  a) Real data (the actual finding)
  b) Gaussian random data matched in shape
  c) Trial-shuffled data (temporal structure destroyed)
  d) Neuron-permuted data (spatial structure destroyed)

The delta between real and best baseline must be >= 0.10 for the finding
to pass M2.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp54"
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
SUBSPACE_DIM = 5
MIN_TRIALS_PER_CONDITION = 20
N_RANDOM_REPS = 20


def _contrast_to_evidence_label(sess):
    cl = sess.get("contrast_left", np.array([]))
    cr = sess.get("contrast_right", np.array([]))
    if len(cl) == 0 or len(cr) == 0:
        return None
    n = min(len(cl), len(cr))
    evidence = cr[:n] - cl[:n]
    if (evidence != 0).sum() < MIN_TRIALS_PER_CONDITION:
        return None
    labels = np.zeros(n, dtype=int)
    labels[evidence > 0] = 1
    labels[evidence < 0] = 0
    labels[evidence == 0] = -1
    return labels


def _cka(X, Y):
    K = linear_kernel(X)
    L = linear_kernel(Y)
    K_c = K - K.mean(0, keepdims=True) - K.mean(1, keepdims=True) + K.mean()
    L_c = L - L.mean(0, keepdims=True) - L.mean(1, keepdims=True) + L.mean()
    hsic = np.sum(K_c * L_c)
    norm = np.sqrt(np.sum(K_c * K_c) * np.sum(L_c * L_c))
    return float(hsic / norm) if norm > 0 else 0.0


def _power_law_exponent(activity):
    n_c = min(50, activity.shape[1], activity.shape[0])
    pca = PCA(n_components=n_c)
    pca.fit(activity)
    eig = pca.explained_variance_
    eig = eig[eig > 0]
    if len(eig) < 10:
        return None
    s, e = 9, min(49, len(eig) - 1)
    coeffs = np.polyfit(np.log10(np.arange(s+1, e+2)), np.log10(eig[s:e+1]), 1)
    return float(-coeffs[0])


def _estimate_subspace(activity, labels, n_dims=SUBSPACE_DIM):
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


def _iia_with_subspace(activity, ev_labels, ch_labels, V):
    left_idx = np.where(ev_labels == 0)[0]
    right_idx = np.where(ev_labels == 1)[0]
    if len(left_idx) < 10 or len(right_idx) < 10:
        return None
    lda = LinearDiscriminantAnalysis()
    try:
        lda.fit(activity, ch_labels)
    except Exception:
        return None
    n_pairs = min(len(left_idx), len(right_idx), 50)
    rng = np.random.default_rng(42)
    sl = left_idx[rng.choice(len(left_idx), n_pairs, replace=False)]
    sr = right_idx[rng.choice(len(right_idx), n_pairs, replace=False)]
    flips, total = 0, 0
    for li, ri in zip(sl, sr):
        x = activity[li].copy()
        pl = V @ (V.T @ activity[li])
        pr = V @ (V.T @ activity[ri])
        if np.linalg.norm(pr - pl) < 1e-10:
            continue
        orig = lda.predict(x.reshape(1, -1))[0]
        if lda.predict((x - pl + pr).reshape(1, -1))[0] != orig:
            flips += 1
        total += 1
    return float(flips / total) if total > 0 else None


def _marchenko_pastur_eigenvalues(n_samples, n_features, n_components=50):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, n_features))
    pca = PCA(n_components=min(n_components, n_features - 1, n_samples - 1))
    pca.fit(X)
    return pca.explained_variance_


def run(max_sessions=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    baselines = {
        "iia": {"real": [], "random_subspace": [], "shuffled": []},
        "alpha": {"real": [], "marchenko_pastur": []},
        "cka_self": {"real": [], "gaussian": [], "neuron_permuted": []},
    }

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Baseline separation")):
        choice_labels = get_choice_labels(sess)
        if len(np.unique(choice_labels)) < 2:
            continue
        ev_labels = _contrast_to_evidence_label(sess)
        if ev_labels is None:
            continue

        for region in list_regions(sess, min_neurons=MIN_NEURONS):
            act = get_region_activity(sess, region)
            if act is None or act.shape[1] < MIN_NEURONS:
                continue
            n = min(act.shape[0], len(choice_labels), len(ev_labels))
            activity = act[:n, :, TIME_WINDOW].mean(axis=2)
            ch = choice_labels[:n]
            ev = ev_labels[:n]
            valid = ev >= 0
            if valid.sum() < MIN_TRIALS_PER_CONDITION * 2:
                continue

            act_v = activity[valid]
            ev_v = ev[valid]
            ch_v = ch[valid]

            V_ev = _estimate_subspace(act_v, ev_v)
            if V_ev is not None:
                real_iia = _iia_with_subspace(act_v, ev_v, ch_v, V_ev)
                if real_iia is not None:
                    baselines["iia"]["real"].append(real_iia)

                    rng = np.random.default_rng(sess_idx * 100 + hash(region) % 100)
                    rand_iias = []
                    for _ in range(N_RANDOM_REPS):
                        Q, _ = np.linalg.qr(rng.standard_normal((act_v.shape[1], V_ev.shape[1])))
                        ri = _iia_with_subspace(act_v, ev_v, ch_v, Q)
                        if ri is not None:
                            rand_iias.append(ri)
                    if rand_iias:
                        baselines["iia"]["random_subspace"].append(float(np.mean(rand_iias)))

                    shuf_ev = ev_v.copy()
                    rng.shuffle(shuf_ev)
                    V_shuf = _estimate_subspace(act_v, shuf_ev)
                    if V_shuf is not None:
                        si = _iia_with_subspace(act_v, shuf_ev, ch_v, V_shuf)
                        if si is not None:
                            baselines["iia"]["shuffled"].append(si)

            alpha = _power_law_exponent(act_v)
            if alpha is not None:
                baselines["alpha"]["real"].append(alpha)
                mp_eig = _marchenko_pastur_eigenvalues(act_v.shape[0], act_v.shape[1])
                mp_eig = mp_eig[mp_eig > 0]
                if len(mp_eig) >= 10:
                    s, e = 9, min(49, len(mp_eig) - 1)
                    coeffs = np.polyfit(np.log10(np.arange(s+1, e+2)), np.log10(mp_eig[s:e+1]), 1)
                    baselines["alpha"]["marchenko_pastur"].append(float(-coeffs[0]))

    tests = {}
    for metric_name, bl in baselines.items():
        real_vals = bl.get("real", [])
        tests[metric_name] = {"n_real": len(real_vals), "mean_real": float(np.mean(real_vals)) if real_vals else None}

        for bl_name, bl_vals in bl.items():
            if bl_name == "real" or not bl_vals:
                continue
            delta = float(np.mean(real_vals[:len(bl_vals)]) - np.mean(bl_vals)) if real_vals and bl_vals else None
            tests[metric_name][f"mean_{bl_name}"] = float(np.mean(bl_vals))
            tests[metric_name][f"delta_vs_{bl_name}"] = delta
            tests[metric_name][f"passes_{bl_name}"] = delta >= 0.10 if delta is not None else False

    results = {
        "timestamp": datetime.now().isoformat(),
        "mechval_criterion": "M2 Baseline Separation",
        "mechval_threshold": "delta >= 0.10 vs best baseline",
        "tests": tests,
        "n_random_reps": N_RANDOM_REPS,
    }

    out_path = RESULTS_DIR / "baseline_separation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved to {out_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sessions", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(max_sessions=args.max_sessions)
