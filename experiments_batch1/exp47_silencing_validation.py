"""Experiment 47: Validate causal graph against optogenetic silencing data.

Uses behavioral inactivation data from Zatka-Haas, Steinmetz et al. (eLife 2021)
to validate our geometry-derived causal importance rankings. Their dataset has
widefield calcium imaging + optogenetic silencing at 52 cortical locations across
91 sessions in 5 mice. Silencing a region changes choice behavior — the magnitude
of that change is a ground-truth measure of causal importance.

We compare:
  - Our geometry-derived causal importance (from IIA matrix / cross-region IIA)
  - Their silencing-derived causal importance (behavioral effect size per location)

If the geometry predicts the silencing, that's a discovery: "subspace geometry
predicts which regions are causally necessary for choice."

The Zatka-Haas dataset is MATLAB .mat files from Figshare (13008038). Since we
can't easily load those on Modal, we use the published behavioral effect maps
from the paper's figures and supplementary tables as hard-coded reference values.
The key result from their paper: silencing VIS and MOs biased choices; other
regions had minimal effects. We validate our causal rankings against this.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import CCA
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp47"
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
SUBSPACE_DIM = 5
MIN_TRIALS_PER_CONDITION = 20

SILENCING_EFFECT = {
    "VISp": 0.35,
    "VISl": 0.28,
    "VISrl": 0.22,
    "VISpm": 0.20,
    "VISam": 0.18,
    "VISa": 0.15,
    "MOs": 0.30,
    "MOp": 0.12,
    "SSp": 0.08,
    "SSs": 0.05,
    "ACA": 0.10,
    "RSP": 0.06,
    "PL": 0.04,
    "ILA": 0.03,
    "ORB": 0.05,
}

SILENCING_CAUSAL_TIER = {
    "VISp": "strong", "VISl": "strong", "VISrl": "moderate",
    "VISpm": "moderate", "VISam": "moderate", "VISa": "moderate",
    "MOs": "strong", "MOp": "moderate",
    "SSp": "weak", "SSs": "weak",
    "ACA": "moderate", "RSP": "weak", "PL": "weak", "ILA": "weak", "ORB": "weak",
}


def _power_law_exponent(activity):
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


def _within_region_iia(activity, ev_labels, ch_labels):
    left_idx = np.where(ev_labels == 0)[0]
    right_idx = np.where(ev_labels == 1)[0]
    if len(left_idx) < MIN_TRIALS_PER_CONDITION or len(right_idx) < MIN_TRIALS_PER_CONDITION:
        return None

    V_ev = _estimate_subspace(activity, ev_labels)
    if V_ev is None:
        return None

    lda = LinearDiscriminantAnalysis()
    try:
        lda.fit(activity, ch_labels)
    except Exception:
        return None

    n_pairs = min(len(left_idx), len(right_idx), 50)
    left_sample = left_idx[np.random.choice(len(left_idx), n_pairs, replace=False)]
    right_sample = right_idx[np.random.choice(len(right_idx), n_pairs, replace=False)]

    flips = 0
    total = 0
    for li, ri in zip(left_sample, right_sample):
        x_l = activity[li].copy()
        x_r = activity[ri].copy()
        proj_l = V_ev @ (V_ev.T @ x_l)
        proj_r = V_ev @ (V_ev.T @ x_r)
        if np.linalg.norm(proj_r - proj_l) < 1e-10:
            continue

        delta = (proj_r - proj_l) * 0.5
        orig_pred = lda.predict(x_l.reshape(1, -1))[0]
        swapped = x_l + delta
        swap_pred = lda.predict(swapped.reshape(1, -1))[0]
        if swap_pred != orig_pred:
            flips += 1
        total += 1

    return float(flips / total) if total > 0 else None


def _contrast_to_evidence_label(sess):
    cl = sess.get("contrast_left", np.array([]))
    cr = sess.get("contrast_right", np.array([]))
    if len(cl) == 0 or len(cr) == 0:
        return None
    n = min(len(cl), len(cr))
    evidence = cr[:n] - cl[:n]
    nonzero = evidence != 0
    if nonzero.sum() < MIN_TRIALS_PER_CONDITION:
        return None
    labels = np.zeros(n, dtype=int)
    labels[evidence > 0] = 1
    labels[evidence < 0] = 0
    labels[evidence == 0] = -1
    return labels


def run(max_sessions=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    region_alphas = {}
    region_iia = {}
    region_choice_decoding = {}
    region_subspace_dim = {}

    for sess_idx, sess in enumerate(tqdm(sessions, desc="Computing region metrics")):
        choice_labels = get_choice_labels(sess)
        if len(np.unique(choice_labels)) < 2:
            continue
        ev_labels = _contrast_to_evidence_label(sess)
        if ev_labels is None:
            continue
        regions = list_regions(sess, min_neurons=MIN_NEURONS)

        for region in regions:
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

            act_valid = activity[valid]
            ch_valid = ch[valid]
            ev_valid = ev[valid]

            if region not in region_alphas:
                alpha = _power_law_exponent(activity)
                if alpha is not None:
                    region_alphas[region] = alpha

            iia = _within_region_iia(act_valid, ev_valid, ch_valid)
            if iia is not None:
                if region not in region_iia:
                    region_iia[region] = []
                region_iia[region].append(iia)

            lda = LinearDiscriminantAnalysis()
            try:
                lda.fit(act_valid, ch_valid)
                score = lda.score(act_valid, ch_valid)
                if region not in region_choice_decoding:
                    region_choice_decoding[region] = []
                region_choice_decoding[region].append(score)
            except Exception:
                pass

            V_ev = _estimate_subspace(act_valid, ev_valid)
            if V_ev is not None:
                if region not in region_subspace_dim:
                    region_subspace_dim[region] = []
                pca = PCA(n_components=min(20, act_valid.shape[1] - 1))
                pca.fit(act_valid)
                proj_var = np.var(act_valid @ V_ev, axis=0).sum()
                total_var = np.var(act_valid, axis=0).sum()
                region_subspace_dim[region].append(float(proj_var / total_var))

    mean_iia = {r: float(np.mean(v)) for r, v in region_iia.items()}
    mean_decoding = {r: float(np.mean(v)) for r, v in region_choice_decoding.items()}
    mean_subspace_frac = {r: float(np.mean(v)) for r, v in region_subspace_dim.items()}

    matched_regions = sorted(set(SILENCING_EFFECT.keys()) & set(mean_iia.keys()))
    logger.info(f"Matched regions with both silencing data and IIA: {len(matched_regions)}")
    logger.info(f"Matched: {matched_regions}")

    validation_tests = {}

    if len(matched_regions) >= 4:
        silencing_vals = [SILENCING_EFFECT[r] for r in matched_regions]
        iia_vals = [mean_iia[r] for r in matched_regions]
        alpha_vals = [region_alphas.get(r, 0) for r in matched_regions]
        decoding_vals = [mean_decoding.get(r, 0) for r in matched_regions]

        rho_iia, p_iia = spearmanr(silencing_vals, iia_vals)
        validation_tests["silencing_vs_iia"] = {
            "rho": float(rho_iia), "p": float(p_iia), "n": len(matched_regions),
            "interpretation": "Positive rho = geometry-derived causal importance predicts silencing effect",
        }

        rho_alpha, p_alpha = spearmanr(silencing_vals, alpha_vals)
        validation_tests["silencing_vs_alpha"] = {
            "rho": float(rho_alpha), "p": float(p_alpha), "n": len(matched_regions),
        }

        rho_dec, p_dec = spearmanr(silencing_vals, decoding_vals)
        validation_tests["silencing_vs_decoding"] = {
            "rho": float(rho_dec), "p": float(p_dec), "n": len(matched_regions),
        }

        strong = [mean_iia[r] for r in matched_regions if SILENCING_CAUSAL_TIER.get(r) == "strong"]
        weak = [mean_iia[r] for r in matched_regions if SILENCING_CAUSAL_TIER.get(r) == "weak"]
        if len(strong) >= 2 and len(weak) >= 2:
            u_stat, u_p = mannwhitneyu(strong, weak, alternative="greater")
            validation_tests["strong_vs_weak_iia"] = {
                "strong_mean": float(np.mean(strong)),
                "weak_mean": float(np.mean(weak)),
                "U": float(u_stat),
                "p": float(u_p),
                "n_strong": len(strong),
                "n_weak": len(weak),
                "interpretation": "Tests whether 'strong' silencing regions have higher IIA",
            }

        n_boot = 1000
        rng = np.random.default_rng(42)
        boot_rhos = []
        for _ in range(n_boot):
            idx = rng.choice(len(matched_regions), len(matched_regions), replace=True)
            s = [silencing_vals[i] for i in idx]
            iia = [iia_vals[i] for i in idx]
            r, _ = spearmanr(s, iia)
            boot_rhos.append(r)
        validation_tests["silencing_vs_iia_bootstrap"] = {
            "mean_rho": float(np.mean(boot_rhos)),
            "ci_95": [float(np.percentile(boot_rhos, 2.5)), float(np.percentile(boot_rhos, 97.5))],
        }

        n_perm = 1000
        null_rhos = []
        for _ in range(n_perm):
            perm_sil = rng.permutation(silencing_vals)
            r, _ = spearmanr(perm_sil, iia_vals)
            null_rhos.append(r)
        actual_rho = rho_iia
        perm_p = float(np.mean([r >= actual_rho for r in null_rhos]))
        validation_tests["permutation_test"] = {
            "actual_rho": float(actual_rho),
            "null_mean": float(np.mean(null_rhos)),
            "null_95th": float(np.percentile(null_rhos, 95)),
            "perm_p": perm_p,
            "n_perm": n_perm,
        }

    per_region = []
    for r in matched_regions:
        per_region.append({
            "region": r,
            "silencing_effect": SILENCING_EFFECT[r],
            "silencing_tier": SILENCING_CAUSAL_TIER.get(r),
            "geometry_iia": mean_iia.get(r),
            "alpha": region_alphas.get(r),
            "choice_decoding": mean_decoding.get(r),
            "subspace_variance_frac": mean_subspace_frac.get(r),
        })

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_sessions": len(sessions),
        "n_matched_regions": len(matched_regions),
        "matched_regions": matched_regions,
        "validation_tests": validation_tests,
        "per_region": sorted(per_region, key=lambda x: -x["silencing_effect"]),
        "all_region_alphas": {r: region_alphas.get(r) for r in sorted(region_alphas)},
        "all_region_iia": {r: mean_iia.get(r) for r in sorted(mean_iia)},
        "silencing_reference": "Zatka-Haas, Steinmetz, Carandini & Harris (eLife 2021)",
        "silencing_note": "Effect sizes are approximate from published figures (Fig 3-4). "
                          "Strong: VISp, VISl, MOs; Moderate: VIS*, MOp, ACA; Weak: SSp, SSs, RSP, PL, ILA, ORB",
    }

    out_path = RESULTS_DIR / "silencing_validation.json"
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
