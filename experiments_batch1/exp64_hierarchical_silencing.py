"""Experiment 64: Hierarchical region grouping for optogenetic validation.

The exp47b optogenetic validation only matched 12 cortical regions between
Steinmetz and Zatka-Haas data. This experiment uses Allen CCF ontology to group
Steinmetz sub-regions under parent areas that match Zatka-Haas silencing sites,
expanding n from 12 to ~20+.

Key idea: Zatka-Haas laser coordinates cover ~1mm radius areas that encompass
multiple Allen CCF sub-regions recorded in Steinmetz. For example, the laser at
(1.5, 2.5) targeting VISp also silences nearby VISrl, VISa, etc. We can:
1. Keep all sub-regions as individual data points, assigning each the parent
   group's silencing effect -> per-sub-region analysis (n ~ 20+)
2. Average IIA within groups weighted by n_sessions -> grouped analysis (n ~ 10)

Both analyses run bootstrap Spearman, BCa CI, permutation p, and delta-rho
bootstrap tests.
"""
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "exp64"

EXP57_RESULTS = Path(__file__).parent.parent / "results" / "exp57" / "exp57_20260621_210317.json"
EXP47B_RESULTS = Path(__file__).parent.parent / "results" / "exp47b" / "silencing_validation_real.json"

# Allen CCF hierarchy: Steinmetz sub-region -> Zatka-Haas silencing group
REGION_TO_GROUP = {
    "VISp": "VIS", "VISl": "VIS", "VISam": "VIS", "VISpm": "VIS",
    "VISrl": "VIS", "VISa": "VIS", "VISal": "VIS",
    "MOs": "MO", "MOp": "MO",
    "SSp": "SS", "SSs": "SS",
    "ACA": "ACA", "PL": "PL", "ORB": "ORB", "RSP": "RSP",
    "ILA": "PL",   # infralimbic is near prelimbic
    "ORBm": "ORB", "ORBl": "ORB",
}

# Map silencing group -> mean silencing effect from exp47b.
# Multiple Zatka-Haas coordinates can map to the same group; we average them.
GROUP_REGION_SOURCES = {
    "VIS": ["VISp", "VISl", "VISam", "VISpm"],
    "MO": ["MOs", "MOp"],
    "SS": ["SSp", "SSs"],
    "ACA": ["ACA"],
    "PL": ["PL"],
    "ORB": ["ORB"],
    "RSP": ["RSP"],
}

N_BOOTSTRAP = 10000
N_PERMUTATIONS = 10000


def _bca_ci(data, stat_fn, n_boot=N_BOOTSTRAP, ci=0.95):
    """Bias-corrected and accelerated (BCa) bootstrap confidence interval."""
    observed = stat_fn(data)
    n = len(data)
    boot_stats = np.empty(n_boot)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n_boot):
            idx = np.random.choice(n, n, replace=True)
            val = stat_fn(data[idx])
            boot_stats[i] = val if np.isfinite(val) else 0.0

    # Bias correction
    z0 = _norm_ppf(np.mean(boot_stats < observed))

    # Acceleration (jackknife)
    jack = np.empty(n)
    for i in range(n):
        jack[i] = stat_fn(np.delete(data, i, axis=0))
    jack_mean = jack.mean()
    num = np.sum((jack_mean - jack) ** 3)
    denom = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    a = num / denom if denom != 0 else 0.0

    alpha = (1 - ci) / 2
    z_lo = _norm_ppf(alpha)
    z_hi = _norm_ppf(1 - alpha)

    a1 = _norm_cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
    a2 = _norm_cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))

    lo = float(np.nanpercentile(boot_stats, 100 * a1))
    hi = float(np.nanpercentile(boot_stats, 100 * a2))
    return lo, hi, boot_stats


def _norm_ppf(p):
    """Normal percent-point function (inverse CDF) using the rational approximation."""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    # Abramowitz and Stegun approximation 26.2.23
    if p < 0.5:
        t = np.sqrt(-2 * np.log(p))
    else:
        t = np.sqrt(-2 * np.log(1 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    val = t - (c0 + c1 * t + c2 * t**2) / (1 + d1 * t + d2 * t**2 + d3 * t**3)
    return val if p >= 0.5 else -val


def _norm_cdf(x):
    """Standard normal CDF via error function."""
    from math import erf
    return 0.5 * (1 + erf(x / np.sqrt(2)))


def _bootstrap_spearman(x, y, n_boot=N_BOOTSTRAP, ci=0.95):
    """Bootstrap Spearman correlation with BCa CI."""
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    n = len(x)
    rho_obs, p_obs = spearmanr(x, y)

    # Paired data for BCa
    paired = np.column_stack([x, y])

    def stat_fn(data):
        return spearmanr(data[:, 0], data[:, 1])[0]

    ci_lo, ci_hi, boot_rhos = _bca_ci(paired, stat_fn, n_boot=n_boot, ci=ci)

    return {
        "rho": float(rho_obs),
        "p": float(p_obs),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "n": n,
        "bootstrap_se": float(np.nanstd(boot_rhos)),
        "bootstrap_mean": float(np.nanmean(boot_rhos)),
    }


def _permutation_test(x, y, n_perm=N_PERMUTATIONS):
    """Two-sided permutation test for Spearman rho."""
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    rho_obs = spearmanr(x, y)[0]
    count = 0
    for _ in range(n_perm):
        perm = np.random.permutation(len(y))
        if abs(spearmanr(x, y[perm])[0]) >= abs(rho_obs):
            count += 1
    return float((count + 1) / (n_perm + 1))


def _delta_rho_bootstrap(x, y, x_orig, y_orig, n_boot=N_BOOTSTRAP, ci=0.95):
    """Bootstrap the difference in Spearman rho between expanded and original sets."""
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    x_orig, y_orig = np.array(x_orig, dtype=float), np.array(y_orig, dtype=float)
    n_exp = len(x)
    n_orig = len(x_orig)

    rho_exp = spearmanr(x, y)[0]
    rho_orig = spearmanr(x_orig, y_orig)[0]
    delta_obs = rho_exp - rho_orig

    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx_exp = np.random.choice(n_exp, n_exp, replace=True)
        idx_orig = np.random.choice(n_orig, n_orig, replace=True)
        r_exp = spearmanr(x[idx_exp], y[idx_exp])[0]
        r_orig = spearmanr(x_orig[idx_orig], y_orig[idx_orig])[0]
        deltas[i] = r_exp - r_orig

    alpha = (1 - ci) / 2
    lo = float(np.nanpercentile(deltas, 100 * alpha))
    hi = float(np.nanpercentile(deltas, 100 * (1 - alpha)))

    return {
        "delta_rho": float(delta_obs),
        "rho_expanded": float(rho_exp),
        "rho_original": float(rho_orig),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "bootstrap_se": float(np.nanstd(deltas)),
        "n_expanded": n_exp,
        "n_original": n_orig,
    }


def _load_exp57_iia():
    """Load per-region VAE IIA from exp57 results."""
    with open(EXP57_RESULTS) as f:
        data = json.load(f)
    return {
        region: {
            "vae_iia": info["vae_iia_mean"],
            "vae_iia_std": info["vae_iia_std"],
            "lda_iia": info["lda_iia_mean"],
            "n_sessions": info["n_sessions"],
            "iia_above_null": info["iia_vae_above_null"],
        }
        for region, info in data["region_results"].items()
    }


def _load_exp47b_silencing():
    """Load per-region silencing effects from exp47b."""
    with open(EXP47B_RESULTS) as f:
        data = json.load(f)
    return data["silencing_effects_used"], data


def _compute_group_silencing(silencing_effects):
    """Average silencing effects across sub-regions to get per-group values."""
    group_effects = {}
    for group, sources in GROUP_REGION_SOURCES.items():
        vals = [silencing_effects[r] for r in sources if r in silencing_effects]
        if vals:
            group_effects[group] = float(np.mean(vals))
    return group_effects


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("Loading exp57 VAE IIA results")
    region_iia = _load_exp57_iia()
    logger.info(f"  {len(region_iia)} regions with VAE IIA")

    logger.info("Loading exp47b silencing effects")
    silencing_effects, exp47b_data = _load_exp47b_silencing()
    logger.info(f"  {len(silencing_effects)} regions with silencing effects")

    # Compute per-group silencing effects
    group_silencing = _compute_group_silencing(silencing_effects)
    logger.info(f"  {len(group_silencing)} groups with silencing effects: {group_silencing}")

    # --- Analysis 1: Per-sub-region (expanded) ---
    # For each Steinmetz region that maps to a group, use the group's silencing effect
    expanded_regions = []
    for region, info in region_iia.items():
        group = REGION_TO_GROUP.get(region)
        if group is None or group not in group_silencing:
            continue
        expanded_regions.append({
            "region": region,
            "group": group,
            "silencing_effect": group_silencing[group],
            "vae_iia": info["vae_iia"],
            "vae_iia_std": info["vae_iia_std"],
            "lda_iia": info["lda_iia"],
            "n_sessions": info["n_sessions"],
            "iia_above_null": info["iia_above_null"],
        })

    expanded_regions.sort(key=lambda x: -x["silencing_effect"])
    logger.info(f"Expanded per-sub-region set: {len(expanded_regions)} regions")
    for r in expanded_regions:
        logger.info(f"  {r['region']} ({r['group']}): sil={r['silencing_effect']:.4f}, "
                     f"vae_iia={r['vae_iia']:.4f}, n_sess={r['n_sessions']}")

    # Get original exp47b matched set for delta-rho comparison
    orig_matched = exp47b_data["matched_regions"]
    orig_per_region = {r["region"]: r for r in exp47b_data["per_region"]}

    # Compute correlations on expanded set
    sil_expanded = [r["silencing_effect"] for r in expanded_regions]
    iia_expanded = [r["vae_iia"] for r in expanded_regions]
    iia_above_null_expanded = [r["iia_above_null"] for r in expanded_regions]

    expanded_tests = {}
    if len(expanded_regions) >= 4:
        # VAE IIA vs silencing
        expanded_tests["vae_iia_vs_silencing"] = _bootstrap_spearman(sil_expanded, iia_expanded)
        expanded_tests["vae_iia_vs_silencing"]["perm_p"] = _permutation_test(
            sil_expanded, iia_expanded)

        # IIA above null vs silencing
        expanded_tests["iia_above_null_vs_silencing"] = _bootstrap_spearman(
            sil_expanded, iia_above_null_expanded)
        expanded_tests["iia_above_null_vs_silencing"]["perm_p"] = _permutation_test(
            sil_expanded, iia_above_null_expanded)

        # LDA IIA vs silencing
        lda_iia_expanded = [r["lda_iia"] for r in expanded_regions]
        expanded_tests["lda_iia_vs_silencing"] = _bootstrap_spearman(
            sil_expanded, lda_iia_expanded)
        expanded_tests["lda_iia_vs_silencing"]["perm_p"] = _permutation_test(
            sil_expanded, lda_iia_expanded)

        logger.info(f"Expanded VAE IIA vs silencing: rho={expanded_tests['vae_iia_vs_silencing']['rho']:.4f}, "
                     f"p={expanded_tests['vae_iia_vs_silencing']['p']:.4f}, "
                     f"BCa CI=[{expanded_tests['vae_iia_vs_silencing']['ci_lo']:.4f}, "
                     f"{expanded_tests['vae_iia_vs_silencing']['ci_hi']:.4f}], "
                     f"perm_p={expanded_tests['vae_iia_vs_silencing']['perm_p']:.4f}")

    # --- Analysis 2: Grouped (weighted average IIA per group) ---
    group_iia = {}
    group_iia_above_null = {}
    group_lda_iia = {}
    group_n_sessions = {}
    for r in expanded_regions:
        g = r["group"]
        group_iia.setdefault(g, []).append((r["vae_iia"], r["n_sessions"]))
        group_iia_above_null.setdefault(g, []).append((r["iia_above_null"], r["n_sessions"]))
        group_lda_iia.setdefault(g, []).append((r["lda_iia"], r["n_sessions"]))
        group_n_sessions.setdefault(g, []).append(r["n_sessions"])

    grouped_data = []
    for g in sorted(group_silencing.keys()):
        if g not in group_iia:
            continue
        # Weighted average by n_sessions
        vals_w = group_iia[g]
        total_w = sum(w for _, w in vals_w)
        w_iia = sum(v * w for v, w in vals_w) / total_w

        vals_w_null = group_iia_above_null[g]
        w_iia_above_null = sum(v * w for v, w in vals_w_null) / total_w

        vals_w_lda = group_lda_iia[g]
        w_lda_iia = sum(v * w for v, w in vals_w_lda) / total_w

        sub_regions = [r["region"] for r in expanded_regions if r["group"] == g]
        grouped_data.append({
            "group": g,
            "silencing_effect": group_silencing[g],
            "weighted_vae_iia": w_iia,
            "weighted_iia_above_null": w_iia_above_null,
            "weighted_lda_iia": w_lda_iia,
            "total_sessions": total_w,
            "sub_regions": sub_regions,
            "n_sub_regions": len(sub_regions),
        })

    grouped_data.sort(key=lambda x: -x["silencing_effect"])
    logger.info(f"Grouped set: {len(grouped_data)} groups")
    for g in grouped_data:
        logger.info(f"  {g['group']} ({g['n_sub_regions']} sub-regions): "
                     f"sil={g['silencing_effect']:.4f}, w_iia={g['weighted_vae_iia']:.4f}")

    grouped_tests = {}
    if len(grouped_data) >= 4:
        g_sil = [g["silencing_effect"] for g in grouped_data]
        g_iia = [g["weighted_vae_iia"] for g in grouped_data]
        g_iia_above_null = [g["weighted_iia_above_null"] for g in grouped_data]
        g_lda_iia = [g["weighted_lda_iia"] for g in grouped_data]

        grouped_tests["vae_iia_vs_silencing"] = _bootstrap_spearman(g_sil, g_iia)
        grouped_tests["vae_iia_vs_silencing"]["perm_p"] = _permutation_test(g_sil, g_iia)

        grouped_tests["iia_above_null_vs_silencing"] = _bootstrap_spearman(g_sil, g_iia_above_null)
        grouped_tests["iia_above_null_vs_silencing"]["perm_p"] = _permutation_test(
            g_sil, g_iia_above_null)

        grouped_tests["lda_iia_vs_silencing"] = _bootstrap_spearman(g_sil, g_lda_iia)
        grouped_tests["lda_iia_vs_silencing"]["perm_p"] = _permutation_test(g_sil, g_lda_iia)

        logger.info(f"Grouped VAE IIA vs silencing: rho={grouped_tests['vae_iia_vs_silencing']['rho']:.4f}, "
                     f"p={grouped_tests['vae_iia_vs_silencing']['p']:.4f}, "
                     f"BCa CI=[{grouped_tests['vae_iia_vs_silencing']['ci_lo']:.4f}, "
                     f"{grouped_tests['vae_iia_vs_silencing']['ci_hi']:.4f}]")

    # --- Delta-rho: compare expanded vs original ---
    delta_rho_tests = {}
    # Build original exp47b arrays (LDA IIA was the metric there)
    orig_sil = [orig_per_region[r]["silencing_effect"] for r in orig_matched if r in orig_per_region]
    orig_iia = [orig_per_region[r]["iia"] for r in orig_matched if r in orig_per_region]

    if len(orig_sil) >= 4 and len(expanded_regions) >= 4:
        # Delta rho for VAE IIA expanded vs original LDA IIA
        delta_rho_tests["expanded_vae_vs_original_lda"] = _delta_rho_bootstrap(
            sil_expanded, iia_expanded, orig_sil, orig_iia)

        # Also compare with VAE IIA on the original 12 regions
        orig_vae_sil = []
        orig_vae_iia = []
        for r in orig_matched:
            if r in region_iia:
                orig_vae_sil.append(silencing_effects[r])
                orig_vae_iia.append(region_iia[r]["vae_iia"])
        if len(orig_vae_sil) >= 4:
            delta_rho_tests["expanded_vae_vs_original_vae"] = _delta_rho_bootstrap(
                sil_expanded, iia_expanded, orig_vae_sil, orig_vae_iia)

            # Original 12 regions with VAE IIA
            delta_rho_tests["original_12_vae"] = _bootstrap_spearman(orig_vae_sil, orig_vae_iia)
            delta_rho_tests["original_12_vae"]["perm_p"] = _permutation_test(
                orig_vae_sil, orig_vae_iia)

        logger.info(f"Delta rho (expanded VAE vs original LDA): "
                     f"{delta_rho_tests['expanded_vae_vs_original_lda']['delta_rho']:.4f}")

    results = {
        "timestamp": timestamp,
        "hierarchy_mapping": REGION_TO_GROUP,
        "group_silencing_effects": group_silencing,
        "expanded_analysis": {
            "n_regions": len(expanded_regions),
            "regions": expanded_regions,
            "tests": expanded_tests,
        },
        "grouped_analysis": {
            "n_groups": len(grouped_data),
            "groups": grouped_data,
            "tests": grouped_tests,
        },
        "delta_rho": delta_rho_tests,
        "original_n": len(orig_matched),
        "original_regions": orig_matched,
        "config": {
            "n_bootstrap": N_BOOTSTRAP,
            "n_permutations": N_PERMUTATIONS,
        },
    }

    out_path = RESULTS_DIR / f"exp64_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved results to {out_path}")

    # Print summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info(f"  Original n: {len(orig_matched)}")
    logger.info(f"  Expanded n (per-sub-region): {len(expanded_regions)}")
    logger.info(f"  Grouped n: {len(grouped_data)}")
    if expanded_tests.get("vae_iia_vs_silencing"):
        t = expanded_tests["vae_iia_vs_silencing"]
        logger.info(f"  Expanded VAE IIA vs silencing: rho={t['rho']:.4f}, "
                     f"perm_p={t.get('perm_p', 'N/A')}, "
                     f"BCa CI=[{t['ci_lo']:.4f}, {t['ci_hi']:.4f}]")
    if grouped_tests.get("vae_iia_vs_silencing"):
        t = grouped_tests["vae_iia_vs_silencing"]
        logger.info(f"  Grouped VAE IIA vs silencing: rho={t['rho']:.4f}, "
                     f"perm_p={t.get('perm_p', 'N/A')}, "
                     f"BCa CI=[{t['ci_lo']:.4f}, {t['ci_hi']:.4f}]")
    if delta_rho_tests.get("expanded_vae_vs_original_lda"):
        t = delta_rho_tests["expanded_vae_vs_original_lda"]
        logger.info(f"  Delta rho (expanded - original): {t['delta_rho']:.4f}, "
                     f"CI=[{t['ci_lo']:.4f}, {t['ci_hi']:.4f}]")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run()
