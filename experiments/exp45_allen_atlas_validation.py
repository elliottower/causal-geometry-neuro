"""Experiment 45: Validate IIA causal graph against Allen Mouse Brain Connectivity Atlas.

Computes cross-region IIA (interchange intervention accuracy) between Steinmetz brain
regions and validates the resulting directed causal influence matrix against anatomical
projection densities from the Allen Mouse Brain Connectivity Atlas.

Tests:
1. Spearman correlation between IIA strength and Allen projection density (matched pairs)
2. Permutation null: shuffle region labels on IIA matrix, compute 1000 shuffled correlations
3. Directed edge validation: do high-asymmetry IIA edges match directed anatomical projections?
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from scipy.stats import binomtest, spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm

from data.steinmetz import get_choice_labels, get_region_activity, list_regions, load_all

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp45"
MIN_NEURONS = 15
TIME_WINDOW = slice(15, 35)
SUBSPACE_DIM = 5
MIN_TRIALS_PER_CONDITION = 20
TOP_N_REGIONS = 15
N_SHUFFLE_NULL = 1000
ALLEN_CACHE_DIR = Path("/results/allen_cache")
ALLEN_CACHE_DIR_LOCAL = Path(__file__).parent.parent / "data" / "cache" / "allen_connectivity"

# Mapping from Steinmetz region acronyms to Allen ontology acronyms.
# Most match directly; a few need translation. 'root' has no Allen equivalent.
STEINMETZ_TO_ALLEN = {
    "VISp": "VISp",
    "VISl": "VISl",
    "VISpm": "VISpm",
    "VISrl": "VISrl",
    "VISa": "VISa",
    "VISam": "VISam",
    "CA1": "CA1",
    "CA3": "CA3",
    "DG": "DG",
    "SUB": "SUB",
    "MOs": "MOs",
    "MOp": "MOp",
    "ACA": "ACA",
    "PL": "PL",
    "ILA": "ILA",
    "RSP": "RSP",
    "ORB": "ORB",
    "SSp": "SSp",
    "SSs": "SSs",
    "AUD": "AUD",
    "CP": "CP",
    "GPe": "GPe",
    "SNr": "SNr",
    "ACB": "ACB",
    "TH": "TH",
    "LP": "LP",
    "LD": "LD",
    "LGd": "LGd",
    "VPL": "VPL",
    "VPM": "VPM",
    "PO": "PO",
    "SCm": "SCm",
    "SCig": "SCig",
    "SCs": "SCs",
    "SCsg": "SCsg",
    "MRN": "MRN",
    "PAG": "PAG",
}

SKIP_REGIONS = {"root"}


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


def _cross_region_iia(act_source, act_target, ev_labels, ch_labels_target, V_ev_source):
    """Compute directional IIA: swap evidence subspace in source, measure choice flip in target."""
    left_idx = np.where(ev_labels == 0)[0]
    right_idx = np.where(ev_labels == 1)[0]
    if len(left_idx) < MIN_TRIALS_PER_CONDITION or len(right_idx) < MIN_TRIALS_PER_CONDITION:
        return None

    lda_target = LinearDiscriminantAnalysis()
    try:
        lda_target.fit(act_target, ch_labels_target)
    except Exception:
        return None

    n_s = act_source.shape[1]
    n_t = act_target.shape[1]
    min_dim = min(n_s, n_t)
    pca_s = PCA(n_components=min(10, min_dim - 1, act_source.shape[0] - 1))
    pca_t = PCA(n_components=min(10, min_dim - 1, act_target.shape[0] - 1))

    try:
        scores_s = pca_s.fit_transform(act_source)
        scores_t = pca_t.fit_transform(act_target)
    except Exception:
        return None

    n_cca = min(3, scores_s.shape[1], scores_t.shape[1], act_source.shape[0] - 1)
    if n_cca < 1:
        return None

    try:
        cca = CCA(n_components=n_cca, max_iter=500)
        proj_s, proj_t = cca.fit_transform(scores_s, scores_t)
    except Exception:
        return None

    V_ev_cca = _estimate_subspace(proj_s, ev_labels, min(3, n_cca))
    if V_ev_cca is None:
        return None

    pinv_y = np.linalg.pinv(cca.y_weights_)

    n_pairs = min(len(left_idx), len(right_idx), 50)
    left_sample = left_idx[np.random.choice(len(left_idx), n_pairs, replace=False)]
    right_sample = right_idx[np.random.choice(len(right_idx), n_pairs, replace=False)]

    flips = 0
    total = 0
    for li, ri in zip(left_sample, right_sample):
        s_l = proj_s[li].copy()
        s_r = proj_s[ri].copy()
        proj_l = V_ev_cca @ (V_ev_cca.T @ s_l)
        proj_r = V_ev_cca @ (V_ev_cca.T @ s_r)
        scale = np.linalg.norm(proj_r - proj_l)
        if scale < 1e-10:
            continue

        delta_cca = (proj_r - proj_l) * 0.5
        delta_pca = (delta_cca.reshape(1, -1) @ pinv_y).flatten()

        orig_t_l = act_target[li]
        orig_pred = lda_target.predict(orig_t_l.reshape(1, -1))[0]

        try:
            scores_t_l = scores_t[li].copy()
            shifted = scores_t_l + delta_pca
            recon = pca_t.inverse_transform(shifted.reshape(1, -1)).flatten()
            swap_pred = lda_target.predict(recon.reshape(1, -1))[0]
        except (ValueError, np.linalg.LinAlgError):
            continue

        if swap_pred != orig_pred:
            flips += 1
        total += 1

    return float(flips / total) if total > 0 else None


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
    labels = np.zeros(n, dtype=int)
    labels[evidence > 0] = 1
    labels[evidence < 0] = 0
    labels[evidence == 0] = -1
    return labels


def _load_steinmetz_region_data(sessions):
    """Load and aggregate per-region data across sessions."""
    region_data = {}
    for sess in tqdm(sessions, desc="Loading sessions"):
        choice_labels = get_choice_labels(sess)
        if len(np.unique(choice_labels)) < 2:
            continue
        ev_labels = _contrast_to_evidence_label(sess)
        if ev_labels is None:
            continue
        regions = list_regions(sess, min_neurons=MIN_NEURONS)
        for region in regions:
            if region in SKIP_REGIONS:
                continue
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
            if region not in region_data:
                region_data[region] = []
            region_data[region].append({
                "activity": activity[valid],
                "choice_labels": ch[valid],
                "evidence_labels": ev[valid],
            })
    return region_data


def _compute_iia_matrix(region_data, target_regions):
    """Compute directed IIA for all ordered pairs of target regions."""
    iia_matrix = {}
    for r_source in tqdm(target_regions, desc="Cross-region IIA"):
        for r_target in target_regions:
            if r_source == r_target:
                continue
            iia_values = []
            for m_s in region_data[r_source]:
                for m_t in region_data[r_target]:
                    n = min(m_s["activity"].shape[0], m_t["activity"].shape[0])
                    if n < MIN_TRIALS_PER_CONDITION * 2:
                        continue
                    act_s = m_s["activity"][:n]
                    act_t = m_t["activity"][:n]
                    ev = m_s["evidence_labels"][:n]
                    ch_t = m_t["choice_labels"][:n]
                    V_ev = _estimate_subspace(act_s, ev)
                    if V_ev is None:
                        continue
                    iia = _cross_region_iia(act_s, act_t, ev, ch_t, V_ev)
                    if iia is not None:
                        iia_values.append(iia)
            if iia_values:
                iia_matrix[(r_source, r_target)] = float(np.mean(iia_values))
    return iia_matrix


def _get_allen_projection_densities(target_regions):
    """Query Allen Mouse Brain Connectivity Atlas for projection densities between regions.

    Returns dict mapping (source_acronym, target_acronym) -> mean projection density.
    """
    # Use Modal volume path if available, else local cache
    if ALLEN_CACHE_DIR.exists():
        manifest = str(ALLEN_CACHE_DIR / "connectivity" / "manifest.json")
    else:
        ALLEN_CACHE_DIR_LOCAL.mkdir(parents=True, exist_ok=True)
        manifest = str(ALLEN_CACHE_DIR_LOCAL / "manifest.json")

    logger.info(f"Allen SDK manifest: {manifest}")
    mcc = MouseConnectivityCache(manifest_file=manifest)
    structure_tree = mcc.get_structure_tree()

    # Resolve Allen structure IDs for each Steinmetz region
    region_to_allen_id = {}
    for region in target_regions:
        allen_acronym = STEINMETZ_TO_ALLEN.get(region)
        if allen_acronym is None:
            logger.warning(f"No Allen mapping for region {region}, skipping")
            continue
        try:
            structures = structure_tree.get_structures_by_acronym([allen_acronym])
            if structures:
                region_to_allen_id[region] = structures[0]["id"]
            else:
                logger.warning(f"Allen structure not found for acronym {allen_acronym}")
        except Exception as e:
            logger.warning(f"Failed to resolve Allen ID for {allen_acronym}: {e}")

    logger.info(f"Resolved {len(region_to_allen_id)}/{len(target_regions)} regions to Allen IDs")

    # For each source region, find injection experiments and get projection densities to targets
    allen_densities = {}
    resolved_regions = [r for r in target_regions if r in region_to_allen_id]

    for source in tqdm(resolved_regions, desc="Querying Allen Atlas"):
        source_id = region_to_allen_id[source]

        # Get experiments injected into this structure
        try:
            experiments = mcc.get_experiments(
                dataframe=True, injection_structure_ids=[source_id]
            )
        except Exception as e:
            logger.warning(f"Failed to get experiments for {source} (id={source_id}): {e}")
            continue

        if experiments is None or len(experiments) == 0:
            logger.debug(f"No injection experiments for {source}")
            continue

        exp_ids = list(experiments["id"])
        logger.debug(f"{source}: {len(exp_ids)} injection experiments")

        # Get unionize data for all experiments from this source
        try:
            unionizes = mcc.get_structure_unionizes(
                experiment_ids=exp_ids, is_injection=False,
                structure_ids=[region_to_allen_id[t] for t in resolved_regions if t != source],
            )
        except Exception as e:
            logger.warning(f"Failed to get unionizes for {source}: {e}")
            continue

        if unionizes is None or len(unionizes) == 0:
            continue

        # Extract mean projection density for each target
        for target in resolved_regions:
            if target == source:
                continue
            target_id = region_to_allen_id[target]
            target_rows = unionizes[unionizes["structure_id"] == target_id]
            if len(target_rows) > 0:
                mean_density = float(target_rows["projection_density"].mean())
                allen_densities[(source, target)] = mean_density

    logger.info(f"Got Allen projection densities for {len(allen_densities)} directed pairs")
    return allen_densities, region_to_allen_id


def _correlate_iia_and_allen(iia_matrix, allen_densities, target_regions):
    """Compute Spearman correlation between IIA and Allen projection density for matched pairs."""
    iia_vals = []
    allen_vals = []
    matched_pairs = []

    for (src, tgt), iia_val in iia_matrix.items():
        if (src, tgt) in allen_densities:
            iia_vals.append(iia_val)
            allen_vals.append(allen_densities[(src, tgt)])
            matched_pairs.append({"source": src, "target": tgt, "iia": iia_val,
                                  "allen_density": allen_densities[(src, tgt)]})

    if len(iia_vals) < 5:
        logger.warning(f"Only {len(iia_vals)} matched pairs, too few for correlation")
        return None

    iia_arr = np.array(iia_vals)
    allen_arr = np.array(allen_vals)
    rho, p = spearmanr(iia_arr, allen_arr)

    return {
        "rho": float(rho),
        "p_value": float(p),
        "n_matched_pairs": len(matched_pairs),
        "matched_pairs": matched_pairs,
    }


def _permutation_null(iia_matrix, allen_densities, target_regions, n_shuffles=N_SHUFFLE_NULL):
    """Null model: shuffle region labels on IIA matrix, recompute correlation."""
    # Build matched arrays
    iia_vals = []
    allen_vals = []
    pair_keys = []
    for (src, tgt), iia_val in iia_matrix.items():
        if (src, tgt) in allen_densities:
            iia_vals.append(iia_val)
            allen_vals.append(allen_densities[(src, tgt)])
            pair_keys.append((src, tgt))

    if len(iia_vals) < 5:
        return None

    iia_arr = np.array(iia_vals)
    allen_arr = np.array(allen_vals)
    actual_rho = spearmanr(iia_arr, allen_arr)[0]

    # Shuffle: permute region labels on IIA matrix, rebuild matched pairs
    regions_in_iia = sorted({r for pair in iia_matrix for r in pair})
    null_rhos = []

    for _ in tqdm(range(n_shuffles), desc="Permutation null"):
        perm = np.random.permutation(len(regions_in_iia))
        label_map = {regions_in_iia[i]: regions_in_iia[perm[i]] for i in range(len(regions_in_iia))}

        shuffled_iia_vals = []
        shuffled_allen_vals = []
        for (src, tgt), iia_val in iia_matrix.items():
            shuffled_src = label_map[src]
            shuffled_tgt = label_map[tgt]
            if (shuffled_src, shuffled_tgt) in allen_densities:
                shuffled_iia_vals.append(iia_val)
                shuffled_allen_vals.append(allen_densities[(shuffled_src, shuffled_tgt)])

        if len(shuffled_iia_vals) >= 5:
            null_rho = spearmanr(shuffled_iia_vals, shuffled_allen_vals)[0]
            null_rhos.append(float(null_rho))

    null_rhos = np.array(null_rhos)
    p_value = float(np.mean(null_rhos >= actual_rho)) if len(null_rhos) > 0 else 1.0

    return {
        "actual_rho": float(actual_rho),
        "null_mean": float(np.mean(null_rhos)) if len(null_rhos) > 0 else None,
        "null_std": float(np.std(null_rhos)) if len(null_rhos) > 0 else None,
        "null_percentile_95": float(np.percentile(null_rhos, 95)) if len(null_rhos) > 0 else None,
        "null_percentile_99": float(np.percentile(null_rhos, 99)) if len(null_rhos) > 0 else None,
        "p_value": p_value,
        "n_shuffles": len(null_rhos),
        "n_requested": n_shuffles,
    }


def _directed_edge_validation(iia_matrix, allen_densities, target_regions):
    """Test whether high-asymmetry IIA edges match directed anatomical projections.

    For each unordered region pair, compare:
    - IIA direction: which direction has higher IIA
    - Allen direction: which direction has higher projection density
    """
    checked_pairs = set()
    concordant = 0
    discordant = 0
    edge_details = []

    for r_a in target_regions:
        for r_b in target_regions:
            if r_a >= r_b:
                continue
            pair = (r_a, r_b)
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)

            iia_ab = iia_matrix.get((r_a, r_b))
            iia_ba = iia_matrix.get((r_b, r_a))
            allen_ab = allen_densities.get((r_a, r_b))
            allen_ba = allen_densities.get((r_b, r_a))

            if iia_ab is None or iia_ba is None or allen_ab is None or allen_ba is None:
                continue

            iia_asymmetry = iia_ab - iia_ba
            allen_asymmetry = allen_ab - allen_ba

            # Only count edges with meaningful asymmetry in both
            iia_threshold = 0.02
            allen_threshold = 1e-6

            if abs(iia_asymmetry) < iia_threshold or abs(allen_asymmetry) < allen_threshold:
                continue

            is_concordant = (iia_asymmetry > 0) == (allen_asymmetry > 0)
            if is_concordant:
                concordant += 1
            else:
                discordant += 1

            edge_details.append({
                "region_a": r_a,
                "region_b": r_b,
                "iia_ab": float(iia_ab),
                "iia_ba": float(iia_ba),
                "iia_asymmetry": float(iia_asymmetry),
                "allen_ab": float(allen_ab),
                "allen_ba": float(allen_ba),
                "allen_asymmetry": float(allen_asymmetry),
                "concordant": is_concordant,
            })

    total = concordant + discordant
    if total == 0:
        return None

    # Binomial test: expected concordance under chance is 0.5
    binom_result = binomtest(concordant, total, 0.5, alternative="greater")

    return {
        "concordant": concordant,
        "discordant": discordant,
        "total": total,
        "concordance_rate": float(concordant / total),
        "binomial_p": float(binom_result.pvalue),
        "edge_details": sorted(edge_details, key=lambda x: -abs(x["iia_asymmetry"])),
    }


def run(max_sessions=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now()
    logger.info(f"[{start_time.isoformat()}] Starting exp45: Allen Atlas validation")

    # Step 1: Load Steinmetz data and compute cross-region IIA
    logger.info(f"[{datetime.now().isoformat()}] Loading Steinmetz data")
    sessions = load_all()
    if max_sessions:
        sessions = sessions[:max_sessions]

    region_data = _load_steinmetz_region_data(sessions)
    regions_with_data = sorted(r for r in region_data if len(region_data[r]) >= 2)
    logger.info(f"{len(regions_with_data)} regions with >= 2 sessions")

    # Select top regions by session count, excluding unmappable ones
    mappable_regions = [r for r in regions_with_data if r in STEINMETZ_TO_ALLEN]
    regions_by_sessions = sorted(mappable_regions, key=lambda r: -len(region_data[r]))
    target_regions = regions_by_sessions[:TOP_N_REGIONS]
    logger.info(f"[{datetime.now().isoformat()}] Using {len(target_regions)} regions: {target_regions}")

    logger.info(f"[{datetime.now().isoformat()}] Computing cross-region IIA matrix")
    iia_matrix = _compute_iia_matrix(region_data, target_regions)
    logger.info(f"Computed IIA for {len(iia_matrix)} directed pairs")

    # Step 2: Query Allen Mouse Brain Connectivity Atlas
    logger.info(f"[{datetime.now().isoformat()}] Querying Allen Connectivity Atlas")
    allen_densities, region_to_allen_id = _get_allen_projection_densities(target_regions)

    # Step 3: Spearman correlation between IIA and Allen projection density
    logger.info(f"[{datetime.now().isoformat()}] Computing IIA-Allen correlation")
    correlation_result = _correlate_iia_and_allen(iia_matrix, allen_densities, target_regions)

    # Step 4: Permutation null model
    logger.info(f"[{datetime.now().isoformat()}] Running permutation null ({N_SHUFFLE_NULL} shuffles)")
    null_result = _permutation_null(iia_matrix, allen_densities, target_regions)

    # Step 5: Directed edge concordance
    logger.info(f"[{datetime.now().isoformat()}] Testing directed edge concordance")
    direction_result = _directed_edge_validation(iia_matrix, allen_densities, target_regions)

    # Build IIA matrix as nested dict for JSON serialization
    iia_matrix_serializable = {}
    for (src, tgt), val in iia_matrix.items():
        if src not in iia_matrix_serializable:
            iia_matrix_serializable[src] = {}
        iia_matrix_serializable[src][tgt] = val

    allen_densities_serializable = {}
    for (src, tgt), val in allen_densities.items():
        if src not in allen_densities_serializable:
            allen_densities_serializable[src] = {}
        allen_densities_serializable[src][tgt] = val

    end_time = datetime.now()
    results = {
        "timestamp": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "n_sessions": len(sessions),
        "n_regions": len(target_regions),
        "regions": target_regions,
        "n_iia_pairs": len(iia_matrix),
        "n_allen_pairs": len(allen_densities),
        "region_to_allen_id": {r: region_to_allen_id.get(r) for r in target_regions},
        "correlation": correlation_result,
        "permutation_null": null_result,
        "directed_edge_concordance": direction_result,
        "iia_matrix": iia_matrix_serializable,
        "allen_densities": allen_densities_serializable,
    }

    out_path = RESULTS_DIR / "allen_atlas_validation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"[{end_time.isoformat()}] Saved results to {out_path}")

    # Print summary
    if correlation_result:
        logger.info(
            f"IIA-Allen correlation: rho={correlation_result['rho']:.3f}, "
            f"p={correlation_result['p_value']:.4f}, "
            f"n={correlation_result['n_matched_pairs']} pairs"
        )
    if null_result:
        logger.info(
            f"Permutation null: actual rho={null_result['actual_rho']:.3f}, "
            f"null mean={null_result['null_mean']:.3f}, "
            f"p={null_result['p_value']:.4f}"
        )
    if direction_result:
        logger.info(
            f"Directed edge concordance: {direction_result['concordant']}/{direction_result['total']} "
            f"({direction_result['concordance_rate']:.1%}), "
            f"binomial p={direction_result['binomial_p']:.4f}"
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp45: Allen Atlas validation of IIA causal graph")
    parser.add_argument("--max-sessions", type=int, default=None,
                        help="Limit number of Steinmetz sessions (for testing)")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(max_sessions=args.max_sessions)
