"""Experiment 6: Gauge correction effect (Allen VBN).

For cross-session comparisons within animals (same neural population across days),
compute raw and gauge-normalized Grassmannian distances. Quantify the variance
reduction from gauge normalization.

Prediction: Gauge normalization reduces variance in cross-session comparisons by >30%.
Falsification: No significant reduction — raw Grassmannian distance is already
gauge-invariant in practice.
"""
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data.allen import list_sessions, load_session_units
from geometry.distances import gauge_normalized_distance, grassmannian_distance
from geometry.subspace import fit_lda_subspace

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp6"
TARGET_AREA = "VISp"
SUBSPACE_K = 5
MIN_UNITS = 15


def run(max_sessions: int | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sessions_meta = list_sessions(min_units=MIN_UNITS, area=TARGET_AREA)
    if max_sessions:
        sessions_meta = sessions_meta[:max_sessions]

    session_data = {}
    for sess in tqdm(sessions_meta, desc=f"Loading {TARGET_AREA}"):
        sid = sess["session_id"]
        mouse = sess["mouse_id"]

        try:
            data = load_session_units(sid, TARGET_AREA)
            if data is None:
                continue

            activity = data["activity"].mean(axis=2)
            if activity.shape[1] < MIN_UNITS:
                continue

            stim = data["stimulus_presentations"]
            orientations = stim.get("orientation")
            if orientations is None or len(orientations.unique()) < 2:
                continue

            labels = (orientations.values > orientations.median()).astype(int)[:activity.shape[0]]
            k = min(SUBSPACE_K, activity.shape[1] - 1)
            U = fit_lda_subspace(activity, labels, k=k)

            session_data[sid] = {
                "subspace": U,
                "activity": activity,
                "mouse_id": mouse,
                "session_id": sid,
            }
        except Exception as e:
            logger.warning(f"Failed {sid}: {e}")

    logger.info(f"Loaded {len(session_data)} sessions")

    mice = {}
    for sid, d in session_data.items():
        m = d["mouse_id"]
        if m not in mice:
            mice[m] = []
        mice[m].append(sid)

    within_animal_raw = []
    within_animal_gauge = []
    cross_animal_raw = []
    cross_animal_gauge = []

    keys = list(session_data.keys())
    for (s1, s2) in tqdm(list(combinations(keys, 2)), desc="Pairwise"):
        d1, d2 = session_data[s1], session_data[s2]
        raw = grassmannian_distance(d1["subspace"], d2["subspace"])
        gauge = gauge_normalized_distance(
            d1["subspace"], d2["subspace"], d1["activity"], d2["activity"]
        )

        same_mouse = d1["mouse_id"] == d2["mouse_id"]
        if same_mouse:
            within_animal_raw.append(raw)
            within_animal_gauge.append(gauge)
        else:
            cross_animal_raw.append(raw)
            cross_animal_gauge.append(gauge)

    def _stats(vals):
        if not vals:
            return {"mean": None, "std": None, "n": 0}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    variance_reduction_within = None
    if within_animal_raw and within_animal_gauge:
        var_raw = float(np.var(within_animal_raw))
        var_gauge = float(np.var(within_animal_gauge))
        if var_raw > 0:
            variance_reduction_within = 1.0 - var_gauge / var_raw

    results = {
        "timestamp": datetime.now().isoformat(),
        "area": TARGET_AREA,
        "k": SUBSPACE_K,
        "n_sessions": len(session_data),
        "n_mice": len(mice),
        "within_animal": {
            "raw": _stats(within_animal_raw),
            "gauge_normalized": _stats(within_animal_gauge),
            "variance_reduction": variance_reduction_within,
            "prediction_threshold": 0.30,
        },
        "cross_animal": {
            "raw": _stats(cross_animal_raw),
            "gauge_normalized": _stats(cross_animal_gauge),
        },
    }

    logger.info(f"\nWithin-animal raw: {_stats(within_animal_raw)}")
    logger.info(f"Within-animal gauge: {_stats(within_animal_gauge)}")
    logger.info(f"Variance reduction: {variance_reduction_within}")

    out_path = RESULTS_DIR / "gauge_correction.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved to {out_path}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sessions", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run(max_sessions=args.max_sessions)
