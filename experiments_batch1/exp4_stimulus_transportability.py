"""Experiment 4: Stimulus subspace transportability across visual areas (Allen VBN).

Fit causal subspaces for stimulus identity in V1, LM, and AL across sessions
and animals. Compare:
  (a) V1 across animals  vs  (b) V1-to-LM within an animal

Is the visual stimulus mechanism more similar within-region-across-animals
or within-animal-across-regions?

Prediction: d_G(within-V1, cross-animal) < d_G(V1-to-LM, same animal)
  (same circuit different instance > different circuit same animal)

Falsification: reverse ordering — circuit is more animal-specific than region-specific.
"""
import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data.allen import list_sessions, load_session_units
from geometry.distances import grassmannian_distance
from geometry.subspace import fit_lda_subspace

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp4"
TARGET_AREAS = ["VISp", "VISl", "VISal"]  # V1, LM, AL
SUBSPACE_K = 5
MIN_UNITS = 15


def run(max_sessions: int | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    session_subspaces = {}

    for area in TARGET_AREAS:
        logger.info(f"\n=== Loading {area} ===")
        sessions_meta = list_sessions(min_units=MIN_UNITS, area=area)
        if max_sessions:
            sessions_meta = sessions_meta[:max_sessions]

        for sess in tqdm(sessions_meta, desc=area):
            sid = sess["session_id"]
            mouse = sess["mouse_id"]

            try:
                data = load_session_units(sid, area)
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

                U = fit_lda_subspace(activity, labels, k=min(SUBSPACE_K, activity.shape[1] - 1))
                session_subspaces[(area, sid, mouse)] = {
                    "subspace": U,
                    "area": area,
                    "session_id": sid,
                    "mouse_id": mouse,
                }
            except Exception as e:
                logger.warning(f"Failed {area}/{sid}: {e}")

    logger.info(f"\nTotal area-sessions: {len(session_subspaces)}")

    distances = {
        "within_region_within_animal": [],
        "within_region_cross_animal": [],
        "cross_region_within_animal": [],
        "cross_region_cross_animal": [],
    }

    keys = list(session_subspaces.keys())
    for (k1, k2) in tqdm(list(combinations(keys, 2)), desc="Pairwise distances"):
        d1, d2 = session_subspaces[k1], session_subspaces[k2]
        d_g = grassmannian_distance(d1["subspace"], d2["subspace"])

        same_area = d1["area"] == d2["area"]
        same_mouse = d1["mouse_id"] == d2["mouse_id"]

        if same_area and same_mouse:
            cat = "within_region_within_animal"
        elif same_area and not same_mouse:
            cat = "within_region_cross_animal"
        elif not same_area and same_mouse:
            cat = "cross_region_within_animal"
        else:
            cat = "cross_region_cross_animal"

        distances[cat].append({
            "distance": d_g,
            "area_1": d1["area"],
            "area_2": d2["area"],
            "mouse_1": str(d1["mouse_id"]),
            "mouse_2": str(d2["mouse_id"]),
        })

    summary = {}
    for cat, vals in distances.items():
        if vals:
            ds = [v["distance"] for v in vals]
            summary[cat] = {"mean": float(np.mean(ds)), "std": float(np.std(ds)), "n": len(ds)}
            logger.info(f"{cat}: d_G = {np.mean(ds):.3f} ± {np.std(ds):.3f} (n={len(ds)})")

    results = {
        "timestamp": datetime.now().isoformat(),
        "k": SUBSPACE_K,
        "areas": TARGET_AREAS,
        "n_sessions": len(session_subspaces),
        "summary": summary,
        "distances": distances,
    }

    out_path = RESULTS_DIR / "stimulus_transportability.json"
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
