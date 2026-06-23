"""Experiment 5: Behavioral state modulation of sensory subspaces (Allen VBN).

The VBN dataset includes passive viewing and active change-detection epochs
for the same stimuli. Test whether the sensory causal subspace is stable
across behavioral states.

Prediction: Small Grassmannian distance (< 15 degrees principal angle)
between passive and active subspaces — subspace is circuit-determined,
not state-dependent.

Falsification: Large distance — subspace shifts with behavioral context,
consistent with top-down modulation changing the mechanistic implementation.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data.allen import get_cache
from geometry.distances import grassmannian_distance, principal_angles
from geometry.subspace import fit_lda_subspace

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "artifacts" / "exp5"
TARGET_AREAS = ["VISp", "VISl", "VISal", "VISpm", "VISam"]
SUBSPACE_K = 5
MIN_UNITS = 15


def _split_by_behavioral_state(session) -> tuple[np.ndarray, np.ndarray] | None:
    """Split stimulus presentations into active and passive epochs.

    Returns (active_indices, passive_indices) into the stimulus table,
    or None if the session doesn't have both states.
    """
    stim = session.stimulus_presentations

    if "active" in stim.columns:
        active_mask = stim["active"].values.astype(bool)
    elif "stimulus_block" in stim.columns:
        active_mask = stim["stimulus_block"].values == 0
    else:
        return None

    active_idx = np.where(active_mask)[0]
    passive_idx = np.where(~active_mask)[0]

    if len(active_idx) < 20 or len(passive_idx) < 20:
        return None

    return active_idx, passive_idx


def run(max_sessions: int | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cache = get_cache()
    sessions_table = cache.get_ecephys_session_table()
    session_ids = list(sessions_table.index)
    if max_sessions:
        session_ids = session_ids[:max_sessions]

    results_list = []

    for sid in tqdm(session_ids, desc="Sessions"):
        try:
            session = cache.get_ecephys_session(ecephys_session_id=sid)
        except Exception as e:
            logger.warning(f"Failed to load session {sid}: {e}")
            continue

        split = _split_by_behavioral_state(session)
        if split is None:
            continue
        active_idx, passive_idx = split

        units = session.get_units()
        spike_times = session.spike_times
        stim = session.stimulus_presentations

        for area in TARGET_AREAS:
            area_units = units[units["ecephys_structure_acronym"] == area]
            if len(area_units) < MIN_UNITS:
                continue

            unit_ids = area_units.index.values
            n_units = len(unit_ids)

            def _bin_trials(trial_indices, bin_size=0.01, pre=0.05, post=0.3):
                n_bins = int((pre + post) / bin_size)
                counts = np.zeros((len(trial_indices), n_units, n_bins), dtype=np.float32)
                for t_idx, stim_idx in enumerate(trial_indices):
                    if stim_idx >= len(stim):
                        continue
                    onset = stim.iloc[stim_idx]["start_time"]
                    for u_idx, uid in enumerate(unit_ids):
                        if uid not in spike_times:
                            continue
                        st = spike_times[uid]
                        mask = (st >= onset - pre) & (st < onset + post)
                        for spike_t in st[mask]:
                            b = int((spike_t - (onset - pre)) / bin_size)
                            if 0 <= b < n_bins:
                                counts[t_idx, u_idx, b] += 1
                return counts

            active_counts = _bin_trials(active_idx)
            passive_counts = _bin_trials(passive_idx)

            active_activity = active_counts.mean(axis=2)
            passive_activity = passive_counts.mean(axis=2)

            orientations = stim.get("orientation")
            if orientations is None:
                continue

            active_labels = (orientations.iloc[active_idx].values > orientations.median()).astype(int)
            passive_labels = (orientations.iloc[passive_idx].values > orientations.median()).astype(int)

            if len(np.unique(active_labels)) < 2 or len(np.unique(passive_labels)) < 2:
                continue

            try:
                k = min(SUBSPACE_K, n_units - 1)
                U_active = fit_lda_subspace(active_activity, active_labels, k=k)
                U_passive = fit_lda_subspace(passive_activity, passive_labels, k=k)

                d_g = grassmannian_distance(U_active, U_passive)
                angles = principal_angles(U_active, U_passive)
                mean_angle_deg = float(np.mean(angles) * 180 / np.pi)

                results_list.append({
                    "session_id": int(sid),
                    "mouse_id": str(sessions_table.loc[sid].get("mouse_id", "unknown")),
                    "area": area,
                    "n_units": n_units,
                    "n_active_trials": len(active_idx),
                    "n_passive_trials": len(passive_idx),
                    "grassmannian_distance": d_g,
                    "mean_principal_angle_deg": mean_angle_deg,
                    "principal_angles_deg": [float(a * 180 / np.pi) for a in angles],
                })
                logger.info(f"{sid}/{area}: d_G={d_g:.3f}, mean angle={mean_angle_deg:.1f}°")
            except Exception as e:
                logger.warning(f"Subspace fitting failed for {sid}/{area}: {e}")

    per_area = {}
    for area in TARGET_AREAS:
        area_results = [r for r in results_list if r["area"] == area]
        if area_results:
            angles = [r["mean_principal_angle_deg"] for r in area_results]
            per_area[area] = {
                "mean_angle_deg": float(np.mean(angles)),
                "std_angle_deg": float(np.std(angles)),
                "n_sessions": len(area_results),
            }

    out = {
        "timestamp": datetime.now().isoformat(),
        "k": SUBSPACE_K,
        "prediction_threshold_deg": 15.0,
        "per_area_summary": per_area,
        "sessions": results_list,
    }

    out_path = RESULTS_DIR / "behavioral_state_modulation.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"\nSaved {len(results_list)} results to {out_path}")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sessions", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run(max_sessions=args.max_sessions)
