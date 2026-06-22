"""Quick debug script to figure out how to get region labels from IBL data.

Run on Modal: uv run modal run modal_run.py --detach --experiment debug_ibl_regions
Or directly: python debug_ibl_regions.py
"""
import json
import logging
import traceback

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = __import__("pathlib").Path(__file__).parent / "results" / "debug_ibl"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run(max_sessions=None):
    from data.ibl import get_one

    one = get_one()
    results = {}

    # Get one session for VISp
    rest_results = one.alyx.rest("insertions", "list", atlas_acronym="VISp")
    sess = rest_results[0]
    eid = sess.get("session") or sess.get("session_info", {}).get("id")
    probe = sess.get("name", "probe00")
    logger.info(f"Test session: eid={eid}, probe={probe}")
    results["eid"] = eid
    results["probe"] = probe

    # 1. Load clusters object and inspect ALL keys
    for collection in [f"alf/{probe}/pykilosort", f"alf/{probe}"]:
        try:
            clusters = one.load_object(eid, "clusters", collection=collection)
            keys = list(clusters.keys()) if hasattr(clusters, "keys") else dir(clusters)
            results[f"clusters_keys_{collection}"] = keys
            logger.info(f"clusters ({collection}): {keys}")
            for k in keys:
                v = clusters[k] if hasattr(clusters, "__getitem__") else getattr(clusters, k)
                results[f"clusters_{k}_type"] = str(type(v).__name__)
                results[f"clusters_{k}_dtype"] = str(getattr(v, "dtype", "N/A"))
                if hasattr(v, "shape"):
                    results[f"clusters_{k}_shape"] = list(v.shape)
                if hasattr(v, "__len__") and len(v) > 0:
                    sample = v[:3]
                    results[f"clusters_{k}_sample"] = str(sample)
            break
        except Exception as e:
            logger.info(f"clusters from {collection} failed: {e}")

    # 2. Load channels object and inspect ALL keys
    for collection in [f"alf/{probe}/pykilosort", f"alf/{probe}"]:
        try:
            channels = one.load_object(eid, "channels", collection=collection)
            keys = list(channels.keys()) if hasattr(channels, "keys") else dir(channels)
            results[f"channels_keys_{collection}"] = keys
            logger.info(f"channels ({collection}): {keys}")
            for k in keys:
                v = channels[k] if hasattr(channels, "__getitem__") else getattr(channels, k)
                results[f"channels_{k}_type"] = str(type(v).__name__)
                results[f"channels_{k}_dtype"] = str(getattr(v, "dtype", "N/A"))
                if hasattr(v, "shape"):
                    results[f"channels_{k}_shape"] = list(v.shape)
                if hasattr(v, "__len__") and len(v) > 0:
                    sample = v[:3]
                    results[f"channels_{k}_sample"] = str(sample)
            break
        except Exception as e:
            logger.info(f"channels from {collection} failed: {e}")

    # 3. Try BrainRegions atlas mapping
    try:
        from iblatlas.regions import BrainRegions
        br = BrainRegions()
        results["BrainRegions_available"] = True
        results["BrainRegions_methods"] = [m for m in dir(br) if not m.startswith("_") and callable(getattr(br, m, None))]
        # Test with a known ID
        test_id = 385  # VISp
        try:
            idx = br.id2index(test_id)
            results["id2index_385"] = str(idx)
            results["acronym_385"] = str(br.acronym[idx[1]])
        except Exception as e:
            results["id2index_error"] = str(e)
        try:
            results["id2acronym_385"] = str(br.id2acronym(385))
        except Exception as e:
            results["id2acronym_error"] = str(e)
        try:
            results["get_385"] = str(br.get(385))
        except Exception as e:
            results["get_error"] = str(e)
    except ImportError as e:
        results["BrainRegions_available"] = False
        results["BrainRegions_import_error"] = str(e)

    # 4. Try loading clusters with explicit dataset names
    try:
        acr = one.load_dataset(eid, "clusters.brainLocationAcronyms_ccf_2017",
                               collection=f"alf/{probe}/pykilosort")
        results["direct_acronyms"] = str(type(acr).__name__)
        if acr is not None:
            results["direct_acronyms_len"] = len(acr)
            results["direct_acronyms_sample"] = str(acr[:5])
    except Exception as e:
        results["direct_acronyms_error"] = str(e)

    # 5. Try brainbox load_channel_locations
    try:
        from brainbox.io.one import load_channel_locations
        ch_locs = load_channel_locations(eid, probe=probe, one=one)
        results["brainbox_available"] = True
        results["brainbox_keys"] = list(ch_locs.keys()) if hasattr(ch_locs, "keys") else str(type(ch_locs))
        if hasattr(ch_locs, "get") and "acronym" in ch_locs:
            results["brainbox_acronym_sample"] = str(ch_locs["acronym"][:5])
    except Exception as e:
        results["brainbox_error"] = str(e)
        results["brainbox_traceback"] = traceback.format_exc()

    # 6. Try using the pid directly with one.load_object
    pid = sess.get("id")
    if pid:
        try:
            from one.alf.io import AlfBunch
            clusters_pid = one.load_object(eid, "clusters",
                                           collection=f"alf/{probe}/pykilosort",
                                           attribute=["brainLocationAcronyms_ccf_2017"])
            results["clusters_pid_keys"] = list(clusters_pid.keys()) if hasattr(clusters_pid, "keys") else str(type(clusters_pid))
        except Exception as e:
            results["clusters_pid_error"] = str(e)

    # 7. List ALL datasets for this session to see what's available
    try:
        datasets = one.list_datasets(eid)
        cluster_datasets = [d for d in datasets if "cluster" in str(d).lower()]
        channel_datasets = [d for d in datasets if "channel" in str(d).lower()]
        results["cluster_datasets"] = [str(d) for d in cluster_datasets[:30]]
        results["channel_datasets"] = [str(d) for d in channel_datasets[:30]]
    except Exception as e:
        results["list_datasets_error"] = str(e)

    out_path = RESULTS_DIR / "debug_output.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved debug output to {out_path}")

    return results
