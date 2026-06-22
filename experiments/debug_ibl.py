"""Debug IBL data access on Modal. Fails loudly on any problem."""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("debug_ibl")


def run(**kwargs):
    results = {"timestamp": datetime.now().isoformat(), "steps": []}

    def step(name, data=None):
        entry = {"step": name, "ts": datetime.now().isoformat()}
        if data:
            entry.update(data)
        results["steps"].append(entry)
        logger.info(f"STEP: {name} — {json.dumps(data or {}, default=str)[:500]}")
        return entry

    # Step 1: Import ONE
    try:
        from one.api import ONE
        step("import_one", {"ok": True, "version": getattr(ONE, "__version__", "unknown")})
    except Exception as e:
        step("import_one", {"ok": False, "error": str(e)})
        results["fatal"] = "Cannot import ONE"
        return results

    # Step 2: Setup credentials
    base_url = "https://openalyx.internationalbrainlab.org"
    one_dir = Path.home() / ".one"
    one_dir.mkdir(exist_ok=True)
    params_file = one_dir / ".alyx.internationalbrainlab.org"
    params_file.write_text(json.dumps({
        "ALYX_URL": base_url,
        "ALYX_LOGIN": "intbrainlab",
        "ALYX_PWD": "international",
        "CACHE_DIR": str(Path.home() / "ibl_cache"),
    }))
    step("write_params", {"path": str(params_file)})

    # Step 3: Try ONE.setup
    try:
        ONE.setup(base_url=base_url, silent=True)
        step("one_setup", {"ok": True})
    except Exception as e:
        step("one_setup", {"ok": False, "error": str(e)})

    # Step 4: Create ONE instance
    try:
        one = ONE(base_url=base_url, password="international", silent=True)
        step("one_init", {"ok": True, "mode": str(getattr(one, 'mode', 'unknown')), "type": type(one).__name__})
    except Exception as e:
        step("one_init", {"ok": False, "error": str(e)})
        results["fatal"] = "Cannot create ONE instance"
        return results

    # Step 5: Check ONE mode and attributes
    step("one_attrs", {
        "mode": str(getattr(one, 'mode', 'unknown')),
        "base_url": str(getattr(one, '_base_url', getattr(one, 'base_url', 'unknown'))),
        "has_alyx": hasattr(one, 'alyx'),
        "type": type(one).__name__,
        "cache_dir": str(getattr(one, 'cache_dir', 'unknown')),
    })

    # Step 6: Try search_insertions with different parameter combos
    test_region = "VISp"

    # 6a: Simplest call
    try:
        result = one.search_insertions(atlas_acronym=test_region)
        result_list = list(result) if result is not None else []
        step("search_simple", {"ok": True, "n_results": len(result_list), "sample": [str(r) for r in result_list[:3]]})
    except Exception as e:
        step("search_simple", {"ok": False, "error": str(e), "type": type(e).__name__})

    # 6b: With datasets filter
    try:
        result = one.search_insertions(atlas_acronym=test_region, datasets="spikes.times.npy")
        result_list = list(result) if result is not None else []
        step("search_with_datasets", {"ok": True, "n_results": len(result_list)})
    except Exception as e:
        step("search_with_datasets", {"ok": False, "error": str(e), "type": type(e).__name__})

    # 6c: With dataset_types instead
    try:
        result = one.search_insertions(atlas_acronym=test_region, dataset_types="spikes.times")
        result_list = list(result) if result is not None else []
        step("search_with_dataset_types", {"ok": True, "n_results": len(result_list)})
    except Exception as e:
        step("search_with_dataset_types", {"ok": False, "error": str(e), "type": type(e).__name__})

    # 6d: With project filter
    try:
        result = one.search_insertions(atlas_acronym=test_region, project="brainwide")
        result_list = list(result) if result is not None else []
        step("search_with_project", {"ok": True, "n_results": len(result_list)})
    except Exception as e:
        step("search_with_project", {"ok": False, "error": str(e), "type": type(e).__name__})

    # Step 7: Try searching sessions instead
    try:
        eids = one.search(atlas_acronym=test_region, task_protocol="ephys")
        eid_list = list(eids) if eids is not None else []
        step("search_sessions", {"ok": True, "n_results": len(eid_list), "sample": [str(e) for e in eid_list[:3]]})
    except Exception as e:
        step("search_sessions", {"ok": False, "error": str(e), "type": type(e).__name__})

    # Step 7b: Even simpler session search
    try:
        eids = one.search(dataset="spikes.times.npy")
        eid_list = list(eids) if eids is not None else []
        step("search_any_spikes", {"ok": True, "n_results": len(eid_list)})
    except Exception as e:
        step("search_any_spikes", {"ok": False, "error": str(e), "type": type(e).__name__})

    # Step 8: Try direct REST API call
    try:
        rest_result = one.alyx.rest("insertions", "list", atlas_acronym=test_region)
        step("rest_insertions", {"ok": True, "n_results": len(rest_result), "sample_keys": list(rest_result[0].keys()) if rest_result else []})
    except Exception as e:
        step("rest_insertions", {"ok": False, "error": str(e), "type": type(e).__name__})

    # Step 9: If we got insertions from REST, try loading one session
    if results["steps"][-1].get("ok") and results["steps"][-1].get("n_results", 0) > 0:
        try:
            insertion = rest_result[0]
            eid = insertion.get("session") or insertion.get("session_info", {}).get("id")
            step("first_insertion", {
                "pid": insertion.get("id"),
                "eid": eid,
                "keys": list(insertion.keys()),
            })

            if eid:
                # Try loading spike data
                spikes = one.load_object(eid, "spikes", collection=f"alf/{insertion.get('name', 'probe00')}/pykilosort")
                step("load_spikes", {
                    "ok": True,
                    "n_spikes": len(spikes.times) if hasattr(spikes, 'times') else "no times",
                    "attrs": list(spikes.keys()) if hasattr(spikes, 'keys') else dir(spikes)[:10],
                })
        except Exception as e:
            step("load_data", {"ok": False, "error": str(e), "type": type(e).__name__})

    # Step 10: Region mapping — the actual bug we're debugging
    # The clusters object has no 'acronym', so we need to map atlas IDs → region names
    import numpy as np
    try:
        insertion = rest_result[0]
        eid = insertion.get("session") or insertion.get("session_info", {}).get("id")
        probe = insertion.get("name", "probe00")
        collection = f"alf/{probe}/pykilosort"

        clusters = one.load_object(eid, "clusters", collection=collection)
        channels = one.load_object(eid, "channels", collection=collection)

        cl_keys = list(clusters.keys()) if hasattr(clusters, "keys") else []
        ch_keys = list(channels.keys()) if hasattr(channels, "keys") else []
        step("loaded_objects", {"cluster_keys": cl_keys, "channel_keys": ch_keys})

        # Check if channels has brainLocationIds
        atlas_ids = getattr(channels, "brainLocationIds_ccf_2017", None)
        if atlas_ids is not None:
            atlas_ids = np.asarray(atlas_ids).astype(int)
            step("channel_atlas_ids", {
                "shape": list(atlas_ids.shape),
                "dtype": str(atlas_ids.dtype),
                "sample": atlas_ids[:10].tolist(),
                "unique_count": int(len(np.unique(atlas_ids))),
            })

        # Check if clusters has a 'channels' field for the mapping
        cl_channels = getattr(clusters, "channels", None)
        if cl_channels is not None:
            cl_channels = np.asarray(cl_channels).astype(int)
            step("cluster_channels", {
                "shape": list(cl_channels.shape),
                "sample": cl_channels[:10].tolist(),
                "max": int(cl_channels.max()),
                "n_channels_available": len(atlas_ids) if atlas_ids is not None else "N/A",
            })

        # 10a: Try iblatlas.regions.BrainRegions
        try:
            from iblatlas.regions import BrainRegions
            br = BrainRegions()
            step("brain_regions_import", {"ok": True, "n_regions": len(br.id)})

            # Test single ID
            test_ids = atlas_ids[:5] if atlas_ids is not None else np.array([385, 312, 997])
            for method_name in ["id2acronym", "id2index", "get"]:
                try:
                    method = getattr(br, method_name, None)
                    if method is None:
                        continue
                    result = method(test_ids[0])
                    step(f"br_{method_name}_single", {"ok": True, "input": int(test_ids[0]), "result": str(result)})
                except Exception as e:
                    step(f"br_{method_name}_single", {"ok": False, "error": str(e)})

            # Test array of IDs
            try:
                idx_result = br.id2index(test_ids)
                step("br_id2index_array", {"ok": True, "result_type": str(type(idx_result)), "result": str(idx_result)})
                if isinstance(idx_result, tuple):
                    mapped_acronyms = br.acronym[idx_result[1]]
                else:
                    mapped_acronyms = br.acronym[idx_result]
                step("br_mapped_acronyms", {"ok": True, "sample": list(mapped_acronyms[:5])})
            except Exception as e:
                step("br_id2index_array", {"ok": False, "error": str(e), "traceback": __import__("traceback").format_exc()})

            # Full mapping: map all channel atlas IDs → acronyms, then cluster → channel → region
            if atlas_ids is not None and cl_channels is not None:
                try:
                    all_idx = br.id2index(atlas_ids)
                    if isinstance(all_idx, tuple):
                        ch_acronyms = br.acronym[all_idx[1]]
                    else:
                        ch_acronyms = br.acronym[all_idx]
                    cluster_regions = ch_acronyms[cl_channels]
                    unique_regions = np.unique(cluster_regions)
                    step("full_mapping", {
                        "ok": True,
                        "n_clusters": len(cluster_regions),
                        "n_unique_regions": int(len(unique_regions)),
                        "sample_regions": list(unique_regions[:20]),
                        "has_VISp": bool("VISp" in unique_regions),
                        "n_VISp": int(np.sum(cluster_regions == "VISp")),
                    })
                except Exception as e:
                    step("full_mapping", {"ok": False, "error": str(e), "traceback": __import__("traceback").format_exc()})

        except ImportError as e:
            step("brain_regions_import", {"ok": False, "error": str(e)})

        # 10b: Try one.load_dataset for direct acronyms
        try:
            acr = one.load_dataset(eid, "clusters.brainLocationAcronyms_ccf_2017", collection=collection)
            step("direct_load_acronyms", {
                "ok": True if acr is not None else False,
                "type": str(type(acr).__name__) if acr is not None else "None",
                "len": len(acr) if acr is not None else 0,
                "sample": list(acr[:5]) if acr is not None and len(acr) > 0 else [],
            })
        except Exception as e:
            step("direct_load_acronyms", {"ok": False, "error": str(e)})

        # 10c: List ALL available datasets for this session
        try:
            all_ds = one.list_datasets(eid)
            cluster_ds = sorted([str(d) for d in all_ds if "cluster" in str(d).lower()])
            channel_ds = sorted([str(d) for d in all_ds if "channel" in str(d).lower()])
            step("available_datasets", {
                "n_total": len(all_ds),
                "cluster_datasets": cluster_ds,
                "channel_datasets": channel_ds,
            })
        except Exception as e:
            step("available_datasets", {"ok": False, "error": str(e)})

    except Exception as e:
        step("region_mapping_error", {"error": str(e), "traceback": __import__("traceback").format_exc()})

    # Summary
    ok_steps = sum(1 for s in results["steps"] if s.get("ok", False))
    fail_steps = sum(1 for s in results["steps"] if s.get("ok") is False)
    results["summary"] = {"ok_steps": ok_steps, "fail_steps": fail_steps, "total": len(results["steps"])}

    return results
