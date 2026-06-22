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

    # Summary
    ok_steps = sum(1 for s in results["steps"] if s.get("ok", False))
    fail_steps = sum(1 for s in results["steps"] if s.get("ok") is False)
    results["summary"] = {"ok_steps": ok_steps, "fail_steps": fail_steps, "total": len(results["steps"])}

    return results
