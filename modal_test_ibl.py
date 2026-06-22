"""Quick test: does IBL ONE API work on Modal?"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("setuptools<71", "wheel", "Cython", "numpy>=1.24,<2")
    .pip_install("ONE-api>=2.7", "ibllib>=2.36", "scipy>=1.11", "tqdm>=4.66")
)

app = modal.App("ibl-test")

@app.function(image=image, timeout=300)
def test_ibl():
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ibl-test")

    from one.api import ONE
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
    )
    logger.info(f"ONE type: {type(one)}")
    logger.info(f"ONE mode: {one.mode}")

    for region in ["VISp", "CA1", "MOs"]:
        try:
            insertions = one.search_insertions(
                atlas_acronym=region,
                datasets="spikes.times.npy",
                project="brainwide",
            )
            if insertions is None:
                insertions = []
            insertions = list(insertions)
            logger.info(f"{region}: {len(insertions)} insertions")

            if len(insertions) > 0:
                pid = insertions[0]
                logger.info(f"  First pid: {pid}")
                info = one.alyx.rest("insertions", "read", id=pid)
                eid = info.get("session")
                logger.info(f"  Session EID: {eid}")
                return {"region": region, "n_insertions": len(insertions), "first_pid": str(pid), "eid": str(eid)}
        except Exception as e:
            logger.error(f"{region} failed: {e}")
            import traceback
            traceback.print_exc()

        try:
            insertions = one.search_insertions(atlas_acronym=region, dataset_types="spikes.times")
            if insertions is None:
                insertions = []
            insertions = list(insertions)
            logger.info(f"{region} (fallback): {len(insertions)} insertions")
            if insertions:
                return {"region": region, "n_insertions": len(insertions), "method": "fallback"}
        except Exception as e:
            logger.error(f"{region} fallback failed: {e}")

    return {"error": "no insertions found for any region"}


@app.local_entrypoint()
def main():
    import json
    result = test_ibl.remote()
    print(json.dumps(result, indent=2, default=str))
