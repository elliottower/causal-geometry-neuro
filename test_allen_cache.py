"""Quick test: does Allen VBN from_s3_cache work on Modal?"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "cmake", "pkg-config")
    .pip_install("setuptools<71", "wheel", "Cython", "numpy>=1.24,<2")
    .pip_install("allensdk>=2.16", "matplotlib>=3.8")
)

app = modal.App("test-allen-cache")
volume = modal.Volume.from_name("neuro-causal-geometry-results", create_if_missing=True)


@app.function(image=image, cpu=2, memory=8192, timeout=600, volumes={"/results": volume})
def test_cache():
    import json
    import logging
    from pathlib import Path
    logging.basicConfig(level=logging.INFO)

    from allensdk.brain_observatory.behavior.behavior_project_cache import (
        VisualBehaviorNeuropixelsProjectCache,
    )

    cache_dir = Path("/results/allen_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Creating cache from S3...")
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=str(cache_dir))

    print("Getting session table...")
    table = cache.get_ecephys_session_table()
    print(f"Sessions: {len(table)}")
    print(f"Columns: {list(table.columns)[:15]}")

    result = {
        "n_sessions": len(table),
        "columns": list(table.columns),
        "sample_row": {k: str(v) for k, v in table.iloc[0].to_dict().items()},
    }

    out = Path("/results/allen_cache_test.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    volume = modal.Volume.from_name("neuro-causal-geometry-results")
    volume.commit()
    print(f"Saved to {out}")
    return result


@app.local_entrypoint()
def main():
    import json
    result = test_cache.remote()
    print(json.dumps(result, indent=2)[:2000])
