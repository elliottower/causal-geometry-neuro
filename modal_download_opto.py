"""Download Zatka-Haas optogenetic inactivation dataset to Modal volume + GCS.

Downloads ~20GB of split zip files from Figshare, extracts them, and stores
the full dataset on both the Modal volume and GCS for use by exp47b.

Usage:
    modal run -d modal_download_opto.py
"""
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import modal

SA_KEY_PATH = Path.home() / ".config" / "gcloud" / "factorization-circuits" / "sa.json"
sa_key_content = SA_KEY_PATH.read_text() if SA_KEY_PATH.exists() else "{}"
gcs_secret = modal.Secret.from_dict({"GCS_SA_JSON": sa_key_content})

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "unzip", "p7zip-full")
    .pip_install(
        "scipy>=1.11",
        "numpy>=1.24,<2",
        "google-cloud-storage>=2.14",
        "tqdm>=4.66",
        "matplotlib>=3.8",
    )
)

app = modal.App("neuro-causal-geometry-opto-download")
volume = modal.Volume.from_name("neuro-causal-geometry-results", create_if_missing=True)

FIGSHARE_URLS = [
    ("Zatka-Haas_et_al_Dataset.zip.001", "https://ndownloader.figshare.com/files/24786056"),
    ("Zatka-Haas_et_al_Dataset.zip.002", "https://ndownloader.figshare.com/files/24786080"),
    ("Zatka-Haas_et_al_Dataset.zip.003", "https://ndownloader.figshare.com/files/24786128"),
    ("Zatka-Haas_et_al_Dataset.zip.004", "https://ndownloader.figshare.com/files/24786167"),
    ("code.zip", "https://ndownloader.figshare.com/files/24786170"),
]

GCS_BUCKET = "neuro-causal-geometry-data"
GCS_PREFIX = "zatka_haas"


@app.function(
    image=image,
    cpu=4,
    memory=32768,
    timeout=3600 * 6,
    volumes={"/results": volume},
    secrets=[gcs_secret],
    ephemeral_disk=600_000,
)
def download_and_extract():
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("opto_download")

    base_dir = Path("/results/zatka_haas")
    base_dir.mkdir(parents=True, exist_ok=True)
    zip_dir = base_dir / "zips"
    zip_dir.mkdir(exist_ok=True)
    extract_dir = base_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)

    mat_check = base_dir / "Inactivation_52Coord.mat"
    if mat_check.exists():
        logger.info(f"Already extracted: {mat_check} ({mat_check.stat().st_size / 1e6:.1f} MB)")
        volume.commit()
        return {"status": "already_done", "path": str(mat_check)}

    for name, url in FIGSHARE_URLS:
        dest = zip_dir / name
        if dest.exists() and dest.stat().st_size > 1_000_000:
            logger.info(f"Already downloaded: {name} ({dest.stat().st_size / 1e6:.1f} MB)")
            continue
        logger.info(f"Downloading {name}...")
        result = subprocess.run(
            ["curl", "-L", "--progress-bar", "-o", str(dest), url],
            timeout=7200,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to download {name}")
        logger.info(f"Downloaded {name}: {dest.stat().st_size / 1e6:.1f} MB")
        volume.commit()

    combined = zip_dir / "Zatka-Haas_et_al_Dataset.zip"
    if not combined.exists():
        logger.info("Combining split zip files...")
        with open(combined, "wb") as out:
            for name, _ in FIGSHARE_URLS:
                if name == "code.zip":
                    continue
                part = zip_dir / name
                logger.info(f"  Appending {name} ({part.stat().st_size / 1e6:.1f} MB)")
                with open(part, "rb") as inp:
                    while True:
                        chunk = inp.read(10 * 1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
        logger.info(f"Combined zip: {combined.stat().st_size / 1e6:.1f} MB")
        volume.commit()

    logger.info("Extracting dataset (this takes a while)...")
    result = subprocess.run(
        ["unzip", "-o", str(combined), "-d", str(extract_dir)],
        timeout=7200,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        logger.warning(f"unzip returned {result.returncode}")
        logger.warning(f"stderr: {result.stderr[:2000]}")
        logger.info("Trying 7z as fallback...")
        subprocess.run(
            ["7z", "x", str(combined), f"-o{extract_dir}", "-y"],
            timeout=7200,
        )

    mat_files = list(extract_dir.rglob("*.mat"))
    logger.info(f"Found {len(mat_files)} .mat files")
    for f in mat_files[:20]:
        logger.info(f"  {f.relative_to(extract_dir)} ({f.stat().st_size / 1e6:.1f} MB)")

    for f in extract_dir.rglob("Inactivation_52Coord.mat"):
        import shutil
        shutil.copy2(f, mat_check)
        logger.info(f"Copied Inactivation_52Coord.mat to {mat_check}")
        break

    coord_set_files = list(extract_dir.rglob("26CoordSet.mat"))
    for f in coord_set_files:
        dest = base_dir / "26CoordSet.mat"
        import shutil
        shutil.copy2(f, dest)
        logger.info(f"Copied 26CoordSet.mat to {dest}")
        break

    svd_files = list(extract_dir.rglob("*SVD*.mat"))
    behav_files = list(extract_dir.rglob("*behav*.mat"))
    logger.info(f"SVD files: {len(svd_files)}, behavioral files: {len(behav_files)}")

    volume.commit()

    sa_json = os.environ.get("GCS_SA_JSON", "{}")
    if sa_json != "{}":
        logger.info("Uploading key files to GCS...")
        try:
            from google.cloud import storage
            sa_path = Path("/tmp/gcs-sa.json")
            sa_path.write_text(sa_json)
            client = storage.Client.from_service_account_json(str(sa_path))
            bucket = client.bucket(GCS_BUCKET)

            key_files = [mat_check]
            coord_dest = base_dir / "26CoordSet.mat"
            if coord_dest.exists():
                key_files.append(coord_dest)

            for local_file in key_files:
                if local_file.exists():
                    blob_name = f"{GCS_PREFIX}/{local_file.name}"
                    blob = bucket.blob(blob_name)
                    blob.upload_from_filename(str(local_file))
                    logger.info(f"Uploaded to gs://{GCS_BUCKET}/{blob_name}")

            all_mat = list(extract_dir.rglob("*.mat"))
            for f in all_mat:
                rel = f.relative_to(extract_dir)
                blob_name = f"{GCS_PREFIX}/all/{rel}"
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(str(f))
            logger.info(f"Uploaded {len(all_mat)} .mat files to GCS")
        except Exception as e:
            logger.warning(f"GCS upload failed: {e}")

    logger.info("Cleaning up split zips to save volume space...")
    for name, _ in FIGSHARE_URLS:
        if name == "code.zip":
            continue
        part = zip_dir / name
        if part.exists():
            part.unlink()
            logger.info(f"Removed {name}")
    if combined.exists():
        combined.unlink()
        logger.info("Removed combined zip")
    volume.commit()

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "mat_files": [str(f.relative_to(extract_dir)) for f in mat_files],
        "inactivation_mat": str(mat_check) if mat_check.exists() else None,
        "svd_count": len(svd_files),
        "behav_count": len(behav_files),
    }
    manifest_path = base_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    volume.commit()

    logger.info("Done!")
    return manifest


@app.local_entrypoint()
def main():
    result = download_and_extract.remote()
    print(json.dumps(result, indent=2, default=str)[:5000])
