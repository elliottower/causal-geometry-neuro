"""Shared GCS data cache for neuro-causal-geometry.

All environments (local, Modal, RunPod) use this to avoid re-downloading
neural datasets from source APIs. Pattern:
  1. Check local cache dir
  2. Check GCS bucket
  3. Download from source API → save locally → upload to GCS

Requires: google-cloud-storage OR gcloud CLI.

GCS bucket: gs://neuro-causal-geometry-data/
  steinmetz/         — .npz files from OSF
  ibl/{eid}/         — per-session spike data
  allen/{session_id}/ — per-session unit data
  results/           — experiment outputs
"""
import json
import logging
import os
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent

_dotenv = _REPO_ROOT / ".env"
if _dotenv.exists():
    for line in _dotenv.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

BUCKET = os.environ.get("GCS_BUCKET")
if not BUCKET:
    raise RuntimeError("GCS_BUCKET not set. Add it to .env or set the environment variable.")
_LOCAL_CACHE = Path(os.environ.get("NEURO_CACHE_DIR", _REPO_ROOT / "cache"))
_sa_env = os.environ.get("GCS_SA_KEY_PATH")
if not _sa_env:
    raise RuntimeError("GCS_SA_KEY_PATH not set. Add it to .env or set the environment variable.")
_SA_KEY_PATHS = [Path(_sa_env)]


def _get_gcs_client():
    try:
        from google.cloud import storage
        for sa_path in _SA_KEY_PATHS:
            if sa_path.exists():
                return storage.Client.from_service_account_json(str(sa_path))
        return storage.Client()
    except ImportError:
        return None


def _gcloud_available() -> bool:
    try:
        subprocess.run(["gcloud", "--version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def local_cache_dir() -> Path:
    _LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
    return _LOCAL_CACHE


def download_from_gcs(gcs_path: str, local_path: Path) -> bool:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob_name = gcs_path.lstrip("/")

    client = _get_gcs_client()
    if client is not None:
        try:
            bucket = client.bucket(BUCKET)
            blob = bucket.blob(blob_name)
            if blob.exists():
                blob.download_to_filename(str(local_path))
                logger.info(f"Downloaded gs://{BUCKET}/{blob_name} → {local_path}")
                return True
            return False
        except Exception as e:
            logger.warning(f"GCS client download failed: {e}")

    if _gcloud_available():
        try:
            result = subprocess.run(
                ["gcloud", "storage", "cp", f"gs://{BUCKET}/{blob_name}", str(local_path)],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                logger.info(f"Downloaded gs://{BUCKET}/{blob_name} → {local_path} (via gcloud)")
                return True
        except subprocess.TimeoutExpired:
            pass

    return False


def upload_to_gcs(local_path: Path, gcs_path: str) -> bool:
    blob_name = gcs_path.lstrip("/")

    client = _get_gcs_client()
    if client is not None:
        try:
            bucket = client.bucket(BUCKET)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(local_path))
            logger.info(f"Uploaded {local_path} → gs://{BUCKET}/{blob_name}")
            return True
        except Exception as e:
            logger.warning(f"GCS client upload failed: {e}")

    if _gcloud_available():
        try:
            result = subprocess.run(
                ["gcloud", "storage", "cp", str(local_path), f"gs://{BUCKET}/{blob_name}"],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                logger.info(f"Uploaded {local_path} → gs://{BUCKET}/{blob_name} (via gcloud)")
                return True
        except subprocess.TimeoutExpired:
            pass

    return False


def exists_on_gcs(gcs_path: str) -> bool:
    blob_name = gcs_path.lstrip("/")

    client = _get_gcs_client()
    if client is not None:
        try:
            bucket = client.bucket(BUCKET)
            return bucket.blob(blob_name).exists()
        except Exception:
            pass

    if _gcloud_available():
        try:
            result = subprocess.run(
                ["gcloud", "storage", "ls", f"gs://{BUCKET}/{blob_name}"],
                capture_output=True, text=True, timeout=30,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            pass

    return False


def upload_dir_to_gcs(local_dir: Path, gcs_prefix: str) -> int:
    count = 0
    for f in local_dir.rglob("*"):
        if f.is_file():
            rel = f.relative_to(local_dir)
            if upload_to_gcs(f, f"{gcs_prefix}/{rel}"):
                count += 1
    return count


def cached_path(gcs_path: str, downloader=None) -> Path | None:
    """Get a local path for a GCS-cached file. Downloads if needed.

    Args:
        gcs_path: Path within the bucket (e.g. "steinmetz/session_0.npz")
        downloader: Optional callable() -> Path that downloads from source
                    and returns the local path. Called if not in GCS either.
    """
    local = local_cache_dir() / gcs_path
    if local.exists():
        return local

    if download_from_gcs(gcs_path, local):
        return local

    if downloader is not None:
        source_path = downloader()
        if source_path is not None and source_path.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            if source_path != local:
                import shutil
                shutil.copy2(source_path, local)
            upload_to_gcs(local, gcs_path)
            return local

    return None


def upload_results(experiment: str, results: dict, tag: str | None = None):
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{experiment}_{tag}_{ts}.json" if tag else f"{experiment}_{ts}.json"
    gcs_path = f"results/{experiment}/{name}"

    local = local_cache_dir() / "results" / experiment / name
    local.parent.mkdir(parents=True, exist_ok=True)
    with open(local, "w") as f:
        json.dump(results, f, indent=2, default=str)

    upload_to_gcs(local, gcs_path)
    logger.info(f"Results saved: {local} + gs://{BUCKET}/{gcs_path}")
    return local
