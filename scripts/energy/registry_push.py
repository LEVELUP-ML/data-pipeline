#!/usr/bin/env python3
"""
Push energy model artifact to GCS model registry.

Packages model + metadata into a versioned tarball and uploads
to GCS under model_registry/energy/{timestamp}/.
Maintains latest.json pointer for rollback.

Usage:
    python registry_push.py
    python registry_push.py rollback [version]
"""

import json
import logging
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    BEST_MODEL_PATH, MODEL_WEIGHTS_JSON, MODELS_DIR,
    VALIDATION_REPORT, BIAS_DETECTION_REPORT, SHAP_REPORT,
    GCS_BUCKET, GCS_REGISTRY_PREFIX, ensure_dirs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MAX_LOCAL_VERSIONS = 3


def create_model_package(version_tag: str) -> Path:
    """Package model + metadata into a versioned tarball."""
    package_dir = MODELS_DIR / "package"
    package_dir.mkdir(parents=True, exist_ok=True)

    tar_path = package_dir / f"model-{version_tag}.tar.gz"

    files_to_include = []
    for path in [BEST_MODEL_PATH, MODEL_WEIGHTS_JSON, VALIDATION_REPORT,
                 BIAS_DETECTION_REPORT, SHAP_REPORT,
                 BEST_MODEL_PATH.parent / "training_metadata.json",
                 BEST_MODEL_PATH.parent / "test_data.json"]:
        if path.exists():
            files_to_include.append(path)
        else:
            log.warning("File not found, skipping: %s", path)

    with tarfile.open(tar_path, "w:gz") as tar:
        for fp in files_to_include:
            tar.add(fp, arcname=fp.name)

    log.info("Created model package: %s (%.1f KB)", tar_path, tar_path.stat().st_size / 1024)
    return tar_path


def _gsutil(args: list) -> bool:
    """Run a gsutil command. Returns True on success."""
    result = subprocess.run(["gsutil"] + args, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("gsutil failed: %s", result.stderr)
        return False
    return True


def push_to_gcs(tar_path: Path, version_tag: str) -> str:
    """Upload model package and metadata to GCS registry."""
    prefix = f"gs://{GCS_BUCKET}/{GCS_REGISTRY_PREFIX}/{version_tag}"
    log.info("Uploading to %s/", prefix)

    # Upload tarball
    if not _gsutil(["cp", str(tar_path), f"{prefix}/{tar_path.name}"]):
        log.warning("GCS upload failed — saving to local registry only.")
        return _save_locally(tar_path, version_tag)

    # Upload individual report files for quick inspection
    for path in [VALIDATION_REPORT, BIAS_DETECTION_REPORT, SHAP_REPORT]:
        if path.exists():
            _gsutil(["cp", str(path), f"{prefix}/{path.name}"])

    # Upload plots
    plots = list((MODELS_DIR / "plots").glob("*.png")) if (MODELS_DIR / "plots").exists() else []
    for plot in plots:
        _gsutil(["cp", str(plot), f"{prefix}/plots/{plot.name}"])

    # Build and upload latest.json
    meta_path = BEST_MODEL_PATH.parent / "training_metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    latest = {
        "version":    version_tag,
        "prefix":     prefix,
        "timestamp":  datetime.utcnow().isoformat(),
        "model_type": metadata.get("best_model_type", "unknown"),
        "cv_mae":     metadata.get("cv_mae"),
        "test_mae":   metadata.get("test_mae"),
        "test_rmse":  metadata.get("test_rmse"),
        "test_r2":    metadata.get("test_r2"),
    }
    latest_local = MODELS_DIR / "latest.json"
    latest_local.write_text(json.dumps(latest, indent=2))
    _gsutil(["cp", str(latest_local),
             f"gs://{GCS_BUCKET}/{GCS_REGISTRY_PREFIX}/latest.json"])

    log.info("✅ Pushed to GCS: %s", prefix)

    # Also save locally
    _save_locally(tar_path, version_tag)
    return prefix


def _save_locally(tar_path: Path, version_tag: str) -> str:
    """Save model to local registry."""
    registry_dir = MODELS_DIR / "registry"
    version_dir  = registry_dir / version_tag
    version_dir.mkdir(parents=True, exist_ok=True)

    dest = version_dir / tar_path.name
    shutil.copy2(tar_path, dest)

    # Update latest.json
    (registry_dir / "latest.json").write_text(json.dumps({
        "version":   version_tag,
        "path":      str(dest),
        "timestamp": datetime.utcnow().isoformat(),
    }, indent=2))

    # Prune old local versions
    versions = sorted(
        [d for d in registry_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime, reverse=True,
    )
    for old in versions[MAX_LOCAL_VERSIONS:]:
        shutil.rmtree(old)
        log.info("Pruned old local version: %s", old.name)

    log.info("Saved to local registry → %s", dest)
    return str(dest)


def rollback(target_version: str | None = None) -> bool:
    """Rollback to a previous local model version."""
    registry_dir = MODELS_DIR / "registry"
    if not registry_dir.exists():
        log.error("No local registry found. Cannot rollback.")
        return False

    versions = sorted(
        [d for d in registry_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime, reverse=True,
    )

    if len(versions) < 2:
        log.error("Not enough versions for rollback (need at least 2).")
        return False

    target = (registry_dir / target_version) if target_version else versions[1]
    if not target.exists():
        log.error("Target version not found: %s", target_version)
        return False

    tarballs = list(target.glob("*.tar.gz"))
    if not tarballs:
        log.error("No package in version %s", target.name)
        return False

    log.info("Rolling back to version: %s", target.name)
    with tarfile.open(tarballs[0], "r:gz") as tar:
        tar.extractall(MODELS_DIR)

    (registry_dir / "latest.json").write_text(json.dumps({
        "version":   target.name,
        "path":      str(tarballs[0]),
        "timestamp": datetime.utcnow().isoformat(),
        "rollback":  True,
    }, indent=2))

    log.info("✅ Rollback complete to %s", target.name)
    return True


def main():
    """Package and push energy model to registry."""
    ensure_dirs()

    # Gate: validation must have passed
    if VALIDATION_REPORT.exists():
        report = json.loads(VALIDATION_REPORT.read_text())
        if report.get("status") != "PASS":
            log.error("❌ Cannot push: validation did not pass (status=%s).", report.get("status"))
            sys.exit(1)
    else:
        log.warning("No validation report found — proceeding anyway.")

    version_tag = f"v{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
    log.info("Energy model version: %s", version_tag)

    tar_path = create_model_package(version_tag)
    location = push_to_gcs(tar_path, version_tag)

    log.info("✅ Energy model pushed: %s", location)
    return location


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "rollback":
        target = sys.argv[2] if len(sys.argv) > 2 else None
        success = rollback(target)
        sys.exit(0 if success else 1)
    else:
        main()