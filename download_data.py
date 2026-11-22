"""Download and verify the official UCI hydraulic systems dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import DATA_DIR, RAW_DIR, SENSORS, ensure_directories

UCI_DATASET_PAGE = "https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip"
ZIP_PATH = DATA_DIR / "condition_monitoring_of_hydraulic_systems.zip"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(force: bool = False) -> Path:
    ensure_directories()
    if ZIP_PATH.exists() and not force:
        print(f"Using existing archive: {ZIP_PATH}")
        return ZIP_PATH
    temporary = ZIP_PATH.with_suffix(".zip.part")
    try:
        print(f"Downloading official UCI archive from:\n{UCI_ZIP_URL}")
        with urllib.request.urlopen(UCI_ZIP_URL, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(ZIP_PATH)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "UCI download failed. Download the archive manually from "
            f"{UCI_DATASET_PAGE} and save it as {ZIP_PATH}. Root cause: {exc}"
        ) from exc
    return ZIP_PATH


def extract(archive: Path, force: bool = False) -> None:
    if force and RAW_DIR.exists():
        # Remove only known extracted dataset files, never the containing project.
        for item in RAW_DIR.iterdir():
            if item.is_file():
                item.unlink()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            target_name = Path(member.filename).name
            if not target_name:
                continue
            target = RAW_DIR / target_name
            if target.exists() and not force:
                continue
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def verify() -> dict:
    missing = [name for name in [*SENSORS, "profile"] if not (RAW_DIR / f"{name}.txt").exists()]
    if missing:
        raise FileNotFoundError(f"Missing extracted files: {missing}")

    profile = np.loadtxt(RAW_DIR / "profile.txt", dtype=int)
    if profile.shape != (2205, 5):
        raise ValueError(f"Expected profile shape (2205, 5), got {profile.shape}")
    if set(np.unique(profile[:, 2])) != {0, 1, 2}:
        raise ValueError("Unexpected pump leakage labels in profile column 3")

    sensor_shapes = {}
    for sensor, (rate, _, _) in SENSORS.items():
        path = RAW_DIR / f"{sensor}.txt"
        first_row = np.loadtxt(path, max_rows=1)
        expected = rate * 60
        if first_row.size != expected:
            raise ValueError(f"{sensor}: expected {expected} samples/cycle, got {first_row.size}")
        # Count rows without parsing all values.
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            rows = sum(1 for line in handle if line.strip())
        if rows != 2205:
            raise ValueError(f"{sensor}: expected 2205 cycles, got {rows}")
        sensor_shapes[sensor] = [rows, int(first_row.size)]

    report = {
        "source": UCI_DATASET_PAGE,
        "download_url": UCI_ZIP_URL,
        "archive_sha256": _sha256(ZIP_PATH) if ZIP_PATH.exists() else None,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "profile_shape": list(profile.shape),
        "pump_label_counts": {str(k): int(v) for k, v in zip(*np.unique(profile[:, 2], return_counts=True))},
        "sensor_shapes": sensor_shapes,
        "sampling_rates_hz": {sensor: values[0] for sensor, values in SENSORS.items()},
    }
    (DATA_DIR / "dataset_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="redownload and re-extract")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    ensure_directories()
    if not args.verify_only:
        archive = download(force=args.force)
        extract(archive, force=args.force)
    verify()


if __name__ == "__main__":
    main()

