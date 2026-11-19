#!/usr/bin/env python3
"""
Validates local dataset directory structure for NIH ChestX-ray14 and
Medical Segmentation Decathlon Task06_Lung prior to running `dvc repro`.
Uses Python standard library only so it runs in any Python 3.11+ environment.
"""

import json
import sys
from pathlib import Path


def check_nih_chestxray14(root: Path) -> bool:
    csv_path = root / "Data_Entry_2017.csv"
    if not csv_path.exists():
        print(f"[MISSING] NIH ChestX-ray14 metadata CSV not found at: {csv_path}")
        return False
    images = list(root.rglob("*.png")) + list(root.rglob("*.jpg"))
    if not images:
        print(f"[MISSING] No Chest X-ray images found under: {root}")
        return False
    print(f"[OK] NIH ChestX-ray14 validated: CSV present, {len(images)} image files found.")
    return True


def check_msd_task06_lung(root: Path) -> bool:
    ds_json = root / "dataset.json"
    if not ds_json.exists():
        print(f"[MISSING] MSD Task06_Lung dataset.json not found at: {ds_json}")
        return False
    try:
        meta = json.loads(ds_json.read_text(encoding="utf-8"))
        train_list = meta.get("training", [])
    except Exception as exc:
        print(f"[ERROR] Failed to parse {ds_json}: {exc}")
        return False
    existing = sum(
        1 for entry in train_list if (root / entry.get("image", "")).exists()
    )
    print(
        f"[OK] MSD Task06_Lung dataset.json parsed: {existing}/{len(train_list)} training NIfTI volumes present."
    )
    return existing > 0


if __name__ == "__main__":
    base = Path("data/raw")
    nih_ok = check_nih_chestxray14(base / "nih_chestxray14")
    msd_ok = check_msd_task06_lung(base / "msd_task06_lung")
    if not (nih_ok and msd_ok):
        print("\nSee docs/DATASET.md for official download and DVC tracking instructions.")
        sys.exit(1)
    print("\nAll dataset directories validated successfully.")
