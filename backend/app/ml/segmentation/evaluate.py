"""
Held-out Test Evaluation for MONAI 3D U-Net Lung Segmentation (Task06_Lung).
Computes Dice coefficient, Intersection over Union (IoU), Sensitivity, and Specificity.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference

from app.ml.segmentation.model import load_trained_segmentation_model
from app.ml.segmentation.train import get_val_transforms


def compute_binary_segmentation_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> Dict[str, float]:
    pred = (pred_mask > 0).astype(np.uint8)
    truth = (true_mask > 0).astype(np.uint8)

    tp = float(np.sum((pred == 1) & (truth == 1)))
    fp = float(np.sum((pred == 1) & (truth == 0)))
    fn = float(np.sum((pred == 0) & (truth == 1)))
    tn = float(np.sum((pred == 0) & (truth == 0)))

    eps = 1e-7
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    sensitivity = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)

    return {
        "dice": dice,
        "iou": iou,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def evaluate_segmentation(
    splits_path: Path,
    checkpoint_path: Path,
    report_path: Path,
) -> Dict:
    if not splits_path.exists():
        raise FileNotFoundError(f"Segmentation split file not found at {splits_path}.")

    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    test_files = splits.get("test", [])
    if not test_files:
        raise ValueError("No held-out test volumes present in splits JSON.")

    test_ds = Dataset(test_files, transform=get_val_transforms())
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    model, device, meta = load_trained_segmentation_model(checkpoint_path)
    roi_size = tuple(meta.get("roi_size", (96, 96, 96)))

    per_volume = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].cpu().numpy()[0, 0]
            logits = sliding_window_inference(inputs, roi_size, 4, model)
            preds = torch.argmax(logits, dim=1).cpu().numpy()[0]
            per_volume.append(compute_binary_segmentation_metrics(preds, labels))

    report = {
        "dataset": "Medical Segmentation Decathlon (Task06_Lung Held-out Test Split)",
        "checkpoint": str(checkpoint_path),
        "model_version": meta.get("version"),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "test_volumes": len(per_volume),
        "mean_dice": round(float(np.mean([m["dice"] for m in per_volume])), 4),
        "mean_iou": round(float(np.mean([m["iou"] for m in per_volume])), 4),
        "sensitivity": round(float(np.mean([m["sensitivity"] for m in per_volume])), 4),
        "specificity": round(float(np.mean([m["specificity"] for m in per_volume])), 4),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MONAI 3D U-Net on Task06_Lung test split")
    parser.add_argument("--splits", type=Path, default=Path("./data/processed/task06_lung_splits.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("./models/monai_unet3d_task06_lung.pth"))
    parser.add_argument("--report", type=Path, default=Path("./reports/segmentation_metrics.json"))
    args = parser.parse_args()

    result = evaluate_segmentation(args.splits, args.checkpoint, args.report)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
