"""
Held-out Test Set Evaluation for NIH ChestX-ray14 DenseNet-121 Classifier.
Computes Per-Class AUROC, Macro AUROC, Precision, Recall, F1-score, and per-class Confusion Matrices.
Outputs a DVC-tracked JSON metrics report.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from app.ml.classification.model import NIH_CHESTXRAY14_LABELS, load_trained_classifier
from app.ml.classification.train import (
    NIHChestXRayDataset,
    get_eval_transforms,
    validate_dataset_directory,
)


def evaluate_classifier(
    data_dir: Path,
    test_split_csv: Path,
    checkpoint_path: Path,
    report_path: Path,
    threshold: float = 0.5,
    batch_size: int = 32,
) -> Dict:
    _, image_index = validate_dataset_directory(data_dir)
    if not test_split_csv.exists():
        raise FileNotFoundError(
            f"Held-out test split CSV not found at {test_split_csv}. Run training split creation first."
        )

    test_df = pd.read_csv(test_split_csv)
    test_ds = NIHChestXRayDataset(test_df, image_index, get_eval_transforms())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model, device, meta = load_trained_classifier(checkpoint_path)

    all_targets, all_probs = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            logits = model(images.to(device))
            all_targets.append(targets.numpy())
            all_probs.append(torch.sigmoid(logits).cpu().numpy())

    y_true = np.vstack(all_targets)
    y_prob = np.vstack(all_probs)
    y_pred = (y_prob >= threshold).astype(np.int32)

    per_class_auroc: Dict[str, float] = {}
    per_class_cm: Dict[str, Dict[str, int]] = {}
    valid_aurocs = []

    for idx, label in enumerate(NIH_CHESTXRAY14_LABELS):
        col_true = y_true[:, idx]
        col_pred = y_pred[:, idx]
        if len(np.unique(col_true)) == 2:
            auc = float(roc_auc_score(col_true, y_prob[:, idx]))
            per_class_auroc[label] = round(auc, 4)
            valid_aurocs.append(auc)
        tn, fp, fn, tp = confusion_matrix(col_true, col_pred, labels=[0, 1]).ravel()
        per_class_cm[label] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    report = {
        "dataset": "NIH ChestX-ray14 (Held-out Patient Test Split)",
        "checkpoint": str(checkpoint_path),
        "model_version": meta.get("version"),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "test_samples": len(test_ds),
        "threshold": threshold,
        "macro_auroc": round(float(np.mean(valid_aurocs)), 4) if valid_aurocs else None,
        "per_class_auroc": per_class_auroc,
        "precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "per_class_confusion_matrix": per_class_cm,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DenseNet-121 on held-out NIH14 test split")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, default=Path("./data/processed/nih14_splits/test_split.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("./models/densenet121_nih14.pth"))
    parser.add_argument("--report", type=Path, default=Path("./reports/classification_metrics.json"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    result = evaluate_classifier(
        data_dir=args.data_dir,
        test_split_csv=args.test_csv,
        checkpoint_path=args.checkpoint,
        report_path=args.report,
        threshold=args.threshold,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
