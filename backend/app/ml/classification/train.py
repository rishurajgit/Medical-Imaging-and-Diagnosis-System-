"""
NIH ChestX-ray14 Multi-Label Training Script (PyTorch + TorchVision DenseNet-121).

Uncertain / Multi-Label Handling Strategy:
- NIH ChestX-ray14 provides pipe-delimited Finding Labels (e.g. "Infiltration|Effusion" or "No Finding").
- "No Finding" maps to an all-zeros 14-dimensional binary target vector.
- Any uncertain label (-1) if using CheXpert-formatted CSVs is mapped explicitly according to
  `--uncertain-policy` (`zeros` = U-Zeros conservative default, or `ones` = U-Ones high-sensitivity).
- Patient-Level Splitting:
  Patients are strictly grouped by `Patient ID` before splitting into Train (70%), Val (15%), and
  Test (15%) using `GroupShuffleSplit` so images from the same patient NEVER cross splits.
  The test split is strictly held out and never touched during training or checkpoint selection.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from app.ml.classification.model import (
    CLASSIFICATION_MODEL_SPEC,
    NIH_CHESTXRAY14_LABELS,
    ChestXRayDenseNet121,
    resolve_device,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def validate_dataset_directory(data_dir: Path) -> Tuple[Path, Dict[str, Path]]:
    csv_path = data_dir / "Data_Entry_2017.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"NIH ChestX-ray14 metadata CSV not found at {csv_path}. "
            "See docs/DATASET.md for official NIH Clinical Center / Kaggle download instructions."
        )

    image_index: Dict[str, Path] = {}
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for img_path in data_dir.rglob(ext):
            image_index[img_path.name] = img_path

    if not image_index:
        raise FileNotFoundError(
            f"No chest X-ray image files found under {data_dir}. Extract images_001..images_012 first."
        )
    return csv_path, image_index


def encode_nih_labels(finding_str: str, uncertain_policy: str = "zeros") -> np.ndarray:
    vec = np.zeros(len(NIH_CHESTXRAY14_LABELS), dtype=np.float32)
    if not isinstance(finding_str, str) or finding_str.strip() == "No Finding":
        return vec
    tokens = {tok.strip() for tok in finding_str.split("|") if tok.strip()}
    for idx, label in enumerate(NIH_CHESTXRAY14_LABELS):
        if label in tokens:
            vec[idx] = 1.0
        elif f"{label}?" in tokens and uncertain_policy == "ones":
            vec[idx] = 1.0
    return vec


def build_patient_level_splits(
    df: pd.DataFrame, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Guarantees zero patient overlap across train, validation, and held-out test sets."""
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(df, groups=df["Patient ID"]))
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=seed)
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df["Patient ID"]))
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    # Assert strict patient disjointness
    train_p = set(train_df["Patient ID"])
    val_p = set(val_df["Patient ID"])
    test_p = set(test_df["Patient ID"])
    assert train_p.isdisjoint(val_p) and train_p.isdisjoint(test_p) and val_p.isdisjoint(test_p)
    return train_df, val_df, test_df


def get_train_transforms() -> transforms.Compose:
    """Training-only augmentation: small rotation, horizontal flip, slight affine/brightness jitter."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transforms() -> transforms.Compose:
    """Deterministic validation/test/inference transforms without augmentation."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class NIHChestXRayDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_index: Dict[str, Path],
        transform: transforms.Compose,
        uncertain_policy: str = "zeros",
    ):
        self.records: List[Tuple[Path, np.ndarray]] = []
        for _, row in df.iterrows():
            img_name = str(row["Image Index"])
            img_path = image_index.get(img_name)
            if img_path is not None:
                target = encode_nih_labels(str(row["Finding Labels"]), uncertain_policy)
                self.records.append((img_path, target))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, target = self.records[idx]
        image = Image.open(img_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, torch.from_numpy(target)


def compute_macro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, Dict[str, float]]:
    per_class: Dict[str, float] = {}
    valid_scores: List[float] = []
    for i, label in enumerate(NIH_CHESTXRAY14_LABELS):
        col_true = y_true[:, i]
        if len(np.unique(col_true)) < 2:
            continue
        score = float(roc_auc_score(col_true, y_prob[:, i]))
        per_class[label] = round(score, 4)
        valid_scores.append(score)
    macro = float(np.mean(valid_scores)) if valid_scores else 0.0
    return round(macro, 4), per_class


def train_classifier(
    data_dir: Path,
    output_checkpoint: Path,
    splits_dir: Path,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    num_workers: int = 4,
    uncertain_policy: str = "zeros",
    seed: int = 42,
) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    csv_path, image_index = validate_dataset_directory(data_dir)
    df = pd.read_csv(csv_path)
    df = df[df["Image Index"].isin(image_index.keys())].reset_index(drop=True)
    if len(df) < 10:
        raise ValueError(f"Insufficient matched images ({len(df)}) in {data_dir} to train.")

    train_df, val_df, test_df = build_patient_level_splits(df, seed=seed)
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(splits_dir / "train_split.csv", index=False)
    val_df.to_csv(splits_dir / "val_split.csv", index=False)
    test_df.to_csv(splits_dir / "test_split.csv", index=False)

    train_ds = NIHChestXRayDataset(train_df, image_index, get_train_transforms(), uncertain_policy)
    val_ds = NIHChestXRayDataset(val_df, image_index, get_eval_transforms(), uncertain_policy)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = resolve_device("auto")
    model = ChestXRayDenseNet121(num_classes=len(NIH_CHESTXRAY14_LABELS), pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auroc = -1.0
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * images.size(0)

        scheduler.step()
        train_loss = running_loss / max(1, len(train_ds))

        # Validation loop (never touches held-out test_df)
        model.eval()
        val_loss_sum = 0.0
        all_targets, all_probs = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                logits = model(images)
                val_loss_sum += float(criterion(logits, targets).item()) * images.size(0)
                all_targets.append(targets.cpu().numpy())
                all_probs.append(torch.sigmoid(logits).cpu().numpy())

        val_loss = val_loss_sum / max(1, len(val_ds))
        y_true = np.vstack(all_targets)
        y_prob = np.vstack(all_probs)
        val_macro_auroc, per_class_auroc = compute_macro_auroc(y_true, y_prob)

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "val_macro_auroc": val_macro_auroc,
            }
        )
        print(
            f"[Epoch {epoch:02d}/{epochs}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_macro_auroc={val_macro_auroc:.4f}"
        )

        if val_macro_auroc > best_auroc:
            best_auroc = val_macro_auroc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_auroc": val_macro_auroc,
                    "per_class_val_auroc": per_class_auroc,
                    "labels": NIH_CHESTXRAY14_LABELS,
                    "version": CLASSIFICATION_MODEL_SPEC.version,
                    "uncertain_policy": uncertain_policy,
                    "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                output_checkpoint,
            )

    summary = {
        "best_val_macro_auroc": best_auroc,
        "checkpoint": str(output_checkpoint),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "held_out_test_samples": len(test_df),
        "history": history,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DenseNet-121 on NIH ChestX-ray14")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("./models/densenet121_nih14.pth"))
    parser.add_argument("--splits-dir", type=Path, default=Path("./data/processed/nih14_splits"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--uncertain-policy", choices=["zeros", "ones"], default="zeros")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = train_classifier(
        data_dir=args.data_dir,
        output_checkpoint=args.output,
        splits_dir=args.splits_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        uncertain_policy=args.uncertain_policy,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
