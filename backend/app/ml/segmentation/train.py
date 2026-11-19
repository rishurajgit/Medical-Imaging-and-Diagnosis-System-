"""
Medical Segmentation Decathlon (Task06_Lung) 3D U-Net Training Pipeline using MONAI + PyTorch.
Implements:
- NIfTI dataset loading from MSD `dataset.json`
- RAS orientation standardization (`Orientationd`)
- Isotropic voxel resampling (`Spacingd`)
- CT Hounsfield Unit window clipping & intensity scaling (`ScaleIntensityRanged`)
- Foreground-balanced 3D patch sampling (`RandCropByPosNegLabeld`)
- DiceCELoss + sliding window validation + best-Dice checkpointing
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.utils import set_determinism

from app.ml.classification.model import resolve_device
from app.ml.segmentation.model import LUNG_CT_SEGMENTATION_SPEC, build_monai_unet3d


def load_msd_task06_splits(data_dir: Path) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    dataset_json = data_dir / "dataset.json"
    if not dataset_json.exists():
        raise FileNotFoundError(
            f"MSD Task06_Lung dataset.json not found at {dataset_json}. "
            "See docs/DATASET.md for official Medical Segmentation Decathlon download instructions."
        )

    meta = json.loads(dataset_json.read_text(encoding="utf-8"))
    training_entries = meta.get("training", [])
    records = []
    for item in training_entries:
        img_path = (data_dir / item["image"]).resolve()
        lbl_path = (data_dir / item["label"]).resolve()
        if img_path.exists() and lbl_path.exists():
            records.append({"image": str(img_path), "label": str(lbl_path)})

    if len(records) < 6:
        raise ValueError(f"Insufficient NIfTI image/label pairs found in {data_dir} ({len(records)}).")

    n_total = len(records)
    n_test = max(1, int(round(n_total * 0.15)))
    n_val = max(1, int(round(n_total * 0.15)))
    n_train = n_total - n_val - n_test

    return records[:n_train], records[n_train : n_train + n_val], records[n_train + n_val :]


def get_train_transforms(roi_size: Tuple[int, int, int]) -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=LUNG_CT_SEGMENTATION_SPEC.hu_min,
                a_max=LUNG_CT_SEGMENTATION_SPEC.hu_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=1,
                neg=1,
                num_samples=2,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_val_transforms() -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=LUNG_CT_SEGMENTATION_SPEC.hu_min,
                a_max=LUNG_CT_SEGMENTATION_SPEC.hu_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def train_segmentation(
    data_dir: Path,
    output_checkpoint: Path,
    splits_path: Path,
    epochs: int = 100,
    batch_size: int = 2,
    lr: float = 1e-4,
    roi_size: Tuple[int, int, int] = (96, 96, 96),
    seed: int = 42,
) -> Dict:
    set_determinism(seed=seed)
    train_files, val_files, test_files = load_msd_task06_splits(data_dir)

    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(
        json.dumps({"train": train_files, "val": val_files, "test": test_files}, indent=2),
        encoding="utf-8",
    )

    train_ds = CacheDataset(train_files, transform=get_train_transforms(roi_size), cache_rate=0.5)
    val_ds = CacheDataset(val_files, transform=get_val_transforms(), cache_rate=0.5)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    device = resolve_device("auto")
    model = build_monai_unet3d().to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    best_dice = -1.0
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / max(1, len(train_loader))

        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                for val_batch in val_loader:
                    val_inputs = val_batch["image"].to(device)
                    val_labels = val_batch["label"].to(device)
                    val_outputs = sliding_window_inference(val_inputs, roi_size, 4, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels)

                mean_dice = float(dice_metric.aggregate().item())
                dice_metric.reset()

            print(f"[Epoch {epoch:03d}/{epochs}] train_loss={avg_loss:.4f} val_mean_dice={mean_dice:.4f}")
            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_mean_dice": round(mean_dice, 4),
                        "roi_size": list(roi_size),
                        "version": LUNG_CT_SEGMENTATION_SPEC.version,
                        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    output_checkpoint,
                )

    return {
        "best_val_mean_dice": round(best_dice, 4),
        "checkpoint": str(output_checkpoint),
        "train_volumes": len(train_files),
        "val_volumes": len(val_files),
        "held_out_test_volumes": len(test_files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MONAI 3D U-Net on MSD Task06_Lung")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("./models/monai_unet3d_task06_lung.pth"))
    parser.add_argument("--splits", type=Path, default=Path("./data/processed/task06_lung_splits.json"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    summary = train_segmentation(
        data_dir=args.data_dir,
        output_checkpoint=args.output,
        splits_path=args.splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
