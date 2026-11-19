from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from monai.inferers import sliding_window_inference

from app.ml.segmentation.model import LUNG_CT_SEGMENTATION_SPEC, load_trained_segmentation_model


def predict_lung_ct_volume(
    volume_tensor: torch.Tensor,
    checkpoint_path: Path,
    voxel_spacing_mm: Tuple[float, float, float] = (1.5, 1.5, 2.0),
    device_pref: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Runs real MONAI 3D U-Net sliding-window inference on a preprocessed [1, 1, H, W, D] CT volume.
    Returns:
      - binary_mask: np.ndarray [H, W, D] uint8
      - foreground_prob: np.ndarray [H, W, D] float32 in [0, 1]
      - summary metadata dict
    Raises ModelWeightsUnavailableError if checkpoint_path is absent.
    """
    model, device, meta = load_trained_segmentation_model(
        checkpoint_path=checkpoint_path,
        spec=LUNG_CT_SEGMENTATION_SPEC,
        device_pref=device_pref,
    )
    roi_size = tuple(meta.get("roi_size", LUNG_CT_SEGMENTATION_SPEC.roi_size))

    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=volume_tensor.to(device),
            roi_size=roi_size,
            sw_batch_size=2,
            predictor=model,
            overlap=0.25,
        )
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # [2, H, W, D]
        foreground_prob = probs[1].astype(np.float32)
        binary_mask = (np.argmax(probs, axis=0) == 1).astype(np.uint8)

    voxel_count = int(np.sum(binary_mask))
    voxel_vol_mm3 = float(voxel_spacing_mm[0] * voxel_spacing_mm[1] * voxel_spacing_mm[2])
    estimated_volume_mm3 = round(voxel_count * voxel_vol_mm3, 2)
    mean_prob = (
        round(float(np.mean(foreground_prob[binary_mask == 1])), 4) if voxel_count > 0 else 0.0
    )

    # Identify axial slice with peak segmented area for default preview overlay
    slice_areas = binary_mask.sum(axis=(0, 1))
    peak_slice_index = int(np.argmax(slice_areas)) if slice_areas.size > 0 else 0

    summary = {
        "model_id": LUNG_CT_SEGMENTATION_SPEC.model_id,
        "model_name": LUNG_CT_SEGMENTATION_SPEC.name,
        "model_version": meta.get("version", LUNG_CT_SEGMENTATION_SPEC.version),
        "lesion_detected": voxel_count > 0,
        "voxel_count": voxel_count,
        "estimated_volume_mm3": estimated_volume_mm3,
        "mean_probability": mean_prob,
        "peak_slice_index": peak_slice_index,
        "num_slices": int(binary_mask.shape[2]) if binary_mask.ndim == 3 else 1,
    }
    return binary_mask, foreground_prob, summary
