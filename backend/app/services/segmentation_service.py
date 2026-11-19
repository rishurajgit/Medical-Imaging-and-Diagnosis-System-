from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

from app.ml.segmentation.predict import predict_lung_ct_volume
from app.services.preprocessing_service import (
    preprocess_ct_volume_from_nifti,
    preprocess_ct_volume_from_slices,
)


def render_segmentation_mask_png(
    binary_slice: np.ndarray,
    prob_slice: np.ndarray,
    output_path: Path,
) -> Path:
    """
    Renders a transparent RGBA overlay PNG combining the foreground probability map (cyan/amber)
    and crisp contour outline of the predicted MONAI 3D U-Net segmentation mask.
    """
    h, w = binary_slice.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Soft cyan fill weighted by foreground probability
    mask_bool = binary_slice > 0
    rgba[mask_bool, 0] = 6
    rgba[mask_bool, 1] = 182
    rgba[mask_bool, 2] = 212
    rgba[mask_bool, 3] = np.clip((prob_slice[mask_bool] * 165.0), 60, 185).astype(np.uint8)

    # Crisp contour outline
    contours, _ = cv2.findContours(
        (binary_slice > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        contour_rgb = rgba[:, :, :3].copy()
        cv2.drawContours(contour_rgb, contours, -1, (245, 158, 11), thickness=2)
        contour_alpha = rgba[:, :, 3].copy()
        cv2.drawContours(contour_alpha, contours, -1, 240, thickness=2)
        rgba[:, :, :3] = contour_rgb
        rgba[:, :, 3] = contour_alpha

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(output_path, format="PNG")
    return output_path


def run_ct_lung_segmentation_on_slices(
    hu_slices: List[np.ndarray],
    checkpoint_path: Path,
    mask_output_path: Path,
    voxel_spacing_mm: Tuple[float, float, float] = (1.5, 1.5, 2.0),
    device_pref: str = "auto",
) -> Dict:
    tensor, _ = preprocess_ct_volume_from_slices(hu_slices)
    binary_mask, foreground_prob, summary = predict_lung_ct_volume(
        volume_tensor=tensor,
        checkpoint_path=checkpoint_path,
        voxel_spacing_mm=voxel_spacing_mm,
        device_pref=device_pref,
    )
    peak_idx = summary["peak_slice_index"]
    render_segmentation_mask_png(
        binary_slice=binary_mask[:, :, peak_idx],
        prob_slice=foreground_prob[:, :, peak_idx],
        output_path=mask_output_path,
    )
    summary["mask_output_path"] = str(mask_output_path)
    return summary


def run_ct_lung_segmentation_on_nifti(
    nifti_path: Path,
    checkpoint_path: Path,
    mask_output_path: Path,
    device_pref: str = "auto",
) -> Dict:
    tensor, _, zooms = preprocess_ct_volume_from_nifti(nifti_path)
    binary_mask, foreground_prob, summary = predict_lung_ct_volume(
        volume_tensor=tensor,
        checkpoint_path=checkpoint_path,
        voxel_spacing_mm=zooms,
        device_pref=device_pref,
    )
    peak_idx = summary["peak_slice_index"]
    render_segmentation_mask_png(
        binary_slice=binary_mask[:, :, peak_idx],
        prob_slice=foreground_prob[:, :, peak_idx],
        output_path=mask_output_path,
    )
    summary["mask_output_path"] = str(mask_output_path)
    return summary
