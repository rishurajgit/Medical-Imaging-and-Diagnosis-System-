from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_CXR_INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def preprocess_2d_xray_from_uint8(grayscale_uint8: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Converts a 2D grayscale numpy array into:
      - DenseNet-121 compatible normalized tensor [1, 3, 224, 224]
      - Original RGB uint8 array [H, W, 3] preserved for Grad-CAM overlay resizing
    """
    if grayscale_uint8.ndim == 2:
        pil_rgb = Image.fromarray(grayscale_uint8, mode="L").convert("RGB")
    else:
        pil_rgb = Image.fromarray(grayscale_uint8).convert("RGB")
    original_rgb = np.array(pil_rgb, dtype=np.uint8)
    tensor = _CXR_INFERENCE_TRANSFORM(pil_rgb).unsqueeze(0)
    return tensor, original_rgb


def preprocess_2d_xray_from_file(image_path: Path) -> Tuple[torch.Tensor, np.ndarray]:
    pil_rgb = Image.open(image_path).convert("RGB")
    original_rgb = np.array(pil_rgb, dtype=np.uint8)
    tensor = _CXR_INFERENCE_TRANSFORM(pil_rgb).unsqueeze(0)
    return tensor, original_rgb


def preprocess_ct_volume_from_slices(
    hu_slices: List[np.ndarray],
    hu_min: float = -1000.0,
    hu_max: float = 400.0,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Stacks ordered 2D Hounsfield Unit slices into a 3D volume [H, W, D], clips to lung/mediastinal
    HU range [-1000, 400], normalizes to [0, 1], and returns [1, 1, H, W, D] tensor for MONAI 3D U-Net.
    """
    if not hu_slices:
        raise ValueError("Cannot build CT volume from an empty slice list.")
    volume_hwd = np.stack(hu_slices, axis=-1).astype(np.float32)
    clipped = np.clip(volume_hwd, hu_min, hu_max)
    norm = (clipped - hu_min) / max(hu_max - hu_min, 1e-5)
    tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W, D]
    return tensor, norm


def preprocess_ct_volume_from_nifti(
    nifti_path: Path,
    hu_min: float = -1000.0,
    hu_max: float = 400.0,
) -> Tuple[torch.Tensor, np.ndarray, Tuple[float, float, float]]:
    """
    Loads a NIfTI CT volume (.nii / .nii.gz), canonicalizes orientation to RAS,
    clips Hounsfield Units to [-1000, 400], and returns ([1, 1, H, W, D], norm_volume, voxel_spacing_mm).
    """
    img = nib.load(str(nifti_path))
    canonical = nib.as_closest_canonical(img)
    data = canonical.get_fdata(dtype=np.float32)
    zooms = tuple(float(z) for z in canonical.header.get_zooms()[:3])
    clipped = np.clip(data, hu_min, hu_max)
    norm = (clipped - hu_min) / max(hu_max - hu_min, 1e-5)
    tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0)
    return tensor, norm, zooms
