from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from monai.networks.layers import Norm
from monai.networks.nets import UNet

from app.ml.classification.model import ModelWeightsUnavailableError, resolve_device


@dataclass(frozen=True)
class SegmentationModelSpec:
    model_id: str
    name: str
    version: str
    architecture: str
    supported_modalities: Tuple[str, ...]
    supported_anatomy: Tuple[str, ...]
    input_format: str
    dataset: str
    in_channels: int
    out_channels: int
    roi_size: Tuple[int, int, int]
    hu_min: float
    hu_max: float


LUNG_CT_SEGMENTATION_SPEC = SegmentationModelSpec(
    model_id="lung_ct_monai_unet3d",
    name="MONAI 3D U-Net Lung Segmentation",
    version="1.0.0-msd-task06",
    architecture="MONAI 3D U-Net (channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2)",
    supported_modalities=("CT",),
    supported_anatomy=("LUNG", "CHEST", "THORAX"),
    input_format="3D Thoracic CT Volume (NIfTI .nii/.nii.gz or DICOM CT series, RAS orientation, HU [-1000, 400])",
    dataset="Medical Segmentation Decathlon — Task06_Lung",
    in_channels=1,
    out_channels=2,  # 0=Background, 1=Lung Tumor/Lesion
    roi_size=(96, 96, 96),
    hu_min=-1000.0,
    hu_max=400.0,
)

# Modular registry allowing future Brain MRI (BraTS) 3D U-Net registration without changing core API
SEGMENTATION_MODEL_REGISTRY: Dict[str, SegmentationModelSpec] = {
    LUNG_CT_SEGMENTATION_SPEC.model_id: LUNG_CT_SEGMENTATION_SPEC,
}


def build_monai_unet3d(
    in_channels: int = LUNG_CT_SEGMENTATION_SPEC.in_channels,
    out_channels: int = LUNG_CT_SEGMENTATION_SPEC.out_channels,
) -> UNet:
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    )


def load_trained_segmentation_model(
    checkpoint_path: Path,
    spec: SegmentationModelSpec = LUNG_CT_SEGMENTATION_SPEC,
    device_pref: str = "auto",
) -> Tuple[UNet, torch.device, Dict]:
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        raise ModelWeightsUnavailableError(
            f"Trained 3D U-Net segmentation checkpoint not found at '{checkpoint_path}'. "
            "Download Medical Segmentation Decathlon Task06_Lung and run "
            "`python -m app.ml.segmentation.train` or mount a trained checkpoint file."
        )

    device = resolve_device(device_pref)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = payload.get("model_state_dict", payload)

    model = build_monai_unet3d(in_channels=spec.in_channels, out_channels=spec.out_channels)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    meta = {
        "epoch": payload.get("epoch"),
        "val_mean_dice": payload.get("val_mean_dice"),
        "version": payload.get("version", spec.version),
        "roi_size": tuple(payload.get("roi_size", spec.roi_size)),
    }
    return model, device, meta
