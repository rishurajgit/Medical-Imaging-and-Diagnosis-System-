from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torchvision import models

NIH_CHESTXRAY14_LABELS: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    name: str
    version: str
    architecture: str
    supported_modalities: Tuple[str, ...]
    supported_anatomy: Tuple[str, ...]
    input_format: str
    dataset: str
    num_classes: int


CLASSIFICATION_MODEL_SPEC = ModelSpec(
    model_id="cxr_densenet121_nih14",
    name="DenseNet-121 Chest X-Ray Classifier",
    version="1.0.0-nih14",
    architecture="TorchVision DenseNet-121 (Transfer Learning, 14-label BCEWithLogitsLoss)",
    supported_modalities=("CR", "DX"),
    supported_anatomy=("CHEST", "THORAX"),
    input_format="2D Grayscale DICOM (.dcm) or PNG/JPEG converted to 3-channel 224x224 ImageNet-normalized tensor",
    dataset="NIH ChestX-ray14",
    num_classes=len(NIH_CHESTXRAY14_LABELS),
)


class ModelWeightsUnavailableError(FileNotFoundError):
    """Raised whenever trained checkpoint weights are absent instead of returning synthetic predictions."""


class ChestXRayDenseNet121(nn.Module):
    """
    Transfer-learning DenseNet-121 classifier for NIH ChestX-ray14 multi-label classification.
    Exposes `get_gradcam_target_layer()` returning the final convolutional block (`features.norm5`)
    for genuine gradient-weighted class activation mapping.
    """

    def __init__(self, num_classes: int = len(NIH_CHESTXRAY14_LABELS), pretrained: bool = False):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.densenet121(weights=weights)
        in_features = backbone.classifier.in_features
        backbone.classifier = nn.Linear(in_features, num_classes)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_gradcam_target_layer(self) -> nn.Module:
        return self.backbone.features.norm5


def resolve_device(device_pref: str = "auto") -> torch.device:
    if device_pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device_pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def load_trained_classifier(
    checkpoint_path: Path, device_pref: str = "auto"
) -> Tuple[ChestXRayDenseNet121, torch.device, Dict]:
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        raise ModelWeightsUnavailableError(
            f"Trained Chest X-ray checkpoint not found at '{checkpoint_path}'. "
            "Download NIH ChestX-ray14 and run `python -m app.ml.classification.train` "
            "or mount a trained checkpoint file."
        )

    device = resolve_device(device_pref)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = payload.get("model_state_dict", payload)
    model = ChestXRayDenseNet121(num_classes=len(NIH_CHESTXRAY14_LABELS), pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    metadata = {
        "epoch": payload.get("epoch"),
        "val_macro_auroc": payload.get("val_macro_auroc"),
        "labels": payload.get("labels", NIH_CHESTXRAY14_LABELS),
        "version": payload.get("version", CLASSIFICATION_MODEL_SPEC.version),
    }
    return model, device, metadata
