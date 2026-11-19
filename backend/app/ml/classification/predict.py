from pathlib import Path
from typing import Dict, List, Tuple

import torch

from app.ml.classification.model import (
    CLASSIFICATION_MODEL_SPEC,
    NIH_CHESTXRAY14_LABELS,
    ChestXRayDenseNet121,
    load_trained_classifier,
)


def predict_chest_xray(
    input_tensor: torch.Tensor,
    checkpoint_path: Path,
    device_pref: str = "auto",
) -> Tuple[Dict, ChestXRayDenseNet121, torch.device]:
    """
    Runs real PyTorch DenseNet-121 multi-label inference on a preprocessed [1, 3, 224, 224] tensor.
    Raises ModelWeightsUnavailableError if checkpoint_path does not exist (never fabricates outputs).
    """
    model, device, meta = load_trained_classifier(checkpoint_path, device_pref=device_pref)
    labels: List[str] = meta.get("labels", NIH_CHESTXRAY14_LABELS)

    with torch.no_grad():
        logits = model(input_tensor.to(device))
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    ranked = sorted(
        [{"label": lbl, "probability": round(float(p), 4)} for lbl, p in zip(labels, probs)],
        key=lambda item: item["probability"],
        reverse=True,
    )

    top_item = ranked[0]
    top_index = labels.index(top_item["label"])
    primary_finding = (
        top_item["label"]
        if top_item["probability"] >= 0.5
        else f"Low-probability finding (highest: {top_item['label']})"
    )

    result = {
        "model_id": CLASSIFICATION_MODEL_SPEC.model_id,
        "model_name": CLASSIFICATION_MODEL_SPEC.name,
        "model_version": meta.get("version", CLASSIFICATION_MODEL_SPEC.version),
        "primary_finding": primary_finding,
        "top_class_index": top_index,
        "confidence": top_item["probability"],
        "class_probabilities": ranked[:6],
        "all_probabilities": ranked,
    }
    return result, model, device
