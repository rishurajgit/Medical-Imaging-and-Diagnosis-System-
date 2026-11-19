import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter

from app.core.config import settings
from app.ml.classification.model import CLASSIFICATION_MODEL_SPEC
from app.ml.segmentation.model import LUNG_CT_SEGMENTATION_SPEC
from app.services.orthanc_service import orthanc_service

router = APIRouter(tags=["health", "models", "metrics"])


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


@router.get("/api/health")
def health_check() -> Dict[str, Any]:
    cxr_ready = settings.cxr_weights_path.exists()
    seg_ready = settings.seg_weights_path.exists()
    return {
        "status": "ok",
        "service": "medvision-fastapi-backend",
        "orthancReachable": orthanc_service.is_reachable(),
        "modelsLoaded": int(cxr_ready) + int(seg_ready),
        "totalRegisteredModels": 2,
        "cxrWeightsLoaded": cxr_ready,
        "segmentationWeightsLoaded": seg_ready,
        "disclaimer": (
            "FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY. NOT FOR CLINICAL USE."
        ),
    }


@router.get("/api/models")
def list_models() -> Dict[str, Any]:
    cxr_ready = settings.cxr_weights_path.exists()
    seg_ready = settings.seg_weights_path.exists()
    return {
        "models": [
            {
                "id": CLASSIFICATION_MODEL_SPEC.model_id,
                "name": CLASSIFICATION_MODEL_SPEC.name,
                "version": CLASSIFICATION_MODEL_SPEC.version,
                "architecture": CLASSIFICATION_MODEL_SPEC.architecture,
                "task": "classification",
                "supportedModalities": list(CLASSIFICATION_MODEL_SPEC.supported_modalities),
                "supportedAnatomy": list(CLASSIFICATION_MODEL_SPEC.supported_anatomy),
                "inputFormat": CLASSIFICATION_MODEL_SPEC.input_format,
                "dataset": CLASSIFICATION_MODEL_SPEC.dataset,
                "explainabilityMethod": "Grad-CAM on features.norm5 convolutional gradients",
                "weightsPath": str(settings.cxr_weights_path),
                "weightsLoaded": cxr_ready,
                "statusMessage": (
                    "Checkpoint loaded and ready for inference."
                    if cxr_ready
                    else f"Weights not found at {settings.cxr_weights_path}. Run classification training first."
                ),
                "limitations": [
                    "Supports frontal 2D Chest X-rays (CR/DX) only.",
                    "NIH ChestX-ray14 NLP labels carry documented noise (~10%).",
                    "Grad-CAM shows model gradient sensitivity, not clinical causation.",
                ],
            },
            {
                "id": LUNG_CT_SEGMENTATION_SPEC.model_id,
                "name": LUNG_CT_SEGMENTATION_SPEC.name,
                "version": LUNG_CT_SEGMENTATION_SPEC.version,
                "architecture": LUNG_CT_SEGMENTATION_SPEC.architecture,
                "task": "segmentation",
                "supportedModalities": list(LUNG_CT_SEGMENTATION_SPEC.supported_modalities),
                "supportedAnatomy": list(LUNG_CT_SEGMENTATION_SPEC.supported_anatomy),
                "inputFormat": LUNG_CT_SEGMENTATION_SPEC.input_format,
                "dataset": LUNG_CT_SEGMENTATION_SPEC.dataset,
                "explainabilityMethod": "Voxel-level foreground probability map and binary mask overlay",
                "weightsPath": str(settings.seg_weights_path),
                "weightsLoaded": seg_ready,
                "statusMessage": (
                    "Checkpoint loaded and ready for volumetric CT inference."
                    if seg_ready
                    else f"Weights not found at {settings.seg_weights_path}. Run MONAI 3D U-Net training first."
                ),
                "limitations": [
                    "Supports thoracic CT volumes only; does not support X-ray or Brain MRI.",
                    "Performance may degrade on thick-slice (>5mm) or severe motion-artifact CT.",
                    "Segmentation contours require radiologist review.",
                ],
            },
        ]
    }


@router.get("/api/metrics")
def get_metrics() -> Dict[str, Any]:
    cxr_path = settings.reports_dir / "classification_metrics.json"
    seg_path = settings.reports_dir / "segmentation_metrics.json"
    cxr_json = _read_json(cxr_path)
    seg_json = _read_json(seg_path)

    return {
        "evaluationAvailable": bool(cxr_json or seg_json),
        "note": (
            "Loaded directly from local DVC evaluation reports."
            if (cxr_json or seg_json)
            else "No local training/evaluation metrics generated yet. Never fabricated."
        ),
        "classification": {
            "status": "evaluated" if cxr_json else "not_evaluated_locally",
            "dataset": "NIH ChestX-ray14",
            "metricsFile": str(cxr_path),
            "macroAuroc": cxr_json.get("macro_auroc") if cxr_json else None,
            "perClassAuroc": cxr_json.get("per_class_auroc") if cxr_json else None,
            "precision": cxr_json.get("precision") if cxr_json else None,
            "recall": cxr_json.get("recall") if cxr_json else None,
            "f1Score": cxr_json.get("f1_score") if cxr_json else None,
            "evaluatedAt": cxr_json.get("evaluated_at") if cxr_json else None,
        },
        "segmentation": {
            "status": "evaluated" if seg_json else "not_evaluated_locally",
            "dataset": "Medical Segmentation Decathlon (Task06_Lung)",
            "metricsFile": str(seg_path),
            "meanDice": seg_json.get("mean_dice") if seg_json else None,
            "meanIoU": seg_json.get("mean_iou") if seg_json else None,
            "sensitivity": seg_json.get("sensitivity") if seg_json else None,
            "specificity": seg_json.get("specificity") if seg_json else None,
            "evaluatedAt": seg_json.get("evaluated_at") if seg_json else None,
        },
    }
