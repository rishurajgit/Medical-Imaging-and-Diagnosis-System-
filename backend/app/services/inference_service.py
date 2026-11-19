import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.models import ImagingStudy, InferenceResult
from app.ml.classification.model import (
    CLASSIFICATION_MODEL_SPEC,
    ModelWeightsUnavailableError,
)
from app.ml.classification.predict import predict_chest_xray
from app.ml.segmentation.model import LUNG_CT_SEGMENTATION_SPEC
from app.services.audit_service import record_audit_event
from app.services.dicom_service import parse_dicom_bytes, parse_dicom_zip
from app.services.gradcam_service import generate_gradcam_artifacts
from app.services.preprocessing_service import (
    preprocess_2d_xray_from_file,
    preprocess_2d_xray_from_uint8,
)
from app.services.segmentation_service import (
    run_ct_lung_segmentation_on_nifti,
    run_ct_lung_segmentation_on_slices,
)

CHEST_ANATOMY_TOKENS = {"CHEST", "THORAX", "LUNG", "PULMONARY", "UNKNOWN", ""}


def route_study_pipeline(modality: str, body_part: str) -> Tuple[Optional[str], str]:
    mod = (modality or "").upper()
    bp = (body_part or "UNKNOWN").upper()

    if mod in CLASSIFICATION_MODEL_SPEC.supported_modalities and bp in CHEST_ANATOMY_TOKENS:
        return CLASSIFICATION_MODEL_SPEC.model_id, CLASSIFICATION_MODEL_SPEC.name

    if mod in LUNG_CT_SEGMENTATION_SPEC.supported_modalities and bp in CHEST_ANATOMY_TOKENS:
        return LUNG_CT_SEGMENTATION_SPEC.model_id, LUNG_CT_SEGMENTATION_SPEC.name

    return (
        None,
        f"Unsupported modality/anatomy combination ({mod} / {bp}). "
        f"Supported pipelines: CR/DX Chest (NIH ChestX-ray14 DenseNet-121) and "
        f"CT Thorax/Lung (MSD Task06_Lung MONAI 3D U-Net).",
    )


def run_study_inference(db: Session, study: ImagingStudy) -> Dict:
    """
    Executes the appropriate real PyTorch/MONAI pipeline for a study.
    - Rejects unsupported modalities gracefully without running mismatched models.
    - Explicitly records `weights_unavailable` when checkpoints are missing (never fabricates outputs).
    - Persists InferenceResult and append-only UTC AuditLog records.
    """
    pipeline_id, pipeline_desc = route_study_pipeline(study.modality, study.body_part)

    record_audit_event(
        db=db,
        study=study,
        event_type="inference_requested",
        actor="Inference Service",
        notes=f"Routed study to pipeline: {pipeline_id or 'none'}",
    )

    if pipeline_id is None:
        study.processing_status = "unsupported_modality"
        study.processing_message = pipeline_desc
        db.commit()
        record_audit_event(
            db=db,
            study=study,
            event_type="inference_blocked",
            actor="Inference Router",
            notes=pipeline_desc,
        )
        return {"status": "unsupported_modality", "message": pipeline_desc}

    raw_path = Path(study.raw_file_path) if study.raw_file_path else None
    if raw_path is None or not raw_path.exists():
        msg = "Source scan file not found on storage volume."
        study.processing_status = "failed"
        study.processing_message = msg
        db.commit()
        return {"status": "failed", "message": msg}

    try:
        if pipeline_id == CLASSIFICATION_MODEL_SPEC.model_id:
            if raw_path.suffix.lower() in {".dcm", ".dicom"}:
                parsed = parse_dicom_bytes(raw_path.read_bytes())
                tensor, original_rgb = preprocess_2d_xray_from_uint8(parsed.display_uint8)
            else:
                tensor, original_rgb = preprocess_2d_xray_from_file(raw_path)

            pred, model, device = predict_chest_xray(
                input_tensor=tensor,
                checkpoint_path=settings.cxr_weights_path,
                device_pref=settings.device,
            )

            heatmap_file = settings.artifacts_dir / f"{study.id}_gradcam.png"
            blended_file = settings.artifacts_dir / f"{study.id}_gradcam_blended.png"
            cam_meta = generate_gradcam_artifacts(
                model=model,
                input_tensor=tensor,
                original_rgb=original_rgb,
                target_class_index=pred["top_class_index"],
                device=device,
                heatmap_output_path=heatmap_file,
                blended_output_path=blended_file,
            )

            inf = InferenceResult(
                id=f"inf-{uuid.uuid4().hex[:12]}",
                study_id=study.id,
                model_name=pred["model_name"],
                model_version=pred["model_version"],
                status="completed",
                primary_finding=pred["primary_finding"],
                confidence=pred["confidence"],
                class_probabilities=pred["class_probabilities"],
                gradcam_path=str(heatmap_file),
                gradcam_summary=(
                    f"Real DenseNet-121 Grad-CAM computed from features.norm5 gradients for top class "
                    f"'{pred['primary_finding']}' (peak normalized XY={cam_meta['peak_normalized_xy']}). "
                    f"{cam_meta['disclaimer']}"
                ),
                processed_at=datetime.now(timezone.utc),
            )
            db.add(inf)
            study.processing_status = "completed"
            study.processing_message = "DenseNet-121 classification and Grad-CAM completed."
            db.commit()

            record_audit_event(
                db=db,
                study=study,
                event_type="inference_completed",
                actor=pred["model_name"],
                model_name=pred["model_name"],
                model_version=pred["model_version"],
                primary_finding=pred["primary_finding"],
                confidence=pred["confidence"],
                notes="Multi-label classification completed via PyTorch DenseNet-121.",
            )
            record_audit_event(
                db=db,
                study=study,
                event_type="heatmap_generated",
                actor="Grad-CAM Service",
                model_name=pred["model_name"],
                model_version=pred["model_version"],
                primary_finding=pred["primary_finding"],
                confidence=pred["confidence"],
                notes=f"Grad-CAM PNG artifact written to {heatmap_file.name}.",
            )
            return {"status": "completed", "inference_id": inf.id}

        # Thoracic CT MONAI 3D U-Net Segmentation pipeline
        mask_file = settings.artifacts_dir / f"{study.id}_seg_mask.png"
        lower_name = raw_path.name.lower()
        if lower_name.endswith(".nii") or lower_name.endswith(".nii.gz"):
            seg_summary = run_ct_lung_segmentation_on_nifti(
                nifti_path=raw_path,
                checkpoint_path=settings.seg_weights_path,
                mask_output_path=mask_file,
                device_pref=settings.device,
            )
        elif raw_path.suffix.lower() == ".zip":
            series = parse_dicom_zip(raw_path.read_bytes())
            hu_slices = [s.hu_array for s in series.slices]
            seg_summary = run_ct_lung_segmentation_on_slices(
                hu_slices=hu_slices,
                checkpoint_path=settings.seg_weights_path,
                mask_output_path=mask_file,
                device_pref=settings.device,
            )
        elif raw_path.suffix.lower() in {".dcm", ".dicom"}:
            parsed = parse_dicom_bytes(raw_path.read_bytes())
            # Single CT slice stacked into minimal depth volume for sliding window inference
            seg_summary = run_ct_lung_segmentation_on_slices(
                hu_slices=[parsed.hu_array],
                checkpoint_path=settings.seg_weights_path,
                mask_output_path=mask_file,
                device_pref=settings.device,
            )
        else:
            raise ValueError(
                "CT lung segmentation requires NIfTI (.nii/.nii.gz) or DICOM (.dcm/.zip) volume input."
            )

        finding = (
            f"Segmented Lung Region ({seg_summary['estimated_volume_mm3']} mm³)"
            if seg_summary["lesion_detected"]
            else "No Foreground Lung Lesion Segmented"
        )
        inf = InferenceResult(
            id=f"inf-{uuid.uuid4().hex[:12]}",
            study_id=study.id,
            model_name=seg_summary["model_name"],
            model_version=seg_summary["model_version"],
            status="completed",
            primary_finding=finding,
            confidence=seg_summary["mean_probability"],
            class_probabilities=[
                {"label": "Foreground Lesion", "probability": seg_summary["mean_probability"]},
                {
                    "label": "Background",
                    "probability": round(1.0 - seg_summary["mean_probability"], 4),
                },
            ],
            segmentation_mask_path=str(mask_file),
            segmentation_metadata=seg_summary,
            gradcam_summary="3D U-Net outputs voxel-level segmentation probability maps rather than 2D classification Grad-CAM.",
            processed_at=datetime.now(timezone.utc),
        )
        db.add(inf)
        study.processing_status = "completed"
        study.processing_message = "MONAI 3D U-Net volumetric lung segmentation completed."
        db.commit()

        record_audit_event(
            db=db,
            study=study,
            event_type="inference_completed",
            actor=seg_summary["model_name"],
            model_name=seg_summary["model_name"],
            model_version=seg_summary["model_version"],
            primary_finding=finding,
            confidence=seg_summary["mean_probability"],
            notes=f"Segmentation mask written to {mask_file.name}.",
        )
        return {"status": "completed", "inference_id": inf.id}

    except ModelWeightsUnavailableError as exc:
        msg = str(exc)
        study.processing_status = "weights_unavailable"
        study.processing_message = msg
        db.commit()
        record_audit_event(
            db=db,
            study=study,
            event_type="inference_blocked",
            actor="Inference Service",
            model_name=pipeline_desc,
            notes=msg,
        )
        return {"status": "weights_unavailable", "message": msg}
    except Exception as exc:
        msg = f"Inference failed: {exc}"
        study.processing_status = "failed"
        study.processing_message = msg
        db.commit()
        record_audit_event(
            db=db,
            study=study,
            event_type="inference_blocked",
            actor="Inference Service",
            model_name=pipeline_desc,
            notes=msg,
        )
        return {"status": "failed", "message": msg}
