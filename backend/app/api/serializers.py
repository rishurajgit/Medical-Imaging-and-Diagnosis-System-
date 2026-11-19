from pathlib import Path
from typing import Dict, Optional

from app.database.models import AuditLog, ImagingStudy


def _artifact_url(file_path: Optional[str]) -> Optional[str]:
    if not file_path:
        return None
    name = Path(file_path).name
    return f"/api/artifacts/{name}"


def serialize_study(study: ImagingStudy) -> Dict:
    latest_inf = (
        sorted(study.inference_results, key=lambda r: r.processed_at, reverse=True)[0]
        if study.inference_results
        else None
    )
    latest_rev = (
        sorted(study.reviews, key=lambda r: r.created_at, reverse=True)[0]
        if study.reviews
        else None
    )

    slice_urls = []
    series_uid = None
    series_desc = f"{study.modality} {study.body_part}"
    if study.series:
        first_series = study.series[0]
        series_uid = first_series.series_instance_uid
        series_desc = first_series.series_description
        slice_urls = [_artifact_url(p) for p in (first_series.slice_preview_paths or []) if p]

    seg_meta = latest_inf.segmentation_metadata if latest_inf else None
    segmentation_out = None
    if seg_meta:
        segmentation_out = {
            "supported": True,
            "lesionDetected": seg_meta.get("lesion_detected"),
            "voxelCount": seg_meta.get("voxel_count"),
            "estimatedVolumeMm3": seg_meta.get("estimated_volume_mm3"),
            "meanProbability": seg_meta.get("mean_probability"),
            "maskImageUrl": _artifact_url(latest_inf.segmentation_mask_path),
            "note": "MONAI 3D U-Net volumetric lung segmentation output.",
        }

    return {
        "id": study.id,
        "accessionNumber": study.accession_number,
        "title": study.title,
        "modality": study.modality,
        "bodyPart": study.body_part,
        "imageUrl": _artifact_url(study.preview_image_path) or "",
        "sliceUrls": slice_urls,
        "sliceCount": study.slice_count,
        "isDemo": study.is_demo,
        "processingStatus": study.processing_status,
        "processingMessage": study.processing_message,
        "modelName": latest_inf.model_name if latest_inf else "Pending / Unassigned",
        "modelVersion": latest_inf.model_version if latest_inf else "N/A",
        "processedAt": latest_inf.processed_at.isoformat() if latest_inf else None,
        "primaryFinding": (
            latest_inf.primary_finding
            if latest_inf
            else (
                "Unsupported Modality — Manual Review Only"
                if study.processing_status == "unsupported_modality"
                else "Awaiting Trained Model Weights"
            )
        ),
        "confidence": latest_inf.confidence if latest_inf else None,
        "classProbabilities": latest_inf.class_probabilities if latest_inf else [],
        "gradCamImageUrl": _artifact_url(latest_inf.gradcam_path) if latest_inf else None,
        "gradCamSummary": latest_inf.gradcam_summary if latest_inf else (study.processing_message or ""),
        "clinicalImpression": (
            "Educational research inference output only — requires attending radiologist confirmation."
        ),
        "heatmapRegions": [],
        "segmentation": segmentation_out,
        "dicom": {
            "studyInstanceUid": study.study_instance_uid,
            "seriesInstanceUid": series_uid,
            "seriesDescription": series_desc,
            "datasetSource": study.source_type,
            "patientId": study.patient_id,
            "patientSex": study.patient_sex,
            "patientAge": study.patient_age,
            "studyDate": study.study_date,
            "sliceThicknessMm": study.slice_thickness_mm,
            "pixelSpacingMm": study.pixel_spacing_mm,
            "windowCenter": study.window_center,
            "windowWidth": study.window_width,
            "photometricInterpretation": study.photometric_interpretation,
            "institutionName": study.institution_name,
            "rows": study.rows,
            "columns": study.columns,
            "bitsAllocated": study.bits_allocated,
        },
        "status": study.review_status,
        "reviewedBy": latest_rev.reviewer_id if latest_rev else None,
        "reviewedAt": latest_rev.created_at.isoformat() if latest_rev else None,
        "reviewerNotes": latest_rev.notes if latest_rev else None,
        "createdAt": study.created_at.isoformat(),
        "sourceType": study.source_type,
    }


def serialize_audit(entry: AuditLog) -> Dict:
    return {
        "id": entry.id,
        "studyId": entry.study_id,
        "accessionNumber": entry.accession_number,
        "modality": entry.modality,
        "modelName": entry.model_name,
        "modelVersion": entry.model_version,
        "primaryFinding": entry.primary_finding,
        "confidence": entry.confidence,
        "decision": entry.event_type,
        "reviewer": entry.actor,
        "notes": entry.notes,
        "isDemo": entry.is_demo,
        "timestamp": entry.timestamp_utc.isoformat(),
    }
