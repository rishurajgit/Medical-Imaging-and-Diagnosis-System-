import base64
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from PIL import Image
from sqlalchemy.orm import Session

from app.api.serializers import serialize_study
from app.core.config import settings
from app.core.security import (
    anonymize_log_identifier,
    resolve_safe_path,
    validate_upload_extension,
    verify_optional_auth,
)
from app.database.models import DicomSeries, ImagingStudy
from app.database.schemas import UploadJsonRequest
from app.database.session import get_db
from app.services.audit_service import record_audit_event
from app.services.dicom_service import (
    DicomValidationError,
    parse_dicom_bytes,
    parse_dicom_zip,
    save_uint8_preview,
)
from app.services.inference_service import run_study_inference
from app.services.orthanc_service import OrthancError, orthanc_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["upload", "orthanc"])


def _ingest_bytes(
    db: Session,
    raw_bytes: bytes,
    filename: str,
    title: Optional[str],
    fallback_modality: str,
    fallback_body_part: str,
    fallback_patient_id: Optional[str],
) -> Dict:
    if not raw_bytes or len(raw_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file must be between 1 byte and {settings.max_upload_mb} MB.",
        )

    ext = validate_upload_extension(filename)
    study_id = f"study-{uuid.uuid4().hex[:10]}"
    accession = f"ACC-{datetime.now(timezone.utc).strftime('%Y')}-{uuid.uuid4().hex[:4].upper()}"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    raw_dest = resolve_safe_path(settings.uploads_dir, f"{study_id}{ext}")
    raw_dest.write_bytes(raw_bytes)

    preview_path: Optional[Path] = None
    slice_previews = []
    orthanc_study_id = None
    modality = (fallback_modality or "CR").upper()
    body_part = (fallback_body_part or "CHEST").upper()
    patient_id = (fallback_patient_id or f"ANON-{uuid.uuid4().hex[:6].upper()}").strip()
    patient_sex = "U"
    patient_age = "000Y"
    study_uid = f"1.2.826.0.1.3680043.8.498.{uuid.uuid4().int % 10**10}"
    series_uid = f"{study_uid}.1"
    series_desc = f"{modality} {body_part} UPLOAD"
    study_date = today
    institution = "Uploaded Educational Study"
    photometric = "MONOCHROME2"
    wc, ww = 128.0, 256.0
    slice_thickness = "N/A"
    pixel_spacing = "N/A"
    rows, cols, bits = None, None, None
    source_type = "upload_image"
    slice_count = 1

    try:
        if ext in {".dcm", ".dicom"}:
            source_type = "upload_dicom"
            parsed = parse_dicom_bytes(raw_bytes)
            modality = parsed.modality if parsed.modality != "OTHER" else modality
            body_part = parsed.body_part if parsed.body_part != "UNKNOWN" else body_part
            patient_id = parsed.patient_id or patient_id
            patient_sex = parsed.patient_sex
            patient_age = parsed.patient_age
            study_uid = parsed.study_instance_uid or study_uid
            series_uid = parsed.series_instance_uid or series_uid
            series_desc = parsed.series_description
            study_date = parsed.study_date if parsed.study_date != "Unknown" else today
            institution = parsed.institution_name
            photometric = parsed.photometric_interpretation
            wc, ww = parsed.window_center, parsed.window_width
            slice_thickness = parsed.slice_thickness_mm
            pixel_spacing = parsed.pixel_spacing_mm
            rows, cols, bits = parsed.rows, parsed.columns, parsed.bits_allocated

            preview_path = save_uint8_preview(
                parsed.display_uint8, settings.artifacts_dir / f"{study_id}_preview.png"
            )
            slice_previews = [str(preview_path)]

            if orthanc_service.is_reachable():
                try:
                    orth_resp = orthanc_service.upload_dicom_instance(raw_bytes)
                    orthanc_study_id = orth_resp.get("ParentStudy")
                except OrthancError as exc:
                    logger.warning("Orthanc forwarding skipped: %s", exc)

        elif ext == ".zip":
            source_type = "upload_zip"
            series = parse_dicom_zip(raw_bytes)
            first = series.slices[0]
            modality = first.modality if first.modality != "OTHER" else modality
            body_part = first.body_part if first.body_part != "UNKNOWN" else body_part
            patient_id = first.patient_id or patient_id
            study_uid = series.study_instance_uid or study_uid
            series_uid = series.series_instance_uid or series_uid
            series_desc = first.series_description
            study_date = first.study_date if first.study_date != "Unknown" else today
            photometric = first.photometric_interpretation
            wc, ww = first.window_center, first.window_width
            slice_thickness = first.slice_thickness_mm
            pixel_spacing = first.pixel_spacing_mm
            rows, cols, bits = first.rows, first.columns, first.bits_allocated
            slice_count = len(series.slices)

            # Save up to 24 evenly sampled preview slices for interactive UI slice scrubbing
            step = max(1, len(series.slices) // 24)
            for idx, sl in enumerate(series.slices[::step]):
                out_p = save_uint8_preview(
                    sl.display_uint8, settings.artifacts_dir / f"{study_id}_slice_{idx:03d}.png"
                )
                slice_previews.append(str(out_p))
            preview_path = Path(slice_previews[len(slice_previews) // 2])

        elif ext in {".png", ".jpg", ".jpeg", ".webp"}:
            source_type = "upload_image"
            img = Image.open(raw_dest).convert("L")
            cols, rows = img.size
            bits = 8
            preview_path = settings.artifacts_dir / f"{study_id}_preview.png"
            img.save(preview_path, format="PNG")
            slice_previews = [str(preview_path)]

        elif ext in {".nii", ".nii.gz"}:
            source_type = "upload_nifti"
            modality = "CT" if modality not in {"CT", "MR"} else modality

    except DicomValidationError as exc:
        raw_dest.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    logger.info(
        "Ingested study %s (modality=%s, patient=%s)",
        study_id,
        modality,
        anonymize_log_identifier(patient_id),
    )

    study = ImagingStudy(
        id=study_id,
        accession_number=accession,
        study_instance_uid=study_uid,
        orthanc_study_id=orthanc_study_id,
        patient_id=patient_id,
        patient_sex=patient_sex,
        patient_age=patient_age,
        title=(title or "").strip() or f"{modality} {body_part} — {filename}",
        modality=modality,
        body_part=body_part,
        study_date=study_date,
        institution_name=institution,
        source_type=source_type,
        is_demo=False,
        raw_file_path=str(raw_dest),
        preview_image_path=str(preview_path) if preview_path else None,
        slice_count=slice_count,
        photometric_interpretation=photometric,
        window_center=wc,
        window_width=ww,
        slice_thickness_mm=slice_thickness,
        pixel_spacing_mm=pixel_spacing,
        rows=rows,
        columns=cols,
        bits_allocated=bits,
        processing_status="queued",
        review_status="pending",
    )
    db.add(study)

    series_row = DicomSeries(
        id=f"ser-{uuid.uuid4().hex[:10]}",
        study_id=study.id,
        series_instance_uid=series_uid,
        series_description=series_desc,
        modality=modality,
        instance_count=slice_count,
        storage_dir=str(settings.uploads_dir),
        slice_preview_paths=slice_previews,
    )
    db.add(series_row)
    db.commit()
    db.refresh(study)

    record_audit_event(
        db=db,
        study=study,
        event_type="dicom_ingested" if source_type in {"upload_dicom", "upload_zip"} else "study_uploaded",
        actor="Upload API",
        notes=f"Stored {filename} ({len(raw_bytes) / 1024:.1f} KB) on disk.",
    )

    inference_outcome = run_study_inference(db, study)
    db.refresh(study)

    return {
        "study": serialize_study(study),
        "inference": inference_outcome,
    }


@router.post("/api/upload", status_code=status.HTTP_201_CREATED)
async def upload_study(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    title: Optional[str] = Form(default=None),
    modality: str = Form(default="CR"),
    bodyPart: str = Form(default="CHEST"),
    patientId: Optional[str] = Form(default=None),
    db: Session = Depends(get_db),
    _auth: dict = Depends(verify_optional_auth),
) -> Dict:
    """
    Accepts either standard multipart/form-data file uploads OR JSON base64 payloads.
    """
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = UploadJsonRequest.model_validate(await request.json())
        b64 = body.fileBase64.split(",", 1)[1] if "," in body.fileBase64 else body.fileBase64
        try:
            raw_bytes = base64.b64decode(b64)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid base64 file payload."
            ) from exc
        return _ingest_bytes(
            db=db,
            raw_bytes=raw_bytes,
            filename=body.fileName,
            title=body.title,
            fallback_modality=body.modality or "CR",
            fallback_body_part=body.bodyPart or "CHEST",
            fallback_patient_id=body.patientId,
        )

    if file is None or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing uploaded file.",
        )

    raw_bytes = await file.read()
    return _ingest_bytes(
        db=db,
        raw_bytes=raw_bytes,
        filename=file.filename,
        title=title,
        fallback_modality=modality,
        fallback_body_part=bodyPart,
        fallback_patient_id=patientId,
    )


@router.post("/api/orthanc/sync")
def sync_orthanc_studies(
    db: Session = Depends(get_db),
    _auth: dict = Depends(verify_optional_auth),
) -> Dict:
    """
    Polls the configured Orthanc DICOM Server for new studies, downloads instances,
    extracts DICOM metadata & pixels, runs supported AI inference, and persists results.
    """
    if not orthanc_service.is_reachable():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Orthanc DICOM Server is unreachable at {settings.orthanc_url}.",
        )

    orthanc_ids = orthanc_service.list_studies()
    ingested = []
    for orth_id in orthanc_ids:
        existing = (
            db.query(ImagingStudy).filter(ImagingStudy.orthanc_study_id == orth_id).first()
        )
        if existing:
            continue
        study_meta = orthanc_service.get_study(orth_id)
        series_ids = study_meta.get("Series", [])
        if not series_ids:
            continue
        first_series = orthanc_service.get_series(series_ids[0])
        instances = first_series.get("Instances", [])
        if not instances:
            continue
        raw_dcm = orthanc_service.get_instance_dicom_bytes(instances[0])
        res = _ingest_bytes(
            db=db,
            raw_bytes=raw_dcm,
            filename=f"orthanc_{orth_id}.dcm",
            title=f"Orthanc Study {orth_id[:8]}",
            fallback_modality="CR",
            fallback_body_part="CHEST",
            fallback_patient_id=None,
        )
        # Attach Orthanc ID explicitly
        db_study = db.query(ImagingStudy).filter(ImagingStudy.id == res["study"]["id"]).first()
        if db_study:
            db_study.orthanc_study_id = orth_id
            db_study.source_type = "orthanc"
            db.commit()
        ingested.append(res["study"]["id"])

    return {"syncedStudies": ingested, "totalOrthancStudies": len(orthanc_ids)}
