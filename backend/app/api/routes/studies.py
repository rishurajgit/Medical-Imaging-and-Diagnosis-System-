from pathlib import Path
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.serializers import serialize_audit, serialize_study
from app.core.security import verify_optional_auth
from app.database.models import AuditLog, ImagingStudy
from app.database.session import get_db

router = APIRouter(prefix="/api/studies", tags=["studies"])


@router.get("")
def list_studies(
    db: Session = Depends(get_db),
    _auth: dict = Depends(verify_optional_auth),
) -> Dict:
    studies = db.query(ImagingStudy).order_by(ImagingStudy.created_at.desc()).all()
    logs = db.query(AuditLog).order_by(AuditLog.timestamp_utc.desc()).limit(250).all()
    return {
        "studies": [serialize_study(s) for s in studies],
        "auditLog": [serialize_audit(a) for a in logs],
    }


@router.get("/{study_id}")
def get_study(
    study_id: str,
    db: Session = Depends(get_db),
    _auth: dict = Depends(verify_optional_auth),
) -> Dict:
    study = db.query(ImagingStudy).filter(ImagingStudy.id == study_id).first()
    if not study:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Study not found.")
    return {"study": serialize_study(study)}


@router.get("/{study_id}/results")
def get_study_results(
    study_id: str,
    db: Session = Depends(get_db),
    _auth: dict = Depends(verify_optional_auth),
) -> Dict:
    study = db.query(ImagingStudy).filter(ImagingStudy.id == study_id).first()
    if not study:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Study not found.")
    serialized = serialize_study(study)
    return {
        "studyId": serialized["id"],
        "processingStatus": serialized["processingStatus"],
        "processingMessage": serialized["processingMessage"],
        "modelName": serialized["modelName"],
        "modelVersion": serialized["modelVersion"],
        "processedAt": serialized["processedAt"],
        "primaryFinding": serialized["primaryFinding"],
        "confidence": serialized["confidence"],
        "classProbabilities": serialized["classProbabilities"],
        "gradCamImageUrl": serialized["gradCamImageUrl"],
        "gradCamSummary": serialized["gradCamSummary"],
        "segmentation": serialized["segmentation"],
    }


@router.get("/{study_id}/heatmap")
def get_study_heatmap(
    study_id: str,
    db: Session = Depends(get_db),
    _auth: dict = Depends(verify_optional_auth),
):
    study = db.query(ImagingStudy).filter(ImagingStudy.id == study_id).first()
    if not study:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Study not found.")
    latest_inf = (
        sorted(study.inference_results, key=lambda r: r.processed_at, reverse=True)[0]
        if study.inference_results
        else None
    )
    if not latest_inf or not latest_inf.gradcam_path or not Path(latest_inf.gradcam_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Grad-CAM heatmap artifact not generated for this study.",
        )
    return FileResponse(latest_inf.gradcam_path, media_type="image/png")


@router.get("/{study_id}/segmentation")
def get_study_segmentation(
    study_id: str,
    db: Session = Depends(get_db),
    _auth: dict = Depends(verify_optional_auth),
):
    study = db.query(ImagingStudy).filter(ImagingStudy.id == study_id).first()
    if not study:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Study not found.")
    latest_inf = (
        sorted(study.inference_results, key=lambda r: r.processed_at, reverse=True)[0]
        if study.inference_results
        else None
    )
    if (
        not latest_inf
        or not latest_inf.segmentation_mask_path
        or not Path(latest_inf.segmentation_mask_path).exists()
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Segmentation mask artifact not generated for this study.",
        )
    return FileResponse(latest_inf.segmentation_mask_path, media_type="image/png")
