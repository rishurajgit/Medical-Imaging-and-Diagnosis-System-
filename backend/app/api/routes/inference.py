from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.serializers import serialize_study
from app.core.security import verify_optional_auth
from app.database.models import ImagingStudy
from app.database.schemas import AnalyzeRequest
from app.database.session import get_db
from app.services.inference_service import run_study_inference

router = APIRouter(tags=["inference"])


@router.post("/api/analyze")
def analyze_study(
    payload: AnalyzeRequest,
    db: Session = Depends(get_db),
    _auth: dict = Depends(verify_optional_auth),
) -> Dict:
    study = db.query(ImagingStudy).filter(ImagingStudy.id == payload.studyId).first()
    if not study:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Study not found.")

    outcome = run_study_inference(db, study)
    db.refresh(study)
    return {
        "study": serialize_study(study),
        "status": outcome.get("status"),
        "message": outcome.get("message", "Inference completed."),
    }
