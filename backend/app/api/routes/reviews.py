import uuid
from datetime import datetime, timezone
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.serializers import serialize_audit, serialize_study
from app.core.security import verify_optional_auth
from app.database.models import ImagingStudy, RadiologistReview
from app.database.schemas import ReviewCreateRequest
from app.database.session import get_db
from app.services.audit_service import record_audit_event

router = APIRouter(tags=["reviews"])


@router.post("/api/studies/{study_id}/review")
def submit_review(
    study_id: str,
    payload: ReviewCreateRequest,
    db: Session = Depends(get_db),
    _auth: dict = Depends(verify_optional_auth),
) -> Dict:
    study = db.query(ImagingStudy).filter(ImagingStudy.id == study_id).first()
    if not study:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Study not found.")

    prior_status = study.review_status
    review = RadiologistReview(
        id=f"rev-{uuid.uuid4().hex[:12]}",
        study_id=study.id,
        decision=payload.decision,
        reviewer_id=payload.reviewer.strip(),
        notes=payload.notes.strip(),
        prior_status=prior_status,
        created_at=datetime.now(timezone.utc),
    )
    db.add(review)
    study.review_status = payload.decision
    db.commit()
    db.refresh(study)

    latest_inf = (
        sorted(study.inference_results, key=lambda r: r.processed_at, reverse=True)[0]
        if study.inference_results
        else None
    )

    event_type = (
        "review_updated"
        if prior_status != "pending" and prior_status != payload.decision
        else payload.decision
    )
    note_text = payload.notes.strip() or (
        "AI output approved on review."
        if payload.decision == "approved"
        else "AI output rejected/overridden on review."
    )
    if event_type == "review_updated":
        note_text = f"[Status changed: {prior_status} -> {payload.decision}] {note_text}"

    audit_entry = record_audit_event(
        db=db,
        study=study,
        event_type=event_type,
        actor=payload.reviewer.strip(),
        model_name=latest_inf.model_name if latest_inf else "Manual Review",
        model_version=latest_inf.model_version if latest_inf else "N/A",
        primary_finding=latest_inf.primary_finding if latest_inf else "Manual Review",
        confidence=latest_inf.confidence if latest_inf else None,
        notes=note_text,
    )

    return {
        "study": serialize_study(study),
        "auditEntry": serialize_audit(audit_entry),
    }
