import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.orm import Session

from app.database.models import AuditLog, ImagingStudy


def record_audit_event(
    db: Session,
    study: ImagingStudy,
    event_type: str,
    actor: str,
    notes: str = "",
    model_name: str = "N/A",
    model_version: str = "N/A",
    primary_finding: str = "",
    confidence: Optional[float] = None,
) -> AuditLog:
    """
    Appends an immutable UTC AuditLog row. Existing rows are never overwritten or deleted.
    """
    entry = AuditLog(
        id=f"aud-{uuid.uuid4().hex[:12]}",
        study_id=study.id,
        accession_number=study.accession_number,
        modality=study.modality,
        model_name=model_name,
        model_version=model_version,
        primary_finding=primary_finding,
        confidence=confidence,
        event_type=event_type,
        actor=actor,
        notes=notes,
        is_demo=study.is_demo,
        timestamp_utc=datetime.now(timezone.utc),
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry
