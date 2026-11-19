from typing import Dict, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.serializers import serialize_audit
from app.core.security import verify_optional_auth
from app.database.models import AuditLog
from app.database.session import get_db

router = APIRouter(tags=["audit"])


@router.get("/api/audit")
def get_audit_log(
    study_id: Optional[str] = Query(default=None),
    event_type: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    db: Session = Depends(get_db),
    _auth: dict = Depends(verify_optional_auth),
) -> Dict:
    query = db.query(AuditLog)
    if study_id:
        query = query.filter(AuditLog.study_id == study_id)
    if event_type:
        query = query.filter(AuditLog.event_type == event_type)

    rows = query.order_by(AuditLog.timestamp_utc.desc()).limit(limit).all()
    return {
        "count": len(rows),
        "auditLog": [serialize_audit(r) for r in rows],
    }
