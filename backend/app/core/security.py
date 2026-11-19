import re
import secrets
from pathlib import Path
from typing import Optional
from fastapi import Header, HTTPException, status

from app.core.config import settings

ALLOWED_EXTENSIONS = {".dcm", ".dicom", ".zip", ".png", ".jpg", ".jpeg", ".webp", ".nii", ".gz"}
SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9._-]")


def sanitize_filename(filename: str) -> str:
    """Strip directory components and unsafe characters to prevent path traversal."""
    base = Path(filename).name
    cleaned = SAFE_FILENAME_RE.sub("_", base)
    if not cleaned or cleaned in {".", ".."}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename provided.",
        )
    return cleaned


def resolve_safe_path(base_dir: Path, relative_name: str) -> Path:
    """Resolve a child path and assert it remains strictly within base_dir."""
    base_resolved = base_dir.resolve()
    target = (base_resolved / sanitize_filename(relative_name)).resolve()
    if not str(target).startswith(str(base_resolved)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Rejected unsafe file path.",
        )
    return target


def validate_upload_extension(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".nii.gz"):
        return ".nii.gz"
    ext = Path(lower).suffix
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )
    return ext


def anonymize_log_identifier(patient_id: Optional[str]) -> str:
    """Avoid writing raw PatientID values to application logs."""
    if not patient_id:
        return "ANON"
    if len(patient_id) <= 4:
        return "****"
    return f"***{patient_id[-4:]}"


def verify_optional_auth(
    authorization: Optional[str] = Header(default=None),
    x_reviewer_role: Optional[str] = Header(default="radiologist"),
) -> dict:
    """
    Optional Bearer token & role guard controlled by REQUIRE_AUTH env var.
    In educational local mode (REQUIRE_AUTH=false), permits local testing while preserving RBAC hooks.
    """
    if not settings.require_auth:
        return {"authenticated": True, "role": x_reviewer_role or "radiologist"}

    if not settings.api_auth_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="REQUIRE_AUTH is enabled but API_AUTH_TOKEN is not configured.",
        )

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer authorization token.",
        )

    token = authorization.split(" ", 1)[1].strip()
    if not secrets.compare_digest(token, settings.api_auth_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid authorization token.",
        )

    return {"authenticated": True, "role": x_reviewer_role or "radiologist"}
