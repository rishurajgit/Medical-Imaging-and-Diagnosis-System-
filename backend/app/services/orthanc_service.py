import logging
from typing import Any, Dict, List, Optional
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class OrthancError(RuntimeError):
    """Raised when Orthanc PACS server communication fails."""


class OrthancService:
    """
    REST & DICOMweb client for integrating with Orthanc DICOM Server.
    Communicates exclusively over internal network credentials defined in environment variables.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: float = 15.0,
    ):
        self.base_url = (base_url or settings.orthanc_url).rstrip("/")
        self.auth = (
            username or settings.orthanc_username,
            password or settings.orthanc_password,
        )
        self.timeout = timeout

    def is_reachable(self) -> bool:
        try:
            with httpx.Client(auth=self.auth, timeout=3.0) as client:
                resp = client.get(f"{self.base_url}/system")
                return resp.status_code == 200
        except Exception:
            return False

    def get_system_info(self) -> Dict[str, Any]:
        with httpx.Client(auth=self.auth, timeout=self.timeout) as client:
            resp = client.get(f"{self.base_url}/system")
            resp.raise_for_status()
            return resp.json()

    def upload_dicom_instance(self, dicom_bytes: bytes) -> Dict[str, Any]:
        """Stores a DICOM instance in Orthanc via POST /instances and returns Orthanc IDs."""
        try:
            with httpx.Client(auth=self.auth, timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/instances",
                    content=dicom_bytes,
                    headers={"Content-Type": "application/dicom"},
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            logger.warning("Orthanc upload failed: %s", exc)
            raise OrthancError(f"Failed to upload DICOM to Orthanc: {exc}") from exc

    def list_studies(self) -> List[str]:
        with httpx.Client(auth=self.auth, timeout=self.timeout) as client:
            resp = client.get(f"{self.base_url}/studies")
            resp.raise_for_status()
            return resp.json()

    def get_study(self, orthanc_study_id: str) -> Dict[str, Any]:
        with httpx.Client(auth=self.auth, timeout=self.timeout) as client:
            resp = client.get(f"{self.base_url}/studies/{orthanc_study_id}")
            resp.raise_for_status()
            return resp.json()

    def get_series(self, orthanc_series_id: str) -> Dict[str, Any]:
        with httpx.Client(auth=self.auth, timeout=self.timeout) as client:
            resp = client.get(f"{self.base_url}/series/{orthanc_series_id}")
            resp.raise_for_status()
            return resp.json()

    def get_instance_dicom_bytes(self, orthanc_instance_id: str) -> bytes:
        with httpx.Client(auth=self.auth, timeout=self.timeout) as client:
            resp = client.get(f"{self.base_url}/instances/{orthanc_instance_id}/file")
            resp.raise_for_status()
            return resp.content

    def get_instance_tags(self, orthanc_instance_id: str) -> Dict[str, Any]:
        with httpx.Client(auth=self.auth, timeout=self.timeout) as client:
            resp = client.get(f"{self.base_url}/instances/{orthanc_instance_id}/simplified-tags")
            resp.raise_for_status()
            return resp.json()


orthanc_service = OrthancService()
