import io
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
from PIL import Image


def build_synthetic_dicom_bytes(
    modality: str = "CR",
    body_part: str = "CHEST",
    photometric: str = "MONOCHROME2",
    rows: int = 64,
    cols: int = 64,
) -> bytes:
    """Builds a standards-compliant Part 10 DICOM binary in memory for automated testing."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset("test.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientID = "TEST-PAT-001"
    ds.PatientSex = "M"
    ds.PatientAge = "045Y"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.Modality = modality
    ds.BodyPartExamined = body_part
    ds.SeriesDescription = f"TEST {modality} {body_part}"
    ds.StudyDate = "20261119"
    ds.PhotometricInterpretation = photometric
    ds.SamplesPerPixel = 1
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.WindowCenter = 2000
    ds.WindowWidth = 4000
    ds.RescaleSlope = 1
    ds.RescaleIntercept = 0

    arr = np.linspace(0, 4095, rows * cols, dtype=np.uint16).reshape((rows, cols))
    ds.PixelData = arr.tobytes()

    buf = io.BytesIO()
    pydicom.dcmwrite(buf, ds, enforce_file_format=True)
    return buf.getvalue()


def test_health_and_registry_endpoints(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "NOT FOR CLINICAL USE" in body["disclaimer"]

    m = client.get("/api/models")
    assert m.status_code == 200
    models = m.json()["models"]
    assert len(models) == 2
    assert any(item["id"] == "cxr_densenet121_nih14" for item in models)
    assert any(item["id"] == "lung_ct_monai_unet3d" for item in models)

    met = client.get("/api/metrics")
    assert met.status_code == 200
    assert met.json()["evaluationAvailable"] is False


def test_upload_validation_rejects_invalid_extension(client):
    resp = client.post(
        "/api/upload",
        files={"file": ("malware.exe", b"MZ12345", "application/octet-stream")},
    )
    assert resp.status_code == 400


def test_dicom_upload_parsing_review_and_audit_trail(client):
    dcm_bytes = build_synthetic_dicom_bytes(modality="CR", body_part="CHEST", photometric="MONOCHROME1")
    resp = client.post(
        "/api/upload",
        files={"file": ("sample_cxr.dcm", dcm_bytes, "application/dicom")},
        data={"title": "Test PA Chest DICOM", "modality": "CR", "bodyPart": "CHEST"},
    )
    assert resp.status_code == 201
    payload = resp.json()
    study = payload["study"]
    study_id = study["id"]

    # Without mounted trained weights, backend must report weights_unavailable honestly
    assert study["processingStatus"] == "weights_unavailable"
    assert study["confidence"] is None
    assert study["dicom"]["photometricInterpretation"] == "MONOCHROME1"
    assert study["dicom"]["rows"] == 64

    # Approve review
    rev_approve = client.post(
        f"/api/studies/{study_id}/review",
        json={"decision": "approved", "reviewer": "Dr. QA Radiologist", "notes": "Verified DICOM metadata."},
    )
    assert rev_approve.status_code == 200
    assert rev_approve.json()["study"]["status"] == "approved"

    # Reject override must preserve historical approval in append-only audit log
    rev_reject = client.post(
        f"/api/studies/{study_id}/review",
        json={"decision": "rejected", "reviewer": "Dr. Second Reader", "notes": "Overriding prior sign-off."},
    )
    assert rev_reject.status_code == 200
    assert rev_reject.json()["study"]["status"] == "rejected"

    audit_resp = client.get(f"/api/audit?study_id={study_id}")
    assert audit_resp.status_code == 200
    events = [e["decision"] for e in audit_resp.json()["auditLog"]]
    assert "dicom_ingested" in events
    assert "approved" in events
    assert "review_updated" in events


def test_unsupported_modality_gated_gracefully(client):
    img = Image.fromarray(np.zeros((64, 64), dtype=np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    resp = client.post(
        "/api/upload",
        files={"file": ("brain_mri.png", buf.getvalue(), "image/png")},
        data={"modality": "MR", "bodyPart": "BRAIN"},
    )
    assert resp.status_code == 201
    assert resp.json()["study"]["processingStatus"] == "unsupported_modality"
