# Orthanc DICOM Server Setup & Integration Guide

## Overview
MedVision AI uses **Orthanc** (`orthancteam/orthanc:24.8.1`) as its PACS/DICOM server inside the internal Docker network (`medvision_internal`).

## Security Hardening
- Orthanc ports `4242` (DICOM C-STORE/C-FIND) and `8042` (HTTP REST & DICOMweb) are bound **strictly to `127.0.0.1`** in `docker-compose.yml` so administrative interfaces are never exposed publicly.
- HTTP authentication is enforced (`AuthenticationEnabled: true`) using `ORTHANC_PASSWORD` from `.env`.
- The FastAPI backend connects internally via `http://orthanc:8042` using `ORTHANC_USERNAME` and `ORTHANC_PASSWORD`.

## Starting Orthanc via Docker Compose
```bash
cp .env.example .env
docker compose up -d postgres orthanc backend frontend
```

## Verifying Orthanc Reachability
```bash
# Direct localhost health check (authenticated)
curl -u orthanc_admin:change_me_orthanc_password http://127.0.0.1:8042/system

# Via FastAPI backend health endpoint
curl http://localhost:8000/api/health
```

## Sending DICOM Studies to Orthanc
### Option A: Upload via MedVision UI / `POST /api/upload`
Uploading a `.dcm` file through the MedVision UI automatically parses metadata/pixels with `pydicom` and forwards the raw DICOM instance to Orthanc (`POST /instances`), storing `orthanc_study_id` in PostgreSQL.

### Option B: Push via Standard DICOM `storescu` (DCMTK)
```bash
storescu -aec MEDVISION 127.0.0.1 4242 path/to/chest_xray.dcm
```
Then trigger backend ingestion & inference routing:
```bash
curl -X POST http://localhost:8000/api/orthanc/sync
```

## DICOM Handling Guarantees (`backend/app/services/dicom_service.py`)
- Preserves `StudyInstanceUID`, `SeriesInstanceUID`, and `SOPInstanceUID`.
- Applies `RescaleSlope` and `RescaleIntercept` before Window Center / Window Width VOI LUT scaling.
- Automatically inverts `MONOCHROME1` images so air renders dark and bone/dense tissue renders bright consistently with `MONOCHROME2`.
- Gracefully rejects corrupted DICOM files, missing `PixelData`, or unsupported modalities without crashing the pipeline.
