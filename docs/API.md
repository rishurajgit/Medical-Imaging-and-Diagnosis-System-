# REST API Documentation — MedVision AI

Base URL:
- FastAPI Backend (Docker / Local): `http://localhost:8000`
- Interactive OpenAPI Swagger UI: `http://localhost:8000/docs`
- Frontend Proxy / Educational Preview: `http://localhost:3000`

Optional Auth Headers (enforced when `REQUIRE_AUTH=true`):
- `Authorization: Bearer <API_AUTH_TOKEN>`
- `X-Reviewer-Role: radiologist`

---

## Endpoints

### `GET /api/health`
Returns service health, Orthanc reachability, loaded model checkpoint count, and mandatory non-clinical disclaimer.

### `GET /api/models`
Lists all registered ML models with explicit supported modalities, anatomies, input formats, datasets, explainability methods, weights path, live `weightsLoaded` status, and documented limitations.

### `GET /api/metrics`
Reads DVC-tracked local evaluation reports (`reports/classification_metrics.json`, `reports/segmentation_metrics.json`). Returns `null` metric fields when no local evaluation has run—never fabricates accuracy numbers.

### `GET /api/studies`
Returns `{ "studies": [...], "auditLog": [...] }` ordered by newest UTC timestamp first.

### `GET /api/studies/{study_id}`
Returns full metadata, DICOM header tags, inference summary, and latest review status for a single study (`404` if not found).

### `POST /api/upload`
Ingests `.dcm`, `.dicom`, `.zip` (DICOM series), `.nii`/`.nii.gz`, or `.png`/`.jpg`/`.webp` scans.
- Accepts `multipart/form-data` (`file`, `title`, `modality`, `bodyPart`, `patientId`) or `application/json` (`fileBase64`, `fileName`, ...).
- Validates extension and file size (`<= MAX_UPLOAD_MB`), parses DICOM headers/pixels (`MONOCHROME1/2`, Window/Level), forwards `.dcm` to Orthanc when reachable, appends audit events, and routes to the appropriate inference pipeline.
- Returns `201 Created` with serialized study and inference outcome (`400` on invalid DICOM/extension, `413` on oversized file).

### `POST /api/orthanc/sync`
Polls Orthanc `GET /studies`, downloads newly arrived DICOM instances not yet in PostgreSQL, extracts metadata/pixels, runs supported inference, and stores results (`503` if Orthanc is unreachable).

### `POST /api/analyze`
Body: `{ "studyId": "<id>" }`.
Triggers inference routing for a persisted study. Returns `completed`, `weights_unavailable`, or `unsupported_modality`.

### `GET /api/studies/{study_id}/results`
Returns structured prediction output, class probabilities, processing timestamp, Grad-CAM summary, and segmentation metadata.

### `GET /api/studies/{study_id}/heatmap`
Returns the generated transparent RGBA Grad-CAM PNG image (`image/png`) from disk (`404` if not generated).

### `GET /api/studies/{study_id}/segmentation`
Returns the generated transparent RGBA MONAI 3D U-Net segmentation mask PNG (`image/png`) (`404` if not generated).

### `POST /api/studies/{study_id}/review`
Body:
```json
{
  "decision": "approved",
  "reviewer": "Dr. A. Mehta, MD",
  "notes": "Verified consolidation pattern on PA chest radiograph."
}
```
Updates study review status and appends an immutable UTC entry (`approved`, `rejected`, or `review_updated`) to `audit_logs` without overwriting prior history.

### `GET /api/audit`
Query parameters: `study_id` (optional), `event_type` (optional), `limit` (default `200`, max `1000`).
Returns append-only UTC audit records.

### `POST /api/explain-output` *(Optional Non-Diagnostic Educational Helper)*
Body: `{ "studyId": "<id>" }`.
Uses Gemini 4 Hydrogen strictly on structured technical metadata (model architecture, probability definitions, DICOM windowing) to produce plain-language educational notes. Never analyzes scan pixels to diagnose patients.
