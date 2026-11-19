# MedVision AI — AI-Assisted Medical Imaging & Radiologist Review Platform

> **FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY. NOT FOR CLINICAL USE.**
> AI-generated outputs are experimental and must not be used to diagnose, treat, or make medical decisions.

---

## 1. Project Overview
MedVision AI is an end-to-end educational medical imaging workstation and deep-learning review platform. It connects a React/TypeScript PACS viewer to a Python FastAPI backend, Orthanc DICOM Server, PostgreSQL audit database, PyTorch DenseNet-121 Chest X-ray classification with gradient-based Grad-CAM explainability, MONAI 3D U-Net thoracic CT segmentation, and DVC data/experiment versioning.

## 2. Problem Statement
Medical imaging volume (X-ray, CT, MRI) continues to outpace radiology reading capacity, while opaque "black-box" AI predictions fail to earn clinician trust or provide traceable governance. Educational and research teams need a transparent, modular reference architecture demonstrating:
- Standards-compliant DICOM ingestion and preprocessing (`MONOCHROME1/2`, Window/Level VOI LUT, HU clipping).
- Modality-gated deep learning inference with visual explainability (Grad-CAM & probability masks).
- Mandatory human-in-the-loop radiologist approval/rejection with an append-only UTC audit trail.

## 3. Objectives
1. Ingest and parse DICOM (`.dcm`), multi-slice DICOM ZIP series, NIfTI volumes, and 2D X-ray images.
2. Separate Chest X-ray multi-label classification (**NIH ChestX-ray14**, DenseNet-121) from thoracic CT 3D segmentation (**MSD Task06_Lung**, MONAI 3D U-Net) with strict modality gating.
3. Compute genuine Grad-CAM heatmaps from final convolutional gradients (`features.norm5`) and voxel-level segmentation overlays.
4. Provide an interactive clinical viewer with Window/Level, Zoom/Pan, Slice navigation, Split comparison, and radiologist sign-off.
5. Persist studies, inference results, reviews, and immutable UTC audit logs in PostgreSQL while keeping imaging binaries on disk storage.
6. Package the entire stack with Docker Compose and DVC for reproducible execution.

## 4. Features
- **DICOM & PACS Integration**: Orthanc DICOM Server (`C-STORE` + REST/DICOMweb) + `pydicom` header/pixel extraction + lightweight TypeScript Part 10 DICOM parser in preview mode.
- **Interactive PACS Viewer**: Original, Heatmap-only, Blended Grad-CAM, Segmentation Mask, and interactive Split views; Window/Level presets and sliders; grayscale inversion; crosshair intensity/HU probe; multi-slice scrubbing.
- **Human-in-the-Loop Sign-Off**: Reviewer ID, clinical/override notes, explicit pre-submission confirmation gate, and append-only UTC audit log with CSV export.
- **Strict Honesty & Demo Mode Separation**: Demo Mode studies are explicitly labeled as illustrative examples. Uploaded studies without mounted trained weights honestly report `weights_unavailable` instead of fabricating predictions.
- **Optional Non-Diagnostic Explainer**: Uses Google Gemini (`gemini-4-hydrogen`, server-side only) strictly to explain technical ML/DICOM terminology in plain language—never as a substitute for medical imaging models or radiologists.

## 5. Architecture
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full Mermaid architecture diagram, data flow, and security boundaries.

## 6. Technology Stack
| Layer | Technologies |
| :--- | :--- |
| **Frontend** | React 19, TypeScript, Vite, Tailwind CSS v4, Lucide React |
| **Backend API** | Python 3.11, FastAPI, Pydantic v2, SQLAlchemy 2.0, PostgreSQL 16 (plus Express reverse-proxy / preview server) |
| **Machine Learning** | PyTorch, TorchVision (DenseNet-121), MONAI (3D U-Net), NumPy, OpenCV, Scikit-learn |
| **Medical Imaging** | Orthanc DICOM Server, `pydicom`, SimpleITK, NiBabel |
| **Explainability** | Real PyTorch Grad-CAM hooks (`features.norm5`), 3D U-Net probability & contour overlays |
| **MLOps & Infra** | DVC, Git, Docker, Docker Compose |

## 7. Dataset Selection
- **Classification**: **NIH ChestX-ray14** (14 thoracic pathology labels, frontal PA/AP radiographs, strict patient-level splits).
- **Segmentation**: **Medical Segmentation Decathlon — Task06_Lung** (3D thoracic CT NIfTI volumes with lung tumor masks).
- **Planned Modular Extension**: BraTS Brain MRI segmentation (explicitly gated as unsupported in active models until trained weights are registered).

## 8. Dataset Download Instructions
Large medical datasets are intentionally excluded from Git and managed via DVC. Follow the exact download and directory verification instructions in [docs/DATASET.md](docs/DATASET.md), then validate locally:
```bash
python3 scripts/validate_datasets.py
```

## 9. Installation Instructions
### Frontend / Node Preview Server
```bash
npm install
npm run dev
```
### Python FastAPI Backend (Local Virtual Environment)
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

## 10. Environment Configuration
Copy `.env.example` to `.env` (root) and `backend/.env.example` to `backend/.env`. Never commit `.env` files or secrets to Git. Key variables:
- `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `DATABASE_URL`
- `ORTHANC_URL`, `ORTHANC_USERNAME`, `ORTHANC_PASSWORD`
- `CXR_WEIGHTS_PATH`, `SEG_WEIGHTS_PATH`
- `CORS_ORIGINS`, `REQUIRE_AUTH`, `API_AUTH_TOKEN`
- `GEMINI_API_KEY` (optional, strictly for non-diagnostic technical explanation)

## 11. Docker Setup
Start PostgreSQL, Orthanc DICOM Server, FastAPI Backend, and React Frontend in one command:
```bash
cp .env.example .env
docker compose up --build
```
- Workstation UI: `http://localhost:3000`
- FastAPI OpenAPI Swagger UI: `http://localhost:8000/docs`
- Orthanc REST / DICOMweb (localhost-only): `http://127.0.0.1:8042`

## 12. Orthanc Configuration
Orthanc is configured in `docker-compose.yml` and `orthanc/orthanc.json` with authentication enabled and ports bound strictly to `127.0.0.1` (`4242` DICOM C-STORE, `8042` HTTP/DICOMweb). See [docs/ORTHANC_SETUP.md](docs/ORTHANC_SETUP.md) for sending DICOM studies via `storescu` and syncing via `POST /api/orthanc/sync`.

## 13. Model Training
Reproduce end-to-end training with DVC (`dvc repro`) or run pipelines directly:
```bash
# Chest X-ray DenseNet-121 (NIH ChestX-ray14, patient-level split, BCEWithLogitsLoss)
PYTHONPATH=backend python3 -m app.ml.classification.train \
  --data-dir data/raw/nih_chestxray14 \
  --output models/densenet121_nih14.pth \
  --epochs 20 --batch-size 32 --uncertain-policy zeros

# Thoracic CT 3D U-Net (MONAI, MSD Task06_Lung, DiceCELoss)
PYTHONPATH=backend python3 -m app.ml.segmentation.train \
  --data-dir data/raw/msd_task06_lung \
  --output models/monai_unet3d_task06_lung.pth \
  --epochs 100 --batch-size 2
```

## 14. Model Evaluation
Held-out test splits are never used during training or model selection:
```bash
PYTHONPATH=backend python3 -m app.ml.classification.evaluate \
  --data-dir data/raw/nih_chestxray14 \
  --checkpoint models/densenet121_nih14.pth \
  --report reports/classification_metrics.json

PYTHONPATH=backend python3 -m app.ml.segmentation.evaluate \
  --checkpoint models/monai_unet3d_task06_lung.pth \
  --report reports/segmentation_metrics.json
```

## 15. Inference Workflow
1. User uploads `.dcm`, `.zip`, `.nii.gz`, or `.png`/`.jpg` via UI or sends DICOM to Orthanc (`POST /api/orthanc/sync`).
2. Backend validates file type/size, extracts DICOM headers (`MONOCHROME1/2`, Window/Level), stores binaries on disk, and writes study metadata to PostgreSQL.
3. Inference router verifies modality/anatomy compatibility (`CR/DX Chest` -> DenseNet-121 + Grad-CAM; `CT Thorax/Lung` -> MONAI 3D U-Net; unsupported modalities -> gated gracefully).
4. Radiologist inspects original scan, heatmap/mask overlays, and confidence scores, then confirms Approve/Reject, writing an immutable UTC audit record.

## 16. API Documentation
Complete endpoint specifications, schemas, and status codes are documented in [docs/API.md](docs/API.md) and live at `http://localhost:8000/docs`.

## 17. Testing Instructions
```bash
# Frontend TypeScript & Vite production build verification
npm run lint
npm run build

# Node.js DICOM binary parser & live API integration tests
node --test tests/api_and_dicom.test.mjs

# Python FastAPI, pydicom, PyTorch DenseNet-121, Grad-CAM & MONAI 3D U-Net tests
PYTHONPATH=backend pytest backend/app/tests -v
```

## 18. Screenshots Section
Capture and attach screenshots from the running workstation (`http://localhost:3000`) covering:
1. Study Worklist & Demo Mode Separation (`Workstation` tab)
2. Grad-CAM Blended & Split Comparison Viewport (`Workstation` tab)
3. Radiologist Sign-Off Confirmation Gate & Immutable Audit Log (`Audit Log` tab)
4. Model Registry, Modality Matrix & DVC Evaluation Status (`Models & Docs` tab)

## 19. Limitations and Ethical Considerations
Detailed analysis of dataset label noise, shortcut learning, demographic/scanner domain shift, Grad-CAM non-causation, and regulatory requirements is documented in [docs/BIAS_AND_LIMITATIONS.md](docs/BIAS_AND_LIMITATIONS.md) and [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

## 20. Future Enhancements
- Register modular **BraTS Brain MRI** 3D U-Net segmentation weights.
- Add temperature scaling / Platt calibration for multi-label probabilities.
- Integrate DICOM Structured Reporting (DICOM-SR) export back to Orthanc.
- Implement OIDC/SSO identity integration with institutional RBAC.
