# Technical Project Report — MedVision AI

## 1. Executive Summary
MedVision AI was engineered from an initial frontend PACS concept into a full-stack, multi-service medical imaging research platform combining React/TypeScript, Python FastAPI, SQLAlchemy + PostgreSQL, Orthanc DICOM Server, PyTorch DenseNet-121 classification with gradient-based Grad-CAM, MONAI 3D U-Net volumetric lung CT segmentation, DVC experiment pipelines, and Docker Compose orchestration.

## 2. Architecture Audit & Upgrades Performed
- **Replaced LLM-as-Diagnostic-Substitute**: Removed raw-image diagnostic calls to Gemini; restricted Gemini exclusively to an optional, clearly labeled non-diagnostic technical explainer (`POST /api/explain-output`).
- **Real ML Pipelines Built**:
  - `backend/app/ml/classification/`: TorchVision `DenseNet-121` for 14-label NIH ChestX-ray14 with patient-level `GroupShuffleSplit`, train-only augmentation, `BCEWithLogitsLoss`, and `evaluate.py` (AUROC, Precision, Recall, F1, confusion matrices).
  - `backend/app/ml/segmentation/`: MONAI 3D `UNet` for MSD `Task06_Lung` with RAS orientation, isotropic resampling, CT HU `[-1000, 400]` intensity scaling, foreground-balanced patch cropping, `DiceCELoss`, sliding-window inference, and `evaluate.py` (Dice, IoU, Sensitivity, Specificity).
  - `backend/app/services/gradcam_service.py`: Genuine forward/backward hooks on `backbone.features.norm5` producing normalized, resized RGBA heatmap overlays.
- **DICOM & Orthanc Integration**: Implemented `pydicom` parsing (`MONOCHROME1/2`, Window/Level VOI LUT, RescaleSlope/Intercept, ZIP series sorting), `OrthancService` REST integration + `POST /api/orthanc/sync`, and a zero-dependency TypeScript Part 10 DICOM parser for the standalone Node preview server.
- **Relational Persistence & Immutable Audit Log**: Defined 6 SQLAlchemy ORM tables (`ImagingStudy`, `DicomSeries`, `InferenceResult`, `ModelVersion`, `RadiologistReview`, `AuditLog`) storing binary scan/overlay files separately on disk volumes.
- **Honest Demo Mode & Weight Gating**: Separated `isDemo: true` illustrative studies from real uploaded studies and ensured missing trained weights raise `ModelWeightsUnavailableError` / display `weights_unavailable` rather than synthetic predictions.

## 3. Verification & Environment Constraints
- **Executed Successfully in Sandbox**:
  - TypeScript typecheck (`npm run lint`) and Vite production build (`npm run build`).
  - Python 3.11 syntax compilation (`python3 -m py_compile`) across all `backend/app/**/*.py` modules.
  - Automated Node.js unit & live API integration tests (`node --test tests/api_and_dicom.test.mjs`) verifying Part 10 DICOM binary parsing, `MONOCHROME1` inversion, PNG encoding, upload validation, honest missing-weights handling, and append-only audit persistence.
- **Blocked in Sandbox (Ready for Local / GPU Workstation)**:
  - Running `pytest backend/app/tests` and `docker compose up --build` requires Python pip packages (`torch`, `monai`, `fastapi`, `pydicom`, `sqlalchemy`) and Docker Engine, which are not installed in the browser preview sandbox.
  - Training checkpoints and generating non-null evaluation metrics require downloading NIH ChestX-ray14 and MSD Task06_Lung (see `docs/DATASET.md`).
