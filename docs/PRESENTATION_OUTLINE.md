# 10-Minute Final Presentation Outline — MedVision AI

Target duration: **10 minutes** (~10 slides + 2-minute live demo)

---

### Slide 1: Title & Problem Statement (0:00 – 1:00)
- **Title**: MedVision AI — AI-Assisted Medical Imaging & Radiologist Review Platform *(Educational / Research Prototype)*
- **Problem**: Rising X-ray/CT imaging volumes create radiology reading bottlenecks, while opaque black-box ML tools lack DICOM integration, explainability, and auditable clinician oversight.

### Slide 2: Proposed Solution & Core Design Principles (1:00 – 1:45)
- End-to-end human-in-the-loop PACS platform: Upload/Orthanc -> Preprocessing -> Modality-Gated AI -> Explainability Overlays -> Radiologist Approve/Reject -> Immutable UTC Audit Trail.
- Strict engineering honesty: zero fabricated predictions, clear Demo Mode separation, and non-clinical disclaimer.

### Slide 3: System Architecture (1:45 – 2:45)
- Walk through the Mermaid diagram (`docs/ARCHITECTURE.md`):
  - React/TypeScript PACS Frontend
  - FastAPI Backend + SQLAlchemy + PostgreSQL (metadata & audit logs)
  - Orthanc DICOM Server (localhost-restricted PACS)
  - Disk Storage Volumes (raw scans, PNG slices, Grad-CAM & segmentation overlays)
  - Docker Compose single-command orchestration.

### Slide 4: Datasets, Preprocessing & DVC Versioning (2:45 – 3:45)
- **NIH ChestX-ray14** (2D CXR classification): `224×224`, ImageNet normalization, strict **patient-level** `GroupShuffleSplit` (zero patient leakage), train-only augmentations, explicit uncertain-label policy (`U-Zeros` / `U-Ones`).
- **MSD Task06_Lung** (3D CT segmentation): NIfTI loading, RAS orientation, isotropic resampling, HU `[-1000, 400]` clipping, foreground-balanced `96³` patch sampling.
- **DVC** (`dvc.yaml`, `params.yaml`) tracks datasets, checkpoints, and JSON metric reports outside Git.

### Slide 5: Machine Learning Models & Modality Gating (3:45 – 4:45)
- **Classifier**: PyTorch/TorchVision `DenseNet-121` transfer learning with `BCEWithLogitsLoss` (14 thoracic labels).
- **Segmenter**: MONAI 3D `UNet` with `DiceCELoss` and sliding-window volumetric inference.
- **Safety Router**: Explicitly pairs each model to its supported modality/anatomy (`CR/DX Chest`, `CT Thorax/Lung`) and gates unsupported scans (e.g., Brain MRI) gracefully.

### Slide 6: Explainability — Grad-CAM & Segmentation Masks (4:45 – 5:45)
- Real PyTorch forward/backward gradient hooks on `DenseNet-121` `features.norm5` -> ReLU weighted CAM -> min-max normalization -> resized RGBA heatmap overlay.
- Volumetric 3D U-Net voxel probability maps + contour masks.
- Crucial caveat: Grad-CAM reveals model sensitivity, **not** clinical causation.

### Slide 7: DICOM & Orthanc PACS Integration (5:45 – 6:30)
- Standards-compliant DICOM handling: `StudyInstanceUID`/`SeriesInstanceUID` preservation, `RescaleSlope/Intercept`, Window Center/Width VOI LUT, and `MONOCHROME1` vs `MONOCHROME2` inversion.
- Automated `POST /api/orthanc/sync` pulls newly arrived PACS studies into the inference workflow.

### Slide 8: Live Demo Walkthrough (6:30 – 8:15)
- Demonstrate Worklist & Demo Mode toggle -> Canvas Viewport (Original/Heatmap/Blended/Mask/Split, Window/Level, Zoom/Pan, Slice scrubbing) -> Upload real `.dcm` -> Radiologist confirmation gate -> Immutable Audit Log & CSV export.

### Slide 9: Testing, Results & Engineering Challenges (8:15 – 9:15)
- Automated verification: TypeScript/Vite build, Python syntax checks, Node DICOM binary & API integration tests, and full pytest suite (`backend/app/tests/`).
- Honest reporting: local training metrics remain `null` until full NIH14/MSD datasets are trained on GPU; key challenges solved include `MONOCHROME1` handling, Zip-Slip/path traversal hardening, and avoiding synthetic predictions.

### Slide 10: Limitations, Bias & Future Scope (9:15 – 10:00)
- **Limitations**: Report-mined label noise, shortcut learning, scanner/demographic domain shift, regulatory clearance needed before clinical use.
- **Next Steps**: Register BraTS Brain MRI 3D U-Net weights, add probability calibration (temperature scaling), export DICOM Structured Reports (DICOM-SR) to Orthanc, and run multi-center external validation.
