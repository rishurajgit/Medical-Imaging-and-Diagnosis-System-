# System Architecture — MedVision AI

## High-Level System Diagram

```mermaid
flowchart TB
    subgraph Client["Clinician / Researcher Browser"]
        UI["React + TypeScript + Vite PACS Workstation\n(Worklist, Canvas Viewer, Grad-CAM/Mask Overlays, Audit Log)"]
    end

    subgraph Edge["Frontend Proxy & Preview Layer (Port 3000)"]
        NODE["Express Server (server.ts)\n- Serves Built SPA\n- Proxies /api/* to FastAPI when configured\n- Optional Non-Diagnostic Gemini Explainer"]
    end

    subgraph Backend["FastAPI Backend Service (Port 8000, Internal Network)"]
        API["REST API Layer\n/upload, /studies, /analyze, /review, /audit, /models, /metrics"]
        DICOM["DICOM & Preprocessing Services\npydicom, MONOCHROME1/2, VOI LUT, HU Clipping"]
        ROUTER["Modality Safety Router\nCR/DX Chest -> Classifier | CT Lung -> 3D U-Net | Other -> Gated"]
        CLF["PyTorch DenseNet-121\n+ Real Grad-CAM (features.norm5)"]
        SEG["MONAI 3D U-Net\nSliding-Window Lung Segmentation"]
        AUDIT["Append-Only UTC Audit Service"]
    end

    subgraph DataInfra["Persistent & PACS Infrastructure (Internal Docker Network)"]
        ORTHANC["Orthanc DICOM Server\nC-STORE :4242 | REST/DICOMweb :8042\n(Bound strictly to 127.0.0.1)"]
        PG[("PostgreSQL 16\nImagingStudy, DicomSeries, InferenceResult,\nModelVersion, RadiologistReview, AuditLog")]
        VOL[("Disk Storage Volumes\n/storage/uploads & /storage/artifacts\n/models (Checkpoints) & /reports")]
    end

    subgraph MLOps["Offline Training & Experiment Tracking"]
        DVC["DVC Pipeline (dvc.yaml + params.yaml)\nNIH ChestX-ray14 & MSD Task06_Lung"]
    end

    UI <--> NODE
    NODE <--> API
    API --> DICOM
    DICOM <--> ORTHANC
    DICOM --> ROUTER
    ROUTER --> CLF
    ROUTER --> SEG
    CLF --> VOL
    SEG --> VOL
    API --> PG
    API --> AUDIT
    AUDIT --> PG
    DVC -.->|"Versioned Checkpoints & Metrics JSON"| VOL
```

## Architectural Principles

1. **Separation of Structured Metadata and Binary Imaging Artifacts**:
   - PostgreSQL stores relational metadata (`ImagingStudy`, `DicomSeries`, `InferenceResult`, `ModelVersion`, `RadiologistReview`, `AuditLog`).
   - Raw DICOM files, PNG slice previews, Grad-CAM heatmaps, and segmentation masks are stored on filesystem volumes (`STORAGE_DIR`) referenced by path—never as large database BLOBs.
2. **Strict Modality Gating**:
   - Models explicitly declare supported modalities (`CR/DX` vs `CT`), anatomies (`CHEST/THORAX/LUNG`), input formats, and source datasets.
   - Unsupported modalities (e.g., Brain MRI prior to BraTS registration) are gracefully gated as `unsupported_modality` rather than silently pushed through an incompatible network.
3. **Zero Synthetic Predictions**:
   - When trained checkpoints (`models/densenet121_nih14.pth` or `models/monai_unet3d_task06_lung.pth`) are absent, inference raises `ModelWeightsUnavailableError` and records `weights_unavailable`.
   - Demo Mode studies are strictly isolated via `isDemo: true` and explicitly labeled as illustrative examples.
4. **Immutable Audit Trail**:
   - Every ingestion, DICOM parse, inference attempt, Grad-CAM export, approval, rejection, and review-status change appends a new UTC-timestamped row to `audit_logs`. Historical records are never overwritten.
5. **Security Boundaries**:
   - Orthanc HTTP/DICOM interfaces are bound to `127.0.0.1` and authenticated via environment credentials.
   - Upload filenames are sanitized and path-resolved strictly inside `STORAGE_DIR` to eliminate path traversal and Zip-Slip attacks.
