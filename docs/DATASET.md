# Datasets, Preprocessing & DVC Versioning Guide

Large medical imaging datasets are never committed to Git. MedVision AI uses **DVC** (`dvc.yaml`, `params.yaml`, `.dvc/config`) to version datasets, patient-level splits, checkpoints, and evaluation reports.

---

## 1. Chest X-Ray Classification — NIH ChestX-ray14

- **Source**: NIH Clinical Center ([NIH Box / Kaggle NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/data))
- **Modality**: 2D Frontal Chest Radiographs (`CR`/`DX`)
- **Labels (14)**: `Atelectasis`, `Cardiomegaly`, `Effusion`, `Infiltration`, `Mass`, `Nodule`, `Pneumonia`, `Pneumothorax`, `Consolidation`, `Edema`, `Emphysema`, `Fibrosis`, `Pleural_Thickening`, `Hernia` (`No Finding` -> all zeros).
- **Expected Directory Layout**:
  ```text
  data/raw/nih_chestxray14/
    Data_Entry_2017.csv
    images_001/images/*.png
    ...
  ```
- **Preprocessing**:
  - Resize to `224 × 224` RGB (`ImageNet` mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`).
  - **Patient-Level Split**: `GroupShuffleSplit` grouped strictly by `Patient ID` (70% Train / 15% Val / 15% Held-Out Test) with assertion verifying zero patient overlap.
  - **Training-Only Augmentation**: Random horizontal flip (`p=0.5`), random rotation (`±7°`), mild brightness/contrast jitter (`0.1`). Validation/test use deterministic transforms only.
  - **Uncertain Label Strategy**: Configurable via `--uncertain-policy` (`zeros` = conservative U-Zeros default; `ones` = high-sensitivity U-Ones).

---

## 2. Thoracic CT Segmentation — Medical Segmentation Decathlon (Task06_Lung)

- **Source**: Medical Segmentation Decathlon ([medicaldecathlon.com](http://medicaldecathlon.com/)) — `Task06_Lung.tar`
- **Modality**: 3D Thoracic CT Volumes in NIfTI (`.nii.gz`)
- **Expected Directory Layout**:
  ```text
  data/raw/msd_task06_lung/
    dataset.json
    imagesTr/*.nii.gz
    labelsTr/*.nii.gz
  ```
- **Preprocessing (MONAI)**:
  - `LoadImaged` + `EnsureChannelFirstd`
  - `Orientationd(axcodes="RAS")` (standardized Right-Anterior-Superior orientation)
  - `Spacingd(pixdim=(1.5, 1.5, 2.0))` (isotropic resampling)
  - `ScaleIntensityRanged(a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True)` (CT Hounsfield Unit windowing)
  - `RandCropByPosNegLabeld(spatial_size=(96, 96, 96))` foreground-balanced patch sampling during training.

---

## 3. Validation & DVC Commands

```bash
# 1. Verify dataset folder structure
python3 scripts/validate_datasets.py

# 2. Initialize DVC (if setting up a new repository clone)
dvc init
dvc remote add -d local_store ../medvision-dvc-storage

# 3. Track raw datasets without adding binaries to Git
dvc add data/raw/nih_chestxray14
dvc add data/raw/msd_task06_lung
git add data/raw/*.dvc data/raw/.gitignore

# 4. Reproduce training and held-out evaluation stages
dvc repro

# 5. Track and inspect metrics & model artifacts
dvc metrics show
dvc push
```
