# Model Card — MedVision AI

> **Intended Use Notice**: FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY. NOT FOR CLINICAL USE. Not cleared or approved by the FDA, EMA, or any regulatory body.

---

## Model 1: Chest X-Ray Multi-Label Classifier (`cxr_densenet121_nih14`)

| Attribute | Specification |
| :--- | :--- |
| **Version** | `1.0.0-nih14` |
| **Architecture** | TorchVision `DenseNet-121` with ImageNet-1K transfer learning + 14-unit linear classification head |
| **Loss Function** | `BCEWithLogitsLoss` (multi-label independent sigmoid probabilities) |
| **Supported Modalities** | `CR`, `DX` (Frontal PA/AP Chest Radiographs only) |
| **Supported Anatomy** | `CHEST`, `THORAX` |
| **Input Format** | 2D Grayscale DICOM or PNG/JPEG resized to `224×224`, 3-channel ImageNet-normalized |
| **Training Dataset** | NIH ChestX-ray14 (Patient-level 70/15/15 split) |
| **Explainability** | Real Grad-CAM computed from `backbone.features.norm5` forward activations & backward gradients |
| **Evaluation Metrics** | Per-class AUROC, Macro AUROC, Macro Precision, Macro Recall, Macro F1, per-class Confusion Matrices |
| **Local Evaluation Status** | Not yet executed in sandbox (datasets & GPU training not provisioned; metrics reported as `null` until `evaluate.py` runs) |

---

## Model 2: Thoracic CT Lung Lesion Segmenter (`lung_ct_monai_unet3d`)

| Attribute | Specification |
| :--- | :--- |
| **Version** | `1.0.0-msd-task06` |
| **Architecture** | MONAI 3D `UNet` (`channels=(16,32,64,128,256)`, `strides=(2,2,2,2)`, `num_res_units=2`, BatchNorm) |
| **Loss Function** | `DiceCELoss(to_onehot_y=True, softmax=True)` |
| **Supported Modalities** | `CT` (Volumetric Thoracic CT only) |
| **Supported Anatomy** | `LUNG`, `CHEST`, `THORAX` |
| **Input Format** | 3D NIfTI (`.nii`/`.nii.gz`) or multi-slice DICOM CT series, RAS orientation, HU clipped `[-1000, 400]` |
| **Training Dataset** | Medical Segmentation Decathlon — `Task06_Lung` |
| **Explainability** | Voxel-level foreground probability map + binary contour mask overlay |
| **Evaluation Metrics** | Mean Dice Coefficient, Mean IoU (Jaccard), Sensitivity, Specificity |
| **Local Evaluation Status** | Not yet executed in sandbox (metrics reported as `null` until `evaluate.py` runs) |

---

## Out-of-Scope Uses
- Any clinical diagnosis, triage, patient treatment, or autonomous screening decision.
- Running the CXR classifier on lateral X-rays, pediatric radiographs, CT slices, or MRI scans.
- Running the Lung CT U-Net on non-thoracic CT or Brain MRI (BraTS MRI is explicitly gated until a dedicated BraTS checkpoint is registered).
