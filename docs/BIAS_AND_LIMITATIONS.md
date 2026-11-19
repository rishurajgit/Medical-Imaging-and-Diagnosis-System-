# Bias, Limitations & Responsible AI Considerations

> **FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY. NOT FOR CLINICAL USE.**

---

## 1. Dataset Limitations & Label Noise
- **Automated NLP Label Noise**: NIH ChestX-ray14 labels were mined from free-text radiology reports via rule-based NLP rather than prospective multi-radiologist consensus. Literature audits show ~8–12% label inaccuracy, particularly for subtle infiltrates, cardiomegaly, and pneumothorax with chest tubes already placed.
- **Small Cohort Size (MSD Task06_Lung)**: Task06_Lung contains 63 labeled training CT volumes, which limits rare tumor morphology coverage and increases overfitting risk without strong 3D augmentation and external validation.

## 2. Population Bias & Under-Representation
- Public cohorts reflect tertiary US/European academic hospital demographics and under-represent pediatric patients, elderly multi-morbid patients, and low-resource global populations.
- Subgroup performance disparities (age, sex, body habitus, comorbidities) can remain hidden behind aggregate Macro AUROC or mean Dice scores.

## 3. Domain Shift & Scanner Generalization
- Models trained on specific X-ray detector manufacturers, kVp/mAs exposure settings, or CT reconstruction kernels frequently degrade when evaluated on external hospitals, portable ICU AP radiographs, or low-dose screening CT protocols.
- Shortcut learning is well-documented in chest radiography: CNNs may exploit chest drains, ECG leads, hospital text markers, or portable view borders rather than genuine parenchymal findings.

## 4. False Positives & False Negatives
- **False Positives** cause alarm fatigue, unnecessary follow-up CT radiation, and biopsy risk (e.g., expiratory effort mimicking cardiomegaly).
- **False Negatives** create dangerous false reassurance if clinicians over-rely on AI ("automation bias"). MedVision AI mandates human-in-the-loop review and logs overrides in an immutable audit table.

## 5. Limitations of Grad-CAM
- DenseNet-121's final feature map (`features.norm5`) is spatially coarse (`7 × 7` for `224 × 224` input) and upsampled smoothly to original scan dimensions.
- **Grad-CAM indicates correlation/gradient sensitivity, NOT clinical causation.** Highlighting a region does not prove the model understood the underlying pathology, and diffuse pathologies (e.g. edema or interstitial fibrosis) often yield poor localization.

## 6. Limitations of 3D Segmentation Masks
- Patch-based sliding-window 3D U-Net inference can lose long-range anatomical context across lung boundaries and produce false-positive islands on high-attenuation vascular bifurcations or pleural scarring.
- Voxel volume estimates depend heavily on accurate DICOM `PixelSpacing` and `SliceThickness` metadata.

## 7. Need for Independent Clinical Validation
Before any medical imaging AI software can assist patient care, it requires prospective multi-center external validation, subgroup fairness audits, probability calibration, cybersecurity hardening (HIPAA/GDPR), QMS traceability, and regulatory clearance (FDA 510(k)/De Novo, EU MDR CE Class IIa/IIb).
