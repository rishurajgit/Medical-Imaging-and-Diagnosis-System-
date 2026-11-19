# Demo Guide — MedVision AI Workstation

## Purpose of Demo Mode
Because full public medical imaging datasets (**NIH ChestX-ray14** ~45 GB, **MSD Task06_Lung** ~8.5 GB) and GPU-trained PyTorch/MONAI checkpoints are not bundled inside Git repositories, MedVision AI includes a strictly isolated **Demo Mode** so reviewers can immediately explore the full PACS workflow without misrepresenting illustrative examples as real inference.

---

## Walkthrough Checklist

### 1. Inspect the Educational Disclaimer & Demo Separation
- The top amber notice explicitly states: *"FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY. NOT FOR CLINICAL USE."*
- In the left **Study Worklist**, use the **Show Demo Mode Studies (ON/OFF)** switch to toggle between illustrative Demo Mode samples and newly uploaded studies.
- Every Demo Mode study is tagged `DEMO-*` and displays an amber *"Illustrative Demo Mode"* callout in the right diagnosis panel.

### 2. Explore the Interactive PACS Viewport
- Select `DEMO-CXR-0842` (Chest X-Ray) or `DEMO-CT-0021` (Thoracic CT).
- Switch between **Original**, **Heatmap**, **Blended**, **Mask**, and **Split** modes.
- Drag the **Original | AI** divider in Split mode, adjust **Window/Level** presets and sliders, toggle **Invert** (`MONOCHROME1/2`), scroll to zoom, drag to pan, and hover over the scan to inspect live crosshair pixel coordinates and intensity/HU estimates.
- For `DEMO-CT-0021`, scrub the **Slice** slider at the bottom of the viewport.
- Select `DEMO-MR-0047` (Brain MRI) to see **modality safety gating** in action: the system explicitly refuses to run Chest X-ray or Lung CT models on Brain MRI.

### 3. Upload a Real Scan (`.dcm`, `.zip`, or `.png`/`.jpg`)
- Click **Upload Scan** in the top bar.
- Drop a `.dcm` DICOM file or X-ray image.
- Observe that DICOM metadata is parsed and displayed honestly; when trained PyTorch/MONAI checkpoints are absent, the study status is recorded as `weights_unavailable` (no fabricated predictions!).

### 4. Test Optional Non-Diagnostic Technical Explainer
- In the right panel, click **Explain Technical Output (Non-Diagnostic)** to invoke Gemini 4 Hydrogen for plain-language explanations of the technical parameters (strictly non-diagnostic).

### 5. Complete a Radiologist Review & Verify Immutable Audit Trail
- Enter a reviewing clinician name and notes in the right panel, click **Approve** or **Reject**, and confirm in the pre-submission confirmation gate.
- Switch to the **Audit Log** tab to verify the new UTC-timestamped event, test filtering by `Uploaded Only` vs `Demo Only`, and export the audit log as CSV.

### 6. Review Live Model Registry & Honest Metrics
- Open **Models & Docs** to view live `GET /api/models` and `GET /api/metrics` output showing exact supported modalities, checkpoint availability, and known bias/limitations.
