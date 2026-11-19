import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pydicom
from PIL import Image
from pydicom.errors import InvalidDicomError
from pydicom.multival import MultiValue


class DicomValidationError(ValueError):
    """Raised when an uploaded file is corrupt, incomplete, or not a supported DICOM image."""


@dataclass
class ParsedDicomSlice:
    study_instance_uid: str
    series_instance_uid: str
    sop_instance_uid: str
    patient_id: str
    patient_sex: str
    patient_age: str
    modality: str
    body_part: str
    series_description: str
    study_date: str
    institution_name: str
    photometric_interpretation: str
    window_center: float
    window_width: float
    slice_thickness_mm: str
    pixel_spacing_mm: str
    rows: int
    columns: int
    bits_allocated: int
    instance_number: int
    slice_location: float
    hu_array: np.ndarray
    display_uint8: np.ndarray


@dataclass
class ParsedDicomSeries:
    study_instance_uid: str
    series_instance_uid: str
    modality: str
    body_part: str
    slices: List[ParsedDicomSlice] = field(default_factory=list)


def _first_float(val, default: float) -> float:
    if val is None:
        return default
    if isinstance(val, MultiValue):
        return float(val[0])
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _format_dicom_date(raw: Optional[str]) -> str:
    if raw and len(str(raw)) == 8 and str(raw).isdigit():
        s = str(raw)
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return "Unknown"


def apply_window_and_photometric(
    pixel_array: np.ndarray,
    slope: float = 1.0,
    intercept: float = 0.0,
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
    photometric_interpretation: str = "MONOCHROME2",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts raw DICOM pixel data to physical units (e.g. Hounsfield Units for CT),
    applies Window Center / Window Width VOI LUT scaling, and corrects MONOCHROME1 inversion.
    Returns (rescaled_float32_array, display_uint8_array).
    """
    rescaled = pixel_array.astype(np.float32) * float(slope) + float(intercept)

    if window_center is not None and window_width is not None and window_width > 0:
        low = window_center - window_width / 2.0
        high = window_center + window_width / 2.0
    else:
        low = float(np.percentile(rescaled, 0.5))
        high = float(np.percentile(rescaled, 99.5))
        if high <= low:
            low, high = float(rescaled.min()), float(rescaled.max())

    span = max(high - low, 1e-5)
    norm = np.clip((rescaled - low) / span, 0.0, 1.0)

    if photometric_interpretation.upper() == "MONOCHROME1":
        norm = 1.0 - norm

    display_uint8 = (norm * 255.0).round().astype(np.uint8)
    return rescaled, display_uint8


def parse_dicom_bytes(raw_bytes: bytes) -> ParsedDicomSlice:
    try:
        ds = pydicom.dcmread(io.BytesIO(raw_bytes), force=False)
    except (InvalidDicomError, Exception) as exc:
        raise DicomValidationError(f"Invalid or corrupted DICOM file: {exc}") from exc

    if "PixelData" not in ds:
        raise DicomValidationError("DICOM dataset does not contain PixelData.")

    try:
        raw_pixels = ds.pixel_array
    except Exception as exc:
        raise DicomValidationError(f"Unable to decode DICOM PixelData: {exc}") from exc

    if raw_pixels.ndim == 3 and raw_pixels.shape[-1] in (3, 4):
        # Convert RGB DICOM secondary capture to luminance grayscale
        raw_pixels = np.mean(raw_pixels[..., :3], axis=-1)
    elif raw_pixels.ndim == 3:
        # Multi-frame DICOM: take middle frame for 2D slice representation
        raw_pixels = raw_pixels[raw_pixels.shape[0] // 2]

    photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).upper()
    slope = _first_float(getattr(ds, "RescaleSlope", None), 1.0)
    intercept = _first_float(getattr(ds, "RescaleIntercept", None), 0.0)
    wc_raw = getattr(ds, "WindowCenter", None)
    ww_raw = getattr(ds, "WindowWidth", None)
    wc = _first_float(wc_raw, 128.0) if wc_raw is not None else None
    ww = _first_float(ww_raw, 256.0) if ww_raw is not None else None

    hu_array, display_uint8 = apply_window_and_photometric(
        raw_pixels,
        slope=slope,
        intercept=intercept,
        window_center=wc,
        window_width=ww,
        photometric_interpretation=photometric,
    )

    pixel_spacing = getattr(ds, "PixelSpacing", None)
    if isinstance(pixel_spacing, MultiValue) and len(pixel_spacing) >= 2:
        spacing_str = f"{float(pixel_spacing[0]):.3f} x {float(pixel_spacing[1]):.3f}"
    else:
        spacing_str = "N/A"

    ipp = getattr(ds, "ImagePositionPatient", None)
    slice_loc = (
        float(ipp[2])
        if isinstance(ipp, MultiValue) and len(ipp) >= 3
        else _first_float(getattr(ds, "SliceLocation", None), 0.0)
    )

    return ParsedDicomSlice(
        study_instance_uid=str(getattr(ds, "StudyInstanceUID", "")),
        series_instance_uid=str(getattr(ds, "SeriesInstanceUID", "")),
        sop_instance_uid=str(getattr(ds, "SOPInstanceUID", "")),
        patient_id=str(getattr(ds, "PatientID", "ANONYMIZED")),
        patient_sex=str(getattr(ds, "PatientSex", "U")),
        patient_age=str(getattr(ds, "PatientAge", "000Y")),
        modality=str(getattr(ds, "Modality", "OTHER")).upper(),
        body_part=str(getattr(ds, "BodyPartExamined", "UNKNOWN")).upper(),
        series_description=str(getattr(ds, "SeriesDescription", "DICOM SERIES")),
        study_date=_format_dicom_date(getattr(ds, "StudyDate", None)),
        institution_name=str(getattr(ds, "InstitutionName", "MedVision Educational PACS")),
        photometric_interpretation=photometric,
        window_center=wc if wc is not None else 128.0,
        window_width=ww if ww is not None else 256.0,
        slice_thickness_mm=str(getattr(ds, "SliceThickness", "N/A")),
        pixel_spacing_mm=spacing_str,
        rows=int(getattr(ds, "Rows", display_uint8.shape[0])),
        columns=int(getattr(ds, "Columns", display_uint8.shape[1])),
        bits_allocated=int(getattr(ds, "BitsAllocated", 16)),
        instance_number=int(getattr(ds, "InstanceNumber", 0) or 0),
        slice_location=slice_loc,
        hu_array=hu_array,
        display_uint8=display_uint8,
    )


def parse_dicom_zip(zip_bytes: bytes, max_slices: int = 512) -> ParsedDicomSeries:
    """Extracts and sorts a multi-slice DICOM series from a ZIP archive."""
    parsed_slices: List[ParsedDicomSlice] = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            members = [m for m in zf.infolist() if not m.is_dir() and not m.filename.startswith("__MACOSX")]
            if not members:
                raise DicomValidationError("Uploaded ZIP archive is empty.")
            if len(members) > max_slices:
                raise DicomValidationError(f"ZIP exceeds maximum slice count ({max_slices}).")

            for member in members:
                # Reject zip-slip traversal paths
                if ".." in Path(member.filename).parts:
                    raise DicomValidationError("Rejected unsafe archive entry path.")
                data = zf.read(member.filename)
                try:
                    parsed_slices.append(parse_dicom_bytes(data))
                except DicomValidationError:
                    continue
    except zipfile.BadZipFile as exc:
        raise DicomValidationError("Uploaded file is not a valid ZIP archive.") from exc

    if not parsed_slices:
        raise DicomValidationError("No readable DICOM slices found inside the uploaded ZIP archive.")

    parsed_slices.sort(key=lambda s: (s.slice_location, s.instance_number))
    first = parsed_slices[0]
    return ParsedDicomSeries(
        study_instance_uid=first.study_instance_uid,
        series_instance_uid=first.series_instance_uid,
        modality=first.modality,
        body_part=first.body_part,
        slices=parsed_slices,
    )


def save_uint8_preview(display_uint8: np.ndarray, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(display_uint8, mode="L").save(output_path, format="PNG")
    return output_path
