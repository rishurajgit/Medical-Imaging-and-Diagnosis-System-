from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.session import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    architecture: Mapped[str] = mapped_column(String(200), nullable=False)
    task: Mapped[str] = mapped_column(String(32), nullable=False)  # classification | segmentation
    supported_modalities: Mapped[list] = mapped_column(JSON, default=list)
    supported_anatomy: Mapped[list] = mapped_column(JSON, default=list)
    input_format: Mapped[str] = mapped_column(String(255), nullable=False)
    dataset: Mapped[str] = mapped_column(String(160), nullable=False)
    weights_path: Mapped[str] = mapped_column(String(300), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ImagingStudy(Base):
    __tablename__ = "imaging_studies"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    accession_number: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    study_instance_uid: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    orthanc_study_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    patient_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    patient_sex: Mapped[str] = mapped_column(String(8), default="U")
    patient_age: Mapped[str] = mapped_column(String(16), default="000Y")
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    modality: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    body_part: Mapped[str] = mapped_column(String(64), default="UNKNOWN")
    study_date: Mapped[str] = mapped_column(String(32), nullable=False)
    institution_name: Mapped[str] = mapped_column(String(160), default="MedVision Educational PACS")
    source_type: Mapped[str] = mapped_column(String(32), default="upload_dicom")
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    # File storage paths on disk (large binaries are never stored in PostgreSQL)
    raw_file_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    preview_image_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    slice_count: Mapped[int] = mapped_column(Integer, default=1)

    # DICOM display metadata
    photometric_interpretation: Mapped[str] = mapped_column(String(32), default="MONOCHROME2")
    window_center: Mapped[float] = mapped_column(Float, default=128.0)
    window_width: Mapped[float] = mapped_column(Float, default=256.0)
    slice_thickness_mm: Mapped[str] = mapped_column(String(64), default="N/A")
    pixel_spacing_mm: Mapped[str] = mapped_column(String(64), default="N/A")
    rows: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    columns: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    bits_allocated: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Processing & Review Status
    processing_status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    processing_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    review_status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    series: Mapped[list["DicomSeries"]] = relationship(
        back_populates="study", cascade="all, delete-orphan"
    )
    inference_results: Mapped[list["InferenceResult"]] = relationship(
        back_populates="study", cascade="all, delete-orphan"
    )
    reviews: Mapped[list["RadiologistReview"]] = relationship(
        back_populates="study", cascade="all, delete-orphan"
    )


class DicomSeries(Base):
    __tablename__ = "dicom_series"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    study_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("imaging_studies.id", ondelete="CASCADE"), index=True
    )
    series_instance_uid: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    orthanc_series_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    series_description: Mapped[str] = mapped_column(String(255), default="DICOM SERIES")
    modality: Mapped[str] = mapped_column(String(16), nullable=False)
    instance_count: Mapped[int] = mapped_column(Integer, default=1)
    storage_dir: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    slice_preview_paths: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    study: Mapped["ImagingStudy"] = relationship(back_populates="series")


class InferenceResult(Base):
    __tablename__ = "inference_results"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    study_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("imaging_studies.id", ondelete="CASCADE"), index=True
    )
    model_version_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("model_versions.id"), nullable=True
    )
    model_name: Mapped[str] = mapped_column(String(160), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    primary_finding: Mapped[str] = mapped_column(String(255), nullable=False)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    class_probabilities: Mapped[list] = mapped_column(JSON, default=list)
    gradcam_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    gradcam_summary: Mapped[str] = mapped_column(Text, default="")
    segmentation_mask_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    segmentation_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    processed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    study: Mapped["ImagingStudy"] = relationship(back_populates="inference_results")


class RadiologistReview(Base):
    __tablename__ = "radiologist_reviews"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    study_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("imaging_studies.id", ondelete="CASCADE"), index=True
    )
    decision: Mapped[str] = mapped_column(String(32), nullable=False)  # approved | rejected
    reviewer_id: Mapped[str] = mapped_column(String(128), nullable=False)
    notes: Mapped[str] = mapped_column(Text, default="")
    prior_status: Mapped[str] = mapped_column(String(32), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    study: Mapped["ImagingStudy"] = relationship(back_populates="reviews")


class AuditLog(Base):
    """
    Append-only UTC audit trail. Historical rows must NEVER be silently updated or deleted.
    """

    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    study_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    accession_number: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    modality: Mapped[str] = mapped_column(String(16), nullable=False)
    model_name: Mapped[str] = mapped_column(String(160), default="N/A")
    model_version: Mapped[str] = mapped_column(String(64), default="N/A")
    primary_finding: Mapped[str] = mapped_column(String(255), default="")
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    event_type: Mapped[str] = mapped_column(String(48), index=True, nullable=False)
    actor: Mapped[str] = mapped_column(String(128), nullable=False)
    notes: Mapped[str] = mapped_column(Text, default="")
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
    timestamp_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True
    )
