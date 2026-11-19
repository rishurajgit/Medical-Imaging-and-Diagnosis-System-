from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ClassProbabilityOut(BaseModel):
    label: str
    probability: float


class DicomMetadataOut(BaseModel):
    studyInstanceUid: str
    seriesInstanceUid: Optional[str] = None
    seriesDescription: str = "DICOM SERIES"
    datasetSource: str = "Uploaded Study"
    patientId: str
    patientSex: str = "U"
    patientAge: str = "000Y"
    studyDate: str
    sliceThicknessMm: str = "N/A"
    pixelSpacingMm: str = "N/A"
    windowCenter: float = 128.0
    windowWidth: float = 256.0
    photometricInterpretation: str = "MONOCHROME2"
    institutionName: str = "MedVision Educational PACS"
    rows: Optional[int] = None
    columns: Optional[int] = None
    bitsAllocated: Optional[int] = None


class SegmentationSummaryOut(BaseModel):
    supported: bool
    lesionDetected: Optional[bool] = None
    voxelCount: Optional[int] = None
    estimatedVolumeMm3: Optional[float] = None
    meanProbability: Optional[float] = None
    maskImageUrl: Optional[str] = None
    note: str


class StudyOut(BaseModel):
    id: str
    accessionNumber: str
    title: str
    modality: str
    bodyPart: str
    imageUrl: str
    sliceUrls: List[str] = Field(default_factory=list)
    sliceCount: int = 1
    isDemo: bool = False
    processingStatus: str
    processingMessage: Optional[str] = None
    modelName: str
    modelVersion: str
    processedAt: Optional[str] = None
    primaryFinding: str
    confidence: Optional[float] = None
    classProbabilities: List[ClassProbabilityOut] = Field(default_factory=list)
    gradCamImageUrl: Optional[str] = None
    gradCamSummary: str = ""
    clinicalImpression: str = ""
    heatmapRegions: List[Dict[str, Any]] = Field(default_factory=list)
    segmentation: Optional[SegmentationSummaryOut] = None
    dicom: DicomMetadataOut
    status: Literal["pending", "approved", "rejected"]
    reviewedBy: Optional[str] = None
    reviewedAt: Optional[str] = None
    reviewerNotes: Optional[str] = None
    createdAt: str
    sourceType: str


class ReviewCreateRequest(BaseModel):
    decision: Literal["approved", "rejected"]
    reviewer: str = Field(..., min_length=2, max_length=128)
    notes: str = Field(default="", max_length=2000)


class AnalyzeRequest(BaseModel):
    studyId: str = Field(..., min_length=1, max_length=64)


class UploadJsonRequest(BaseModel):
    fileBase64: str
    fileName: str
    mimeType: Optional[str] = None
    title: Optional[str] = None
    modality: Optional[str] = "CR"
    bodyPart: Optional[str] = "CHEST"
    patientId: Optional[str] = None


class AuditLogOut(BaseModel):
    id: str
    studyId: str
    accessionNumber: str
    modality: str
    modelName: str
    modelVersion: str
    primaryFinding: str
    confidence: Optional[float] = None
    decision: str
    reviewer: str
    notes: str
    isDemo: bool = False
    timestamp: str


class ModelDescriptorOut(BaseModel):
    id: str
    name: str
    version: str
    architecture: str
    task: Literal["classification", "segmentation"]
    supportedModalities: List[str]
    supportedAnatomy: List[str]
    inputFormat: str
    dataset: str
    explainabilityMethod: str
    weightsPath: str
    weightsLoaded: bool
    statusMessage: str
    limitations: List[str]
