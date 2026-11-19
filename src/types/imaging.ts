export type Modality = 'CR' | 'DX' | 'CT' | 'MR' | 'OTHER';

export type ReviewStatus = 'pending' | 'approved' | 'rejected';

export type ProcessingStatus =
  | 'demo_illustrative'
  | 'completed'
  | 'weights_unavailable'
  | 'unsupported_modality'
  | 'failed'
  | 'queued';

export type OverlayMode = 'off' | 'heatmap_only' | 'gradcam' | 'segmentation' | 'split';

export interface HeatmapRegion {
  /** Normalized center X (0..1) relative to scan image width */
  x: number;
  /** Normalized center Y (0..1) relative to scan image height */
  y: number;
  /** Normalized horizontal radius (0..1) */
  rx: number;
  /** Normalized vertical radius (0..1) */
  ry: number;
  /** Peak activation intensity (0..1) */
  intensity: number;
  /** Anatomical region label */
  label: string;
  /** Optional contour polygon points in normalized (0..1) coordinates for segmentation mask */
  contour?: Array<{ x: number; y: number }>;
}

export interface ClassProbability {
  label: string;
  probability: number;
}

export interface DicomMetadata {
  studyInstanceUid: string;
  seriesInstanceUid?: string;
  sopInstanceUid?: string;
  seriesDescription: string;
  datasetSource: string;
  patientId: string;
  patientSex: string;
  patientAge: string;
  studyDate: string;
  sliceThicknessMm: string;
  pixelSpacingMm: string;
  windowCenter: number;
  windowWidth: number;
  photometricInterpretation: string;
  institutionName: string;
  rows?: number;
  columns?: number;
  bitsAllocated?: number;
  transferSyntaxUid?: string;
}

export interface SegmentationSummary {
  supported: boolean;
  lesionDetected?: boolean;
  voxelCount?: number;
  estimatedVolumeMm3?: number;
  meanProbability?: number;
  maskImageUrl?: string | null;
  note: string;
}

export interface ImagingStudy {
  id: string;
  accessionNumber: string;
  title: string;
  modality: Modality;
  bodyPart: string;
  imageUrl: string;
  /** Optional multi-slice URLs for CT/MRI series navigation */
  sliceUrls?: string[];
  sliceCount: number;
  /** True strictly for clearly labeled illustrative Demo Mode examples */
  isDemo: boolean;
  processingStatus: ProcessingStatus;
  processingMessage?: string;
  modelName: string;
  modelVersion: string;
  supportedPipeline?: string;
  processedAt?: string;
  primaryFinding: string;
  confidence: number | null;
  classProbabilities: ClassProbability[];
  gradCamImageUrl?: string | null;
  gradCamSummary: string;
  clinicalImpression: string;
  heatmapRegions: HeatmapRegion[];
  segmentation?: SegmentationSummary;
  dicom: DicomMetadata;
  status: ReviewStatus;
  reviewedBy?: string;
  reviewedAt?: string;
  reviewerNotes?: string;
  createdAt: string;
  sourceType: 'demo' | 'upload_dicom' | 'upload_image' | 'upload_zip' | 'orthanc';
}

export type AuditEventType =
  | 'study_uploaded'
  | 'dicom_ingested'
  | 'inference_requested'
  | 'inference_completed'
  | 'inference_blocked'
  | 'heatmap_generated'
  | 'approved'
  | 'rejected'
  | 'review_updated';

export interface AuditLogEntry {
  id: string;
  studyId: string;
  accessionNumber: string;
  modality: Modality;
  modelName: string;
  modelVersion: string;
  primaryFinding: string;
  confidence: number | null;
  decision: AuditEventType;
  reviewer: string;
  notes: string;
  isDemo: boolean;
  timestamp: string;
}

export interface ModelDescriptor {
  id: string;
  name: string;
  version: string;
  architecture: string;
  task: 'classification' | 'segmentation';
  supportedModalities: Modality[];
  supportedAnatomy: string[];
  inputFormat: string;
  dataset: string;
  explainabilityMethod: string;
  weightsPath: string;
  weightsLoaded: boolean;
  statusMessage: string;
  limitations: string[];
}

export interface ModelMetricsResponse {
  evaluationAvailable: boolean;
  note: string;
  classification: {
    status: 'not_evaluated_locally' | 'evaluated';
    dataset: string;
    metricsFile: string;
    macroAuroc: number | null;
    perClassAuroc: Record<string, number> | null;
    precision: number | null;
    recall: number | null;
    f1Score: number | null;
    evaluatedAt: string | null;
  };
  segmentation: {
    status: 'not_evaluated_locally' | 'evaluated';
    dataset: string;
    metricsFile: string;
    meanDice: number | null;
    meanIoU: number | null;
    sensitivity: number | null;
    specificity: number | null;
    evaluatedAt: string | null;
  };
}
