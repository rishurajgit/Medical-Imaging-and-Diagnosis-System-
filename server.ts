import 'dotenv/config';
import express from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI } from '@google/genai';
import { DEMO_AUDIT_LOG, DEMO_STUDIES } from './src/data/seedStudies.ts';
import type {
  AuditEventType,
  AuditLogEntry,
  ImagingStudy,
  Modality,
  ModelDescriptor,
  ModelMetricsResponse,
} from './src/types/imaging.ts';
import { parseDicomBuffer } from './src/utils/dicomParser.ts';
import { encodeGrayscalePng } from './src/utils/pngEncoder.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DATA_DIR = path.join(__dirname, 'data');
const UPLOADS_DIR = path.join(DATA_DIR, 'uploads');
const STORE_PATH = path.join(DATA_DIR, 'pacs-store-v2.json');
const FASTAPI_BACKEND_URL = (process.env.FASTAPI_BACKEND_URL || '').replace(/\/+$/, '');
const MAX_UPLOAD_BYTES = 20 * 1024 * 1024; // 20 MB

const CXR_WEIGHTS_PATH = path.join(__dirname, 'models', 'densenet121_nih14.pth');
const SEG_WEIGHTS_PATH = path.join(__dirname, 'models', 'monai_unet3d_task06_lung.pth');
const CXR_METRICS_PATH = path.join(__dirname, 'reports', 'classification_metrics.json');
const SEG_METRICS_PATH = path.join(__dirname, 'reports', 'segmentation_metrics.json');

interface PacsStore {
  studies: ImagingStudy[];
  auditLog: AuditLogEntry[];
}

function ensureStore(): PacsStore {
  if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
  }
  if (!fs.existsSync(UPLOADS_DIR)) {
    fs.mkdirSync(UPLOADS_DIR, { recursive: true });
  }
  if (!fs.existsSync(STORE_PATH)) {
    const initial: PacsStore = {
      studies: DEMO_STUDIES,
      auditLog: DEMO_AUDIT_LOG,
    };
    fs.writeFileSync(STORE_PATH, JSON.stringify(initial, null, 2), 'utf-8');
    return initial;
  }
  try {
    const raw = fs.readFileSync(STORE_PATH, 'utf-8');
    const parsed = JSON.parse(raw) as PacsStore;
    if (!Array.isArray(parsed.studies) || !Array.isArray(parsed.auditLog)) {
      throw new Error('Invalid store format');
    }
    return parsed;
  } catch {
    const fallback: PacsStore = {
      studies: DEMO_STUDIES,
      auditLog: DEMO_AUDIT_LOG,
    };
    fs.writeFileSync(STORE_PATH, JSON.stringify(fallback, null, 2), 'utf-8');
    return fallback;
  }
}

function writeStore(store: PacsStore): void {
  fs.writeFileSync(STORE_PATH, JSON.stringify(store, null, 2), 'utf-8');
}

function sanitizeId(input: string): string {
  return input.replace(/[^a-zA-Z0-9_-]/g, '');
}

function appendAudit(
  store: PacsStore,
  study: ImagingStudy,
  decision: AuditEventType,
  reviewer: string,
  notes: string
): AuditLogEntry {
  const entry: AuditLogEntry = {
    id: `aud-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
    studyId: study.id,
    accessionNumber: study.accessionNumber,
    modality: study.modality,
    modelName: study.modelName,
    modelVersion: study.modelVersion,
    primaryFinding: study.primaryFinding,
    confidence: study.confidence,
    decision,
    reviewer,
    notes,
    isDemo: study.isDemo,
    timestamp: new Date().toISOString(),
  };
  store.auditLog.unshift(entry);
  return entry;
}

function getModelRegistry(): ModelDescriptor[] {
  const cxrLoaded = fs.existsSync(CXR_WEIGHTS_PATH);
  const segLoaded = fs.existsSync(SEG_WEIGHTS_PATH);

  return [
    {
      id: 'cxr_densenet121_nih14',
      name: 'DenseNet-121 Chest X-Ray Classifier',
      version: '1.0.0-nih14',
      architecture: 'TorchVision DenseNet-121 (Transfer Learning, 14-label BCEWithLogitsLoss)',
      task: 'classification',
      supportedModalities: ['CR', 'DX'],
      supportedAnatomy: ['CHEST', 'THORAX'],
      inputFormat: '2D Grayscale DICOM (.dcm) or PNG/JPEG resized to 224×224',
      dataset: 'NIH ChestX-ray14 (Patient-level split)',
      explainabilityMethod: 'Grad-CAM on features.denseblock4 / features.norm5',
      weightsPath: 'models/densenet121_nih14.pth',
      weightsLoaded: cxrLoaded,
      statusMessage: cxrLoaded
        ? 'Trained checkpoint detected on disk. Start the FastAPI/PyTorch backend to run live inference.'
        : 'Trained checkpoint not found. Train via `python -m app.ml.classification.train` or mount models/densenet121_nih14.pth.',
      limitations: [
        'Trained strictly on frontal PA/AP Chest X-rays; not valid for lateral views, CT, MRI, or pediatric scans.',
        'NIH ChestX-ray14 NLP-extracted labels carry documented label noise (~10%).',
        'Grad-CAM highlights convolutional sensitivity and does not prove clinical causation.',
      ],
    },
    {
      id: 'lung_ct_monai_unet3d',
      name: 'MONAI 3D U-Net Lung Segmentation',
      version: '1.0.0-msd-task06',
      architecture: 'MONAI 3D U-Net (5-stage encoder/decoder, DiceCELoss, sliding-window inference)',
      task: 'segmentation',
      supportedModalities: ['CT'],
      supportedAnatomy: ['LUNG', 'CHEST', 'THORAX'],
      inputFormat: '3D CT Volume (NIfTI .nii/.nii.gz or multi-slice DICOM series, RAS orientation, HU [-1000, 400])',
      dataset: 'Medical Segmentation Decathlon — Task06_Lung',
      explainabilityMethod: 'Voxel-level foreground probability map + binary segmentation mask overlay',
      weightsPath: 'models/monai_unet3d_task06_lung.pth',
      weightsLoaded: segLoaded,
      statusMessage: segLoaded
        ? 'Trained checkpoint detected on disk. Start the FastAPI/MONAI backend to run volumetric segmentation.'
        : 'Trained checkpoint not found. Train via `python -m app.ml.segmentation.train` or mount models/monai_unet3d_task06_lung.pth.',
      limitations: [
        'Supports thoracic CT volumes only; does not support X-ray or Brain MRI (BraTS is a modular future extension).',
        'Performance degrades on non-standard slice thickness (>5mm) or severe motion artifacts.',
        'Segmentation boundaries require manual radiologist verification.',
      ],
    },
  ];
}

function readJsonSafe(filePath: string): Record<string, unknown> | null {
  try {
    if (!fs.existsSync(filePath)) return null;
    return JSON.parse(fs.readFileSync(filePath, 'utf-8')) as Record<string, unknown>;
  } catch {
    return null;
  }
}

function getMetricsReport(): ModelMetricsResponse {
  const cxrJson = readJsonSafe(CXR_METRICS_PATH);
  const segJson = readJsonSafe(SEG_METRICS_PATH);

  return {
    evaluationAvailable: Boolean(cxrJson || segJson),
    note:
      cxrJson || segJson
        ? 'Loaded from local DVC evaluation JSON artifacts.'
        : 'No local training/evaluation runs have been executed yet. Metrics remain null until datasets are downloaded and DVC evaluation stages complete — never fabricated.',
    classification: {
      status: cxrJson ? 'evaluated' : 'not_evaluated_locally',
      dataset: 'NIH ChestX-ray14',
      metricsFile: 'reports/classification_metrics.json',
      macroAuroc: typeof cxrJson?.macro_auroc === 'number' ? cxrJson.macro_auroc : null,
      perClassAuroc:
        cxrJson?.per_class_auroc && typeof cxrJson.per_class_auroc === 'object'
          ? (cxrJson.per_class_auroc as Record<string, number>)
          : null,
      precision: typeof cxrJson?.precision === 'number' ? cxrJson.precision : null,
      recall: typeof cxrJson?.recall === 'number' ? cxrJson.recall : null,
      f1Score: typeof cxrJson?.f1_score === 'number' ? cxrJson.f1_score : null,
      evaluatedAt: typeof cxrJson?.evaluated_at === 'string' ? cxrJson.evaluated_at : null,
    },
    segmentation: {
      status: segJson ? 'evaluated' : 'not_evaluated_locally',
      dataset: 'Medical Segmentation Decathlon (Task06_Lung)',
      metricsFile: 'reports/segmentation_metrics.json',
      meanDice: typeof segJson?.mean_dice === 'number' ? segJson.mean_dice : null,
      meanIoU: typeof segJson?.mean_iou === 'number' ? segJson.mean_iou : null,
      sensitivity: typeof segJson?.sensitivity === 'number' ? segJson.sensitivity : null,
      specificity: typeof segJson?.specificity === 'number' ? segJson.specificity : null,
      evaluatedAt: typeof segJson?.evaluated_at === 'string' ? segJson.evaluated_at : null,
    },
  };
}

function determinePipeline(modality: Modality, bodyPart: string): {
  supported: boolean;
  pipelineId: string;
  modelName: string;
  modelVersion: string;
  reason?: string;
} {
  const bp = bodyPart.toUpperCase();
  const isChest = ['CHEST', 'THORAX', 'LUNG', 'PULMONARY', 'UNKNOWN', ''].includes(bp);

  if ((modality === 'CR' || modality === 'DX') && isChest) {
    return {
      supported: true,
      pipelineId: 'cxr_densenet121_nih14',
      modelName: 'DenseNet-121 Chest X-Ray Classifier',
      modelVersion: '1.0.0-nih14',
    };
  }
  if (modality === 'CT' && isChest) {
    return {
      supported: true,
      pipelineId: 'lung_ct_monai_unet3d',
      modelName: 'MONAI 3D U-Net Lung Segmentation',
      modelVersion: '1.0.0-msd-task06',
    };
  }
  return {
    supported: false,
    pipelineId: 'none',
    modelName: 'Unsupported Modality / Anatomy',
    modelVersion: 'N/A',
    reason: `Modality "${modality}" (Anatomy: "${bodyPart}") has no registered active model. Supported pipelines: Chest X-ray (CR/DX) -> DenseNet-121 NIH14; Thoracic CT -> MONAI 3D U-Net Task06_Lung.`,
  };
}

function getGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY is not configured in environment.');
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

async function startServer() {
  const app = express();
  const PORT = Number(process.env.PORT) || 3000;

  app.use(express.json({ limit: '28mb' }));
  ensureStore();

  // Optional FastAPI reverse proxy when running alongside Docker / FastAPI backend
  if (FASTAPI_BACKEND_URL) {
    app.use('/api', async (req, res, next) => {
      if (req.path === '/explain-output') {
        next();
        return;
      }
      try {
        const targetUrl = `${FASTAPI_BACKEND_URL}/api${req.url}`;
        const headers: Record<string, string> = {};
        if (req.headers['content-type']) {
          headers['content-type'] = String(req.headers['content-type']);
        }
        if (req.headers.authorization) {
          headers.authorization = String(req.headers.authorization);
        }
        const fetchRes = await fetch(targetUrl, {
          method: req.method,
          headers,
          body:
            req.method !== 'GET' && req.method !== 'HEAD'
              ? JSON.stringify(req.body)
              : undefined,
        });
        const contentType = fetchRes.headers.get('content-type') || 'application/json';
        res.status(fetchRes.status).setHeader('content-type', contentType);
        const arrayBuf = await fetchRes.arrayBuffer();
        res.send(Buffer.from(arrayBuf));
      } catch (err) {
        console.warn('FastAPI backend unreachable, falling back to Node preview mode:', err);
        next();
      }
    });
  }

  // Serve uploaded study files safely (no directory traversal)
  app.use('/api/uploads', express.static(UPLOADS_DIR, { dotfiles: 'deny', index: false }));

  app.get('/api/health', (_req, res) => {
    const models = getModelRegistry();
    res.json({
      status: 'ok',
      service: 'medvision-pacs',
      runtime: FASTAPI_BACKEND_URL ? 'node_proxy_fallback' : 'node_educational_preview',
      fastapiConfigured: Boolean(FASTAPI_BACKEND_URL),
      orthancReachable: false,
      databaseMode: 'local_json_preview (PostgreSQL configured in backend/ + docker-compose.yml)',
      modelsLoaded: models.filter((m) => m.weightsLoaded).length,
      totalRegisteredModels: models.length,
      disclaimer:
        'FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY. NOT FOR CLINICAL USE.',
    });
  });

  app.get('/api/models', (_req, res) => {
    res.json({ models: getModelRegistry() });
  });

  app.get('/api/metrics', (_req, res) => {
    res.json(getMetricsReport());
  });

  app.get('/api/studies', (_req, res) => {
    const store = ensureStore();
    res.json(store);
  });

  app.get('/api/studies/:id', (req, res) => {
    const id = sanitizeId(req.params.id);
    const store = ensureStore();
    const study = store.studies.find((s) => s.id === id);
    if (!study) {
      res.status(404).json({ error: 'Study not found.' });
      return;
    }
    res.json({ study });
  });

  app.get('/api/studies/:id/results', (req, res) => {
    const id = sanitizeId(req.params.id);
    const store = ensureStore();
    const study = store.studies.find((s) => s.id === id);
    if (!study) {
      res.status(404).json({ error: 'Study not found.' });
      return;
    }
    res.json({
      studyId: study.id,
      isDemo: study.isDemo,
      processingStatus: study.processingStatus,
      processingMessage: study.processingMessage,
      modelName: study.modelName,
      modelVersion: study.modelVersion,
      processedAt: study.processedAt ?? null,
      primaryFinding: study.primaryFinding,
      confidence: study.confidence,
      classProbabilities: study.classProbabilities,
      gradCamSummary: study.gradCamSummary,
      segmentation: study.segmentation ?? null,
    });
  });

  app.get('/api/studies/:id/heatmap', (req, res) => {
    const id = sanitizeId(req.params.id);
    const store = ensureStore();
    const study = store.studies.find((s) => s.id === id);
    if (!study) {
      res.status(404).json({ error: 'Study not found.' });
      return;
    }
    res.json({
      studyId: study.id,
      isDemo: study.isDemo,
      available: study.heatmapRegions.length > 0 || Boolean(study.gradCamImageUrl),
      heatmapImageUrl: study.gradCamImageUrl ?? null,
      heatmapRegions: study.heatmapRegions,
      disclaimer:
        'Grad-CAM highlights convolutional layer sensitivity and does not establish clinical causation.',
    });
  });

  app.get('/api/studies/:id/segmentation', (req, res) => {
    const id = sanitizeId(req.params.id);
    const store = ensureStore();
    const study = store.studies.find((s) => s.id === id);
    if (!study) {
      res.status(404).json({ error: 'Study not found.' });
      return;
    }
    res.json({
      studyId: study.id,
      isDemo: study.isDemo,
      segmentation: study.segmentation ?? {
        supported: false,
        note: 'No segmentation mask available for this study.',
      },
    });
  });

  app.get('/api/audit', (_req, res) => {
    const store = ensureStore();
    res.json({ auditLog: store.auditLog });
  });

  /**
   * POST /api/upload
   * Accepts real DICOM (.dcm), PNG/JPEG/WebP X-rays, or ZIP series metadata.
   * Parses DICOM binary headers & pixels when .dcm is provided.
   * Never fabricates AI predictions when trained PyTorch/MONAI checkpoints are absent.
   */
  app.post('/api/upload', (req, res) => {
    try {
      const {
        fileBase64,
        fileName,
        mimeType,
        title,
        modality: requestedModality,
        bodyPart: requestedBodyPart,
        patientId: requestedPatientId,
      } = req.body as {
        fileBase64?: string;
        fileName?: string;
        mimeType?: string;
        title?: string;
        modality?: Modality;
        bodyPart?: string;
        patientId?: string;
      };

      if (!fileBase64 || !fileName) {
        res.status(400).json({ error: 'Missing required fileBase64 or fileName.' });
        return;
      }

      const base64Clean = fileBase64.includes(',') ? fileBase64.split(',')[1] : fileBase64;
      const rawBuffer = Buffer.from(base64Clean, 'base64');

      if (rawBuffer.length === 0 || rawBuffer.length > MAX_UPLOAD_BYTES) {
        res.status(413).json({
          error: `File size must be between 1 byte and ${Math.round(MAX_UPLOAD_BYTES / (1024 * 1024))} MB.`,
        });
        return;
      }

      const ext = path.extname(fileName).toLowerCase();
      const allowedExts = new Set(['.dcm', '.dicom', '.png', '.jpg', '.jpeg', '.webp', '.zip']);
      if (!allowedExts.has(ext)) {
        res.status(400).json({
          error: `Unsupported file extension "${ext}". Accepted: .dcm, .dicom, .zip (DICOM series), .png, .jpg, .webp.`,
        });
        return;
      }

      const studyId = `study-${Date.now()}`;
      const accession = `ACC-2026-${String(Math.floor(1000 + Math.random() * 8999))}`;
      const now = new Date().toISOString();

      let finalModality: Modality =
        requestedModality && ['CR', 'DX', 'CT', 'MR', 'OTHER'].includes(requestedModality)
          ? requestedModality
          : 'CR';
      let finalBodyPart = (requestedBodyPart || 'CHEST').trim().toUpperCase();
      let imageUrl = '';
      let sourceType: ImagingStudy['sourceType'] = 'upload_image';
      const warnings: string[] = [];

      let dicomMeta: ImagingStudy['dicom'] = {
        studyInstanceUid: `1.2.826.0.1.3680043.8.498.${Date.now()}`,
        seriesInstanceUid: `1.2.826.0.1.3680043.8.498.${Date.now()}.1`,
        seriesDescription: `${finalModality} ${finalBodyPart} UPLOADED`,
        datasetSource: 'User Upload',
        patientId: (requestedPatientId || `ANON-${Math.floor(1000 + Math.random() * 9000)}`).trim(),
        patientSex: 'U',
        patientAge: '000Y',
        studyDate: now.slice(0, 10),
        sliceThicknessMm: finalModality === 'CT' || finalModality === 'MR' ? '1.00' : 'N/A (2D Projection)',
        pixelSpacingMm: 'Unstandardized Upload',
        windowCenter: 128,
        windowWidth: 256,
        photometricInterpretation: 'MONOCHROME2',
        institutionName: 'Uploaded Educational Study',
      };

      if (ext === '.dcm' || ext === '.dicom') {
        sourceType = 'upload_dicom';
        const parsed = parseDicomBuffer(rawBuffer);
        if (!parsed.isValidDicom) {
          res.status(400).json({
            error: parsed.warnings[0] || 'Invalid DICOM Part 10 file (missing DICM header).',
          });
          return;
        }
        warnings.push(...parsed.warnings);
        fs.writeFileSync(path.join(UPLOADS_DIR, `${studyId}.dcm`), rawBuffer);

        if (parsed.modality !== 'OTHER') {
          finalModality = parsed.modality;
        }
        if (parsed.bodyPart && parsed.bodyPart !== 'UNKNOWN') {
          finalBodyPart = parsed.bodyPart;
        }

        dicomMeta = {
          ...dicomMeta,
          ...parsed.metadata,
          datasetSource: 'Uploaded DICOM (.dcm)',
        };

        if (parsed.grayscalePreview) {
          const pngBuf = encodeGrayscalePng(
            parsed.grayscalePreview.width,
            parsed.grayscalePreview.height,
            parsed.grayscalePreview.pixels
          );
          const pngFile = `${studyId}.png`;
          fs.writeFileSync(path.join(UPLOADS_DIR, pngFile), pngBuf);
          imageUrl = `/api/uploads/${pngFile}`;
        } else {
          warnings.push(
            'DICOM metadata was parsed, but compressed/encapsulated pixel frames require the Python pydicom + Orthanc backend to render.'
          );
        }
      } else if (ext === '.zip') {
        sourceType = 'upload_zip';
        // Verify PK zip magic header
        if (rawBuffer.length < 4 || rawBuffer[0] !== 0x50 || rawBuffer[1] !== 0x4b) {
          res.status(400).json({ error: 'Invalid ZIP archive header.' });
          return;
        }
        fs.writeFileSync(path.join(UPLOADS_DIR, `${studyId}.zip`), rawBuffer);
        dicomMeta.datasetSource = 'Uploaded DICOM Series (.zip)';
        warnings.push(
          'ZIP DICOM series stored on disk. Full multi-slice extraction and Orthanc C-STORE forwarding execute in the Docker FastAPI + Orthanc stack.'
        );
      } else {
        const cleanExt = mimeType?.includes('png')
          ? '.png'
          : mimeType?.includes('webp')
          ? '.webp'
          : '.jpg';
        const storedFile = `${studyId}${cleanExt}`;
        fs.writeFileSync(path.join(UPLOADS_DIR, storedFile), rawBuffer);
        imageUrl = `/api/uploads/${storedFile}`;
      }

      const pipeline = determinePipeline(finalModality, finalBodyPart);

      const processingStatus: ImagingStudy['processingStatus'] = !pipeline.supported
        ? 'unsupported_modality'
        : 'weights_unavailable';

      const processingMessage = !pipeline.supported
        ? pipeline.reason
        : `Study ingested and validated for ${pipeline.modelName} (${pipeline.modelVersion}), but trained PyTorch/MONAI model weights are not loaded in this preview runtime. Predictions are NOT fabricated. Run Docker Compose with trained checkpoints mounted in ./models/ to execute live inference and real gradient-based Grad-CAM.`;

      const newStudy: ImagingStudy = {
        id: studyId,
        accessionNumber: accession,
        title: (title || '').trim() || `${finalModality} ${finalBodyPart} — ${fileName}`,
        modality: finalModality,
        bodyPart: finalBodyPart,
        imageUrl,
        sliceCount: 1,
        isDemo: false,
        processingStatus,
        processingMessage:
          warnings.length > 0 ? `${processingMessage} Notes: ${warnings.join(' ')}` : processingMessage,
        modelName: pipeline.modelName,
        modelVersion: pipeline.modelVersion,
        supportedPipeline: pipeline.pipelineId,
        processedAt: now,
        primaryFinding: !pipeline.supported
          ? 'Unsupported Modality — Manual Review Only'
          : 'Awaiting Trained Model Weights (No Synthetic Prediction)',
        confidence: null,
        classProbabilities: [],
        gradCamSummary: !pipeline.supported
          ? 'No explainability overlay generated because no registered model supports this modality/anatomy.'
          : 'Grad-CAM requires actual forward/backward gradients from the trained PyTorch DenseNet-121 checkpoint. No synthetic heatmap coordinates were generated.',
        clinicalImpression:
          'No AI impression generated while trained weights are unavailable. Radiologist manual inspection and audit sign-off remain active.',
        heatmapRegions: [],
        segmentation: {
          supported: pipeline.pipelineId === 'lung_ct_monai_unet3d',
          note:
            pipeline.pipelineId === 'lung_ct_monai_unet3d'
              ? 'MONAI 3D U-Net Task06_Lung weights not loaded in preview runtime.'
              : 'Segmentation pipeline applies to Thoracic CT volumes only.',
        },
        dicom: dicomMeta,
        status: 'pending',
        createdAt: now,
        sourceType,
      };

      const store = ensureStore();
      store.studies.unshift(newStudy);

      const uploadAudit = appendAudit(
        store,
        newStudy,
        sourceType === 'upload_dicom' ? 'dicom_ingested' : 'study_uploaded',
        'Ingestion Service',
        `File "${fileName}" (${(rawBuffer.length / 1024).toFixed(1)} KB) validated and stored on disk.`
      );
      const inferAudit = appendAudit(
        store,
        newStudy,
        'inference_blocked',
        'Inference Router',
        processingMessage || 'Inference deferred.'
      );
      writeStore(store);

      res.status(201).json({
        study: newStudy,
        auditEntries: [inferAudit, uploadAudit],
        warnings,
      });
    } catch (error: unknown) {
      const msg = error instanceof Error ? error.message : 'Upload processing failed.';
      console.error('Upload error:', error);
      res.status(500).json({ error: msg });
    }
  });

  /**
   * POST /api/analyze
   * Explicitly triggers or re-checks model inference for a study_id.
   * Refuses to fabricate predictions if trained PyTorch/MONAI checkpoints are unavailable.
   */
  app.post('/api/analyze', (req, res) => {
    const { studyId } = req.body as { studyId?: string };
    if (!studyId) {
      res.status(400).json({ error: 'studyId is required.' });
      return;
    }
    const store = ensureStore();
    const study = store.studies.find((s) => s.id === sanitizeId(studyId));
    if (!study) {
      res.status(404).json({ error: 'Study not found.' });
      return;
    }
    if (study.isDemo) {
      res.json({
        study,
        status: 'demo_illustrative',
        message:
          'This is an illustrative Demo Mode study. Upload a real study and connect the FastAPI + PyTorch/MONAI backend with trained weights to execute live model inference.',
      });
      return;
    }

    const pipeline = determinePipeline(study.modality, study.bodyPart);
    const auditEntry = appendAudit(
      store,
      study,
      'inference_blocked',
      'Inference Router',
      pipeline.supported
        ? `Live inference requested for ${pipeline.modelName}, but checkpoint is not loaded in Node preview runtime.`
        : pipeline.reason || 'Unsupported modality.'
    );
    writeStore(store);

    res.status(200).json({
      study,
      auditEntry,
      status: pipeline.supported ? 'weights_unavailable' : 'unsupported_modality',
      message: pipeline.supported
        ? `Trained weights for ${pipeline.modelName} are not mounted in this preview runtime. Start Docker Compose (\`docker compose up --build\`) with trained weights in \`models/\` to run live PyTorch/MONAI inference.`
        : pipeline.reason,
    });
  });

  /**
   * POST /api/studies/:id/review
   * Appends an immutable UTC radiologist sign-off decision to the audit log.
   */
  app.post('/api/studies/:id/review', (req, res) => {
    const id = sanitizeId(req.params.id);
    const { decision, reviewer, notes } = req.body as {
      decision?: 'approved' | 'rejected';
      reviewer?: string;
      notes?: string;
    };

    if (decision !== 'approved' && decision !== 'rejected') {
      res.status(400).json({ error: 'Decision must be "approved" or "rejected".' });
      return;
    }

    const cleanReviewer = (reviewer ?? '').trim();
    if (!cleanReviewer) {
      res.status(400).json({ error: 'Reviewer identification is required.' });
      return;
    }

    const cleanNotes = (notes ?? '').trim();
    const timestamp = new Date().toISOString();

    const store = ensureStore();
    const studyIndex = store.studies.findIndex((s) => s.id === id);
    if (studyIndex === -1) {
      res.status(404).json({ error: 'Study not found.' });
      return;
    }

    const existing = store.studies[studyIndex];
    const priorStatus = existing.status;

    const updatedStudy: ImagingStudy = {
      ...existing,
      status: decision,
      reviewedBy: cleanReviewer,
      reviewedAt: timestamp,
      reviewerNotes: cleanNotes || existing.reviewerNotes,
    };
    store.studies[studyIndex] = updatedStudy;

    const auditNote =
      cleanNotes ||
      (decision === 'approved'
        ? 'Study finding approved by reviewing radiologist.'
        : 'Study finding rejected / overridden by reviewing radiologist.');

    const auditEntry = appendAudit(
      store,
      updatedStudy,
      priorStatus !== 'pending' && priorStatus !== decision ? 'review_updated' : decision,
      cleanReviewer,
      priorStatus !== 'pending' && priorStatus !== decision
        ? `[Status changed: ${priorStatus} -> ${decision}] ${auditNote}`
        : auditNote
    );

    writeStore(store);
    res.json({ study: updatedStudy, auditEntry });
  });

  /**
   * POST /api/explain-output
   * Optional NON-DIAGNOSTIC educational helper using Gemini 4 Hydrogen.
   * Translates technical model terminology (Grad-CAM, probability calibration, DICOM windowing,
   * Dice/AUROC metrics) into clear educational prose. NEVER inspects raw pixels to diagnose.
   */
  app.post('/api/explain-output', async (req, res) => {
    try {
      const { studyId } = req.body as { studyId?: string };
      if (!studyId) {
        res.status(400).json({ error: 'studyId is required.' });
        return;
      }
      const store = ensureStore();
      const study = store.studies.find((s) => s.id === sanitizeId(studyId));
      if (!study) {
        res.status(404).json({ error: 'Study not found.' });
        return;
      }

      const ai = getGeminiClient();
      const contextPayload = {
        isDemoExample: study.isDemo,
        modality: study.modality,
        bodyPart: study.bodyPart,
        processingStatus: study.processingStatus,
        modelName: study.modelName,
        modelVersion: study.modelVersion,
        primaryFindingLabel: study.primaryFinding,
        confidenceScore: study.confidence,
        topClassProbabilities: study.classProbabilities.slice(0, 4),
        dicomWindowCenter: study.dicom.windowCenter,
        dicomWindowWidth: study.dicom.windowWidth,
        photometricInterpretation: study.dicom.photometricInterpretation,
      };

      const response = await ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: `Explain the following technical AI medical imaging workstation metadata in plain, educational language (max 140 words, 3 concise bullet points):
${JSON.stringify(contextPayload, null, 2)}

Rules:
- STRICTLY non-diagnostic. Do NOT give medical advice, clinical diagnosis, or treatment recommendations.
- Explain what the model architecture, probability scores, DICOM windowing, or Grad-CAM / segmentation status mean technically.
- Remind the reader that Grad-CAM shows model sensitivity (not causation) and that trained radiologist review is required.`,
        config: {
          systemInstruction:
            'You are an educational medical imaging ML explainer. You translate technical AI pipeline outputs into clear educational language without ever diagnosing patients or substituting for a radiologist.',
        },
      });

      const explanation = (response.text ?? '').trim();
      if (!explanation) {
        res.status(502).json({ error: 'Empty educational explanation returned.' });
        return;
      }

      res.json({
        explanation,
        nonDiagnosticNotice:
          'Educational technical explanation generated by Gemini 4 Hydrogen — strictly non-diagnostic and not a clinical assessment.',
      });
    } catch (error: unknown) {
      const errObj = error as { status?: number; message?: string };
      const message =
        errObj?.message || 'Unable to generate educational technical explanation.';
      const statusCode =
        typeof errObj?.status === 'number' && errObj.status >= 400 && errObj.status < 600
          ? errObj.status
          : 500;
      res.status(statusCode).json({ error: message });
    }
  });

  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.join(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('/{*splat}', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`MedVision PACS server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();
