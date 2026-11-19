import React, { useEffect, useState } from 'react';
import { ArrowRight, Check, Copy } from 'lucide-react';
import type { ModelDescriptor, ModelMetricsResponse } from '../types/imaging';

type CommandTab = 'docker' | 'dvc' | 'train';

const COMMAND_SNIPPETS: Record<CommandTab, { label: string; filename: string; code: string }> = {
  docker: {
    label: 'Docker Compose Startup',
    filename: 'docker-compose.yml',
    code: `# 1. Configure environment secrets (never commit .env)
cp .env.example .env

# 2. Launch Frontend + FastAPI Backend + PostgreSQL + Orthanc DICOM Server
docker compose up --build

# Service Endpoints:
# - React PACS Workstation : http://localhost:3000
# - FastAPI OpenAPI Docs   : http://localhost:8000/docs
# - Orthanc DICOMweb (int) : http://127.0.0.1:8042 (restricted from public exposure)
# - Orthanc DICOM C-STORE  : 127.0.0.1:4242`,
  },
  dvc: {
    label: 'DVC Dataset & Pipeline',
    filename: 'dvc.yaml',
    code: `# Initialize DVC and track external datasets without committing raw scans to Git
dvc init
dvc add data/raw/nih_chestxray14
dvc add data/raw/msd_task06_lung

# Reproduce preprocessing, training, and evaluation stages deterministically
dvc repro

# Inspect tracked evaluation metrics
dvc metrics show`,
  },
  train: {
    label: 'PyTorch & MONAI Training',
    filename: 'backend/app/ml/',
    code: `# Chest X-ray 14-label DenseNet-121 (Patient-level split, BCEWithLogitsLoss)
python -m app.ml.classification.train \\
  --data-dir ./data/raw/nih_chestxray14 \\
  --epochs 20 --batch-size 32 \\
  --output ./models/densenet121_nih14.pth

python -m app.ml.classification.evaluate \\
  --data-dir ./data/raw/nih_chestxray14 \\
  --checkpoint ./models/densenet121_nih14.pth \\
  --report ./reports/classification_metrics.json

# Thoracic CT 3D U-Net (MONAI, MSD Task06_Lung, DiceCELoss)
python -m app.ml.segmentation.train \\
  --data-dir ./data/raw/msd_task06_lung \\
  --epochs 100 --output ./models/monai_unet3d_task06_lung.pth`,
  },
};

export const PipelineDocsView: React.FC = () => {
  const [models, setModels] = useState<ModelDescriptor[]>([]);
  const [metrics, setMetrics] = useState<ModelMetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<CommandTab>('docker');
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    let active = true;
    async function fetchDocsData() {
      setLoading(true);
      setError(null);
      try {
        const [modelsRes, metricsRes] = await Promise.all([
          fetch('/api/models'),
          fetch('/api/metrics'),
        ]);
        if (!modelsRes.ok || !metricsRes.ok) {
          throw new Error('Failed to load model registry and metrics from backend API.');
        }
        const modelsJson = (await modelsRes.json()) as { models: ModelDescriptor[] };
        const metricsJson = (await metricsRes.json()) as ModelMetricsResponse;
        if (active) {
          setModels(modelsJson.models);
          setMetrics(metricsJson);
        }
      } catch (err: unknown) {
        if (active) {
          setError(err instanceof Error ? err.message : 'Unable to fetch backend metadata.');
        }
      } finally {
        if (active) setLoading(false);
      }
    }
    fetchDocsData();
    return () => {
      active = false;
    };
  }, []);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(COMMAND_SNIPPETS[activeTab].code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 space-y-10">
      {/* Prominent Educational & Responsible AI Disclaimer */}
      <section className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-5">
        <h1 className="text-xl font-semibold text-amber-200">
          FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY — NOT FOR CLINICAL USE
        </h1>
        <p className="mt-1.5 text-sm text-amber-100/85 leading-relaxed max-w-4xl">
          AI-generated outputs are experimental and must not be used to diagnose, treat, or make
          medical decisions. MedVision AI enforces explicit modality gating, separates illustrative
          Demo Mode studies from real model inference, and refuses to fabricate predictions when
          trained checkpoints are absent.
        </p>
      </section>

      {/* Registered Models (Live from GET /api/models) */}
      <section className="space-y-4">
        <div>
          <h2 className="text-lg font-semibold text-slate-100">
            01. Registered ML Pipelines &amp; Supported Modalities
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Each model explicitly declares its supported modality, anatomy, input format, dataset, and live checkpoint status.
          </p>
        </div>

        {loading ? (
          <div className="text-xs font-mono text-slate-400 py-6">Loading model registry...</div>
        ) : error ? (
          <div className="text-xs text-rose-300 bg-rose-500/10 border border-rose-500/30 rounded p-4">
            {error}
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            {models.map((m) => (
              <div
                key={m.id}
                className="bg-[#0B0F17] border border-slate-800/80 rounded-lg p-5 space-y-4"
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <h3 className="text-base font-semibold text-slate-100">{m.name}</h3>
                    <div className="text-xs font-mono text-slate-400 mt-0.5">
                      Version {m.version} · {m.task.toUpperCase()}
                    </div>
                  </div>
                  <div
                    className={`text-xs font-mono whitespace-nowrap ${
                      m.weightsLoaded ? 'text-emerald-400' : 'text-amber-400'
                    }`}
                  >
                    {m.weightsLoaded ? '● Weights Loaded' : '○ Weights Not Loaded'}
                  </div>
                </div>

                <dl className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs border-t border-slate-800/70 pt-3">
                  <div>
                    <dt className="text-slate-500">Supported Modalities</dt>
                    <dd className="font-mono text-cyan-300 mt-0.5">
                      {m.supportedModalities.join(' / ')}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-slate-500">Supported Anatomy</dt>
                    <dd className="font-mono text-slate-200 mt-0.5">
                      {m.supportedAnatomy.join(', ')}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-slate-500">Dataset</dt>
                    <dd className="text-slate-200 mt-0.5">{m.dataset}</dd>
                  </div>
                  <div>
                    <dt className="text-slate-500">Explainability</dt>
                    <dd className="text-slate-200 mt-0.5">{m.explainabilityMethod}</dd>
                  </div>
                  <div className="sm:col-span-2">
                    <dt className="text-slate-500">Architecture &amp; Input Format</dt>
                    <dd className="text-slate-300 mt-0.5">
                      {m.architecture} — {m.inputFormat}
                    </dd>
                  </div>
                </dl>

                <div className="text-xs text-slate-400 bg-[#07090E] border border-slate-800 rounded p-3">
                  <span className="font-mono text-slate-300">Checkpoint: </span>
                  {m.statusMessage}
                </div>

                <ul className="text-xs text-slate-400 space-y-1 list-disc pl-4">
                  {m.limitations.map((lim) => (
                    <li key={lim}>{lim}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Honest Evaluation Metrics (Live from GET /api/metrics) */}
      <section className="space-y-4">
        <div>
          <h2 className="text-lg font-semibold text-slate-100">
            02. Local Model Evaluation Metrics (DVC Tracked)
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            {metrics?.note ||
              'Metrics are read strictly from generated evaluation reports and are never fabricated.'}
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <div className="bg-[#0B0F17] border border-slate-800/80 rounded-lg p-5 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-slate-100">
                Classification — NIH ChestX-ray14 (DenseNet-121)
              </h3>
              <span className="text-xs font-mono text-amber-400">
                {metrics?.classification.status === 'evaluated'
                  ? '● Evaluated'
                  : '○ Not Evaluated Locally'}
              </span>
            </div>
            <div className="grid grid-cols-4 gap-3 text-xs font-mono tabular-nums pt-2 border-t border-slate-800/70">
              <div>
                <div className="text-slate-500">Macro AUROC</div>
                <div className="text-base text-slate-200 mt-0.5">
                  {metrics?.classification.macroAuroc !== null &&
                  metrics?.classification.macroAuroc !== undefined
                    ? metrics.classification.macroAuroc.toFixed(4)
                    : '—'}
                </div>
              </div>
              <div>
                <div className="text-slate-500">Precision</div>
                <div className="text-base text-slate-200 mt-0.5">
                  {metrics?.classification.precision !== null &&
                  metrics?.classification.precision !== undefined
                    ? metrics.classification.precision.toFixed(4)
                    : '—'}
                </div>
              </div>
              <div>
                <div className="text-slate-500">Recall</div>
                <div className="text-base text-slate-200 mt-0.5">
                  {metrics?.classification.recall !== null &&
                  metrics?.classification.recall !== undefined
                    ? metrics.classification.recall.toFixed(4)
                    : '—'}
                </div>
              </div>
              <div>
                <div className="text-slate-500">F1 Score</div>
                <div className="text-base text-slate-200 mt-0.5">
                  {metrics?.classification.f1Score !== null &&
                  metrics?.classification.f1Score !== undefined
                    ? metrics.classification.f1Score.toFixed(4)
                    : '—'}
                </div>
              </div>
            </div>
          </div>

          <div className="bg-[#0B0F17] border border-slate-800/80 rounded-lg p-5 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-slate-100">
                Segmentation — MSD Task06_Lung (MONAI 3D U-Net)
              </h3>
              <span className="text-xs font-mono text-amber-400">
                {metrics?.segmentation.status === 'evaluated'
                  ? '● Evaluated'
                  : '○ Not Evaluated Locally'}
              </span>
            </div>
            <div className="grid grid-cols-4 gap-3 text-xs font-mono tabular-nums pt-2 border-t border-slate-800/70">
              <div>
                <div className="text-slate-500">Mean Dice</div>
                <div className="text-base text-slate-200 mt-0.5">
                  {metrics?.segmentation.meanDice !== null &&
                  metrics?.segmentation.meanDice !== undefined
                    ? metrics.segmentation.meanDice.toFixed(4)
                    : '—'}
                </div>
              </div>
              <div>
                <div className="text-slate-500">Mean IoU</div>
                <div className="text-base text-slate-200 mt-0.5">
                  {metrics?.segmentation.meanIoU !== null &&
                  metrics?.segmentation.meanIoU !== undefined
                    ? metrics.segmentation.meanIoU.toFixed(4)
                    : '—'}
                </div>
              </div>
              <div>
                <div className="text-slate-500">Sensitivity</div>
                <div className="text-base text-slate-200 mt-0.5">
                  {metrics?.segmentation.sensitivity !== null &&
                  metrics?.segmentation.sensitivity !== undefined
                    ? metrics.segmentation.sensitivity.toFixed(4)
                    : '—'}
                </div>
              </div>
              <div>
                <div className="text-slate-500">Specificity</div>
                <div className="text-base text-slate-200 mt-0.5">
                  {metrics?.segmentation.specificity !== null &&
                  metrics?.segmentation.specificity !== undefined
                    ? metrics.segmentation.specificity.toFixed(4)
                    : '—'}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Architecture Flow */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-slate-100">
          03. End-to-End Architecture &amp; Infrastructure
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[
            {
              step: 'Stage 1',
              title: 'Orthanc DICOM PACS',
              detail:
                'Ingests DICOM C-STORE / REST studies, preserves StudyInstanceUID & SeriesInstanceUID, isolates admin API to internal Docker network.',
            },
            {
              step: 'Stage 2',
              title: 'pydicom & MONAI Preprocessing',
              detail:
                'Handles MONOCHROME1/2 inversion, Window/Level VOI LUT, 224×224 normalization for CXR, and RAS resampling + HU clipping for CT volumes.',
            },
            {
              step: 'Stage 3',
              title: 'PyTorch + MONAI + Grad-CAM',
              detail:
                'DenseNet-121 multi-label classification with real convolutional Grad-CAM hooks, plus MONAI 3D U-Net volumetric lung segmentation.',
            },
            {
              step: 'Stage 4',
              title: 'PostgreSQL Audit & React Viewer',
              detail:
                'Stores metadata and append-only UTC audit records in PostgreSQL while saving large scan/overlay artifacts to disk storage.',
            },
          ].map((stage, idx) => (
            <div
              key={stage.step}
              className="bg-[#0B0F17] border border-slate-800/80 rounded-lg p-4 flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between text-xs font-mono text-cyan-400">
                  <span>{stage.step}</span>
                  {idx < 3 && <ArrowRight className="w-3.5 h-3.5 text-slate-600 hidden md:block" />}
                </div>
                <h3 className="mt-2 text-sm font-semibold text-slate-100">{stage.title}</h3>
                <p className="mt-1.5 text-xs text-slate-400 leading-relaxed">{stage.detail}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Reproducible Commands */}
      <section className="bg-[#0B0F17] border border-slate-800/80 rounded-lg overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-3.5 border-b border-slate-800">
          <div className="flex items-center gap-1 p-1 bg-[#07090E] border border-slate-800 rounded">
            {(Object.keys(COMMAND_SNIPPETS) as CommandTab[]).map((tabKey) => (
              <button
                key={tabKey}
                type="button"
                onClick={() => setActiveTab(tabKey)}
                className={`px-3 py-1.5 text-xs font-medium rounded transition-colors cursor-pointer whitespace-nowrap ${
                  activeTab === tabKey
                    ? 'bg-slate-800 text-cyan-300'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {COMMAND_SNIPPETS[tabKey].label}
              </button>
            ))}
          </div>

          <button
            type="button"
            onClick={handleCopy}
            className="px-3 py-1.5 rounded bg-[#07090E] border border-slate-800 hover:border-cyan-500 text-xs font-mono text-slate-300 flex items-center gap-1.5 transition-colors cursor-pointer"
          >
            {copied ? (
              <Check className="w-3.5 h-3.5 text-emerald-400" />
            ) : (
              <Copy className="w-3.5 h-3.5 text-slate-400" />
            )}
            {copied ? 'Copied' : `Copy ${COMMAND_SNIPPETS[activeTab].filename}`}
          </button>
        </div>

        <pre className="p-5 text-xs font-mono text-slate-300 overflow-x-auto leading-relaxed bg-[#05070B]">
          <code>{COMMAND_SNIPPETS[activeTab].code}</code>
        </pre>
      </section>

      {/* Limitations & Bias Governance */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-slate-100">
          04. Bias, Domain Shift &amp; Explainability Limitations
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-[#0B0F17] border border-slate-800/80 rounded-lg p-5 space-y-2">
            <h3 className="text-sm font-semibold text-slate-100">
              Dataset Label Noise &amp; Shortcut Learning
            </h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              NIH ChestX-ray14 labels are text-mined from radiology reports and contain substantial
              label noise. CNNs can exploit chest tubes, portable AP markers, or hospital watermarks
              rather than true parenchymal pathology.
            </p>
          </div>
          <div className="bg-[#0B0F17] border border-slate-800/80 rounded-lg p-5 space-y-2">
            <h3 className="text-sm font-semibold text-slate-100">
              Population Bias &amp; Scanner Domain Shift
            </h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Public cohorts under-represent pediatric groups, underrepresented demographics, and
              varied scanner vendors or reconstruction kernels. Independent multi-center clinical
              validation is mandatory before any real-world deployment.
            </p>
          </div>
          <div className="bg-[#0B0F17] border border-slate-800/80 rounded-lg p-5 space-y-2">
            <h3 className="text-sm font-semibold text-slate-100">
              Limits of Grad-CAM &amp; Segmentation Masks
            </h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Grad-CAM produces low-resolution spatial sensitivity maps that indicate where a model
              looked — not why a pathology exists. False positives and false negatives remain common
              without radiologist oversight.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};
