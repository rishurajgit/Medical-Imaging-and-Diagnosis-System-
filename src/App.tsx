import { useCallback, useEffect, useMemo, useState } from 'react';
import { Upload } from 'lucide-react';
import { AuditLogView } from './components/AuditLogView';
import { DicomViewport } from './components/DicomViewport';
import { DiagnosisPanel } from './components/DiagnosisPanel';
import { PipelineDocsView } from './components/PipelineDocsView';
import { UploadScanModal } from './components/UploadScanModal';
import { WorklistSidebar } from './components/WorklistSidebar';
import type { AuditLogEntry, ImagingStudy, Modality, OverlayMode } from './types/imaging';

type ActiveTab = 'workstation' | 'audit' | 'pipeline';

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('workstation');
  const [studies, setStudies] = useState<ImagingStudy[]>([]);
  const [auditLog, setAuditLog] = useState<AuditLogEntry[]>([]);
  const [selectedId, setSelectedId] = useState<string>('');
  const [showDemoStudies, setShowDemoStudies] = useState<boolean>(true);
  const [overlayMode, setOverlayMode] = useState<OverlayMode>('gradcam');
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadStudies = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/studies');
      if (!res.ok) {
        throw new Error(`Failed to load studies (${res.status})`);
      }
      const data = (await res.json()) as {
        studies: ImagingStudy[];
        auditLog: AuditLogEntry[];
      };
      setStudies(data.studies);
      setAuditLog(data.auditLog);
      setSelectedId((prev) =>
        prev && data.studies.some((s) => s.id === prev)
          ? prev
          : data.studies[0]?.id ?? ''
      );
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unable to reach PACS backend.';
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadStudies();
  }, [loadStudies]);

  const visibleStudies = useMemo(
    () => (showDemoStudies ? studies : studies.filter((s) => !s.isDemo)),
    [studies, showDemoStudies]
  );

  const selectedStudy = useMemo(
    () => visibleStudies.find((s) => s.id === selectedId) ?? visibleStudies[0] ?? null,
    [visibleStudies, selectedId]
  );

  const handleSubmitReview = async (
    studyId: string,
    decision: 'approved' | 'rejected',
    reviewer: string,
    notes: string
  ) => {
    const res = await fetch(`/api/studies/${encodeURIComponent(studyId)}/review`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ decision, reviewer, notes }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data?.error || 'Failed to save review.');
    }
    setStudies((prev) => prev.map((s) => (s.id === studyId ? data.study : s)));
    setAuditLog((prev) => [data.auditEntry, ...prev]);
  };

  const handleReanalyze = async (studyId: string): Promise<string> => {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ studyId }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data?.error || 'Failed to invoke analysis endpoint.');
    }
    if (data.study) {
      setStudies((prev) => prev.map((s) => (s.id === studyId ? data.study : s)));
    }
    if (data.auditEntry) {
      setAuditLog((prev) => [data.auditEntry, ...prev]);
    }
    return data.message || 'Inference pipeline checked.';
  };

  const handleUploadComplete = async (payload: {
    fileBase64: string;
    fileName: string;
    mimeType: string;
    title: string;
    modality: Modality;
    bodyPart: string;
    patientId: string;
  }) => {
    const res = await fetch('/api/upload', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data?.error || 'Upload failed.');
    }
    setStudies((prev) => [data.study, ...prev]);
    if (Array.isArray(data.auditEntries) && data.auditEntries.length > 0) {
      setAuditLog((prev) => [...data.auditEntries, ...prev]);
    }
    setSelectedId(data.study.id);
    setOverlayMode('off');
    setActiveTab('workstation');
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#07090E] text-slate-100">
      {/* 3-Zone Top Navigation Bar */}
      <header className="h-14 px-6 bg-[#0B0F17] border-b border-slate-800/80 flex items-center justify-between gap-8 shrink-0">
        <button
          type="button"
          onClick={() => setActiveTab('workstation')}
          className="text-base font-display font-bold tracking-tight text-slate-100 whitespace-nowrap shrink-0 cursor-pointer"
        >
          MedVision PACS
        </button>

        <nav className="flex items-center gap-7 text-sm font-medium">
          <button
            type="button"
            onClick={() => setActiveTab('workstation')}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 cursor-pointer border-b-2 ${
              activeTab === 'workstation'
                ? 'text-slate-100 border-cyan-400'
                : 'text-slate-400 hover:text-slate-200 border-transparent'
            }`}
          >
            Workstation
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('audit')}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 cursor-pointer border-b-2 ${
              activeTab === 'audit'
                ? 'text-slate-100 border-cyan-400'
                : 'text-slate-400 hover:text-slate-200 border-transparent'
            }`}
          >
            Audit Log
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('pipeline')}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 cursor-pointer border-b-2 ${
              activeTab === 'pipeline'
                ? 'text-slate-100 border-cyan-400'
                : 'text-slate-400 hover:text-slate-200 border-transparent'
            }`}
          >
            Models &amp; Docs
          </button>
        </nav>

        <div className="shrink-0">
          <button
            type="button"
            onClick={() => setIsUploadOpen(true)}
            className="px-4 py-2 rounded bg-cyan-600 hover:bg-cyan-500 text-xs font-semibold text-white flex items-center gap-2 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <Upload className="w-3.5 h-3.5" />
            Upload Scan
          </button>
        </div>
      </header>

      {/* Mandatory Prominent Non-Clinical Educational Disclaimer */}
      <div className="px-6 py-2 bg-amber-500/10 border-b border-amber-500/25 text-xs text-amber-200 flex flex-wrap items-center justify-between gap-2 shrink-0">
        <span>
          FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY. NOT FOR CLINICAL USE. AI-generated outputs are
          experimental and must not be used to diagnose, treat, or make medical decisions.
        </span>
        <span className="font-mono text-[11px] text-amber-300/90 whitespace-nowrap">
          {showDemoStudies ? 'Demo Mode Active' : 'Uploaded Studies Only'}
        </span>
      </div>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col min-h-0">
        {loading ? (
          <div className="flex-1 flex items-center justify-center p-10">
            <div className="flex flex-col items-center gap-3 text-slate-400 text-xs font-mono">
              <div className="w-7 h-7 border-2 border-slate-700 border-t-cyan-400 rounded-full animate-spin" />
              <span>Initializing PACS study worklist...</span>
            </div>
          </div>
        ) : error ? (
          <div className="flex-1 flex items-center justify-center p-8">
            <div className="max-w-md bg-[#0B0F17] border border-slate-800 rounded-lg p-6 text-center space-y-3">
              <h2 className="text-base font-semibold text-slate-100">Unable to load studies</h2>
              <p className="text-xs text-slate-400">{error}</p>
              <button
                type="button"
                onClick={loadStudies}
                className="px-4 py-2 rounded bg-cyan-600 hover:bg-cyan-500 text-xs font-semibold text-white cursor-pointer"
              >
                Retry Connection
              </button>
            </div>
          </div>
        ) : activeTab === 'workstation' ? (
          <div className="flex-1 flex flex-col lg:flex-row min-h-0 lg:h-[calc(100vh-56px-36px-32px)]">
            <WorklistSidebar
              studies={studies}
              selectedId={selectedStudy?.id ?? ''}
              showDemoStudies={showDemoStudies}
              onToggleDemoStudies={setShowDemoStudies}
              onSelect={(id) => setSelectedId(id)}
            />
            {selectedStudy ? (
              <>
                <div className="flex-1 min-w-0 min-h-[460px] lg:min-h-0">
                  <DicomViewport
                    study={selectedStudy}
                    overlayMode={overlayMode}
                    onOverlayModeChange={setOverlayMode}
                  />
                </div>
                <DiagnosisPanel
                  study={selectedStudy}
                  onSubmitReview={handleSubmitReview}
                  onReanalyze={handleReanalyze}
                />
              </>
            ) : (
              <div className="flex-1 flex items-center justify-center p-8 text-center">
                <div className="space-y-3 max-w-sm">
                  <p className="text-sm text-slate-300">
                    No studies currently shown with Demo Mode disabled.
                  </p>
                  <div className="flex items-center justify-center gap-3">
                    <button
                      type="button"
                      onClick={() => setIsUploadOpen(true)}
                      className="px-4 py-2 rounded bg-cyan-600 hover:bg-cyan-500 text-xs font-semibold text-white cursor-pointer"
                    >
                      Upload DICOM / X-Ray
                    </button>
                    <button
                      type="button"
                      onClick={() => setShowDemoStudies(true)}
                      className="px-4 py-2 rounded bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-200 cursor-pointer"
                    >
                      Show Demo Mode
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : activeTab === 'audit' ? (
          <div className="flex-1 overflow-y-auto">
            <AuditLogView
              auditLog={auditLog}
              studies={studies}
              onOpenStudy={(studyId) => {
                const target = studies.find((s) => s.id === studyId);
                if (target?.isDemo) setShowDemoStudies(true);
                setSelectedId(studyId);
                setActiveTab('workstation');
              }}
            />
          </div>
        ) : (
          <div className="flex-1 overflow-y-auto">
            <PipelineDocsView />
          </div>
        )}
      </main>

      {/* Quiet Footer */}
      <footer className="h-8 px-6 bg-[#07090E] border-t border-slate-800/70 flex items-center justify-between text-[11px] text-slate-500 shrink-0">
        <span>MedVision AI · Educational &amp; Research Medical Imaging Prototype</span>
        <span className="hidden sm:inline font-mono">
          NIH ChestX-ray14 (DenseNet-121) · MSD Task06_Lung (MONAI 3D U-Net) · Orthanc PACS
        </span>
      </footer>

      <UploadScanModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onUploadComplete={handleUploadComplete}
      />
    </div>
  );
}
