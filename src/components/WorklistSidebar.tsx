import React, { useMemo, useState } from 'react';
import { Search } from 'lucide-react';
import type { ImagingStudy, Modality, ProcessingStatus, ReviewStatus } from '../types/imaging';

interface WorklistSidebarProps {
  studies: ImagingStudy[];
  selectedId: string;
  showDemoStudies: boolean;
  onToggleDemoStudies: (show: boolean) => void;
  onSelect: (id: string) => void;
}

const STATUS_DISPLAY: Record<ReviewStatus, { glyph: string; label: string; colorClass: string }> = {
  pending: { glyph: '○', label: 'Pending', colorClass: 'text-amber-400' },
  approved: { glyph: '●', label: 'Approved', colorClass: 'text-emerald-400' },
  rejected: { glyph: '✕', label: 'Rejected', colorClass: 'text-rose-400' },
};

const PROCESSING_LABEL: Record<ProcessingStatus, { text: string; tone: string }> = {
  demo_illustrative: { text: 'Illustrative Demo', tone: 'text-amber-300/90' },
  completed: { text: 'AI Completed', tone: 'text-cyan-300' },
  weights_unavailable: { text: 'Weights Missing', tone: 'text-slate-400' },
  unsupported_modality: { text: 'Unsupported Modality', tone: 'text-rose-300' },
  failed: { text: 'Inference Error', tone: 'text-rose-400' },
  queued: { text: 'Queued', tone: 'text-slate-400' },
};

export const WorklistSidebar: React.FC<WorklistSidebarProps> = ({
  studies,
  selectedId,
  showDemoStudies,
  onToggleDemoStudies,
  onSelect,
}) => {
  const [query, setQuery] = useState('');
  const [modalityFilter, setModalityFilter] = useState<'ALL' | 'CXR' | 'CT' | 'MR'>('ALL');
  const [statusFilter, setStatusFilter] = useState<'ALL' | ReviewStatus>('ALL');

  const visiblePool = useMemo(
    () => (showDemoStudies ? studies : studies.filter((s) => !s.isDemo)),
    [studies, showDemoStudies]
  );

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return visiblePool.filter((s) => {
      if (modalityFilter === 'CXR' && s.modality !== 'CR' && s.modality !== 'DX') return false;
      if (modalityFilter === 'CT' && s.modality !== 'CT') return false;
      if (modalityFilter === 'MR' && s.modality !== 'MR') return false;
      if (statusFilter !== 'ALL' && s.status !== statusFilter) return false;
      if (!q) return true;
      return (
        s.accessionNumber.toLowerCase().includes(q) ||
        s.primaryFinding.toLowerCase().includes(q) ||
        s.dicom.patientId.toLowerCase().includes(q) ||
        s.bodyPart.toLowerCase().includes(q) ||
        s.dicom.datasetSource.toLowerCase().includes(q)
      );
    });
  }, [visiblePool, query, modalityFilter, statusFilter]);

  const counts = useMemo(
    () => ({
      total: visiblePool.length,
      pending: visiblePool.filter((s) => s.status === 'pending').length,
      uploadedCount: studies.filter((s) => !s.isDemo).length,
    }),
    [visiblePool, studies]
  );

  const modalityLabel = (m: Modality) => (m === 'CR' || m === 'DX' ? `${m}/X-Ray` : m);

  return (
    <aside className="w-full lg:w-[300px] xl:w-[320px] shrink-0 bg-[#0B0F17] border-b lg:border-b-0 lg:border-r border-slate-800/80 flex flex-col max-h-[380px] lg:max-h-none">
      {/* Header & Controls */}
      <div className="p-4 border-b border-slate-800/80 space-y-3">
        <div className="flex items-baseline justify-between">
          <h2 className="text-sm font-semibold text-slate-100">Study Worklist</h2>
          <span className="font-mono text-xs text-slate-400 tabular-nums">
            {counts.pending} pending · {counts.total} shown
          </span>
        </div>

        {/* Demo Mode Separation Toggle */}
        <div className="flex items-center justify-between text-xs py-1.5 px-2.5 bg-[#07090E] border border-slate-800 rounded">
          <span className="text-slate-300">Show Demo Mode Studies</span>
          <button
            type="button"
            role="switch"
            aria-checked={showDemoStudies}
            onClick={() => onToggleDemoStudies(!showDemoStudies)}
            className={`px-2 py-0.5 rounded font-mono text-[11px] transition-colors cursor-pointer ${
              showDemoStudies
                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                : 'bg-slate-800 text-slate-400 border border-slate-700'
            }`}
          >
            {showDemoStudies ? 'ON' : 'OFF'}
          </button>
        </div>

        <div className="relative">
          <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search accession, patient ID, anatomy..."
            className="w-full bg-[#07090E] border border-slate-800 rounded-md pl-8 pr-3 py-1.5 text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-cyan-500"
          />
        </div>

        {/* Modality Segmented Filter */}
        <div className="grid grid-cols-4 gap-1 p-1 bg-[#07090E] border border-slate-800 rounded-md">
          {(['ALL', 'CXR', 'CT', 'MR'] as const).map((mod) => (
            <button
              key={mod}
              type="button"
              onClick={() => setModalityFilter(mod)}
              className={`py-1 text-xs font-medium rounded transition-colors cursor-pointer whitespace-nowrap ${
                modalityFilter === mod
                  ? 'bg-slate-800 text-cyan-300'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {mod === 'ALL' ? 'All' : mod === 'CXR' ? 'X-Ray' : mod}
            </button>
          ))}
        </div>

        {/* Review Status Filter */}
        <div className="flex items-center justify-between text-xs gap-1">
          {(['ALL', 'pending', 'approved', 'rejected'] as const).map((st) => (
            <button
              key={st}
              type="button"
              onClick={() => setStatusFilter(st)}
              className={`px-2 py-1 rounded transition-colors cursor-pointer capitalize whitespace-nowrap ${
                statusFilter === st
                  ? 'text-cyan-300 bg-cyan-500/10'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {st === 'ALL' ? 'All' : st}
            </button>
          ))}
        </div>
      </div>

      {/* Scrollable Worklist Rows */}
      <div className="flex-1 overflow-y-auto divide-y divide-slate-800/60">
        {filtered.length === 0 ? (
          <div className="p-6 text-center text-xs text-slate-400 space-y-1.5">
            <p>No studies match the current filters.</p>
            {!showDemoStudies && counts.uploadedCount === 0 && (
              <p className="text-slate-500">
                Upload a DICOM or X-ray scan, or enable Demo Mode studies above.
              </p>
            )}
          </div>
        ) : (
          filtered.map((study) => {
            const isSelected = study.id === selectedId;
            const statusInfo = STATUS_DISPLAY[study.status];
            const procInfo = PROCESSING_LABEL[study.processingStatus];
            return (
              <button
                key={study.id}
                type="button"
                onClick={() => onSelect(study.id)}
                className={`w-full text-left px-4 py-3.5 transition-colors cursor-pointer block ${
                  isSelected ? 'bg-cyan-500/10' : 'hover:bg-slate-900/60'
                }`}
              >
                <div className="flex items-center justify-between gap-2 text-[11px] font-mono">
                  <span className={isSelected ? 'text-cyan-300 font-medium' : 'text-slate-300'}>
                    {study.accessionNumber}
                  </span>
                  <span className={`${statusInfo.colorClass} shrink-0`}>
                    {statusInfo.glyph} {statusInfo.label}
                  </span>
                </div>

                <div className="mt-1 text-xs font-mono text-slate-400 flex items-center gap-1.5 truncate">
                  <span>{study.dicom.patientId}</span>
                  <span aria-hidden="true">·</span>
                  <span>{modalityLabel(study.modality)}</span>
                  <span aria-hidden="true">·</span>
                  <span>{study.bodyPart}</span>
                </div>

                <div className="mt-1.5 text-sm font-medium text-slate-100 leading-snug line-clamp-2">
                  {study.primaryFinding}
                </div>

                <div className="mt-1.5 flex items-center justify-between text-[11px]">
                  <span className={`${procInfo.tone} truncate pr-2`}>{procInfo.text}</span>
                  <span className="font-mono text-slate-400 shrink-0 tabular-nums">
                    {study.confidence !== null
                      ? `${(study.confidence * 100).toFixed(1)}%`
                      : study.dicom.studyDate}
                  </span>
                </div>
              </button>
            );
          })
        )}
      </div>
    </aside>
  );
};
