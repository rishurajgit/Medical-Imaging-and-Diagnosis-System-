import React, { useMemo, useState } from 'react';
import { Download, Search } from 'lucide-react';
import type { AuditEventType, AuditLogEntry, ImagingStudy } from '../types/imaging';

interface AuditLogViewProps {
  auditLog: AuditLogEntry[];
  studies: ImagingStudy[];
  onOpenStudy: (studyId: string) => void;
}

const EVENT_DISPLAY: Record<AuditEventType, { label: string; colorClass: string }> = {
  approved: { label: '● Approved', colorClass: 'text-emerald-400' },
  rejected: { label: '✕ Rejected', colorClass: 'text-rose-400' },
  review_updated: { label: '▲ Review Updated', colorClass: 'text-amber-300' },
  study_uploaded: { label: '○ Study Uploaded', colorClass: 'text-cyan-300' },
  dicom_ingested: { label: '○ DICOM Ingested', colorClass: 'text-cyan-300' },
  inference_requested: { label: '○ Inference Req.', colorClass: 'text-slate-300' },
  inference_completed: { label: '● AI Completed', colorClass: 'text-cyan-400' },
  inference_blocked: { label: '▲ AI Deferred', colorClass: 'text-amber-400' },
  heatmap_generated: { label: '○ Heatmap Gen.', colorClass: 'text-cyan-300' },
};

export const AuditLogView: React.FC<AuditLogViewProps> = ({
  auditLog,
  studies,
  onOpenStudy,
}) => {
  const [query, setQuery] = useState('');
  const [decisionFilter, setDecisionFilter] = useState<
    'all' | 'reviews' | 'ingestion' | 'approved' | 'rejected'
  >('all');
  const [scopeFilter, setScopeFilter] = useState<'all' | 'real' | 'demo'>('all');

  const summary = useMemo(() => {
    const approved = studies.filter((s) => s.status === 'approved').length;
    const rejected = studies.filter((s) => s.status === 'rejected').length;
    const pending = studies.filter((s) => s.status === 'pending').length;
    return {
      events: auditLog.length,
      approved,
      rejected,
      pending,
    };
  }, [auditLog.length, studies]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return auditLog.filter((entry) => {
      if (scopeFilter === 'real' && entry.isDemo) return false;
      if (scopeFilter === 'demo' && !entry.isDemo) return false;
      if (decisionFilter === 'approved' && entry.decision !== 'approved') return false;
      if (decisionFilter === 'rejected' && entry.decision !== 'rejected') return false;
      if (
        decisionFilter === 'reviews' &&
        !['approved', 'rejected', 'review_updated'].includes(entry.decision)
      ) {
        return false;
      }
      if (
        decisionFilter === 'ingestion' &&
        !['study_uploaded', 'dicom_ingested', 'inference_completed', 'inference_blocked', 'heatmap_generated'].includes(
          entry.decision
        )
      ) {
        return false;
      }
      if (!q) return true;
      return (
        entry.accessionNumber.toLowerCase().includes(q) ||
        entry.studyId.toLowerCase().includes(q) ||
        entry.primaryFinding.toLowerCase().includes(q) ||
        entry.modelName.toLowerCase().includes(q) ||
        entry.reviewer.toLowerCase().includes(q) ||
        entry.notes.toLowerCase().includes(q)
      );
    });
  }, [auditLog, query, decisionFilter, scopeFilter]);

  const handleExportCsv = () => {
    const headers = [
      'Timestamp (UTC)',
      'Audit ID',
      'Study ID',
      'Accession',
      'Scope',
      'Modality',
      'Model',
      'Model Version',
      'Inference Result',
      'Confidence',
      'Event / Decision',
      'Actor / Reviewer',
      'Notes',
    ];
    const escape = (val: string | number) => `"${String(val).replace(/"/g, '""')}"`;
    const rows = filtered.map((e) =>
      [
        e.timestamp,
        e.id,
        e.studyId,
        e.accessionNumber,
        e.isDemo ? 'DEMO' : 'UPLOADED',
        e.modality,
        e.modelName,
        e.modelVersion,
        e.primaryFinding,
        e.confidence !== null ? `${(e.confidence * 100).toFixed(1)}%` : 'N/A',
        e.decision.toUpperCase(),
        e.reviewer,
        e.notes,
      ]
        .map(escape)
        .join(',')
    );
    const csv = [headers.join(','), ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `medvision-audit-log-${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold text-slate-100">
            Immutable Radiologist Review &amp; Pipeline Audit Log
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Append-only UTC record of study uploads, DICOM ingestion, model inference events, and radiologist sign-off decisions.
          </p>
        </div>

        <button
          type="button"
          onClick={handleExportCsv}
          className="px-4 py-2 rounded bg-[#0B0F17] border border-slate-700 hover:border-cyan-500 text-xs font-medium text-slate-200 flex items-center gap-2 transition-colors cursor-pointer whitespace-nowrap shrink-0"
        >
          <Download className="w-4 h-4 text-cyan-400" />
          Download CSV
        </button>
      </div>

      {/* Summary KPI Strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-[#0B0F17] border border-slate-800/80 rounded-lg p-4">
          <div className="text-xs text-slate-400">Append-Only Audit Events</div>
          <div className="mt-1 text-2xl font-mono font-semibold text-slate-100 tabular-nums">
            {summary.events}
          </div>
        </div>
        <div className="bg-[#0B0F17] border border-slate-800/80 rounded-lg p-4">
          <div className="text-xs text-slate-400">Approved Studies</div>
          <div className="mt-1 text-2xl font-mono font-semibold text-emerald-400 tabular-nums">
            {summary.approved}
          </div>
        </div>
        <div className="bg-[#0B0F17] border border-slate-800/80 rounded-lg p-4">
          <div className="text-xs text-slate-400">Rejected / Overridden</div>
          <div className="mt-1 text-2xl font-mono font-semibold text-rose-400 tabular-nums">
            {summary.rejected}
          </div>
        </div>
        <div className="bg-[#0B0F17] border border-slate-800/80 rounded-lg p-4">
          <div className="text-xs text-slate-400">Awaiting Sign-Off</div>
          <div className="mt-1 text-2xl font-mono font-semibold text-amber-400 tabular-nums">
            {summary.pending}
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-[#0B0F17] border border-slate-800/80 rounded-lg p-3">
        <div className="relative flex-1 min-w-[220px]">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter by study ID, accession, model, reviewer, or notes..."
            className="w-full bg-[#07090E] border border-slate-800 rounded pl-9 pr-3 py-1.5 text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-cyan-500"
          />
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          <div className="flex items-center gap-1 p-1 bg-[#07090E] border border-slate-800 rounded">
            {(
              [
                { id: 'all', label: 'All Events' },
                { id: 'reviews', label: 'Reviews' },
                { id: 'ingestion', label: 'Pipeline' },
                { id: 'approved', label: 'Approved' },
                { id: 'rejected', label: 'Rejected' },
              ] as const
            ).map((d) => (
              <button
                key={d.id}
                type="button"
                onClick={() => setDecisionFilter(d.id)}
                className={`px-2.5 py-1 text-xs font-medium rounded transition-colors cursor-pointer whitespace-nowrap ${
                  decisionFilter === d.id
                    ? 'bg-slate-800 text-cyan-300'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {d.label}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-1 p-1 bg-[#07090E] border border-slate-800 rounded">
            {(
              [
                { id: 'all', label: 'All Data' },
                { id: 'real', label: 'Uploaded Only' },
                { id: 'demo', label: 'Demo Only' },
              ] as const
            ).map((s) => (
              <button
                key={s.id}
                type="button"
                onClick={() => setScopeFilter(s.id)}
                className={`px-2.5 py-1 text-xs font-medium rounded transition-colors cursor-pointer whitespace-nowrap ${
                  scopeFilter === s.id
                    ? 'bg-slate-800 text-amber-300'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* High-Density Audit Table */}
      <div className="bg-[#0B0F17] border border-slate-800/80 rounded-lg overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-[11px] font-mono text-slate-400 bg-[#07090E]">
                <th className="py-3 px-4 font-medium">Timestamp (UTC)</th>
                <th className="py-3 px-4 font-medium">Study / Accession</th>
                <th className="py-3 px-4 font-medium">Model &amp; Version</th>
                <th className="py-3 px-4 font-medium">Inference Result</th>
                <th className="py-3 px-4 font-medium text-right">Score</th>
                <th className="py-3 px-4 font-medium">Event / Decision</th>
                <th className="py-3 px-4 font-medium">Reviewer / Actor</th>
                <th className="py-3 px-4 font-medium">Notes</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-xs">
              {filtered.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-10 text-center text-slate-400">
                    No audit log entries match the active filters.
                  </td>
                </tr>
              ) : (
                filtered.map((entry) => {
                  const ev = EVENT_DISPLAY[entry.decision] ?? {
                    label: entry.decision,
                    colorClass: 'text-slate-300',
                  };
                  return (
                    <tr
                      key={entry.id}
                      onClick={() => onOpenStudy(entry.studyId)}
                      className="hover:bg-slate-900/60 transition-colors cursor-pointer"
                    >
                      <td className="py-3 px-4 font-mono text-slate-400 whitespace-nowrap tabular-nums">
                        {entry.timestamp.replace('T', ' ').slice(0, 19)}
                      </td>
                      <td className="py-3 px-4 font-mono whitespace-nowrap">
                        <div className="text-cyan-300">{entry.accessionNumber}</div>
                        <div className="text-[10px] text-slate-500">
                          {entry.modality} · {entry.isDemo ? 'DEMO' : 'UPLOADED'}
                        </div>
                      </td>
                      <td className="py-3 px-4 whitespace-nowrap">
                        <div className="text-slate-200 max-w-[180px] truncate">{entry.modelName}</div>
                        <div className="text-[10px] font-mono text-slate-500">{entry.modelVersion}</div>
                      </td>
                      <td className="py-3 px-4 text-slate-100 font-medium max-w-[210px] truncate">
                        {entry.primaryFinding}
                      </td>
                      <td className="py-3 px-4 font-mono text-slate-200 text-right whitespace-nowrap tabular-nums">
                        {entry.confidence !== null ? `${(entry.confidence * 100).toFixed(1)}%` : '—'}
                      </td>
                      <td className={`py-3 px-4 whitespace-nowrap font-medium ${ev.colorClass}`}>
                        {ev.label}
                      </td>
                      <td className="py-3 px-4 text-slate-300 whitespace-nowrap">{entry.reviewer}</td>
                      <td
                        className="py-3 px-4 text-slate-400 max-w-[260px] truncate"
                        title={entry.notes}
                      >
                        {entry.notes}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
