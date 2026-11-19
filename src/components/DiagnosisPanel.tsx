import React, { useEffect, useState } from 'react';
import { BookOpen, Check, ChevronDown, ChevronRight, RefreshCw, X } from 'lucide-react';
import type { ImagingStudy } from '../types/imaging';

interface DiagnosisPanelProps {
  study: ImagingStudy;
  onSubmitReview: (
    studyId: string,
    decision: 'approved' | 'rejected',
    reviewer: string,
    notes: string
  ) => Promise<void>;
  onReanalyze: (studyId: string) => Promise<string>;
}

export const DiagnosisPanel: React.FC<DiagnosisPanelProps> = ({
  study,
  onSubmitReview,
  onReanalyze,
}) => {
  const [reviewer, setReviewer] = useState('Dr. A. Mehta, MD');
  const [notes, setNotes] = useState(study.reviewerNotes ?? '');
  const [pendingConfirmation, setPendingConfirmation] = useState<'approved' | 'rejected' | null>(
    null
  );
  const [submitting, setSubmitting] = useState(false);
  const [reanalyzing, setReanalyzing] = useState(false);
  const [feedback, setFeedback] = useState<{ type: 'success' | 'error' | 'info'; message: string } | null>(
    null
  );
  const [showDicomTags, setShowDicomTags] = useState(false);

  // Optional non-diagnostic Gemini technical explainer state
  const [explaining, setExplaining] = useState(false);
  const [explanationText, setExplanationText] = useState<string | null>(null);
  const [explanationNotice, setExplanationNotice] = useState<string | null>(null);
  const [explanationError, setExplanationError] = useState<string | null>(null);

  useEffect(() => {
    setNotes(study.reviewerNotes ?? '');
    setFeedback(null);
    setPendingConfirmation(null);
    setExplanationText(null);
    setExplanationNotice(null);
    setExplanationError(null);
  }, [study.id, study.reviewerNotes]);

  const confirmReview = async () => {
    if (!pendingConfirmation) return;
    const cleanReviewer = reviewer.trim();
    if (!cleanReviewer) {
      setFeedback({ type: 'error', message: 'Please enter the reviewing clinician name or ID.' });
      setPendingConfirmation(null);
      return;
    }

    setSubmitting(true);
    setFeedback(null);
    try {
      await onSubmitReview(study.id, pendingConfirmation, cleanReviewer, notes);
      setFeedback({
        type: 'success',
        message:
          pendingConfirmation === 'approved'
            ? 'Study approved and appended to immutable UTC audit log.'
            : 'Study rejected and override appended to immutable UTC audit log.',
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to record radiologist review.';
      setFeedback({ type: 'error', message: msg });
    } finally {
      setSubmitting(false);
      setPendingConfirmation(null);
    }
  };

  const handleRunInferenceCheck = async () => {
    setReanalyzing(true);
    setFeedback(null);
    try {
      const msg = await onReanalyze(study.id);
      setFeedback({ type: 'info', message: msg });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Inference request failed.';
      setFeedback({ type: 'error', message: msg });
    } finally {
      setReanalyzing(false);
    }
  };

  const handleExplainOutput = async () => {
    setExplaining(true);
    setExplanationError(null);
    try {
      const res = await fetch('/api/explain-output', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ studyId: study.id }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data?.error || 'Could not generate educational explanation.');
      }
      setExplanationText(data.explanation);
      setExplanationNotice(data.nonDiagnosticNotice);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Explanation service error.';
      setExplanationError(msg);
    } finally {
      setExplaining(false);
    }
  };

  const confidencePct = study.confidence !== null ? (study.confidence * 100).toFixed(1) : null;

  return (
    <aside className="w-full lg:w-[360px] xl:w-[390px] shrink-0 bg-[#0B0F17] border-t lg:border-t-0 lg:border-l border-slate-800/80 flex flex-col overflow-y-auto">
      {/* Demo / Processing Status Notice */}
      {study.isDemo && (
        <div className="px-5 py-3 bg-amber-500/10 border-b border-amber-500/30 text-xs text-amber-200 leading-relaxed">
          <span className="font-semibold">Illustrative Demo Mode:</span> Example outputs for
          workflow demonstration only — not live PyTorch/MONAI model inference.
        </div>
      )}

      {!study.isDemo && study.processingStatus !== 'completed' && (
        <div className="px-5 py-3 bg-slate-900 border-b border-slate-800 text-xs text-slate-300 space-y-2">
          <div className="font-semibold text-amber-300">
            {study.processingStatus === 'unsupported_modality'
              ? 'Unsupported Modality / Anatomy'
              : 'Trained Model Weights Not Loaded'}
          </div>
          <p className="text-slate-400 leading-relaxed">{study.processingMessage}</p>
          <button
            type="button"
            onClick={handleRunInferenceCheck}
            disabled={reanalyzing}
            className="px-3 py-1.5 rounded bg-[#07090E] border border-slate-700 hover:border-cyan-500 text-[11px] font-medium text-cyan-300 inline-flex items-center gap-1.5 transition-colors cursor-pointer"
          >
            <RefreshCw className={`w-3 h-3 ${reanalyzing ? 'animate-spin' : ''}`} />
            {reanalyzing ? 'Checking Pipeline...' : 'Re-check Inference Pipeline'}
          </button>
        </div>
      )}

      {/* Primary AI Prediction Header */}
      <div className="p-5 border-b border-slate-800/80">
        <div className="text-xs text-slate-400 flex items-center justify-between gap-2">
          <span className="truncate">{study.modelName}</span>
          <span className="font-mono text-slate-500 shrink-0">{study.modelVersion}</span>
        </div>

        <h2 className="mt-2 text-base font-semibold text-slate-100 leading-snug">
          {study.primaryFinding}
        </h2>

        <div className="mt-1 text-[11px] font-mono text-slate-500">
          {study.processedAt
            ? `Processed (UTC): ${study.processedAt.replace('T', ' ').slice(0, 19)}`
            : `Ingested (UTC): ${study.createdAt.replace('T', ' ').slice(0, 19)}`}
        </div>

        <div className="mt-4 flex items-baseline justify-between border-t border-slate-800/60 pt-3">
          <div>
            <div className="text-xs text-slate-400">
              {study.isDemo ? 'Illustrative Score' : 'Model Confidence'}
            </div>
            <div className="mt-0.5 flex items-baseline gap-1.5">
              {confidencePct !== null ? (
                <>
                  <span className="text-3xl font-mono font-semibold text-cyan-300 tabular-nums">
                    {confidencePct}
                  </span>
                  <span className="text-xs font-mono text-slate-400">%</span>
                </>
              ) : (
                <span className="text-lg font-mono text-slate-500">N/A</span>
              )}
            </div>
          </div>

          <div className="text-right">
            <div className="text-xs text-slate-400">Review Status</div>
            <div
              className={`mt-1 text-sm font-medium ${
                study.status === 'approved'
                  ? 'text-emerald-400'
                  : study.status === 'rejected'
                  ? 'text-rose-400'
                  : 'text-amber-400'
              }`}
            >
              {study.status === 'approved'
                ? '● Approved'
                : study.status === 'rejected'
                ? '✕ Rejected'
                : '○ Pending Review'}
            </div>
          </div>
        </div>
      </div>

      {/* Differential Class Probabilities */}
      <div className="p-5 border-b border-slate-800/80 space-y-3">
        <h3 className="text-xs font-semibold text-slate-300">
          Predicted Classes &amp; Probabilities
        </h3>
        {study.classProbabilities.length === 0 ? (
          <p className="text-xs text-slate-500 leading-relaxed">
            No class probability vector generated. Predictions are only populated by real trained
            model inference or clearly marked Demo Mode examples.
          </p>
        ) : (
          <div className="space-y-2.5">
            {study.classProbabilities.map((item, idx) => {
              const pct = Math.round(item.probability * 1000) / 10;
              const isTop = idx === 0;
              return (
                <div key={item.label} className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className={isTop ? 'text-slate-100 font-medium' : 'text-slate-300'}>
                      {item.label}
                    </span>
                    <span className="font-mono text-slate-300 tabular-nums">
                      {pct.toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-1.5 w-full bg-slate-800/90 rounded-sm overflow-hidden">
                    <div
                      className={`h-full transition-all duration-300 ${
                        isTop ? 'bg-cyan-400' : 'bg-slate-500'
                      }`}
                      style={{ width: `${Math.max(2, Math.min(100, pct))}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Explainability, Segmentation & Optional Non-Diagnostic Technical Explainer */}
      <div className="p-5 border-b border-slate-800/80 space-y-4">
        <div>
          <h3 className="text-xs font-semibold text-slate-300">
            Grad-CAM &amp; Explainability Notes
          </h3>
          <p className="mt-1.5 text-xs text-slate-300 leading-relaxed">{study.gradCamSummary}</p>
          <p className="mt-1.5 text-[11px] text-slate-500 leading-relaxed">
            Note: Grad-CAM highlights regions of convolutional sensitivity and does not establish
            clinical causation or pathological proof.
          </p>
        </div>

        {study.segmentation && (
          <div className="pt-2 border-t border-slate-800/60">
            <h3 className="text-xs font-semibold text-slate-300">Segmentation Summary</h3>
            <p className="mt-1 text-xs text-slate-400 leading-relaxed">
              {study.segmentation.note}
            </p>
            {study.segmentation.supported && study.segmentation.estimatedVolumeMm3 !== undefined && (
              <div className="mt-2 grid grid-cols-2 gap-2 text-[11px] font-mono bg-[#07090E] border border-slate-800/80 rounded p-2.5">
                <div>
                  <div className="text-slate-500">Voxels</div>
                  <div className="text-slate-200 tabular-nums">
                    {study.segmentation.voxelCount?.toLocaleString()}
                  </div>
                </div>
                <div>
                  <div className="text-slate-500">Est. Volume</div>
                  <div className="text-slate-200 tabular-nums">
                    {study.segmentation.estimatedVolumeMm3.toFixed(1)} mm³
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Optional Non-Diagnostic Gemini Educational Technical Explainer */}
        <div className="pt-2 border-t border-slate-800/60 space-y-2">
          <button
            type="button"
            onClick={handleExplainOutput}
            disabled={explaining}
            className="w-full px-3 py-2 rounded bg-[#07090E] border border-slate-800 hover:border-cyan-500/70 text-xs font-medium text-slate-200 flex items-center justify-center gap-2 transition-colors cursor-pointer disabled:opacity-50"
          >
            <BookOpen className="w-3.5 h-3.5 text-cyan-400" />
            {explaining ? 'Generating Educational Summary...' : 'Explain Technical Output (Non-Diagnostic)'}
          </button>

          {explanationError && (
            <div className="text-[11px] text-rose-300 bg-rose-500/10 border border-rose-500/30 rounded p-2.5">
              {explanationError}
            </div>
          )}

          {explanationText && (
            <div className="bg-[#07090E] border border-slate-800 rounded p-3 space-y-2">
              <div className="text-[11px] font-medium text-cyan-300">
                Educational Technical Explanation
              </div>
              <div className="text-xs text-slate-300 whitespace-pre-line leading-relaxed">
                {explanationText}
              </div>
              {explanationNotice && (
                <div className="text-[10px] text-slate-500 pt-1 border-t border-slate-800/80">
                  {explanationNotice}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Collapsible DICOM Tag Inspector */}
        <div className="pt-1">
          <button
            type="button"
            onClick={() => setShowDicomTags((v) => !v)}
            className="flex items-center gap-1.5 text-xs font-medium text-cyan-400 hover:text-cyan-300 transition-colors cursor-pointer"
          >
            {showDicomTags ? (
              <ChevronDown className="w-3.5 h-3.5" />
            ) : (
              <ChevronRight className="w-3.5 h-3.5" />
            )}
            DICOM Header Metadata
          </button>

          {showDicomTags && (
            <dl className="mt-3 space-y-1.5 text-[11px] font-mono bg-[#07090E] border border-slate-800/80 rounded p-3">
              <div className="flex justify-between gap-2">
                <dt className="text-slate-500">StudyUID</dt>
                <dd
                  className="text-slate-300 truncate max-w-[180px]"
                  title={study.dicom.studyInstanceUid}
                >
                  {study.dicom.studyInstanceUid}
                </dd>
              </div>
              {study.dicom.seriesInstanceUid && (
                <div className="flex justify-between gap-2">
                  <dt className="text-slate-500">SeriesUID</dt>
                  <dd
                    className="text-slate-300 truncate max-w-[180px]"
                    title={study.dicom.seriesInstanceUid}
                  >
                    {study.dicom.seriesInstanceUid}
                  </dd>
                </div>
              )}
              <div className="flex justify-between gap-2">
                <dt className="text-slate-500">Series</dt>
                <dd className="text-slate-300 truncate">{study.dicom.seriesDescription}</dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt className="text-slate-500">Photometric</dt>
                <dd className="text-slate-300">{study.dicom.photometricInterpretation}</dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt className="text-slate-500">Dimensions</dt>
                <dd className="text-slate-300">
                  {study.dicom.rows && study.dicom.columns
                    ? `${study.dicom.columns}×${study.dicom.rows} (${study.dicom.bitsAllocated ?? 8}-bit)`
                    : 'N/A'}
                </dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt className="text-slate-500">Pixel Spacing</dt>
                <dd className="text-slate-300">{study.dicom.pixelSpacingMm} mm</dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt className="text-slate-500">Default W/L</dt>
                <dd className="text-slate-300">
                  {study.dicom.windowWidth} / {study.dicom.windowCenter}
                </dd>
              </div>
            </dl>
          )}
        </div>
      </div>

      {/* Radiologist Decision & Audit Sign-Off with Explicit Confirmation */}
      <div className="p-5 space-y-3.5 mt-auto">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-semibold text-slate-200">Radiologist Sign-Off</h3>
          {study.reviewedBy && study.reviewedAt && (
            <span className="text-[11px] font-mono text-slate-400 truncate max-w-[160px]">
              Last: {study.reviewedBy}
            </span>
          )}
        </div>

        <div className="space-y-2">
          <label className="block">
            <span className="text-[11px] text-slate-400 block mb-1">Reviewing Clinician</span>
            <input
              type="text"
              value={reviewer}
              onChange={(e) => setReviewer(e.target.value)}
              placeholder="Dr. Name, MD"
              className="w-full bg-[#07090E] border border-slate-800 rounded px-3 py-2 text-xs text-slate-100 focus:outline-none focus:border-cyan-500"
            />
          </label>

          <label className="block">
            <span className="text-[11px] text-slate-400 block mb-1">
              Review Notes / Override Justification
            </span>
            <textarea
              rows={3}
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Document agreement, false-positive artifact, or manual finding..."
              className="w-full bg-[#07090E] border border-slate-800 rounded px-3 py-2 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-cyan-500 resize-none"
            />
          </label>
        </div>

        {feedback && (
          <div
            className={`text-xs px-3 py-2 rounded border ${
              feedback.type === 'success'
                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
                : feedback.type === 'info'
                ? 'bg-cyan-500/10 border-cyan-500/30 text-cyan-200'
                : 'bg-rose-500/10 border-rose-500/30 text-rose-300'
            }`}
          >
            {feedback.message}
          </div>
        )}

        {/* Confirmation Gate before committing to Audit Log */}
        {pendingConfirmation ? (
          <div className="p-3 bg-[#07090E] border border-amber-500/40 rounded space-y-2.5">
            <div className="text-xs text-slate-200">
              Confirm permanent audit log entry:{' '}
              <span
                className={
                  pendingConfirmation === 'approved'
                    ? 'font-semibold text-emerald-400'
                    : 'font-semibold text-rose-400'
                }
              >
                {pendingConfirmation.toUpperCase()}
              </span>{' '}
              by <span className="font-mono text-cyan-300">{reviewer || 'Unspecified'}</span>?
            </div>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                disabled={submitting}
                onClick={() => setPendingConfirmation(null)}
                className="py-2 rounded bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-200 cursor-pointer"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={submitting}
                onClick={confirmReview}
                className={`py-2 rounded text-xs font-semibold text-white cursor-pointer ${
                  pendingConfirmation === 'approved'
                    ? 'bg-emerald-600 hover:bg-emerald-500'
                    : 'bg-rose-600 hover:bg-rose-500'
                }`}
              >
                {submitting ? 'Recording...' : 'Confirm Sign-Off'}
              </button>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-2.5 pt-1">
            <button
              type="button"
              disabled={submitting}
              onClick={() => setPendingConfirmation('approved')}
              className="min-h-[42px] px-4 py-2 rounded bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-xs font-semibold flex items-center justify-center gap-1.5 transition-colors cursor-pointer whitespace-nowrap"
            >
              <Check className="w-4 h-4" />
              Approve
            </button>

            <button
              type="button"
              disabled={submitting}
              onClick={() => setPendingConfirmation('rejected')}
              className="min-h-[42px] px-4 py-2 rounded bg-rose-600/90 hover:bg-rose-500 disabled:opacity-50 text-white text-xs font-semibold flex items-center justify-center gap-1.5 transition-colors cursor-pointer whitespace-nowrap"
            >
              <X className="w-4 h-4" />
              Reject
            </button>
          </div>
        )}
      </div>
    </aside>
  );
};
