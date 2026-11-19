import React, { useRef, useState } from 'react';
import { FileCode2, Upload, X } from 'lucide-react';
import type { Modality } from '../types/imaging';

interface UploadScanModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUploadComplete: (payload: {
    fileBase64: string;
    fileName: string;
    mimeType: string;
    title: string;
    modality: Modality;
    bodyPart: string;
    patientId: string;
  }) => Promise<void>;
}

const ACCEPTED_EXTENSIONS = ['.dcm', '.dicom', '.zip', '.png', '.jpg', '.jpeg', '.webp'];
const MAX_SIZE_BYTES = 20 * 1024 * 1024;

export const UploadScanModal: React.FC<UploadScanModalProps> = ({
  isOpen,
  onClose,
  onUploadComplete,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [fileBase64, setFileBase64] = useState<string | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [mimeType, setMimeType] = useState<string>('application/dicom');
  const [fileName, setFileName] = useState<string>('');
  const [fileSizeKb, setFileSizeKb] = useState<number>(0);
  const [modality, setModality] = useState<Modality>('CR');
  const [bodyPart, setBodyPart] = useState('CHEST');
  const [title, setTitle] = useState('');
  const [patientId, setPatientId] = useState(`ANON-${Math.floor(1000 + Math.random() * 9000)}`);
  const [progress, setProgress] = useState<number>(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleFile = (file: File) => {
    setError(null);
    const lowerName = file.name.toLowerCase();
    const isAllowedExt = ACCEPTED_EXTENSIONS.some((ext) => lowerName.endsWith(ext));
    if (!isAllowedExt) {
      setError(
        `Unsupported file format. Accepted: ${ACCEPTED_EXTENSIONS.join(', ')}`
      );
      return;
    }
    if (file.size === 0 || file.size > MAX_SIZE_BYTES) {
      setError('File size must be between 1 byte and 20 MB.');
      return;
    }

    setFileName(file.name);
    setFileSizeKb(Math.round((file.size / 1024) * 10) / 10);
    setMimeType(
      file.type ||
        (lowerName.endsWith('.dcm') || lowerName.endsWith('.dicom')
          ? 'application/dicom'
          : lowerName.endsWith('.zip')
          ? 'application/zip'
          : 'application/octet-stream')
    );

    if (!title) {
      setTitle(file.name.replace(/\.[^/.]+$/, '').replace(/[_-]+/g, ' '));
    }

    const reader = new FileReader();
    reader.onprogress = (ev) => {
      if (ev.lengthComputable) {
        setProgress(Math.round((ev.loaded / ev.total) * 50));
      }
    };
    reader.onload = () => {
      const result = reader.result as string;
      setFileBase64(result);
      setImagePreviewUrl(file.type.startsWith('image/') ? result : null);
      setProgress(50);
    };
    reader.onerror = () => {
      setError('Failed to read selected file from disk.');
    };
    reader.readAsDataURL(file);
  };

  const resetForm = () => {
    setFileBase64(null);
    setImagePreviewUrl(null);
    setFileName('');
    setFileSizeKb(0);
    setTitle('');
    setProgress(0);
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!fileBase64 || !fileName) {
      setError('Please choose a DICOM (.dcm), ZIP DICOM series, or X-ray image file.');
      return;
    }
    setLoading(true);
    setError(null);
    setProgress(75);
    try {
      await onUploadComplete({
        fileBase64,
        fileName,
        mimeType,
        title: title.trim() || `${modality} ${bodyPart} Study`,
        modality,
        bodyPart,
        patientId,
      });
      setProgress(100);
      resetForm();
      onClose();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Study upload failed.';
      setError(msg);
      setProgress(0);
    } finally {
      setLoading(false);
    }
  };

  const isDicomOrZip =
    fileName.toLowerCase().endsWith('.dcm') ||
    fileName.toLowerCase().endsWith('.dicom') ||
    fileName.toLowerCase().endsWith('.zip');

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="upload-modal-title"
    >
      <div className="w-full max-w-lg bg-[#0B0F17] border border-slate-800 rounded-lg overflow-hidden">
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-800">
          <div>
            <h2 id="upload-modal-title" className="text-base font-semibold text-slate-100">
              Upload Medical Imaging Study
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Supports DICOM (.dcm), DICOM series (.zip), and 2D Chest X-ray images (.png / .jpg).
            </p>
          </div>
          <button
            type="button"
            onClick={() => {
              resetForm();
              onClose();
            }}
            disabled={loading}
            className="p-1.5 rounded text-slate-400 hover:text-white hover:bg-slate-800 transition-colors cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-5 space-y-4">
          {/* Drag-and-drop Zone */}
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const dropped = e.dataTransfer.files?.[0];
              if (dropped) handleFile(dropped);
            }}
            onClick={() => fileInputRef.current?.click()}
            className="border border-dashed border-slate-700 hover:border-cyan-500/70 bg-[#07090E] rounded-md p-5 text-center cursor-pointer transition-colors"
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".dcm,.dicom,.zip,image/png,image/jpeg,image/webp"
              className="hidden"
              onChange={(e) => {
                const chosen = e.target.files?.[0];
                if (chosen) handleFile(chosen);
              }}
            />

            {fileBase64 ? (
              <div className="flex items-center gap-4 text-left">
                {imagePreviewUrl ? (
                  <img
                    src={imagePreviewUrl}
                    alt="Uploaded medical scan preview"
                    referrerPolicy="no-referrer"
                    className="w-20 h-20 object-cover rounded border border-slate-800 bg-black shrink-0"
                  />
                ) : (
                  <div className="w-20 h-20 rounded border border-slate-800 bg-slate-900 flex flex-col items-center justify-center shrink-0">
                    <FileCode2 className="w-6 h-6 text-cyan-400" />
                    <span className="text-[10px] font-mono text-slate-400 mt-1">
                      {fileName.toLowerCase().endsWith('.zip') ? 'ZIP' : 'DICOM'}
                    </span>
                  </div>
                )}
                <div className="min-w-0 flex-1">
                  <p className="text-xs font-medium text-slate-100 truncate">{fileName}</p>
                  <p className="text-[11px] font-mono text-slate-400 mt-0.5 tabular-nums">
                    {fileSizeKb.toFixed(1)} KB
                  </p>
                  <p className="text-[11px] font-mono text-cyan-400 mt-1">
                    Validated · Click to select another file
                  </p>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-2 py-2">
                <Upload className="w-6 h-6 text-cyan-400" />
                <div className="text-xs text-slate-200 font-medium">
                  Drop a DICOM (.dcm), ZIP series, or X-ray image here, or click to browse
                </div>
                <div className="text-[11px] text-slate-500">
                  Max 20 MB · DICOM tags (Modality, Window/Level, Photometric) auto-extracted
                </div>
              </div>
            )}
          </div>

          {/* Modality & Anatomy Fallback Metadata */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <span className="text-[11px] text-slate-400 block mb-1.5">
                Modality {isDicomOrZip ? '(Fallback if missing in DICOM)' : ''}
              </span>
              <div className="grid grid-cols-4 gap-1 p-1 bg-[#07090E] border border-slate-800 rounded">
                {(
                  [
                    { id: 'CR', label: 'CR' },
                    { id: 'DX', label: 'DX' },
                    { id: 'CT', label: 'CT' },
                    { id: 'MR', label: 'MR' },
                  ] as const
                ).map((m) => (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => {
                      setModality(m.id);
                      if (m.id === 'MR' && bodyPart === 'CHEST') setBodyPart('BRAIN');
                      if ((m.id === 'CR' || m.id === 'DX') && bodyPart === 'BRAIN')
                        setBodyPart('CHEST');
                    }}
                    className={`py-1.5 text-xs font-mono rounded transition-colors cursor-pointer ${
                      modality === m.id
                        ? 'bg-cyan-500/20 text-cyan-300'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
            </div>

            <label className="block">
              <span className="text-[11px] text-slate-400 block mb-1.5">Anatomical Region</span>
              <input
                type="text"
                value={bodyPart}
                onChange={(e) => setBodyPart(e.target.value.toUpperCase())}
                placeholder="CHEST / LUNG / BRAIN"
                className="w-full bg-[#07090E] border border-slate-800 rounded px-3 py-2 text-xs text-slate-100 font-mono focus:outline-none focus:border-cyan-500"
              />
            </label>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <label className="block">
              <span className="text-[11px] text-slate-400 block mb-1.5">Study Description</span>
              <input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="e.g. PA Chest Follow-up"
                className="w-full bg-[#07090E] border border-slate-800 rounded px-3 py-2 text-xs text-slate-100 focus:outline-none focus:border-cyan-500"
              />
            </label>

            <label className="block">
              <span className="text-[11px] text-slate-400 block mb-1.5">Anonymized Patient ID</span>
              <input
                type="text"
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
                className="w-full bg-[#07090E] border border-slate-800 rounded px-3 py-2 text-xs text-slate-100 font-mono focus:outline-none focus:border-cyan-500"
              />
            </label>
          </div>

          {/* Progress bar */}
          {(loading || progress > 0) && (
            <div className="space-y-1">
              <div className="flex justify-between text-[11px] font-mono text-slate-400">
                <span>{loading ? 'Ingesting & validating study...' : 'File buffered'}</span>
                <span className="tabular-nums">{progress}%</span>
              </div>
              <div className="h-1.5 bg-slate-800 rounded overflow-hidden">
                <div
                  className="h-full bg-cyan-400 transition-all duration-200"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}

          {error && (
            <div className="text-xs px-3 py-2 rounded bg-rose-500/10 border border-rose-500/30 text-rose-300">
              {error}
            </div>
          )}

          <div className="text-[11px] text-slate-500 leading-relaxed">
            Do not upload real patient-identifiable health information (PHI). If trained PyTorch/MONAI
            checkpoints are not loaded, the study is ingested honestly without fabricating predictions.
          </div>

          <div className="flex items-center justify-end gap-3 pt-2 border-t border-slate-800">
            <button
              type="button"
              onClick={() => {
                resetForm();
                onClose();
              }}
              disabled={loading}
              className="px-4 py-2 text-xs font-medium text-slate-300 hover:text-white transition-colors cursor-pointer"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading || !fileBase64}
              className="px-5 py-2.5 rounded bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-xs font-semibold text-white transition-colors cursor-pointer whitespace-nowrap"
            >
              {loading ? 'Ingesting Study...' : 'Upload & Route to Pipeline'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
