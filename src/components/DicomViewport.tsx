import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Contrast,
  Crosshair,
  Eye,
  EyeOff,
  Info,
  Layers,
  RotateCcw,
  SlidersHorizontal,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import type { HeatmapRegion, ImagingStudy, OverlayMode } from '../types/imaging';

interface DicomViewportProps {
  study: ImagingStudy;
  overlayMode: OverlayMode;
  onOverlayModeChange: (mode: OverlayMode) => void;
}

type WindowPreset = 'default' | 'lung' | 'soft' | 'bone';

const PRESETS: Record<WindowPreset, { label: string; brightness: number; contrast: number }> = {
  default: { label: 'Default', brightness: 100, contrast: 105 },
  lung: { label: 'Lung / Parenchyma', brightness: 114, contrast: 132 },
  soft: { label: 'Soft Tissue', brightness: 96, contrast: 118 },
  bone: { label: 'High Contrast', brightness: 90, contrast: 155 },
};

/**
 * Jet/Turbo-inspired thermal lookup table for illustrative Demo Mode overlays.
 * Real PyTorch Grad-CAM images from the FastAPI backend are rendered directly via study.gradCamImageUrl.
 */
function thermalColor(t: number): [number, number, number, number] {
  const v = Math.max(0, Math.min(1, t));
  if (v < 0.08) return [0, 0, 0, 0];

  let r = 0;
  let g = 0;
  let b = 0;

  if (v < 0.35) {
    const k = (v - 0.08) / 0.27;
    r = 0;
    g = Math.round(210 * k);
    b = Math.round(180 + 75 * k);
  } else if (v < 0.6) {
    const k = (v - 0.35) / 0.25;
    r = Math.round(255 * k);
    g = 235;
    b = Math.round(255 * (1 - k));
  } else if (v < 0.82) {
    const k = (v - 0.6) / 0.22;
    r = 255;
    g = Math.round(235 - 150 * k);
    b = 0;
  } else {
    const k = (v - 0.82) / 0.18;
    r = 255;
    g = Math.round(85 * (1 - k));
    b = Math.round(35 * k);
  }

  const alpha = Math.round(Math.min(255, Math.pow(v, 0.82) * 235));
  return [r, g, b, alpha];
}

function renderDemoHeatmapCanvas(
  canvas: HTMLCanvasElement,
  width: number,
  height: number,
  regions: HeatmapRegion[]
) {
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  ctx.clearRect(0, 0, width, height);
  if (regions.length === 0) return;

  const imgData = ctx.createImageData(width, height);
  const data = imgData.data;

  for (let py = 0; py < height; py++) {
    const ny = py / height;
    for (let px = 0; px < width; px++) {
      const nx = px / width;
      let activation = 0;

      for (let i = 0; i < regions.length; i++) {
        const reg = regions[i];
        const dx = (nx - reg.x) / Math.max(0.03, reg.rx);
        const dy = (ny - reg.y) / Math.max(0.03, reg.ry);
        const distSq = dx * dx + dy * dy;
        if (distSq < 4.5) {
          const val = reg.intensity * Math.exp(-0.95 * distSq);
          if (val > activation) activation = val;
        }
      }

      const idx = (py * width + px) * 4;
      const [r, g, b, a] = thermalColor(activation);
      data[idx] = r;
      data[idx + 1] = g;
      data[idx + 2] = b;
      data[idx + 3] = a;
    }
  }

  ctx.putImageData(imgData, 0, 0);
}

export const DicomViewport: React.FC<DicomViewportProps> = ({
  study,
  overlayMode,
  onOverlayModeChange,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const baseCanvasRef = useRef<HTMLCanvasElement>(null);
  const heatCanvasRef = useRef<HTMLCanvasElement>(null);
  const segCanvasRef = useRef<HTMLCanvasElement>(null);

  const [sliceIndex, setSliceIndex] = useState(0);
  const [imageObj, setImageObj] = useState<HTMLImageElement | null>(null);
  const [imgLoading, setImgLoading] = useState(true);
  const [imgError, setImgError] = useState(false);

  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const panOriginRef = useRef({ x: 0, y: 0 });

  const [preset, setPreset] = useState<WindowPreset>('default');
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(105);
  const [inverted, setInverted] = useState(false);
  const [overlayOpacity, setOverlayOpacity] = useState(0.68);
  const [showDicomHud, setShowDicomHud] = useState(true);
  const [showAnnotations, setShowAnnotations] = useState(true);

  const [splitRatio, setSplitRatio] = useState(0.5);
  const [isDraggingSplit, setIsDraggingSplit] = useState(false);

  const [probe, setProbe] = useState<{
    x: number;
    y: number;
    intensity: number;
    huApprox: number;
  } | null>(null);

  const activeSliceUrl = useMemo(() => {
    if (study.sliceUrls && study.sliceUrls.length > 0) {
      const clamped = Math.max(0, Math.min(study.sliceUrls.length - 1, sliceIndex));
      return study.sliceUrls[clamped];
    }
    return study.imageUrl;
  }, [study.imageUrl, study.sliceUrls, sliceIndex]);

  const totalSlices = Math.max(1, study.sliceUrls?.length ?? study.sliceCount ?? 1);
  const hasRealGradCamImage = Boolean(study.gradCamImageUrl);
  const hasRealSegMaskImage = Boolean(study.segmentation?.maskImageUrl);
  const hasAnyOverlay =
    hasRealGradCamImage || hasRealSegMaskImage || study.heatmapRegions.length > 0;

  // Reset slice & viewport transforms when switching studies
  useEffect(() => {
    setSliceIndex(0);
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setPreset('default');
    setBrightness(100);
    setContrast(105);
    setInverted(false);
  }, [study.id]);

  // Load current slice image
  useEffect(() => {
    if (!activeSliceUrl) {
      setImageObj(null);
      setImgLoading(false);
      setImgError(false);
      return;
    }

    setImgLoading(true);
    setImgError(false);
    setImageObj(null);

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      setImageObj(img);
      setImgLoading(false);
    };
    img.onerror = () => {
      setImgError(true);
      setImgLoading(false);
    };
    img.src = activeSliceUrl;
  }, [activeSliceUrl]);

  const canvasSize = 640;

  // Draw base image onto baseCanvas
  useEffect(() => {
    const canvas = baseCanvasRef.current;
    if (!canvas || !imageObj) return;

    canvas.width = canvasSize;
    canvas.height = canvasSize;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#04060A';
    ctx.fillRect(0, 0, canvasSize, canvasSize);

    const scale = Math.min(canvasSize / imageObj.width, canvasSize / imageObj.height);
    const drawW = imageObj.width * scale;
    const drawH = imageObj.height * scale;
    const dx = (canvasSize - drawW) / 2;
    const dy = (canvasSize - drawH) / 2;

    ctx.drawImage(imageObj, dx, dy, drawW, drawH);
  }, [imageObj]);

  // Draw demo Grad-CAM canvas when no backend-generated Grad-CAM PNG is present
  useEffect(() => {
    const canvas = heatCanvasRef.current;
    if (!canvas) return;
    if (hasRealGradCamImage) {
      const ctx = canvas.getContext('2d');
      ctx?.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    renderDemoHeatmapCanvas(canvas, 256, 256, study.heatmapRegions);
  }, [study.id, study.heatmapRegions, hasRealGradCamImage]);

  // Draw demo segmentation contours when no backend-generated mask PNG is present
  useEffect(() => {
    const canvas = segCanvasRef.current;
    if (!canvas) return;
    canvas.width = canvasSize;
    canvas.height = canvasSize;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvasSize, canvasSize);
    if (hasRealSegMaskImage) return;

    study.heatmapRegions.forEach((region, idx) => {
      const isPrimary = idx === 0;
      const strokeColor = isPrimary ? '#06B6D4' : '#F59E0B';
      const fillColor = isPrimary ? 'rgba(6, 182, 212, 0.22)' : 'rgba(245, 158, 11, 0.16)';

      ctx.save();
      ctx.beginPath();
      if (region.contour && region.contour.length >= 3) {
        const pts = region.contour.map((p) => ({
          x: p.x * canvasSize,
          y: p.y * canvasSize,
        }));
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 0; i < pts.length; i++) {
          const curr = pts[i];
          const next = pts[(i + 1) % pts.length];
          const midX = (curr.x + next.x) / 2;
          const midY = (curr.y + next.y) / 2;
          ctx.quadraticCurveTo(curr.x, curr.y, midX, midY);
        }
        ctx.closePath();
      } else {
        ctx.ellipse(
          region.x * canvasSize,
          region.y * canvasSize,
          region.rx * canvasSize * 0.85,
          region.ry * canvasSize * 0.85,
          0,
          0,
          Math.PI * 2
        );
      }

      ctx.fillStyle = fillColor;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.setLineDash(isPrimary ? [] : [5, 4]);
      ctx.strokeStyle = strokeColor;
      ctx.stroke();

      const cx = region.x * canvasSize;
      const cy = region.y * canvasSize;
      ctx.setLineDash([]);
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(cx - 7, cy);
      ctx.lineTo(cx + 7, cy);
      ctx.moveTo(cx, cy - 7);
      ctx.lineTo(cx, cy + 7);
      ctx.stroke();
      ctx.restore();
    });
  }, [study.id, study.heatmapRegions, hasRealSegMaskImage]);

  const applyPreset = (p: WindowPreset) => {
    setPreset(p);
    setBrightness(PRESETS[p].brightness);
    setContrast(PRESETS[p].contrast);
  };

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    applyPreset('default');
    setInverted(false);
    setSplitRatio(0.5);
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY < 0 ? 0.14 : -0.14;
    setZoom((z) => Math.max(0.75, Math.min(4, Number((z + delta).toFixed(2)))));
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (isDraggingSplit) return;
    setIsPanning(true);
    panOriginRef.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
  };

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (rect && isDraggingSplit) {
        const nextRatio = Math.max(0.05, Math.min(0.95, (e.clientX - rect.left) / rect.width));
        setSplitRatio(nextRatio);
        return;
      }

      if (isPanning) {
        setPan({
          x: e.clientX - panOriginRef.current.x,
          y: e.clientY - panOriginRef.current.y,
        });
      }

      const canvas = baseCanvasRef.current;
      if (canvas && imageObj) {
        const cRect = canvas.getBoundingClientRect();
        if (
          e.clientX >= cRect.left &&
          e.clientX <= cRect.right &&
          e.clientY >= cRect.top &&
          e.clientY <= cRect.bottom
        ) {
          const nx = (e.clientX - cRect.left) / cRect.width;
          const ny = (e.clientY - cRect.top) / cRect.height;
          const px = Math.max(0, Math.min(canvas.width - 1, Math.floor(nx * canvas.width)));
          const py = Math.max(0, Math.min(canvas.height - 1, Math.floor(ny * canvas.height)));
          const ctx = canvas.getContext('2d');
          if (ctx) {
            const pixel = ctx.getImageData(px, py, 1, 1).data;
            const lum = Math.round(0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]);
            const huApprox = Math.round((lum / 255) * 2000 - 1000);
            setProbe({ x: px, y: py, intensity: lum, huApprox });
          }
        } else {
          setProbe(null);
        }
      }
    },
    [isDraggingSplit, isPanning, imageObj]
  );

  const handleMouseUp = () => {
    setIsPanning(false);
    setIsDraggingSplit(false);
  };

  const filterStyle = useMemo(
    () => `brightness(${brightness}%) contrast(${contrast}%) ${inverted ? 'invert(1)' : ''}`,
    [brightness, contrast, inverted]
  );

  const showHeatmapLayer =
    overlayMode === 'gradcam' || overlayMode === 'heatmap_only' || overlayMode === 'split';
  const showSegLayer = overlayMode === 'segmentation' || overlayMode === 'split';

  return (
    <div className="flex flex-col h-full bg-[#04060A] select-none overflow-hidden">
      {/* Top Viewport Instrument Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-2.5 bg-[#0B0F17] border-b border-slate-800/80">
        {/* Overlay Mode Segmented Switch */}
        <div className="flex items-center gap-1 p-1 bg-[#07090E] border border-slate-800 rounded-md overflow-x-auto">
          {(
            [
              { id: 'off', label: 'Original' },
              { id: 'heatmap_only', label: 'Heatmap' },
              { id: 'gradcam', label: 'Blended' },
              { id: 'segmentation', label: 'Mask' },
              { id: 'split', label: 'Split' },
            ] as const
          ).map((mode) => (
            <button
              key={mode.id}
              type="button"
              onClick={() => onOverlayModeChange(mode.id)}
              className={`px-2.5 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                overlayMode === mode.id
                  ? 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/40'
                  : 'text-slate-400 hover:text-slate-200 border border-transparent'
              }`}
            >
              {mode.label}
            </button>
          ))}
        </div>

        {/* Window / Level & Zoom Controls */}
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-1.5">
            <SlidersHorizontal className="w-3.5 h-3.5 text-slate-400" />
            <select
              aria-label="Window Level Preset"
              value={preset}
              onChange={(e) => applyPreset(e.target.value as WindowPreset)}
              className="bg-[#07090E] border border-slate-800 text-xs text-slate-200 rounded px-2.5 py-1.5 focus:outline-none focus:border-cyan-500"
            >
              {Object.entries(PRESETS).map(([key, val]) => (
                <option key={key} value={key}>
                  W/L: {val.label}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-1 border-l border-slate-800 pl-3">
            <button
              type="button"
              onClick={() => setZoom((z) => Math.max(0.75, Number((z - 0.2).toFixed(2))))}
              title="Zoom Out"
              className="p-1.5 rounded text-slate-300 hover:bg-slate-800 hover:text-white transition-colors cursor-pointer"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
            <span className="font-mono text-xs text-slate-300 w-12 text-center tabular-nums">
              {Math.round(zoom * 100)}%
            </span>
            <button
              type="button"
              onClick={() => setZoom((z) => Math.min(4, Number((z + 0.2).toFixed(2))))}
              title="Zoom In"
              className="p-1.5 rounded text-slate-300 hover:bg-slate-800 hover:text-white transition-colors cursor-pointer"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
          </div>

          <div className="flex items-center gap-1 border-l border-slate-800 pl-3">
            <button
              type="button"
              onClick={() => setInverted((v) => !v)}
              title="Invert Grayscale (MONOCHROME1/2)"
              className={`px-2.5 py-1.5 rounded text-xs font-medium flex items-center gap-1.5 transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                inverted
                  ? 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/40'
                  : 'text-slate-300 hover:bg-slate-800 border border-transparent'
              }`}
            >
              <Contrast className="w-3.5 h-3.5" />
              Invert
            </button>
            <button
              type="button"
              onClick={() => setShowAnnotations((v) => !v)}
              title="Toggle ROI Annotations"
              className={`p-1.5 rounded transition-colors cursor-pointer ${
                showAnnotations
                  ? 'text-cyan-300 bg-cyan-500/10'
                  : 'text-slate-400 hover:bg-slate-800'
              }`}
            >
              {showAnnotations ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
            </button>
            <button
              type="button"
              onClick={() => setShowDicomHud((v) => !v)}
              title="Toggle DICOM Metadata HUD"
              className={`p-1.5 rounded transition-colors cursor-pointer ${
                showDicomHud ? 'text-cyan-300 bg-cyan-500/10' : 'text-slate-400 hover:bg-slate-800'
              }`}
            >
              <Info className="w-4 h-4" />
            </button>
            <button
              type="button"
              onClick={resetView}
              title="Reset Viewport"
              className="p-1.5 rounded text-slate-300 hover:bg-slate-800 hover:text-white transition-colors cursor-pointer"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Main Canvas Viewing Area */}
      <div
        ref={containerRef}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => {
          handleMouseUp();
          setProbe(null);
        }}
        className="relative flex-1 flex items-center justify-center overflow-hidden bg-[#04060A] cursor-crosshair"
      >
        {imgLoading && (
          <div className="flex flex-col items-center gap-2 text-slate-400 text-xs font-mono">
            <div className="w-7 h-7 border-2 border-slate-700 border-t-cyan-400 rounded-full animate-spin" />
            <span>Decoding DICOM slice buffer...</span>
          </div>
        )}

        {!activeSliceUrl && !imgLoading && (
          <div className="max-w-md text-center p-6 border border-slate-800 bg-[#0B0F17] rounded-lg space-y-2">
            <p className="text-sm font-medium text-slate-200">
              DICOM Metadata Stored — Compressed Frame / Series Archive
            </p>
            <p className="text-xs text-slate-400 leading-relaxed">
              This uploaded file requires the Python pydicom + Orthanc backend to decode compressed
              transfer syntaxes or multi-slice ZIP series into preview slices. DICOM header tags remain
              available in the right panel.
            </p>
          </div>
        )}

        {imgError && !imgLoading && activeSliceUrl && (
          <div className="max-w-sm text-center p-6 border border-slate-800 bg-[#0B0F17] rounded-lg">
            <p className="text-sm font-medium text-slate-200">Scan image could not be rendered</p>
            <p className="text-xs text-slate-400 mt-1">
              Verify the uploaded DICOM/PNG file or run via the FastAPI pydicom pipeline.
            </p>
          </div>
        )}

        {/* Transformable Scan Stack */}
        <div
          className={`relative transition-opacity duration-150 ${
            imgLoading || imgError || !activeSliceUrl
              ? 'opacity-0 pointer-events-none absolute'
              : 'opacity-100'
          }`}
          style={{
            width: 'min(82%, 560px)',
            aspectRatio: '1 / 1',
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: 'center center',
          }}
        >
          {/* Base Grayscale Scan Canvas */}
          <canvas
            ref={baseCanvasRef}
            className="w-full h-full block rounded-sm transition-opacity duration-150"
            style={{
              filter: filterStyle,
              opacity: overlayMode === 'heatmap_only' && hasAnyOverlay ? 0.16 : 1,
            }}
          />

          {/* Real Backend Grad-CAM PNG (if available) */}
          {hasRealGradCamImage && study.gradCamImageUrl && (
            <img
              src={study.gradCamImageUrl}
              alt="Grad-CAM model activation heatmap"
              referrerPolicy="no-referrer"
              className="absolute inset-0 w-full h-full object-contain pointer-events-none transition-opacity duration-150"
              style={{
                opacity: showHeatmapLayer
                  ? overlayMode === 'heatmap_only'
                    ? 0.95
                    : overlayOpacity
                  : 0,
                mixBlendMode: overlayMode === 'heatmap_only' ? 'normal' : 'screen',
                clipPath:
                  overlayMode === 'split'
                    ? `inset(0 0 0 ${Math.round(splitRatio * 100)}%)`
                    : undefined,
              }}
            />
          )}

          {/* Illustrative Demo Heatmap Canvas */}
          {!hasRealGradCamImage && (
            <canvas
              ref={heatCanvasRef}
              className="absolute inset-0 w-full h-full pointer-events-none transition-opacity duration-150"
              style={{
                opacity: showHeatmapLayer
                  ? overlayMode === 'heatmap_only'
                    ? 0.95
                    : overlayOpacity
                  : 0,
                mixBlendMode: overlayMode === 'heatmap_only' ? 'normal' : 'screen',
                clipPath:
                  overlayMode === 'split'
                    ? `inset(0 0 0 ${Math.round(splitRatio * 100)}%)`
                    : undefined,
              }}
            />
          )}

          {/* Real Backend Segmentation Mask PNG (if available) */}
          {hasRealSegMaskImage && study.segmentation?.maskImageUrl && (
            <img
              src={study.segmentation.maskImageUrl}
              alt="MONAI 3D U-Net segmentation mask"
              referrerPolicy="no-referrer"
              className="absolute inset-0 w-full h-full object-contain pointer-events-none transition-opacity duration-150"
              style={{
                opacity: showSegLayer ? Math.min(1, overlayOpacity + 0.15) : 0,
                clipPath:
                  overlayMode === 'split'
                    ? `inset(0 0 0 ${Math.round(splitRatio * 100)}%)`
                    : undefined,
              }}
            />
          )}

          {/* Demo Contour Mask Canvas */}
          {!hasRealSegMaskImage && (
            <canvas
              ref={segCanvasRef}
              className="absolute inset-0 w-full h-full pointer-events-none transition-opacity duration-150"
              style={{
                opacity: showSegLayer ? Math.min(1, overlayOpacity + 0.15) : 0,
                clipPath:
                  overlayMode === 'split'
                    ? `inset(0 0 0 ${Math.round(splitRatio * 100)}%)`
                    : undefined,
              }}
            />
          )}

          {/* Annotated ROI Labels */}
          {showAnnotations && overlayMode !== 'off' && study.heatmapRegions.length > 0 && (
            <div className="absolute inset-0 pointer-events-none">
              {study.heatmapRegions.map((reg, i) => (
                <div
                  key={`${reg.label}-${i}`}
                  className="absolute -translate-x-1/2 -translate-y-full pb-2"
                  style={{
                    left: `${reg.x * 100}%`,
                    top: `${Math.max(10, (reg.y - reg.ry) * 100)}%`,
                  }}
                >
                  <div className="px-2 py-0.5 bg-[#07090E]/90 border border-slate-700/80 text-[11px] font-mono text-cyan-200 whitespace-nowrap rounded-sm shadow-sm">
                    {reg.label} · {Math.round(reg.intensity * 100)}%
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Interactive Split Divider Handle */}
        {overlayMode === 'split' && !imgLoading && !imgError && activeSliceUrl && (
          <div
            onMouseDown={(e) => {
              e.stopPropagation();
              setIsDraggingSplit(true);
            }}
            style={{ left: `${splitRatio * 100}%` }}
            className="absolute inset-y-0 w-0.5 bg-cyan-400 cursor-ew-resize z-20"
          >
            <div className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 px-2 py-1 bg-[#0B0F17] border border-cyan-500/60 text-[10px] font-mono text-cyan-300 rounded whitespace-nowrap">
              Original | AI
            </div>
          </div>
        )}

        {/* Notice when user toggles overlay on a study without real model outputs */}
        {overlayMode !== 'off' && !hasAnyOverlay && activeSliceUrl && !imgLoading && (
          <div className="absolute top-14 left-1/2 -translate-x-1/2 px-3 py-1.5 bg-[#0B0F17]/95 border border-amber-500/40 rounded text-[11px] font-mono text-amber-200 pointer-events-none whitespace-nowrap">
            No live overlay generated —{' '}
            {study.processingStatus === 'unsupported_modality'
              ? 'unsupported modality for active models'
              : 'trained PyTorch/MONAI weights not loaded'}
          </div>
        )}

        {/* DICOM Corner HUD Overlays */}
        {showDicomHud && (
          <>
            <div className="absolute top-3 left-3.5 pointer-events-none text-[11px] font-mono leading-relaxed text-slate-300/90 space-y-0.5">
              <div className="text-cyan-300 font-medium">
                {study.dicom.patientId} {study.isDemo ? '(DEMO)' : ''}
              </div>
              <div>
                {study.modality} · {study.bodyPart} · {study.dicom.patientSex}/{study.dicom.patientAge}
              </div>
              <div className="text-slate-400">{study.dicom.seriesDescription}</div>
              <div className="text-slate-400">{study.dicom.studyDate}</div>
            </div>

            <div className="absolute top-3 right-3.5 pointer-events-none text-right text-[11px] font-mono leading-relaxed text-slate-300/90 space-y-0.5">
              <div>{study.accessionNumber}</div>
              <div className="text-slate-400">
                Slice {sliceIndex + 1}/{totalSlices} · {study.dicom.sliceThicknessMm}
              </div>
              <div className="text-slate-400">Spacing: {study.dicom.pixelSpacingMm} mm</div>
              <div className="text-slate-400">
                {study.dicom.photometricInterpretation} · W:{Math.round(contrast * 2.4)} / L:
                {Math.round(brightness * 1.28)}
              </div>
            </div>
          </>
        )}

        {/* Bottom-Left Crosshair Intensity Probe */}
        <div className="absolute bottom-3 left-3.5 pointer-events-none flex items-center gap-2 text-[11px] font-mono text-slate-300 bg-[#07090E]/85 border border-slate-800/80 px-2.5 py-1 rounded">
          <Crosshair className="w-3.5 h-3.5 text-cyan-400" />
          {probe ? (
            <span className="tabular-nums">
              XY ({probe.x}, {probe.y}) · Gray {probe.intensity}/255 ·{' '}
              {study.modality === 'CT'
                ? `${probe.huApprox} HU (est.)`
                : `Norm ${(probe.intensity / 255).toFixed(2)}`}
            </span>
          ) : (
            <span className="text-slate-400">Drag to pan · Scroll to zoom · Hover to probe</span>
          )}
        </div>

        {/* Bottom-Right Grad-CAM Color Scale Legend */}
        {overlayMode !== 'off' && hasAnyOverlay && (
          <div className="absolute bottom-3 right-3.5 pointer-events-none bg-[#07090E]/85 border border-slate-800/80 px-2.5 py-1.5 rounded flex items-center gap-2.5">
            <span className="text-[10px] font-mono text-slate-400">
              {study.isDemo ? 'Demo' : 'Model'} 0.0
            </span>
            <div
              className="w-20 h-2 rounded-sm"
              style={{
                background:
                  overlayMode === 'segmentation'
                    ? 'linear-gradient(90deg, rgba(6,182,212,0.2) 0%, #06B6D4 65%, #F59E0B 100%)'
                    : 'linear-gradient(90deg, #0077B6 0%, #00F5D4 35%, #FEE440 65%, #FF4D4D 100%)',
              }}
            />
            <span className="text-[10px] font-mono text-slate-200">1.0</span>
          </div>
        )}
      </div>

      {/* Bottom Fine Adjustment & Multi-Slice Navigation Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 px-4 py-2.5 bg-[#0B0F17] border-t border-slate-800/80 text-xs">
        <div className="flex items-center gap-5 flex-wrap">
          <label className="flex items-center gap-2 text-slate-300">
            <span className="text-slate-400 w-14">Window</span>
            <input
              type="range"
              min={50}
              max={220}
              value={contrast}
              onChange={(e) => setContrast(Number(e.target.value))}
              className="w-20 cursor-pointer"
            />
            <span className="font-mono text-slate-300 w-9 text-right tabular-nums">{contrast}%</span>
          </label>
          <label className="flex items-center gap-2 text-slate-300">
            <span className="text-slate-400 w-10">Level</span>
            <input
              type="range"
              min={50}
              max={180}
              value={brightness}
              onChange={(e) => setBrightness(Number(e.target.value))}
              className="w-20 cursor-pointer"
            />
            <span className="font-mono text-slate-300 w-9 text-right tabular-nums">
              {brightness}%
            </span>
          </label>

          {totalSlices > 1 && (
            <label className="flex items-center gap-2 text-slate-300 border-l border-slate-800 pl-4">
              <Layers className="w-3.5 h-3.5 text-cyan-400" />
              <span className="text-slate-400">Slice</span>
              <input
                type="range"
                min={0}
                max={totalSlices - 1}
                value={sliceIndex}
                onChange={(e) => setSliceIndex(Number(e.target.value))}
                className="w-20 cursor-pointer"
              />
              <span className="font-mono text-slate-300 tabular-nums">
                {sliceIndex + 1}/{totalSlices}
              </span>
            </label>
          )}
        </div>

        <label className="flex items-center gap-2.5 text-slate-300">
          <span className="text-slate-400">Overlay Opacity</span>
          <input
            type="range"
            min={0}
            max={100}
            value={Math.round(overlayOpacity * 100)}
            onChange={(e) => setOverlayOpacity(Number(e.target.value) / 100)}
            disabled={overlayMode === 'off'}
            className="w-24 cursor-pointer disabled:opacity-40"
          />
          <span className="font-mono text-slate-300 w-9 text-right tabular-nums">
            {Math.round(overlayOpacity * 100)}%
          </span>
        </label>
      </div>
    </div>
  );
};
