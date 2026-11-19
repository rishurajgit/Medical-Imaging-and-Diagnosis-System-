import type { DicomMetadata, Modality } from '../types/imaging.ts';

export interface ParsedDicomResult {
  isValidDicom: boolean;
  modality: Modality;
  bodyPart: string;
  metadata: Partial<DicomMetadata>;
  /** Raw uncompressed 8-bit grayscale RGBA PNG-compatible pixel buffer if extractable */
  grayscalePreview?: {
    width: number;
    height: number;
    pixels: Uint8Array;
  };
  warnings: string[];
}

const EXPLICIT_LONG_VRS = new Set(['OB', 'OD', 'OF', 'OL', 'OW', 'SQ', 'UC', 'UN', 'UR', 'UT']);

function readAscii(buf: Buffer, offset: number, length: number): string {
  return buf
    .subarray(offset, offset + length)
    .toString('latin1')
    .replace(/\0+$/, '')
    .trim();
}

function normalizeModality(raw?: string): Modality {
  const m = (raw ?? '').toUpperCase().trim();
  if (m === 'CR' || m === 'DX' || m === 'CT' || m === 'MR') return m;
  return 'OTHER';
}

/**
 * Lightweight, zero-dependency DICOM Part 10 binary parser for the Node.js preview server.
 * Extracts standard DICOM metadata tags (Explicit VR Little Endian & Implicit VR Little Endian)
 * and decodes uncompressed 8-bit/16-bit MONOCHROME1/MONOCHROME2 PixelData when present.
 * Note: Full compressed transfer syntax decoding (JPEG2000, RLE) is handled by pydicom in backend/.
 */
export function parseDicomBuffer(buffer: Buffer): ParsedDicomResult {
  const warnings: string[] = [];

  if (buffer.length < 132 || readAscii(buffer, 128, 4) !== 'DICM') {
    return {
      isValidDicom: false,
      modality: 'OTHER',
      bodyPart: 'UNKNOWN',
      metadata: {},
      warnings: ['Missing DICOM Part 10 "DICM" preamble at byte offset 128.'],
    };
  }

  const tags: Record<string, string | number> = {};
  let pixelDataOffset = -1;
  let pixelDataLength = 0;
  let offset = 132;
  let implicitVr = false;

  while (offset + 8 <= buffer.length) {
    const group = buffer.readUInt16LE(offset);
    const element = buffer.readUInt16LE(offset + 2);
    const tagKey = `${group.toString(16).padStart(4, '0')},${element.toString(16).padStart(4, '0')}`;
    offset += 4;

    // Group 0002 is always Explicit VR Little Endian
    const useExplicit = group === 0x0002 || !implicitVr;
    let vr = '';
    let valueLength = 0;

    if (useExplicit) {
      if (offset + 2 > buffer.length) break;
      vr = readAscii(buffer, offset, 2);
      offset += 2;
      if (EXPLICIT_LONG_VRS.has(vr)) {
        if (offset + 6 > buffer.length) break;
        offset += 2; // reserved 2 bytes
        valueLength = buffer.readUInt32LE(offset);
        offset += 4;
      } else {
        if (offset + 2 > buffer.length) break;
        valueLength = buffer.readUInt16LE(offset);
        offset += 2;
      }
    } else {
      if (offset + 4 > buffer.length) break;
      valueLength = buffer.readUInt32LE(offset);
      offset += 4;
    }

    // Undefined length (sequences or encapsulated pixel data)
    if (valueLength === 0xffffffff) {
      if (tagKey === '7fe0,0010') {
        warnings.push('Encapsulated/compressed PixelData detected; use the Python pydicom backend for full decoding.');
        break;
      }
      // Scan forward for Sequence Delimitation Item (fffe,e0dd)
      let foundDelim = false;
      while (offset + 8 <= buffer.length) {
        const g = buffer.readUInt16LE(offset);
        const e = buffer.readUInt16LE(offset + 2);
        if (g === 0xfffe && e === 0xe0dd) {
          offset += 8;
          foundDelim = true;
          break;
        }
        offset += 2;
      }
      if (!foundDelim) break;
      continue;
    }

    if (offset + valueLength > buffer.length) break;

    if (tagKey === '7fe0,0010') {
      pixelDataOffset = offset;
      pixelDataLength = valueLength;
      break;
    }

    // Extract key metadata tags
    switch (tagKey) {
      case '0002,0010': {
        const tsUid = readAscii(buffer, offset, valueLength);
        tags.transferSyntaxUid = tsUid;
        if (tsUid === '1.2.840.10008.1.2') {
          implicitVr = true;
        }
        break;
      }
      case '0008,0018':
        tags.sopInstanceUid = readAscii(buffer, offset, valueLength);
        break;
      case '0008,0020':
        tags.studyDate = readAscii(buffer, offset, valueLength);
        break;
      case '0008,0050':
        tags.accessionNumber = readAscii(buffer, offset, valueLength);
        break;
      case '0008,0060':
        tags.modality = readAscii(buffer, offset, valueLength);
        break;
      case '0008,0080':
        tags.institutionName = readAscii(buffer, offset, valueLength);
        break;
      case '0008,103e':
        tags.seriesDescription = readAscii(buffer, offset, valueLength);
        break;
      case '0010,0020':
        tags.patientId = readAscii(buffer, offset, valueLength);
        break;
      case '0010,0040':
        tags.patientSex = readAscii(buffer, offset, valueLength);
        break;
      case '0010,1010':
        tags.patientAge = readAscii(buffer, offset, valueLength);
        break;
      case '0018,0015':
        tags.bodyPart = readAscii(buffer, offset, valueLength);
        break;
      case '0018,0050':
        tags.sliceThickness = readAscii(buffer, offset, valueLength);
        break;
      case '0020,000d':
        tags.studyInstanceUid = readAscii(buffer, offset, valueLength);
        break;
      case '0020,000e':
        tags.seriesInstanceUid = readAscii(buffer, offset, valueLength);
        break;
      case '0028,0004':
        tags.photometricInterpretation = readAscii(buffer, offset, valueLength);
        break;
      case '0028,0010':
        if (valueLength >= 2) tags.rows = buffer.readUInt16LE(offset);
        break;
      case '0028,0011':
        if (valueLength >= 2) tags.columns = buffer.readUInt16LE(offset);
        break;
      case '0028,0030':
        tags.pixelSpacing = readAscii(buffer, offset, valueLength);
        break;
      case '0028,0100':
        if (valueLength >= 2) tags.bitsAllocated = buffer.readUInt16LE(offset);
        break;
      case '0028,0103':
        if (valueLength >= 2) tags.pixelRepresentation = buffer.readUInt16LE(offset);
        break;
      case '0028,1050':
        tags.windowCenter = readAscii(buffer, offset, valueLength);
        break;
      case '0028,1051':
        tags.windowWidth = readAscii(buffer, offset, valueLength);
        break;
      case '0028,1052':
        tags.rescaleIntercept = readAscii(buffer, offset, valueLength);
        break;
      case '0028,1053':
        tags.rescaleSlope = readAscii(buffer, offset, valueLength);
        break;
      default:
        break;
    }

    offset += valueLength;
  }

  const rows = Number(tags.rows || 0);
  const cols = Number(tags.columns || 0);
  const bitsAllocated = Number(tags.bitsAllocated || 8);
  const pixelRep = Number(tags.pixelRepresentation || 0);
  const photometric = String(tags.photometricInterpretation || 'MONOCHROME2').toUpperCase();
  const slope = Number(String(tags.rescaleSlope || '1').split('\\')[0]) || 1;
  const intercept = Number(String(tags.rescaleIntercept || '0').split('\\')[0]) || 0;
  const rawWc = Number(String(tags.windowCenter || '').split('\\')[0]);
  const rawWw = Number(String(tags.windowWidth || '').split('\\')[0]);

  let grayscalePreview: ParsedDicomResult['grayscalePreview'];

  if (pixelDataOffset > 0 && rows > 0 && cols > 0 && rows <= 4096 && cols <= 4096) {
    const pixelCount = rows * cols;
    const bytesPerPixel = bitsAllocated <= 8 ? 1 : 2;
    if (pixelDataLength >= pixelCount * bytesPerPixel) {
      const floatVals = new Float32Array(pixelCount);
      let minVal = Infinity;
      let maxVal = -Infinity;

      for (let i = 0; i < pixelCount; i++) {
        const pOff = pixelDataOffset + i * bytesPerPixel;
        const raw =
          bytesPerPixel === 1
            ? buffer.readUInt8(pOff)
            : pixelRep === 1
            ? buffer.readInt16LE(pOff)
            : buffer.readUInt16LE(pOff);
        const val = raw * slope + intercept;
        floatVals[i] = val;
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
      }

      const useWindow = Number.isFinite(rawWc) && Number.isFinite(rawWw) && rawWw > 0;
      const low = useWindow ? rawWc - rawWw / 2 : minVal;
      const high = useWindow ? rawWc + rawWw / 2 : maxVal;
      const span = Math.max(1e-5, high - low);

      const pixels = new Uint8Array(pixelCount);
      for (let i = 0; i < pixelCount; i++) {
        let norm = (floatVals[i] - low) / span;
        norm = Math.max(0, Math.min(1, norm));
        if (photometric === 'MONOCHROME1') {
          norm = 1 - norm;
        }
        pixels[i] = Math.round(norm * 255);
      }

      grayscalePreview = { width: cols, height: rows, pixels };
    }
  }

  const rawDate = String(tags.studyDate || '');
  const formattedDate =
    rawDate.length === 8
      ? `${rawDate.slice(0, 4)}-${rawDate.slice(4, 6)}-${rawDate.slice(6, 8)}`
      : new Date().toISOString().slice(0, 10);

  return {
    isValidDicom: true,
    modality: normalizeModality(String(tags.modality || '')),
    bodyPart: String(tags.bodyPart || 'UNKNOWN').toUpperCase(),
    metadata: {
      studyInstanceUid: String(tags.studyInstanceUid || `1.2.826.0.1.3680043.${Date.now()}`),
      seriesInstanceUid: String(tags.seriesInstanceUid || `1.2.826.0.1.3680043.${Date.now()}.1`),
      sopInstanceUid: String(tags.sopInstanceUid || `1.2.826.0.1.3680043.${Date.now()}.2`),
      seriesDescription: String(tags.seriesDescription || 'UPLOADED DICOM SERIES'),
      datasetSource: 'Uploaded DICOM Study',
      patientId: String(tags.patientId || `ANON-${Math.floor(1000 + Math.random() * 9000)}`),
      patientSex: String(tags.patientSex || 'U'),
      patientAge: String(tags.patientAge || '000Y'),
      studyDate: formattedDate,
      sliceThicknessMm: String(tags.sliceThickness || 'N/A'),
      pixelSpacingMm: String(tags.pixelSpacing || 'N/A').replace('\\', ' × '),
      windowCenter: Number.isFinite(rawWc) ? rawWc : 128,
      windowWidth: Number.isFinite(rawWw) && rawWw > 0 ? rawWw : 256,
      photometricInterpretation: photometric,
      institutionName: String(tags.institutionName || 'Uploaded DICOM Source'),
      rows: rows || undefined,
      columns: cols || undefined,
      bitsAllocated: bitsAllocated || undefined,
      transferSyntaxUid: tags.transferSyntaxUid ? String(tags.transferSyntaxUid) : undefined,
    },
    grayscalePreview,
    warnings,
  };
}
