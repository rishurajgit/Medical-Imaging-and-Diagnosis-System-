import assert from 'node:assert/strict';
import test from 'node:test';
import { parseDicomBuffer } from '../src/utils/dicomParser.ts';
import { encodeGrayscalePng } from '../src/utils/pngEncoder.ts';

/**
 * Helper: constructs a valid Part 10 Explicit VR Little Endian DICOM file in memory
 * with PatientID, Modality, BodyPartExamined, PhotometricInterpretation, Rows, Cols, and PixelData.
 */
function buildPart10Dicom({
  modality = 'CR',
  bodyPart = 'CHEST',
  photometric = 'MONOCHROME1',
  rows = 8,
  cols = 8,
} = {}) {
  const chunks = [];
  // 128-byte preamble + DICM
  const preamble = Buffer.alloc(132, 0);
  preamble.write('DICM', 128, 'ascii');
  chunks.push(preamble);

  function addStringTag(group, elem, vr, val) {
    let padded = val;
    if (padded.length % 2 === 1) padded += ' ';
    const valBuf = Buffer.from(padded, 'latin1');
    const hdr = Buffer.alloc(8);
    hdr.writeUInt16LE(group, 0);
    hdr.writeUInt16LE(elem, 2);
    hdr.write(vr, 4, 'ascii');
    hdr.writeUInt16LE(valBuf.length, 6);
    chunks.push(hdr, valBuf);
  }

  function addUInt16Tag(group, elem, val) {
    const buf = Buffer.alloc(10);
    buf.writeUInt16LE(group, 0);
    buf.writeUInt16LE(elem, 2);
    buf.write('US', 4, 'ascii');
    buf.writeUInt16LE(2, 6);
    buf.writeUInt16LE(val, 8);
    chunks.push(buf);
  }

  addStringTag(0x0002, 0x0010, 'UI', '1.2.840.10008.1.2.1');
  addStringTag(0x0008, 0x0020, 'DA', '20261119');
  addStringTag(0x0008, 0x0060, 'CS', modality);
  addStringTag(0x0010, 0x0020, 'LO', 'UNIT-PAT-99');
  addStringTag(0x0018, 0x0015, 'CS', bodyPart);
  addStringTag(0x0020, 0x000d, 'UI', '1.2.826.0.1.3680043.8.498.123456');
  addStringTag(0x0028, 0x0004, 'CS', photometric);
  addUInt16Tag(0x0028, 0x0010, rows);
  addUInt16Tag(0x0028, 0x0011, cols);
  addUInt16Tag(0x0028, 0x0100, 8);
  addUInt16Tag(0x0028, 0x0103, 0);
  addStringTag(0x0028, 0x1050, 'DS', '128');
  addStringTag(0x0028, 0x1051, 'DS', '256');

  // PixelData (7fe0,0010) OB with 2 reserved bytes + 4-byte length
  const pixels = Buffer.alloc(rows * cols);
  for (let i = 0; i < pixels.length; i++) pixels[i] = i * 4;
  const pixHdr = Buffer.alloc(12);
  pixHdr.writeUInt16LE(0x7fe0, 0);
  pixHdr.writeUInt16LE(0x0010, 2);
  pixHdr.write('OB', 4, 'ascii');
  pixHdr.writeUInt16LE(0, 6);
  pixHdr.writeUInt32LE(pixels.length, 8);
  chunks.push(pixHdr, pixels);

  return Buffer.concat(chunks);
}

test('DICOM binary parser extracts metadata and inverts MONOCHROME1 pixels', () => {
  const dcm = buildPart10Dicom({ modality: 'CR', bodyPart: 'CHEST', photometric: 'MONOCHROME1' });
  const parsed = parseDicomBuffer(dcm);
  assert.equal(parsed.isValidDicom, true);
  assert.equal(parsed.modality, 'CR');
  assert.equal(parsed.bodyPart, 'CHEST');
  assert.equal(parsed.metadata.patientId, 'UNIT-PAT-99');
  assert.equal(parsed.metadata.photometricInterpretation, 'MONOCHROME1');
  assert.equal(parsed.metadata.rows, 8);
  assert.equal(parsed.metadata.columns, 8);
  assert.ok(parsed.grayscalePreview);
  // First raw pixel is 0; under MONOCHROME1 with W=256/L=128 it must invert to 255
  assert.equal(parsed.grayscalePreview.pixels[0], 255);

  const png = encodeGrayscalePng(8, 8, parsed.grayscalePreview.pixels);
  assert.equal(png.subarray(0, 8).toString('hex'), '89504e470d0a1a0a');
});

test('DICOM binary parser rejects invalid non-DICOM payload', () => {
  const parsed = parseDicomBuffer(Buffer.from('not a dicom file'));
  assert.equal(parsed.isValidDicom, false);
});

const BASE_URL = 'http://127.0.0.1:3000';

test('Live API: GET /api/health, /api/models, /api/metrics return honest state', async () => {
  const healthRes = await fetch(`${BASE_URL}/api/health`);
  assert.equal(healthRes.status, 200);
  const health = await healthRes.json();
  assert.equal(health.status, 'ok');
  assert.match(health.disclaimer, /NOT FOR CLINICAL USE/);

  const modelsRes = await fetch(`${BASE_URL}/api/models`);
  assert.equal(modelsRes.status, 200);
  const { models } = await modelsRes.json();
  assert.equal(models.length, 2);
  assert.equal(models[0].weightsLoaded, false);

  const metricsRes = await fetch(`${BASE_URL}/api/metrics`);
  assert.equal(metricsRes.status, 200);
  const metrics = await metricsRes.json();
  assert.equal(metrics.evaluationAvailable, false);
  assert.equal(metrics.classification.macroAuroc, null);
  assert.equal(metrics.segmentation.meanDice, null);
});

test('Live API: POST /api/upload parses real DICOM, gates missing weights honestly, and persists review + audit log', async () => {
  const dcm = buildPart10Dicom({ modality: 'CR', bodyPart: 'CHEST', photometric: 'MONOCHROME2' });
  const uploadRes = await fetch(`${BASE_URL}/api/upload`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      fileBase64: dcm.toString('base64'),
      fileName: 'integration_test_cxr.dcm',
      mimeType: 'application/dicom',
      title: 'Automated Test Chest DICOM',
      modality: 'CR',
      bodyPart: 'CHEST',
    }),
  });
  assert.equal(uploadRes.status, 201);
  const { study } = await uploadRes.json();
  assert.equal(study.isDemo, false);
  assert.equal(study.processingStatus, 'weights_unavailable');
  assert.equal(study.confidence, null);
  assert.equal(study.dicom.patientId, 'UNIT-PAT-99');
  assert.ok(study.imageUrl.endsWith('.png'));

  // Reject unsupported extension
  const badRes = await fetch(`${BASE_URL}/api/upload`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      fileBase64: Buffer.from('abc').toString('base64'),
      fileName: 'bad_script.sh',
    }),
  });
  assert.equal(badRes.status, 400);

  // Approve then Reject to confirm append-only audit log
  const approveRes = await fetch(`${BASE_URL}/api/studies/${study.id}/review`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      decision: 'approved',
      reviewer: 'Dr. Automated Test',
      notes: 'Initial approval check.',
    }),
  });
  assert.equal(approveRes.status, 200);
  assert.equal((await approveRes.json()).study.status, 'approved');

  const rejectRes = await fetch(`${BASE_URL}/api/studies/${study.id}/review`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      decision: 'rejected',
      reviewer: 'Dr. Automated Override',
      notes: 'Testing append-only status update.',
    }),
  });
  assert.equal(rejectRes.status, 200);
  const rejectBody = await rejectRes.json();
  assert.equal(rejectBody.study.status, 'rejected');
  assert.equal(rejectBody.auditEntry.decision, 'review_updated');

  const auditRes = await fetch(`${BASE_URL}/api/audit`);
  assert.equal(auditRes.status, 200);
  const { auditLog } = await auditRes.json();
  const studyEvents = auditLog.filter((e) => e.studyId === study.id).map((e) => e.decision);
  assert.ok(studyEvents.includes('dicom_ingested'));
  assert.ok(studyEvents.includes('approved'));
  assert.ok(studyEvents.includes('review_updated'));
});
