import { copyFile, mkdir, access, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { DRAWING_MODEL } from '../src/informative-drawings-config.js';
import { constants } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const browserPortDir = resolve(scriptDir, '..');
const repoDir = resolve(browserPortDir, '..');
const source = resolve(repoDir, 'outputs', 'onnx', 'virtual_sketching_step.onnx');
const destination = resolve(browserPortDir, 'public', 'models', 'virtual_sketching_step.onnx');
const sampleSource = resolve(repoDir, 'sample_inputs', 'clean_line_drawings', 'duck.png');
const sampleDestination = resolve(browserPortDir, 'public', 'samples', 'duck.png');
const ortDist = resolve(browserPortDir, 'node_modules', 'onnxruntime-web', 'dist');
const ortPublicDir = resolve(browserPortDir, 'public', 'ort-wasm');
const wasmFiles = [
  'ort-wasm-simd-threaded.asyncify.wasm',
  'ort-wasm-simd-threaded.jsep.wasm',
  'ort-wasm-simd-threaded.wasm',
  'ort-wasm-simd-threaded.asyncify.mjs',
  'ort-wasm-simd-threaded.jsep.mjs',
  'ort-wasm-simd-threaded.mjs',
];

try {
  await access(source, constants.R_OK);
} catch {
  throw new Error(`ONNX model not found at ${source}. Run the exporter/conversion first.`);
}

await mkdir(dirname(destination), { recursive: true });
await copyFile(source, destination);
console.log(`Copied ONNX model to ${destination}`);
await mkdir(dirname(sampleDestination), { recursive: true });
await copyFile(sampleSource, sampleDestination);
console.log(`Copied bundled sample to ${sampleDestination}`);

const photoSource = resolve(repoDir, 'outputs', 'onnx', 'virtual_sketching_faces_step.onnx');
const photoDestination = resolve(browserPortDir, 'public', 'models', 'virtual_sketching_faces_step.onnx');
await access(photoSource, constants.R_OK).catch(() => {
  throw new Error(`Photo ONNX model missing: ${photoSource}. See README photo export instructions.`);
});
await copyFile(photoSource, photoDestination);
await copyFile(resolve(repoDir, 'sample_inputs', 'faces', '1390.png'), resolve(browserPortDir, 'public', 'samples', 'portrait.png'));
console.log('Copied photo model and portrait sample to public/');

const roughSource = resolve(repoDir, 'outputs', 'onnx', 'virtual_sketching_rough_step.onnx');
await access(roughSource, constants.R_OK).catch(() => {
  throw new Error(`Rough-sketch ONNX model missing: ${roughSource}. See README export instructions.`);
});
await copyFile(roughSource, resolve(browserPortDir, 'public', 'models', 'virtual_sketching_rough_step.onnx'));
await copyFile(resolve(repoDir, 'sample_inputs', 'rough_sketches', 'penguin1.png'), resolve(browserPortDir, 'public', 'samples', 'penguin1.png'));
console.log('Copied rough-sketch model and penguin sample to public/');

const drawingCache = resolve(repoDir, 'outputs', 'onnx', DRAWING_MODEL.filename);
const validDrawing = data => createHash('sha256').update(data).digest('hex') === DRAWING_MODEL.sha256;
let drawingData = await readFile(drawingCache).catch(() => null);
if (!drawingData || !validDrawing(drawingData)) {
  console.log('Downloading pinned Informative Drawings model (17 MB)…');
  const response = await fetch(DRAWING_MODEL.url);
  if (!response.ok) throw new Error(`Drawing model download failed (HTTP ${response.status}).`);
  drawingData = Buffer.from(await response.arrayBuffer());
  if (!validDrawing(drawingData)) throw new Error('Drawing model checksum mismatch.');
  await mkdir(dirname(drawingCache), { recursive: true });
  await writeFile(drawingCache, drawingData);
}
await writeFile(resolve(browserPortDir, 'public', 'models', DRAWING_MODEL.filename), drawingData);
console.log('Copied verified Informative Drawings model to public/');

await mkdir(ortPublicDir, { recursive: true });
for (const wasmFile of wasmFiles) {
  await copyFile(resolve(ortDist, wasmFile), resolve(ortPublicDir, wasmFile));
}
console.log(`Copied ONNX Runtime Web WASM assets to ${ortPublicDir}`);
