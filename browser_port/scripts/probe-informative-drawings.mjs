// Local feasibility experiment; does not change the app or send images anywhere.
// Uses the existing 256x256 RGB portrait parity fixture and our installed WASM runtime.
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import * as ort from 'onnxruntime-web';
import { BrowserVectorizer } from '../src/model.js';
import { buildStrokeSvg } from '../src/svg.js';

const revision = 'd38eccbd448cdcd228fb81d708506e5e60b41ccb';
const modelUrl = `https://huggingface.co/rocca/informative-drawings-line-art-onnx/resolve/${revision}/model.onnx`;
const expectedHash = '1fef40b8f7126d827e30fbebccf95ae9b0b391795df926bf9366a821bad4f498';
const output = new URL('../../outputs/informative-drawings-probe/', import.meta.url);
await mkdir(output, { recursive: true });
const modelFile = new URL('model.onnx', output);
let bytes;
try {
  bytes = await readFile(modelFile);
} catch (error) {
  if (error.code !== 'ENOENT') throw error;
  console.log('Downloading public ONNX weights (17.2 MB)...');
  const response = await fetch(modelUrl);
  if (!response.ok) throw new Error(`Model download failed: HTTP ${response.status}`);
  bytes = Buffer.from(await response.arrayBuffer());
  assert.equal(createHash('sha256').update(bytes).digest('hex'), expectedHash);
  await writeFile(modelFile, bytes);
}
assert.equal(createHash('sha256').update(bytes).digest('hex'), expectedHash);
ort.env.wasm.numThreads = 1;
const startLoad = performance.now();
const session = await ort.InferenceSession.create(new Uint8Array(bytes), { executionProviders: ['wasm'] });
const metrics = {
  modelUrl, sha256: expectedHash, bytes: bytes.length, runtime: 'onnxruntime-web 1.23.2 / WASM / Node / 1 thread',
  loadSeconds: (performance.now() - startLoad) / 1000,
  inputs: session.inputNames, outputs: session.outputNames, shapeChecks: [], vectorization: [],
};
let drawingSessionReleased = false;
try {
  for (const [height, width] of [[64, 96], [65, 97]]) {
    const input = new ort.Tensor('float32', new Float32Array(3 * height * width).fill(.5), [1, 3, height, width]);
    let outputs;
    try {
      outputs = await session.run({ input });
      metrics.shapeChecks.push({ input: input.dims, output: outputs.output.dims });
    } catch (error) {
      metrics.shapeChecks.push({ input: input.dims, error: String(error.message) });
    } finally {
      input.dispose();
      Object.values(outputs || {}).forEach(tensor => tensor.dispose());
    }
  }
  const fixtureBytes = await readFile(new URL('../../outputs/photo-parity/photo.f32', import.meta.url));
  const rgb = new Float32Array(fixtureBytes.buffer.slice(fixtureBytes.byteOffset, fixtureBytes.byteOffset + fixtureBytes.byteLength));
  const size = 256, pixels = size * size;
  assert.equal(rgb.length, pixels * 3);
  const chw = new Float32Array(rgb.length);
  for (let i = 0; i < pixels; i++) for (let c = 0; c < 3; c++) chw[c * pixels + i] = rgb[i * 3 + c];
  const input = new ort.Tensor('float32', chw, [1, 3, size, size]);
  let outputs;
  let drawing;
  const startInference = performance.now();
  try {
    outputs = await session.run({ input });
    metrics.drawingSeconds = (performance.now() - startInference) / 1000;
    assert.deepEqual(outputs.output.dims, [1, 1, size, size]);
    drawing = new Float32Array(outputs.output.data);
    assert.ok(drawing.every(value => Number.isFinite(value) && value >= 0 && value <= 1));
  } finally {
    input.dispose();
    Object.values(outputs || {}).forEach(tensor => tensor.dispose());
  }
  metrics.nonwhitePercent = Object.fromEntries([1, .995, .98, .95, .9, .5].map(threshold => [threshold, 100 * drawing.filter(value => value < threshold).length / pixels]));
  console.log('Drawing model:', JSON.stringify({ seconds: metrics.drawingSeconds, shapes: metrics.shapeChecks, nonwhitePercent: metrics.nonwhitePercent }));
  const pgm = values => Buffer.concat([Buffer.from(`P5\n${size} ${size}\n255\n`), Buffer.from(values.map(value => Math.round(value * 255)))]);
  await writeFile(new URL('drawing.pgm', output), pgm(drawing));
  // Release the first model before creating the vectorizer session, as the UI should.
  await session.release();
  drawingSessionReleased = true;
  const vectorSession = await ort.InferenceSession.create(new Uint8Array(await readFile(new URL('../../outputs/onnx/virtual_sketching_step.onnx', import.meta.url))), { executionProviders: ['wasm'] });
  try {
    for (const threshold of [1, .995, .98]) {
      const image = { data: drawing.map(value => value >= threshold ? 1 : value), width: size, height: size, channels: 1 };
      const vectorizer = new BrowserVectorizer({ mode: 'line' });
      vectorizer.session = vectorSession;
      vectorizer.setImage(image);
      let seed = 42;
      const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
      const startVectorize = performance.now();
      const result = await vectorizer.vectorize({ maxRounds: 4, maxStepsPerRound: 128, random });
      const seconds = (performance.now() - startVectorize) / 1000;
      const svg = buildStrokeSvg(vectorizer.strokes, size);
      metrics.vectorization.push({ threshold, roundsLimit: 4, stepsPerRoundLimit: 128, steps: result.totalSteps, strokes: result.strokeCount, paths: (svg.match(/<path\b/g) || []).length, seconds });
      console.log('Vectorization:', JSON.stringify(metrics.vectorization.at(-1)));
      await writeFile(new URL(`vector-${threshold}.svg`, output), svg);
      await writeFile(new URL(`input-${threshold}.pgm`, output), pgm(image.data));
    }
  } finally {
    await vectorSession.release();
  }
} finally {
  if (!drawingSessionReleased) await session.release();
}
await writeFile(new URL('metrics.json', output), JSON.stringify(metrics, null, 2));
console.log(JSON.stringify(metrics, null, 2));
