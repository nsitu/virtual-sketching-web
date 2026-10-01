import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import * as ort from 'onnxruntime-web';
import { BrowserVectorizer } from '../src/model.js';
import { resizeArea, cropAndResize, normalizeM1To1 } from '../src/preprocess.js';
import { renderStrokePatch, pastePatch } from '../src/raster.js';

const root = new URL('../../outputs/parity/', import.meta.url);
const ref = JSON.parse(readFileSync(new URL('reference.json', root)));
function read(spec) {
  const bytes = readFileSync(new URL(spec.file, root));
  return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}
function error(actual, expected) {
  assert.equal(actual.length, expected.length);
  let max = 0, sum = 0;
  for (let i = 0; i < actual.length; i++) {
    const diff = Math.abs(actual[i] - expected[i]);
    assert.ok(Number.isFinite(diff));
    max = Math.max(max, diff); sum += diff;
  }
  return { max, mean: sum / actual.length };
}
function check(label, actual, spec, tolerance) {
  const diff = error(actual, read(spec));
  console.log(label, diff);
  assert.ok(diff.max <= tolerance, `${label}: ${diff.max} > ${tolerance}`);
  return diff;
}
const image = { data: read(ref.photo), width: ref.size, height: ref.size, channels: 1 };
for (const [index, stroke] of ref.strokes.entries()) {
  const patch = renderStrokePatch(stroke.params, stroke.width);
  check(`OpenCV stroke ${index}`, patch, stroke.patch, 0);
  const canvas = new Float32Array(ref.size ** 2);
  pastePatch(canvas, patch, stroke.cursor, ref.size, stroke.window);
  check(`TF paste ${index}`, canvas, stroke.pasted, .0002);
}

for (const [index, test] of ref.cases.entries()) {
  check(`TF AREA ${index}`, resizeArea(image, 128, 128).data, test.inputs['step_entire_photo:0'], .0001);
  check(`TF crop ${index}`, normalizeM1To1(cropAndResize(image, test.cursor, test.window, 128, 1).data), test.inputs['step_patch_photo:0'], .0003);
}

// Replay Python's saved trajectory through the production JS raster/paste code.
const replay = new Float32Array(ref.size ** 2);
let index = 0;
for (const [round, length] of ref.sequence.lengths.entries()) {
  let cursor = ref.sequence.cursors[round];
  let window = 128, width = .01;
  for (let step = 0; step < length; step++, index++) {
    const [pen, ...params] = ref.sequence.params[index];
    if (!pen) pastePatch(replay, renderStrokePatch(params, width), cursor, ref.size, window);
    const nextWindow = Math.max(32, Math.min(ref.size, params[5] * window));
    width = params[4] * window / nextWindow;
    cursor = [Math.max(0, Math.min(ref.size - 1, cursor[0] * ref.size + params[3] * window / 2)) / ref.size,
              Math.max(0, Math.min(ref.size - 1, cursor[1] * ref.size + params[2] * window / 2)) / ref.size];
    window = nextWindow;
  }
}
const replayError = error(replay, read(ref.sequence.canvas));
console.log('Python trajectory -> JS canvas', replayError);
assert.ok(replayError.mean < .002);

ort.env.wasm.numThreads = 1;
const session = await ort.InferenceSession.create(new Uint8Array(readFileSync(new URL('../../outputs/onnx/virtual_sketching_step.onnx', import.meta.url))), { executionProviders: ['wasm'] });
let maxPredictionError = 0;
for (const [index, test] of ref.cases.entries()) {
  const feeds = Object.fromEntries(Object.entries(test.inputs).map(([name, spec]) => [name, typeof spec === 'number'
    ? new ort.Tensor('int32', new Int32Array([spec]), []) : new ort.Tensor('float32', read(spec), spec.dims)]));
  const outputs = await session.run(feeds);
  for (const [name, spec] of Object.entries(test.outputs)) {
    const diff = check(`TF1 -> ONNX WASM ${index} ${name}`, outputs[name].data, spec, .0001);
    maxPredictionError = Math.max(maxPredictionError, diff.max);
  }
  for (const tensor of [...Object.values(feeds), ...Object.values(outputs)]) tensor.dispose();

  // Also validate the complete production JS preprocessing + inference step.
  const vectorizer = new BrowserVectorizer();
  vectorizer.session = session;
  vectorizer.setImage(image);
  vectorizer.resetState(test.cursor);
  vectorizer.canvas = read(test.canvas);
  vectorizer.previousWindowSize = test.window;
  vectorizer.state = read(test.inputs['step_state_in:0']);
  const result = await vectorizer.step();
  check(`JS preprocessing + WASM ${index}`, result.params, test.outputs['other_params:0'], .001);
}

const vectorizer = new BrowserVectorizer();
vectorizer.session = session;
vectorizer.setImage(image);
const start = performance.now();
let seed = 42;
const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
const result = await vectorizer.vectorize({ maxRounds: 10, maxStepsPerRound: 500, random });
const seconds = (performance.now() - start) / 1000;
function coverage(canvas) {
  let target = 0, overlap = 0, ink = 0;
  for (let i = 0; i < canvas.length; i++) {
    const t = image.data[i] < 1;
    const p = canvas[i] > 0;
    if (t) target++;
    if (p) ink++;
    if (t && p) overlap++;
  }
  return { recall: overlap / target, precision: overlap / ink };
}
// Pixel recall is low even in Python because its traced lines are thinner than
// the source's antialiasing. Compare against the actual Python baseline, not
// an arbitrary absolute recall target. Stochastic full runs aren't bitwise equal.
const baseline = coverage(read(ref.sequence.canvas));
const metrics = { ...coverage(result.canvas), baseline, steps: result.totalSteps, seconds, replayError, maxPredictionError };
console.log('Autonomous JS/WASM duck', metrics);
writeFileSync(new URL('browser.pgm', root), Buffer.concat([Buffer.from(`P5\n${ref.size} ${ref.size}\n255\n`), Buffer.from(result.canvas.map(v => Math.round((1-v)*255)))]));
writeFileSync(new URL('metrics.json', root), JSON.stringify(metrics, null, 2));
assert.ok(metrics.recall >= baseline.recall - .05, 'Duck coverage should be comparable to Python');
assert.ok(metrics.precision >= baseline.precision - .05, 'Duck stray ink should be comparable to Python');
await session.release();
console.log('Parity checks passed. Artifacts:', fileURLToPath(root));
