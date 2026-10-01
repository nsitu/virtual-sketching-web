import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import * as ort from 'onnxruntime-web';
import { BrowserVectorizer } from '../src/model.js';
import { preparePhoto, prepareRoughSketch, chooseRoughCursor, resizeArea, cropAndResize, normalizeM1To1 } from '../src/preprocess.js';
import { renderStrokePatch, pastePatch } from '../src/raster.js';

const rough = process.argv.includes('--rough');
const mode = rough ? 'rough' : 'photo';
const root = new URL(`../../outputs/${mode}-parity/`, import.meta.url);
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
for (const [index, test] of ref.preprocessing.entries()) {
  const image = { data: read(test.raw), width: test.width, height: test.height, channels: 3 };
  check(`Python RGB preprocessing ${index}`, (rough ? prepareRoughSketch : preparePhoto)(image).data, test.expected, 0);
}
const size = ref.size ?? 256;
const image = { data: read(ref.photo), width: size, height: size, channels: 3 };
for (const move of ref.moves ?? []) {
  let index = 0;
  const input = move.blank ? { ...image, data: new Float32Array(image.data.length).fill(1) } : image;
  const actual = chooseRoughCursor(input, move.cursor, 128, () => {
    assert.ok(index < move.random.length, 'Cursor consumed too many random draws');
    return move.random[index++];
  });
  assert.equal(index, move.random.length);
  assert.deepEqual(actual, move.expected);
}
ort.env.wasm.numThreads = 1;
const session = await ort.InferenceSession.create(new Uint8Array(readFileSync(new URL(`../../outputs/onnx/virtual_sketching_${rough ? 'rough' : 'faces'}_step.onnx`, import.meta.url))), { executionProviders: ['wasm'] });
let maxPredictionError = 0;
try {
  for (const [index, test] of ref.cases.entries()) {
    check(`RGB AREA ${index}`, resizeArea(image, 128, 128).data, test.inputs['step_entire_photo:0'], .0001);
    check(`RGB crop ${index}`, normalizeM1To1(cropAndResize(image, test.cursor, test.window, 128, 1).data), test.inputs['step_patch_photo:0'], .0003);
    const feeds = Object.fromEntries(Object.entries(test.inputs).map(([name, spec]) => [name, typeof spec === 'number'
      ? new ort.Tensor('int32', new Int32Array([spec]), []) : new ort.Tensor('float32', read(spec), spec.dims)]));
    const outputs = await session.run(feeds);
    for (const [name, spec] of Object.entries(test.outputs)) {
      maxPredictionError = Math.max(maxPredictionError, check(`${mode} TF1 -> WASM ${index} ${name}`, outputs[name].data, spec, .0001).max);
    }
    for (const tensor of [...Object.values(feeds), ...Object.values(outputs)]) tensor.dispose();
    const vectorizer = new BrowserVectorizer({ mode });
    vectorizer.session = session;
    vectorizer.setImage(image);
    vectorizer.resetState(test.cursor);
    vectorizer.canvas = read(test.canvas);
    vectorizer.previousWindowSize = test.window;
    vectorizer.state = read(test.inputs['step_state_in:0']);
    const result = await vectorizer.step();
    check(`${mode} JS preprocessing + WASM ${index}`, result.params, test.outputs['other_params:0'], .001);
  }
  const replay = new Float32Array(size ** 2);
  let index = 0;
  for (const [round, length] of (ref.sequence.lengths ?? [100]).entries()) {
    let cursor = ref.sequence.cursors?.[round] ?? [.5, .5], window = 128, width = .01;
    for (let step = 0; step < length; step++, index++) {
      const [pen, ...params] = ref.sequence.params[index];
      if (!pen) pastePatch(replay, renderStrokePatch(params, width), cursor, size, window);
      const nextWindow = Math.max(32, Math.min(size, params[5] * window));
      width = params[4] * window / nextWindow;
      cursor = [Math.max(0, Math.min(size - 1, cursor[0] * size + params[3] * window / 2)) / size,
                Math.max(0, Math.min(size - 1, cursor[1] * size + params[2] * window / 2)) / size];
      window = nextWindow;
    }
  }
  const replayError = error(replay, read(ref.sequence.canvas));
  assert.ok(replayError.mean < .002);
  const vectorizer = new BrowserVectorizer({ mode });
  vectorizer.session = session;
  vectorizer.setImage(image);
  let seed = 42;
  const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
  const result = await vectorizer.vectorize({ random: rough ? random : () => .5 });
  if (rough) {
    assert.equal(new Set(result.steps.map(s => s.round)).size, 10);
    assert.ok(result.totalSteps <= 1280);
  } else assert.equal(result.totalSteps, 100);
  assert.ok(result.strokeCount > 0);
  const trajectoryError = rough ? null : error(result.steps.flatMap(s => s.params), ref.sequence.params.flatMap(p => p.slice(1)));
  const canvasError = error(result.canvas, read(ref.sequence.canvas));
  // Recurrent feedback can diverge; require meaningful agreement with Python.
  assert.ok(canvasError.mean < .03, `${mode} canvas drift: ${canvasError.mean}`);
  const metrics = { maxPredictionError, replayError, trajectoryError, canvasError, steps: result.totalSteps, strokes: result.strokeCount };
  writeFileSync(new URL('browser.pgm', root), Buffer.concat([Buffer.from(`P5\n${size} ${size}\n255\n`), Buffer.from(result.canvas.map(v => Math.round((1 - v) * 255)))]));
  writeFileSync(new URL('metrics.json', root), JSON.stringify(metrics, null, 2));
  console.log(`${mode} parity passed`, metrics);
} finally {
  await session.release();
}
