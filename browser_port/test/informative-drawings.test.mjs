import test from 'node:test';
import assert from 'node:assert/strict';
import { drawingTensorInput, cropDrawingOutput, cleanDrawing, squareDrawing } from '../src/informative-drawings.js';
import { getMode } from '../src/modes.js';

test('RGB planes retain polarity and channel order; odd edges replicate to multiples of four', () => {
  const width = 33, height = 35;
  const data = Float32Array.from({ length: width * height * 3 }, (_, i) => (i % 251) / 250);
  const tensor = drawingTensorInput({ data, width, height, channels: 3 });
  assert.deepEqual(tensor.dims, [1, 3, 36, 36]);
  for (const [x, y] of [[0, 0], [12, 19], [32, 34], [35, 35]]) for (let c = 0; c < 3; c++) {
    assert.equal(tensor.data[c * 36 * 36 + y * 36 + x], data[(Math.min(y, 34) * 33 + Math.min(x, 32)) * 3 + c]);
  }
});

test('tiny RGB images are padded safely and invalid input is rejected', () => {
  const tensor = drawingTensorInput({ data: new Float32Array([0, .5, 1]), width: 1, height: 1, channels: 3 });
  assert.deepEqual(tensor.dims, [1, 3, 32, 32]);
  assert.ok(tensor.data.subarray(0, 1024).every(v => v === 0));
  assert.ok(tensor.data.subarray(1024, 2048).every(v => v === .5));
  assert.ok(tensor.data.subarray(2048).every(v => v === 1));
  assert.throws(() => drawingTensorInput({ data: [], width: 0, height: 0, channels: 3 }));
});

test('crop discards network padding before square white padding for vectorization', () => {
  const data = Float32Array.from({ length: 16 }, (_, i) => i / 16);
  const drawing = cropDrawingOutput(data, [1, 1, 4, 4], 3, 2);
  assert.deepEqual(Array.from(drawing.data), [0, 1/16, 2/16, 4/16, 5/16, 6/16]);
  const square = squareDrawing(drawing);
  assert.equal(square.channels, 1);
  assert.deepEqual([square.width, square.height], [3, 3]);
  assert.deepEqual(Array.from(square.data), [...drawing.data, 1, 1, 1]);
  assert.throws(() => cropDrawingOutput(data, [1, 3, 4, 4], 3, 2));
});

test('cleanup snaps near-white to exact white, keeps dark detail and is reversible from cached output', () => {
  const original = { data: new Float32Array([0, .5, .96, .99, 1]), width: 5, height: 1, channels: 1 };
  const conservative = cleanDrawing(original);
  assert.equal(conservative.data[2], original.data[2]);
  assert.equal(conservative.data[3], 1);
  assert.equal(cleanDrawing(original, .95).data[2], 1);
  assert.equal(cleanDrawing(original, 1).data[3], original.data[3]);
  assert.notEqual(original.data[3], 1);
  assert.notEqual(original.data, conservative.data);
  assert.throws(() => cleanDrawing(original, NaN));
});

test('two-stage mode uses the clean-line checkpoint and a multi-round budget', () => {
  const mode = getMode('drawing');
  assert.equal(mode.modelUrl, getMode('line').modelUrl);
  assert.equal(mode.vectorMode, 'line');
  assert.equal(mode.channels, 1);
  assert.ok(mode.rounds > 1);
});
