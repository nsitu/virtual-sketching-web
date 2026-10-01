import test from 'node:test';
import assert from 'node:assert/strict';
import { clamp, cropAndResize, normalizeM1To1, resizeBilinear, resizeArea, chooseInitialCursor, chooseUndrawnCursor } from '../src/preprocess.js';

test('clamp and normalization match the model input convention', () => {
  assert.equal(clamp(-1, 0, 1), 0);
  assert.equal(clamp(2, 0, 1), 1);
  assert.deepEqual(Array.from(normalizeM1To1(new Float32Array([0, 0.5, 1]))), [-1, 0, 1]);
});

test('area downsampling preserves a thin line missed by bilinear sampling', () => {
  const image = { data: new Float32Array([0, 1, 1, 1, 1]), width: 5, height: 1, channels: 1 };
  assert.ok(Math.abs(resizeArea(image, 1, 1).data[0] - .8) < 1e-7);
  assert.equal(resizeBilinear(image, 1, 1).data[0], 1);
});

test('photo crops extrapolate white while ink crops extrapolate zero', () => {
  const image = { data: new Float32Array(16).fill(.5), width: 4, height: 4, channels: 1 };
  assert.equal(cropAndResize(image, [0, 0], 4, 4, 1).data[0], 1);
  assert.equal(cropAndResize(image, [0, 0], 4, 4).data[0], 0);
});

test('blank images terminate and undrawn selection uses middle-of-cell exploration', () => {
  const image = { data: new Float32Array(256 * 256).fill(1), width: 256, height: 256, channels: 1 };
  const canvas = new Float32Array(image.data.length);
  assert.equal(chooseInitialCursor(image), null);
  assert.equal(chooseUndrawnCursor(image, canvas), null);
  // Faint ink still counts, as in Python's nonzero-ink rounding.
  for (let x = 5; x < 20; x++) image.data[x] = .99;
  assert.deepEqual(chooseUndrawnCursor(image, canvas, 128, {}, () => .5), [.25, .25]);
  for (let x = 5; x < 20; x++) canvas[x] = .01;
  assert.equal(chooseUndrawnCursor(image, canvas), null);
});

test('bilinear resize preserves a constant image', () => {
  const image = {
    data: new Float32Array([0.25, 0.25, 0.25, 0.25]),
    width: 2,
    height: 2,
    channels: 1,
  };
  const resized = resizeBilinear(image, 4, 4);
  assert.equal(resized.data.length, 16);
  assert.ok(Array.from(resized.data).every((value) => Math.abs(value - 0.25) < 1e-6));
});

test('crop resize returns zero for out-of-bounds samples', () => {
  const image = {
    data: new Float32Array([1, 1, 1, 1]),
    width: 2,
    height: 2,
    channels: 1,
  };
  const crop = cropAndResize(image, [0, 0], 2, 3);
  assert.equal(crop.data[0], 0);
  assert.equal(crop.data[crop.data.length - 1], 1);
});
