import test from 'node:test';
import assert from 'node:assert/strict';
import { BrowserVectorizer } from '../src/model.js';
import { preparePhoto, chooseInitialCursor, resizePhotoBilinear } from '../src/preprocess.js';
import { getMode } from '../src/modes.js';

test('modes select distinct checkpoints, channel counts and sampling defaults', () => {
  assert.notEqual(getMode('line').modelUrl, getMode('photo').modelUrl);
  assert.equal(getMode('line').channels, 1);
  assert.equal(getMode('photo').channels, 3);
  assert.equal(getMode('photo').steps, 100);
  assert.equal(getMode('photo').rounds, 1);
  assert.throws(() => getMode('invalid'), /Unknown input mode/);
});

test('photo pads right and bottom black without shifting or losing RGB', () => {
  for (const [width, height] of [[256, 128], [128, 256]]) {
    const data = new Float32Array(width * height * 3);
    for (let i = 0; i < data.length; i += 3) data.set([1, .5, .25], i);
    const photo = preparePhoto({ data, width, height, channels: 3 });
    assert.equal(photo.width, 256);
    assert.equal(photo.channels, 3);
    assert.deepEqual(Array.from(photo.data.subarray(0, 3)), [1, .5, .25]);
    assert.deepEqual(Array.from(photo.data.slice(-3)), [0, 0, 0]);
  }
});

test('photo antialiased resize retains constant RGB channels', () => {
  const data = new Float32Array(513 * 513 * 3);
  for (let i = 0; i < data.length; i += 3) data.set([1, 0, 128 / 255], i);
  const resized = resizePhotoBilinear({ data, width: 513, height: 513, channels: 3 });
  for (let i = 0; i < resized.data.length; i += 3) {
    assert.equal(resized.data[i], 1);
    assert.equal(resized.data[i + 1], 0);
    assert.equal(resized.data[i + 2], Math.fround(128 / 255));
  }
});

test('RGB cursor eligibility checks green/blue too', () => {
  const data = new Float32Array(4 * 4 * 3).fill(1);
  data[1] = 0;
  assert.deepEqual(chooseInitialCursor({ data, width: 4, height: 4, channels: 3 }, 128, () => .5), [.5, .5]);
});

test('runtime rejects incompatible input channels and photo sizes', () => {
  const line = new BrowserVectorizer();
  const photo = new BrowserVectorizer({ mode: 'photo' });
  const rgb = { data: new Float32Array(256 * 256 * 3), width: 256, height: 256, channels: 3 };
  assert.throws(() => line.setImage(rgb), /expects 1/);
  assert.throws(() => photo.setImage({ ...rgb, channels: 1 }), /expects 3/);
  assert.throws(() => photo.setImage({ ...rgb, width: 128, height: 128 }), /256/);
  photo.setImage(rgb);
  assert.equal(photo.fullPhotoSmall.length, 128 * 128 * 3);
});

test('photo uses 100 steps through pen-ups, one pass and 48-step state resets', async () => {
  const photo = new BrowserVectorizer({ mode: 'photo' });
  photo.session = {}; // Loop-only test; no weights needed.
  photo.setImage({ data: new Float32Array(256 * 256 * 3), width: 256, height: 256, channels: 3 });
  const before = [];
  photo.step = async () => {
    before.push(photo.state[0]);
    photo.state[0] += 1;
    return { canvas: photo.canvas, params: [], stroke: false };
  };
  const result = await photo.vectorize({ maxRounds: 10, eosPatience: 1, random: () => .5 });
  assert.equal(result.totalSteps, 100);
  assert.equal(result.strokeCount, 0);
  assert.deepEqual([before[0], before[47], before[48], before[95], before[96]], [0, 47, 0, 47, 0]);
  assert.ok(result.steps.every(s => s.round === 0));
});

test('line mode still stops on blank inputs and consecutive pen-ups', async () => {
  const line = new BrowserVectorizer();
  line.session = {};
  const image = { data: new Float32Array(128 * 128).fill(1), width: 128, height: 128, channels: 1 };
  line.setImage(image);
  line.step = async () => ({ canvas: line.canvas, params: [], stroke: false });
  assert.equal((await line.vectorize()).totalSteps, 0);
  image.data[0] = 0;
  line.setImage(image);
  assert.equal((await line.vectorize({ maxRounds: 1, random: () => .5 })).totalSteps, 12);
});
