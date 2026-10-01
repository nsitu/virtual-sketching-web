import test from 'node:test';
import assert from 'node:assert/strict';
import { BrowserVectorizer } from '../src/model.js';
import { getMode } from '../src/modes.js';
import { prepareRoughSketch, chooseRoughCursor } from '../src/preprocess.js';

test('rough mode uses a separate RGB checkpoint and 10 × 128 defaults', () => {
  const config = getMode('rough');
  assert.equal(config.channels, 3);
  assert.equal(config.rounds, 10);
  assert.equal(config.steps, 128);
  assert.notEqual(config.modelUrl, getMode('photo').modelUrl);
  assert.notEqual(config.modelUrl, getMode('line').modelUrl);
});

test('rough preprocessing preserves native RGB and samples padding ten pixels in', () => {
  for (const [width, height] of [[32, 20], [20, 32]]) {
    const data = new Float32Array(width * height * 3).fill(1);
    data.set([.25, .5, .75], ((height - 10) * width + width - 10) * 3);
    const image = prepareRoughSketch({ data, width, height, channels: 3 });
    assert.equal(image.width, 32);
    assert.equal(image.height, 32);
    assert.deepEqual(Array.from(image.data.slice(-3)), [.25, .5, .75]);
    for (let y = 0; y < height; y++) {
      assert.deepEqual(image.data.slice(y * 32 * 3, (y * 32 + width) * 3), data.slice(y * width * 3, (y + 1) * width * 3));
    }
  }
});

test('rough preprocessing handles tiny images and rejects grayscale', () => {
  const input = { data: new Float32Array([.5, .5, .5, 1, 1, 1]), width: 1, height: 2, channels: 3 };
  assert.ok(prepareRoughSketch(input).data.every(Number.isFinite));
  assert.throws(() => prepareRoughSketch({ ...input, channels: 1 }), /RGB/);
});

test('rough cursor uses signed offsets from last cursor and clamps at edges', () => {
  const image = { data: new Float32Array(640 * 640 * 3), width: 640, height: 640, channels: 3 };
  let values = [.5, .5, 0, .99];
  const cursor = chooseRoughCursor(image, [.5, .5], 128, () => values.shift());
  assert.deepEqual(cursor, [Math.fround(.2), Math.fround(.8)]);
  values = [.99, .99, .99, .99];
  assert.deepEqual(chooseRoughCursor(image, [.99, .99], 128, () => values.shift()), [Math.fround(639 / 640), Math.fround(639 / 640)]);
});

test('rough cursor bounds retries for white input', () => {
  const image = { data: new Float32Array(128 * 128 * 3).fill(1), width: 128, height: 128, channels: 3 };
  let calls = 0;
  chooseRoughCursor(image, [.5, .5], 128, () => { calls++; return .5; });
  assert.equal(calls, 80); // 20 trials × two offsets + two signs
});

test('rough sampler uses all rounds, stops on pen-ups and resets per round', async () => {
  const vectorizer = new BrowserVectorizer({ mode: 'rough' });
  vectorizer.session = {};
  vectorizer.setImage({ data: new Float32Array(128 * 128 * 3), width: 128, height: 128, channels: 3 });
  const rounds = [];
  vectorizer.step = async () => {
    vectorizer.state[0]++;
    vectorizer.cursor = [.8, .8];
    // A full ink canvas must not trigger the clean sampler's coverage stop.
    vectorizer.canvas.fill(1);
    return { canvas: vectorizer.canvas, params: [], stroke: false };
  };
  const result = await vectorizer.vectorize({ random: () => .5, onRound: ({ cursor }) => {
    assert.equal(vectorizer.state[0], 0);
    rounds.push(cursor);
  } });
  assert.equal(rounds.length, 10);
  assert.equal(result.totalSteps, 120);
  assert.equal(result.steps[0].round, 0);
  assert.equal(result.steps.at(-1).round, 9);
  assert.ok(rounds[1][0] > .8, 'Next round starts relative to previous final cursor');
});

test('rough sampler respects 128-step cap and 48-step recurrent resets', async () => {
  const v = new BrowserVectorizer({ mode: 'rough' });
  v.session = {};
  v.setImage({ data: new Float32Array(128 * 128 * 3), width: 128, height: 128, channels: 3 });
  const states = [];
  v.step = async () => {
    states.push(v.state[0]); v.state[0]++;
    return { canvas: v.canvas, params: [], stroke: true };
  };
  const result = await v.vectorize({ maxRounds: 1, random: () => .5 });
  assert.equal(result.totalSteps, 128);
  assert.deepEqual([states[47], states[48], states[95], states[96]], [47, 0, 47, 0]);
});
