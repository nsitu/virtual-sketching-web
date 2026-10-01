import test from 'node:test';
import assert from 'node:assert/strict';
import { appendStrokePathElement, buildStrokeSvg, strokeToSvgPath } from '../src/svg.js';
import { BrowserVectorizer } from '../src/model.js';

const stroke = {
  params: [.25, .75, -.5, 1, .2, 1],
  previousWidth: .01,
  cursor: [.25, .75],
  imageSize: 640,
  windowSize: 128,
};

test('SVG follows Python converter coordinates with row/column swapped to x/y', () => {
  // Start (160,480), offset (64,-32), control fraction (.75,.25).
  assert.equal(strokeToSvgPath(stroke), 'M160.000 480.000Q208.000 472.000 224.000 448.000');
});

test('SVG retains fractional coordinates and unclamped endpoints at image edges', () => {
  assert.equal(strokeToSvgPath({
    ...stroke, cursor: [0, .99], imageSize: 100,
    windowSize: 33, params: [.5, .25, .5, -1, .2, 1],
  }), 'M0.000 99.000Q-4.125 103.125 -16.500 107.250');
});

test('SVG discards model width and taper', () => {
  const differentWidth = { ...stroke, previousWidth: .9, params: [.25, .75, -.5, 1, .8, 1] };
  assert.equal(buildStrokeSvg([stroke], 640), buildStrokeSvg([differentWidth], 640));
});

test('export keeps segments separate with open, uniformly stroked quadratic paths', () => {
  const svg = buildStrokeSvg([stroke, { ...stroke, cursor: [.8, .1] }], 640, 480);
  assert.match(svg, /viewBox="0 0 640 480"/);
  const paths = svg.match(/<path\b[^>]*\/>/g);
  assert.equal(paths.length, 2);
  for (const path of paths) {
    assert.match(path, /fill="none"/);
    assert.match(path, /stroke="#000"/);
    assert.match(path, /stroke-width="3.5"/);
    assert.match(path, /stroke-linecap="round"/);
    assert.match(path, /stroke-linejoin="round"/);
    const d = path.match(/ d="([^"]+)"/)[1];
    assert.deepEqual(d.match(/[A-Za-z]/g), ['M', 'Q']);
  }
  assert.doesNotMatch(svg, /<image\b/);
});

const continuation = { ...stroke, cursor: [224 / 640, 448 / 640] };

test('joining preserves both quadratic commands and removes only the shared move', () => {
  const svg = buildStrokeSvg([stroke, continuation], 640);
  assert.equal((svg.match(/<path\b/g) || []).length, 1);
  assert.ok(svg.includes(`d="${strokeToSvgPath(stroke)}${strokeToSvgPath(continuation).slice(strokeToSvgPath(continuation).indexOf('Q'))}"`));
});

test('pen boundaries and clamped or nearby endpoints cannot introduce connecting lines', () => {
  for (const next of [
    { ...continuation, startsNewPath: true },
    { ...continuation, cursor: [224.01 / 640, 448 / 640] },
    // Mimics a next cursor clamped away from the previous curve endpoint.
    { ...continuation, cursor: [223 / 640, 448 / 640] },
  ]) {
    assert.equal((buildStrokeSvg([stroke, next], 640).match(/<path\b/g) || []).length, 2);
  }
  assert.doesNotMatch(buildStrokeSvg([], 640), /<path\b/);
});

test('incremental preview and export use identical joined geometry and style', () => {
  const originalDocument = globalThis.document;
  globalThis.document = { createElementNS: () => ({
    attributes: {},
    setAttribute(name, value) { this.attributes[name] = value; },
    getAttribute(name) { return this.attributes[name]; },
  }) };
  try {
    const layer = { children: [], append(path) { this.children.push(path); },
      get lastElementChild() { return this.children.at(-1); } };
    const strokes = [stroke, continuation, { ...stroke, startsNewPath: true }];
    for (let i = 0; i < strokes.length; i++) appendStrokePathElement(layer, strokes[i], strokes[i - 1]);
    const exported = [...buildStrokeSvg(strokes, 640).matchAll(/<path\b([^>]+)\/>/g)]
      .map(match => Object.fromEntries([...match[1].matchAll(/([\w-]+)="([^"]*)"/g)].map(m => [m[1], m[2]])));
    assert.deepEqual(layer.children.map(path => path.attributes), exported);
    assert.equal(layer.children.length, 2);
  } finally {
    if (originalDocument === undefined) delete globalThis.document;
    else globalThis.document = originalDocument;
  }
});

test('real runtime records pen-up and round boundaries even when endpoints coincide', async () => {
  const vectorizer = new BrowserVectorizer();
  vectorizer.setImage({ data: new Float32Array(128 * 128), width: 128, height: 128, channels: 1 });
  const penStates = [0, 0, 1, 0, 0, 0];
  vectorizer.session = { run: async () => {
    const penUp = penStates.shift();
    const result = data => ({ data: new Float32Array(data), dispose() {} });
    return {
      'other_params:0': result([.5, .5, 0, penUp ? 0 : .125, .01, 1]),
      'pen_ras:0': result(penUp ? [0, 1] : [1, 0]),
      'state_out:0': result(new Float32Array(1024)),
    };
  } };
  for (let i = 0; i < 5; i++) await vectorizer.step();
  vectorizer.resetState(vectorizer.cursor); // New round at exactly the same point.
  await vectorizer.step();
  assert.deepEqual(vectorizer.strokes.map(s => s.startsNewPath), [true, false, true, false, true]);
  assert.equal((buildStrokeSvg(vectorizer.strokes, 128).match(/<path\b/g) || []).length, 3);
});
