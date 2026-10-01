import test from 'node:test';
import assert from 'node:assert/strict';
import { appendStrokePathElement, buildStrokeSvg, strokeToSvgPath, joinQuadraticSegments, quadraticPathData, buildQuadraticSvg, fitQuadraticPath, redrawQuadraticPaths, sampleQuadraticPath } from '../src/svg.js';
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

const line = (start, end) => ({ start, end, control: [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2], startsNewPath: true });

test('positive joining bridges an inclusive distance threshold without changing source geometry', () => {
  const segments = [line([0, 0], [10, 0]), line([13, 0], [25, 0])];
  const original = JSON.stringify(segments);
  assert.equal(joinQuadraticSegments(segments, 2.99).length, 2);
  const paths = joinQuadraticSegments(segments, 3, { midpoint: false });
  assert.equal(paths.length, 1);
  assert.equal(quadraticPathData(paths[0]), 'M0.000 0.000Q5.000 0.000 10.000 0.000L13.000 0.000Q19.000 0.000 25.000 0.000');
  assert.equal(JSON.stringify(segments), original);
  assert.equal(joinQuadraticSegments(segments, 0).length, 2);
  assert.equal(joinQuadraticSegments(segments, 3).length, 1);
});

test('nearby joining handles all endpoint orientations and unordered paths', () => {
  const a = line([0, 0], [10, 0]), b = line([12, 0], [25, 0]);
  const reverse = s => ({ ...s, start: s.end, end: s.start });
  for (const source of [[a, b], [a, reverse(b)], [reverse(a), b], [reverse(a), reverse(b)], [b, a]]) {
    const paths = joinQuadraticSegments(source, 2, { midpoint: false });
    assert.equal(paths.length, 1);
    assert.equal(paths[0].length, 2);
    assert.deepEqual(paths[0].map(s => s.control).sort((p, q) => p[0] - q[0]), [[5, 0], [18.5, 0]]);
    const ends = [paths[0][0].start[0], paths[0].at(-1).end[0]].sort((p, q) => p - q);
    assert.deepEqual(ends, [0, 25]);
    assert.equal((quadraticPathData(paths[0]).match(/L/g) || []).length, 1);
  }
});

test('nearest endpoint wins and the algorithm does not introduce branches or close cycles', () => {
  const segments = [line([0, 0], [10, 0]), line([11, 0], [20, 0]), line([10, 2], [10, 12])];
  const paths = joinQuadraticSegments(segments, 2, { midpoint: false });
  assert.equal(paths.length, 2);
  assert.deepEqual(paths[0].map(s => s.end), [[10, 0], [20, 0]]);
  // Three edges surrounding a triangle have six free endpoints. Only two joins
  // may be accepted, leaving an open path instead of forcing a closing edge.
  const triangle = [line([0, 0], [10, 0]), line([11, 1], [5, 10]), line([4, 9], [0, 1])];
  const joined = joinQuadraticSegments(triangle, 2, { midpoint: false });
  assert.equal(joined.length, 1);
  assert.equal(joined[0].length, 3);
  assert.equal((quadraticPathData(joined[0]).match(/L/g) || []).length, 2);
});

test('closed contours remain closed and distance validation rejects invalid tolerances', () => {
  const segments = [line([0, 0], [10, 0]), { ...line([10, 0], [0, 0]), startsNewPath: false }, line([1, 0], [20, 0])];
  assert.equal(joinQuadraticSegments(segments, 5).length, 2);
  for (const distance of [-1, NaN, Infinity]) assert.throws(() => joinQuadraticSegments(segments, distance), RangeError);
});

test('midpoint mode defaults on and replaces endpoints without moving controls or source records', () => {
  const a = { ...line([0, 0], [10, 2]), control: [4, 7] };
  const b = { ...line([14, 4], [30, 10]), control: [24, -2] };
  const reverse = s => ({ ...s, start: s.end, end: s.start });
  for (const input of [[a, b], [a, reverse(b)], [reverse(a), b], [reverse(a), reverse(b)], [b, a]]) {
    const original = JSON.stringify(input);
    const [joined] = joinQuadraticSegments(input, 5);
    assert.equal(joined.length, 2);
    assert.deepEqual(joined[0].end, [12, 3]);
    assert.deepEqual(joined[1].start, [12, 3]);
    assert.deepEqual(joined.map(s => s.control).sort((p, q) => p[0] - q[0]), [[4, 7], [24, -2]]);
    assert.doesNotMatch(quadraticPathData(joined), /L/);
    assert.equal(JSON.stringify(input), original);
    const bridge = joinQuadraticSegments(input, 5, { midpoint: false });
    assert.match(quadraticPathData(bridge[0]), /L/);
    assert.equal(joinQuadraticSegments(input, 0).length, 2);
  }
});

test('a middle segment can snap both ends using the original endpoint matches', () => {
  const input = [line([0, 0], [10, 0]), line([12, 0], [20, 0]), line([24, 0], [30, 0])];
  const original = JSON.stringify(input);
  const [joined] = joinQuadraticSegments(input, 4);
  assert.deepEqual(joined.map(s => [s.start, s.end]), [
    [[0, 0], [11, 0]], [[11, 0], [22, 0]], [[22, 0], [30, 0]],
  ]);
  assert.equal(JSON.stringify(input), original);
  assert.equal(joinQuadraticSegments(input, 3).length, 2);
});

test('configured export serializes the same nearby paths as preview geometry with no background', () => {
  const segments = [line([0, 0], [10, 0]), line([12, 0], [20, 0])];
  for (const midpoint of [true, false]) {
    const expected = joinQuadraticSegments(segments, 2, { midpoint }).map(quadraticPathData);
    const svg = buildQuadraticSvg(segments, 100, 80, { joinDistance: 2, midpoint });
    assert.deepEqual([...svg.matchAll(/ d="([^"]+)"/g)].map(m => m[1]), expected);
    assert.doesNotMatch(svg, /<rect\b/);
  }
});

test('redraw samples joined quadratics and emits cubic curves with preserved endpoints', () => {
  const segments = [line([0, 0], [10, 0]), { ...line([10, 0], [20, 10]), control: [12, 8] }];
  const points = sampleQuadraticPath(segments);
  assert.equal(points.length, 33);
  const fitted = fitQuadraticPath(segments, 1);
  assert.ok(fitted.length >= 1);
  assert.deepEqual(fitted[0].start, [0, 0]);
  assert.deepEqual(fitted.at(-1).end, [20, 10]);
  assert.ok(fitted.every(segment => segment.type === 'cubic'));
  const svg = buildQuadraticSvg(segments, 100, 100, { redraw: true, fitTolerance: 1 });
  assert.match(svg, /C/);
  assert.doesNotMatch(svg, /<rect\b/);
});

test('fidelity tolerance controls the number of fitted cubic segments', () => {
  const points = Array.from({ length: 41 }, (_, i) => {
    const x = i * 2.5;
    return [x, Math.sin(i / 3) * 8 + i * .35];
  });
  const source = points.slice(0, -1).map((start, i) => ({
    start, control: [start[0] + .8, start[1]], end: points[i + 1],
  }));
  const low = redrawQuadraticPaths([source], .25)[0];
  const high = redrawQuadraticPaths([source], 16)[0];
  assert.ok(low.length >= high.length, `${low.length} should be >= ${high.length}`);
  assert.deepEqual(low[0].start, source[0].start);
  assert.deepEqual(high.at(-1).end, source.at(-1).end);
});

test('invalid redraw tolerance is rejected', () => {
  assert.throws(() => buildQuadraticSvg([], 100, 100, { redraw: true, fitTolerance: -1 }), RangeError);
  assert.throws(() => redrawQuadraticPaths([], NaN), RangeError);
});

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
