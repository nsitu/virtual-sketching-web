import test from 'node:test';
import assert from 'node:assert/strict';
import { buildStrokeSvg, strokeToSvgPath } from '../src/svg.js';

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
