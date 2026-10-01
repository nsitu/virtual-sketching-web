const SVG_NS = 'http://www.w3.org/2000/svg';
import fitCurve from 'fit-curve';

// Use the uniform width and continuous path structure of Python's cluster export.
const PATH_STYLE = {
  fill: 'none',
  stroke: '#000',
  'stroke-width': '3.5',
  'stroke-linecap': 'round',
  'stroke-linejoin': 'round',
};

// Keep preview colors stable while the joining controls are adjusted. The
// golden-angle hue spacing gives neighboring paths visibly different colors
// without using random values that would jump on every redraw.
export function previewPathColor(index) {
  const hue = (19 + Number(index) * 137.508) % 360;
  return `hsl(${hue.toFixed(1)}, 72%, 48%)`;
}

// Match the Python SVG converter's continuous coordinates, not the quantized
// circle stamps used by the inference rasterizer. Model parameters are in
// (row, column) order; the saved cursor and SVG coordinates are (x, y).
export function strokeToQuadratic(stroke) {
  const { params, cursor, imageSize, windowSize } = stroke;
  const [controlRow, controlColumn, endRow, endColumn] = params;
  const startX = cursor[0] * imageSize;
  const startY = cursor[1] * imageSize;
  const dx = endColumn * windowSize / 2;
  const dy = endRow * windowSize / 2;
  const controlX = startX + dx * controlColumn;
  const controlY = startY + dy * controlRow;
  // Only the next inference cursor is clamped; the curve endpoint is not.
  return {
    start: [startX, startY], control: [controlX, controlY],
    end: [startX + dx, startY + dy], startsNewPath: stroke.startsNewPath,
  };
}

const pointText = point => point.map(value => value.toFixed(3)).join(' ');
const curveText = segment => `Q${pointText(segment.control)} ${pointText(segment.end)}`;

function cubicText(segment) {
  return `C${pointText(segment.control1)} ${pointText(segment.control2)} ${pointText(segment.end)}`;
}

function canJoin(previous, next) {
  // Equality at export precision only: never bridge gaps or alter controls.
  // Legacy records/SVGs without pen metadata can use endpoint continuity alone.
  return previous && !next.startsNewPath && pointText(previous.end) === pointText(next.start);
}

export function joinQuadraticSegments(segments, joinDistance = 0, { midpoint = true } = {}) {
  if (!Number.isFinite(joinDistance) || joinDistance < 0) throw new RangeError('Join distance must be a finite, nonnegative number.');
  const paths = [];
  let previous;
  for (const segment of segments) {
    if (canJoin(previous, segment)) paths.at(-1).push(segment);
    else paths.push([segment]);
    previous = segment;
  }
  return joinDistance > 0 ? joinNearbyPaths(paths, joinDistance, midpoint) : paths;
}

// Match original path endpoints once, nearest first. Each endpoint gets at most
// one partner; union-find prevents cycles. Spatial buckets avoid checking every
// pair for the common case of many widely separated paths.
function joinNearbyPaths(paths, distance, midpoint) {
  const endpoints = paths.flatMap(path => [path[0].start, path.at(-1).end]);
  const buckets = new Map(), candidates = [];
  for (let i = 0; i < endpoints.length; i++) {
    // A closed contour has no free end to attach to.
    if (pointText(endpoints[i & ~1]) === pointText(endpoints[i | 1])) continue;
    const point = endpoints[i];
    const x = Math.floor(point[0] / distance), y = Math.floor(point[1] / distance);
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
      for (const j of buckets.get(`${x + dx},${y + dy}`) || []) {
        if ((i >> 1) === (j >> 1)) continue;
        const gap = Math.hypot(point[0] - endpoints[j][0], point[1] - endpoints[j][1]);
        if (gap <= distance) candidates.push({ i: j, j: i, gap });
      }
    }
    const key = `${x},${y}`;
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key).push(i);
  }
  candidates.sort((a, b) => a.gap - b.gap || a.i - b.i || a.j - b.j);
  const parents = paths.map((_, i) => i);
  function find(i) {
    while (parents[i] !== i) { parents[i] = parents[parents[i]]; i = parents[i]; }
    return i;
  }
  const partners = new Array(endpoints.length).fill(-1);
  for (const { i, j } of candidates) {
    if (partners[i] !== -1 || partners[j] !== -1) continue;
    const a = find(i >> 1), b = find(j >> 1);
    if (a === b) continue;
    partners[i] = j; partners[j] = i; parents[b] = a;
  }
  const joined = [], visited = new Set();
  for (let endpoint = 0; endpoint < endpoints.length; endpoint++) {
    if (partners[endpoint] !== -1 || visited.has(endpoint >> 1)) continue;
    const path = [];
    let entry = endpoint;
    while (entry !== -1) {
      const index = entry >> 1;
      visited.add(index);
      const source = paths[index];
      // Clone records before editing endpoints so toggling modes or lowering
      // the threshold always recomputes from the untouched model geometry.
      const part = entry % 2 === 0 ? source.map(segment => ({ ...segment }))
        : source.slice().reverse().map(segment => ({
        ...segment, start: segment.end, end: segment.start,
      }));
      if (midpoint && path.length) {
        const a = path.at(-1).end, b = part[0].start;
        const shared = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
        path.at(-1).end = shared;
        part[0].start = shared;
      }
      path.push(...part);
      entry = partners[entry ^ 1];
    }
    joined.push(path);
  }
  return joined;
}

export function quadraticPathData(segments) {
  if (!segments.length) return '';
  return `M${pointText(segments[0].start)}${segments.map((segment, i) => {
    const bridge = i && pointText(segments[i - 1].end) !== pointText(segment.start)
      ? `L${pointText(segment.start)}` : '';
    return bridge + (segment.type === 'cubic' ? cubicText(segment) : curveText(segment));
  }).join('')}`;
}

function sampleQuadratic(segment, sampleCount = 16) {
  const points = [];
  for (let i = 0; i <= sampleCount; i++) {
    const t = i / sampleCount, u = 1 - t;
    points.push([
      u * u * segment.start[0] + 2 * u * t * segment.control[0] + t * t * segment.end[0],
      u * u * segment.start[1] + 2 * u * t * segment.control[1] + t * t * segment.end[1],
    ]);
  }
  return points;
}

export function sampleQuadraticPath(segments, samplesPerSegment = 16) {
  const points = [];
  for (const segment of segments) {
    const samples = sampleQuadratic(segment, samplesPerSegment);
    if (points.length) samples.shift();
    points.push(...samples);
  }
  return points;
}

export function fitQuadraticPath(segments, maxError = 1) {
  if (!segments.length) return [];
  const points = sampleQuadraticPath(segments);
  if (points.length < 2) return [];
  const fitted = fitCurve(points, maxError);
  return fitted.map(([start, control1, control2, end]) => ({
    type: 'cubic', start, control1, control2, end,
  }));
}

export function redrawQuadraticPaths(paths, maxError = 1) {
  if (!Number.isFinite(maxError) || maxError < 0) throw new RangeError('Fit tolerance must be a finite, nonnegative number.');
  return paths.map(path => fitQuadraticPath(path, maxError)).filter(path => path.length);
}

export function strokeToSvgPath(stroke) {
  return quadraticPathData([strokeToQuadratic(stroke)]);
}

export function createStrokePathElement(stroke, options = {}) {
  return createQuadraticPathElement([strokeToQuadratic(stroke)], options);
}

export function createQuadraticPathElement(segments, options = {}) {
  return createPathElement(segments, options);
}

export function createPathElement(segments, { stroke } = {}) {
  const path = document.createElementNS(SVG_NS, 'path');
  path.setAttribute('d', quadraticPathData(segments));
  for (const [name, value] of Object.entries({ ...PATH_STYLE, ...(stroke ? { stroke } : {}) })) {
    path.setAttribute(name, value);
  }
  return path;
}

// Append only new segments during live inference, extending the final DOM path
// when possible. Use the same continuity rule as the downloaded SVG.
export function appendStrokePathElement(layer, stroke, previousStroke) {
  const segment = strokeToQuadratic(stroke);
  const previous = previousStroke && strokeToQuadratic(previousStroke);
  if (layer.lastElementChild && canJoin(previous, segment)) {
    const path = layer.lastElementChild;
    path.setAttribute('d', path.getAttribute('d') + curveText(segment));
  } else {
    layer.append(createStrokePathElement(stroke));
  }
}

export function buildQuadraticSvg(segments, width, height = width, { joinDistance = 0, midpoint = true, redraw = false, fitTolerance = 1 } = {}) {
  const style = Object.entries(PATH_STYLE).map(([name, value]) => `${name}="${value}"`).join(' ');
  const joined = joinQuadraticSegments(segments, joinDistance, { midpoint });
  const paths = redraw ? redrawQuadraticPaths(joined, fitTolerance) : joined;
  const body = paths.map(path => `<path ${style} d="${quadraticPathData(path)}"/>`).join('');
  return `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="${SVG_NS}" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">${body}</svg>`;
}

export function buildStrokeSvg(strokes, width, height = width, options = {}) {
  return buildQuadraticSvg(strokes.map(strokeToQuadratic), width, height, options);
}

export function downloadStrokeSvg(strokes, width, height = width, filename = 'virtual-sketching.svg', options = {}) {
  const blob = new Blob([buildStrokeSvg(strokes, width, height, options)], { type: 'image/svg+xml;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
