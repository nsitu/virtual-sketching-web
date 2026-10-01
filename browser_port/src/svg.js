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

function cross(a, b) {
  return a[0] * b[1] - a[1] * b[0];
}

function lineIntersection(a, b, c, d) {
  const directionA = [b[0] - a[0], b[1] - a[1]];
  const directionB = [d[0] - c[0], d[1] - c[1]];
  const denominator = cross(directionA, directionB);
  if (Math.abs(denominator) < 1e-9) return null;
  const offset = [c[0] - a[0], c[1] - a[1]];
  const t = cross(offset, directionB) / denominator;
  const u = cross(offset, directionA) / denominator;
  if (t < -1e-8 || t > 1 + 1e-8 || u < -1e-8 || u > 1 + 1e-8) return null;
  return {
    t: Math.max(0, Math.min(1, t)),
    u: Math.max(0, Math.min(1, u)),
    point: [a[0] + t * directionA[0], a[1] + t * directionA[1]],
  };
}

function splitQuadraticSegment(segment, t) {
  const lerp = (a, b, amount) => [a[0] + (b[0] - a[0]) * amount, a[1] + (b[1] - a[1]) * amount];
  const first = lerp(segment.start, segment.control, t);
  const second = lerp(segment.control, segment.end, t);
  const shared = lerp(first, second, t);
  return [
    { ...segment, start: segment.start, control: first, end: shared },
    { ...segment, start: shared, control: second, end: segment.end, startsNewPath: false },
  ];
}

function splitPathAtIntersections(path, cuts) {
  const result = [], current = [];
  for (let index = 0; index < path.length; index++) {
    const segment = path[index];
    const parameters = [...new Set((cuts.get(index) || []).sort((a, b) => a - b))];
    let pieces = [segment], previous = 0;
    for (const parameter of parameters) {
      const local = (parameter - previous) / (1 - previous);
      const [left, right] = splitQuadraticSegment(pieces.at(-1), local);
      pieces[pieces.length - 1] = left;
      pieces.push(right);
      previous = parameter;
    }
    for (let pieceIndex = 0; pieceIndex < pieces.length; pieceIndex++) {
      current.push({ ...pieces[pieceIndex], startsNewPath: current.length ? pieces[pieceIndex].startsNewPath : true });
      if (pieceIndex < pieces.length - 1) {
        result.push(current.splice(0));
      }
    }
  }
  if (current.length) result.push(current.splice(0));
  return result;
}

// Detect interior crossings using a finely sampled polyline approximation,
// then split the second participating path at each crossing. Endpoint
// contacts and collinear overlaps are left alone because they are already
// represented by path continuity or need a separate overlap policy.
export function splitIntersectingPaths(paths, samplesPerSegment = 24) {
  if (!Number.isInteger(samplesPerSegment) || samplesPerSegment < 4) throw new RangeError('Intersection sampling must use at least four subdivisions.');
  const records = paths.flatMap((path, pathIndex) => path.map((segment, segmentIndex) => ({ pathIndex, segmentIndex, segment, points: sampleQuadratic(segment, samplesPerSegment) })));
  const cuts = new Map();
  const intersections = [];
  const addCut = (pathIndex, segmentIndex, parameter) => {
    if (!cuts.has(pathIndex)) cuts.set(pathIndex, new Map());
    const pathCuts = cuts.get(pathIndex);
    if (!pathCuts.has(segmentIndex)) pathCuts.set(segmentIndex, []);
    pathCuts.get(segmentIndex).push(parameter);
  };
  for (let a = 0; a < records.length; a++) {
    const first = records[a];
    for (let b = a + 1; b < records.length; b++) {
      const second = records[b];
      if (first.pathIndex === second.pathIndex && Math.abs(first.segmentIndex - second.segmentIndex) <= 1) continue;
      const pairIntersections = [];
      for (let i = 0; i < samplesPerSegment; i++) for (let j = 0; j < samplesPerSegment; j++) {
        const hit = lineIntersection(first.points[i], first.points[i + 1], second.points[j], second.points[j + 1]);
        if (!hit) continue;
        const t = (i + hit.t) / samplesPerSegment, u = (j + hit.u) / samplesPerSegment;
        if (t <= 1e-3 || t >= 1 - 1e-3 || u <= 1e-3 || u >= 1 - 1e-3) continue;
        if (pairIntersections.some(previous => Math.abs(previous.t - t) < 2 / samplesPerSegment && Math.abs(previous.u - u) < 2 / samplesPerSegment)) continue;
        pairIntersections.push({ t, u, point: hit.point });
      }
      for (const intersection of pairIntersections) {
        // Splitting the second path keeps the operation deterministic and
        // avoids doubling the number of new endpoints at a crossing.
        addCut(second.pathIndex, second.segmentIndex, intersection.u);
        intersections.push({ ...intersection, first, second });
      }
    }
  }
  if (!intersections.length) return { paths: paths.map(path => path.slice()), intersections: 0 };
  const splitPaths = paths.flatMap((path, pathIndex) => splitPathAtIntersections(path, cuts.get(pathIndex) || new Map()));
  return { paths: splitPaths, intersections: intersections.length };
}

function segmentLength(segment, sampleCount = 16) {
  const points = sampleQuadratic(segment, sampleCount);
  let length = 0;
  for (let i = 1; i < points.length; i++) {
    length += Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]);
  }
  return length;
}

function segmentTangent(segment, t = .5) {
  const first = [segment.control[0] - segment.start[0], segment.control[1] - segment.start[1]];
  const second = [segment.end[0] - segment.control[0], segment.end[1] - segment.control[1]];
  const tangent = [2 * ((1 - t) * first[0] + t * second[0]), 2 * ((1 - t) * first[1] + t * second[1])];
  if (Math.hypot(...tangent) > 1e-8) return tangent;
  return [segment.end[0] - segment.start[0], segment.end[1] - segment.start[1]];
}

function angleBetween(a, b) {
  const aLength = Math.hypot(...a), bLength = Math.hypot(...b);
  if (aLength < 1e-8 || bLength < 1e-8) return Math.PI;
  return Math.acos(Math.max(-1, Math.min(1, (a[0] * b[0] + a[1] * b[1]) / (aLength * bLength))));
}

function bounds(points) {
  return points.reduce((box, [x, y]) => ({
    minX: Math.min(box.minX, x), maxX: Math.max(box.maxX, x),
    minY: Math.min(box.minY, y), maxY: Math.max(box.maxY, y),
  }), { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity });
}

function boxesMayOverlap(a, b, padding) {
  return a.minX <= b.maxX + padding && a.maxX + padding >= b.minX
    && a.minY <= b.maxY + padding && a.maxY + padding >= b.minY;
}

function sameContinuousPath(segments) {
  const ids = [];
  let pathId = -1, previous;
  for (const segment of segments) {
    if (!canJoin(previous, segment)) pathId++;
    ids.push(pathId);
    previous = segment;
  }
  return ids;
}

function coversSegment(candidate, keeper, tolerance) {
  // A near crossing is not redundant: the centerline directions must agree.
  if (Math.abs(Math.cos(angleBetween(candidate.tangent, keeper.tangent))) < Math.cos(Math.PI / 3)) return false;
  // Avoid deleting a visibly curved corner in favor of a straight neighbor.
  const candidateTurn = angleBetween(segmentTangent(candidate.segment, 0), segmentTangent(candidate.segment, 1));
  if (candidateTurn > Math.PI / 2) return false;
  if (!boxesMayOverlap(candidate.box, keeper.box, tolerance)) return false;
  const distances = candidate.points.map(point => Math.min(...keeper.points.map(other => Math.hypot(point[0] - other[0], point[1] - other[1]))));
  const coverage = distances.filter(distance => distance <= tolerance).length / distances.length;
  const average = distances.reduce((sum, distance) => sum + distance, 0) / distances.length;
  return coverage >= .8 && average <= tolerance * .8;
}

// Remove whole source segments only when a longer segment from another
// continuous path already covers nearly all of their centerline. The original
// records are left untouched so changing the tolerance recomputes from the
// model output rather than accumulating deletions.
export function removeRedundantSegments(segments, tolerance = 3) {
  if (!Number.isFinite(tolerance) || tolerance < 0) throw new RangeError('Redundancy tolerance must be a finite, nonnegative number.');
  if (tolerance === 0 || segments.length < 2) return segments.slice();
  const pathIds = sameContinuousPath(segments);
  const entries = segments.map((segment, index) => {
    const points = sampleQuadratic(segment, 12);
    return {
      segment, index, pathId: pathIds[index], points, box: bounds(points),
      tangent: segmentTangent(segment), length: segmentLength(segment, 12),
    };
  }).sort((a, b) => b.length - a.length || a.index - b.index);
  const kept = [];
  for (const candidate of entries) {
    const redundant = kept.some(keeper => keeper.pathId !== candidate.pathId && coversSegment(candidate, keeper, tolerance));
    if (!redundant) kept.push(candidate);
  }
  return kept.sort((a, b) => a.index - b.index).map(entry => entry.segment);
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

export function buildQuadraticSvg(segments, width, height = width, { joinDistance = 0, midpoint = true, redraw = false, fitTolerance = 1, removeRedundant = false, redundancyTolerance = 3, splitIntersections = false } = {}) {
  const style = Object.entries(PATH_STYLE).map(([name, value]) => `${name}="${value}"`).join(' ');
  const source = removeRedundant ? removeRedundantSegments(segments, redundancyTolerance) : segments;
  const joined = joinQuadraticSegments(source, joinDistance, { midpoint });
  const split = splitIntersections ? splitIntersectingPaths(joined) : { paths: joined, intersections: 0 };
  const paths = redraw ? redrawQuadraticPaths(split.paths, fitTolerance) : split.paths;
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
