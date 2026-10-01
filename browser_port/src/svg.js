const SVG_NS = 'http://www.w3.org/2000/svg';

// Use the uniform width and continuous path structure of Python's cluster export.
const PATH_STYLE = {
  fill: 'none',
  stroke: '#000',
  'stroke-width': '3.5',
  'stroke-linecap': 'round',
  'stroke-linejoin': 'round',
};

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

function canJoin(previous, next) {
  // Equality at export precision only: never bridge gaps or alter controls.
  // Legacy records/SVGs without pen metadata can use endpoint continuity alone.
  return previous && !next.startsNewPath && pointText(previous.end) === pointText(next.start);
}

export function joinQuadraticSegments(segments) {
  const paths = [];
  let previous;
  for (const segment of segments) {
    if (canJoin(previous, segment)) paths.at(-1).push(segment);
    else paths.push([segment]);
    previous = segment;
  }
  return paths;
}

export function quadraticPathData(segments) {
  if (!segments.length) return '';
  return `M${pointText(segments[0].start)}${segments.map(curveText).join('')}`;
}

export function strokeToSvgPath(stroke) {
  return quadraticPathData([strokeToQuadratic(stroke)]);
}

export function createStrokePathElement(stroke) {
  const path = document.createElementNS(SVG_NS, 'path');
  path.setAttribute('d', strokeToSvgPath(stroke));
  for (const [name, value] of Object.entries(PATH_STYLE)) path.setAttribute(name, value);
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

export function buildQuadraticSvg(segments, width, height = width) {
  const style = Object.entries(PATH_STYLE).map(([name, value]) => `${name}="${value}"`).join(' ');
  const body = joinQuadraticSegments(segments).map(path => `<path ${style} d="${quadraticPathData(path)}"/>`).join('');
  return `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="${SVG_NS}" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><rect width="100%" height="100%" fill="#fff"/>${body}</svg>`;
}

export function buildStrokeSvg(strokes, width, height = width) {
  return buildQuadraticSvg(strokes.map(strokeToQuadratic), width, height);
}

export function downloadStrokeSvg(strokes, width, height = width, filename = 'virtual-sketching.svg') {
  const blob = new Blob([buildStrokeSvg(strokes, width, height)], { type: 'image/svg+xml;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
