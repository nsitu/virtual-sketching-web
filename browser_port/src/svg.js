const SVG_NS = 'http://www.w3.org/2000/svg';

// Use the uniform width from tools/svg_conversion.py's cluster export, while
// keeping one path per model segment (the Python converter's single mode).
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
export function strokeToSvgPath(stroke) {
  const { params, cursor, imageSize, windowSize } = stroke;
  const [controlRow, controlColumn, endRow, endColumn] = params;
  const startX = cursor[0] * imageSize;
  const startY = cursor[1] * imageSize;
  const dx = endColumn * windowSize / 2;
  const dy = endRow * windowSize / 2;
  const controlX = startX + dx * controlColumn;
  const controlY = startY + dy * controlRow;
  // Only the next inference cursor is clamped; the curve endpoint is not.
  return `M${startX.toFixed(3)} ${startY.toFixed(3)}Q${controlX.toFixed(3)} ${controlY.toFixed(3)} ${(startX + dx).toFixed(3)} ${(startY + dy).toFixed(3)}`;
}

export function createStrokePathElement(stroke) {
  const path = document.createElementNS(SVG_NS, 'path');
  path.setAttribute('d', strokeToSvgPath(stroke));
  for (const [name, value] of Object.entries(PATH_STYLE)) path.setAttribute(name, value);
  return path;
}

export function buildStrokeSvg(strokes, width, height = width) {
  const style = Object.entries(PATH_STYLE).map(([name, value]) => `${name}="${value}"`).join(' ');
  const body = strokes.map(stroke => `<path ${style} d="${strokeToSvgPath(stroke)}"/>`).join('');
  return `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="${SVG_NS}" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><rect width="100%" height="100%" fill="#fff"/>${body}</svg>`;
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
