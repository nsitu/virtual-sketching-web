const SVG_NS = 'http://www.w3.org/2000/svg';

function pasteTransform(cursor, imageSize, windowSize, patchSize) {
  const centerX = cursor[0] * imageSize;
  const centerY = cursor[1] * imageSize;
  const left = centerX - windowSize / 2;
  const top = centerY - windowSize / 2;
  const xFloor = Math.floor(left);
  const yFloor = Math.floor(top);
  const xCeil = Math.ceil(centerX + windowSize / 2);
  const yCeil = Math.ceil(centerY + windowSize / 2);
  const supportWidth = Math.max(1, xCeil - xFloor);
  const supportHeight = Math.max(1, yCeil - yFloor);
  const sourceCenterX = (((xFloor + xCeil) / 2 - left) / windowSize) * patchSize;
  const sourceCenterY = (((yFloor + yCeil) / 2 - top) / windowSize) * patchSize;
  const sourceWidth = patchSize * supportWidth / windowSize;
  const sourceHeight = patchSize * supportHeight / windowSize;
  const sourceLeft = sourceCenterX - (sourceWidth - 1) / 2;
  const sourceTop = sourceCenterY - (sourceHeight - 1) / 2;
  const scaleX = supportWidth > 1 ? (supportWidth - 1) / (sourceWidth - 1) : 0;
  const scaleY = supportHeight > 1 ? (supportHeight - 1) / (sourceHeight - 1) : 0;
  return {
    x: patchX => xFloor + (patchX - sourceLeft) * scaleX,
    y: patchY => yFloor + (patchY - sourceTop) * scaleY,
    scaleX,
    scaleY,
  };
}

// Convert one model quadratic stroke into a vector outline. The original
// rasterizer stamps 100 filled circles while linearly changing radius. Encode
// those circles as subpaths in one SVG path so the exported image stays vector.
export function strokeToSvgPath(stroke, patchSize = 128) {
  const { params, previousWidth, cursor, imageSize, windowSize } = stroke;
  const [controlRow, controlColumn, endRowParam, endColumnParam, endWidth] = params;
  const startRow = 128;
  const startColumn = 128;
  const endRow = Math.trunc(((endRowParam + 1) / 2) * (patchSize * 2 - 1) + .5);
  const endColumn = Math.trunc(((endColumnParam + 1) / 2) * (patchSize * 2 - 1) + .5);
  const controlRowPixel = startRow + (endRow - startRow) * controlRow;
  const controlColumnPixel = startColumn + (endColumn - startColumn) * controlColumn;
  const startRadius = 1 + Math.floor(previousWidth * patchSize / 2);
  const endRadius = 1 + Math.floor(endWidth * patchSize / 2);
  const map = pasteTransform(cursor, imageSize, windowSize, patchSize);
  const circles = [];

  for (let i = 0; i < 100; i++) {
    const t = i / 100;
    const u = 1 - t;
    const row = Math.trunc(u * u * startRow + 2 * t * u * controlRowPixel + t * t * endRow);
    const column = Math.trunc(u * u * startColumn + 2 * t * u * controlColumnPixel + t * t * endColumn);
    const radius = Math.trunc(u * startRadius + t * endRadius) / 2;
    const cx = map.x((column - .5) / 2);
    const cy = map.y((row - .5) / 2);
    const rx = Math.max(.001, radius * map.scaleX);
    const ry = Math.max(.001, radius * map.scaleY);
    const left = cx - rx;
    const right = cx + rx;
    circles.push(`M${left.toFixed(3)} ${cy.toFixed(3)}A${rx.toFixed(3)} ${ry.toFixed(3)} 0 1 0 ${right.toFixed(3)} ${cy.toFixed(3)}A${rx.toFixed(3)} ${ry.toFixed(3)} 0 1 0 ${left.toFixed(3)} ${cy.toFixed(3)}Z`);
  }
  return circles.join('');
}

export function createStrokePathElement(stroke) {
  const path = document.createElementNS(SVG_NS, 'path');
  path.setAttribute('d', strokeToSvgPath(stroke));
  path.setAttribute('fill', '#000');
  path.setAttribute('fill-rule', 'nonzero');
  return path;
}

export function buildStrokeSvg(strokes, width, height = width) {
  const body = strokes.map(stroke => `<path fill="#000" fill-rule="nonzero" d="${strokeToSvgPath(stroke)}"/>`).join('');
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
