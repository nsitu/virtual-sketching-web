export function renderStrokePatch(params, previousWidth, size = 128) {
  // utils.draw: model coordinates are (row, column), the start is the
  // patch CENTER, and OpenCV stamps 100 integer-radius filled circles.
  // Use numeric buffers rather than allocating two browser canvases per step.
  const highSize = size * 2;
  const high = new Float32Array(highSize * highSize);
  const x0 = 0.5;
  const y0 = 0.5;
  const x2 = (params[2] + 1) / 2;
  const y2 = (params[3] + 1) / 2;
  const x1 = x0 + (x2 - x0) * params[0];
  const y1 = y0 + (y2 - y0) * params[1];
  const z0 = 1 + Math.floor((previousWidth * size) / 2);
  const z2 = 1 + Math.floor((params[4] * size) / 2);
  const normal = (v) => Math.trunc(v * (highSize - 1) + 0.5);
  const [r0, c0, r1, c1, r2, c2] = [x0, y0, x1, y1, x2, y2].map(normal);
  for (let i = 0; i < 100; i += 1) {
    const t = i * 0.01;
    const row = Math.trunc((1 - t) * (1 - t) * r0 + 2 * t * (1 - t) * r1 + t * t * r2);
    const col = Math.trunc((1 - t) * (1 - t) * c0 + 2 * t * (1 - t) * c1 + t * t * c2);
    filledCircle(high, highSize, col, row, Math.trunc((1 - t) * z0 + t * z2));
  }
  // OpenCV's default bilinear resize at exactly 2:1 averages each 2x2 block.
  const output = new Float32Array(size * size);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const p = 2 * y * highSize + 2 * x;
      output[y * size + x] = (high[p] + high[p + 1] + high[p + highSize] + high[p + highSize + 1]) / 4;
    }
  }
  return output;
}

function filledCircle(buffer, size, cx, cy, radius) {
  const span = (y, left, right) => {
    if (y < 0 || y >= size) return;
    left = Math.max(0, left);
    right = Math.min(size - 1, right);
    if (left <= right) buffer.fill(1, y * size + left, y * size + right + 1);
  };
  let dx = radius;
  let dy = 0;
  let error = 0;
  let plus = 1;
  let minus = 2 * radius - 1;
  while (dx >= dy) {
    span(cy - dy, cx - dx, cx + dx);
    span(cy + dy, cx - dx, cx + dx);
    span(cy - dx, cx - dy, cx + dy);
    span(cy + dx, cx - dy, cx + dy);
    dy += 1;
    error += plus;
    plus += 2;
    if (error > 0) {
      error -= minus;
      dx -= 1;
      minus -= 2;
    }
  }
}

export function pastePatch(canvas, patch, cursor, imageSize, windowSize, patchSize = 128) {
  // Match DiffPastingV3.image_pasting_sampling_v3 in model_common_test.py.
  // It first snaps the destination support to integer floor/ceil bounds,
  // resamples the patch into that support with crop_and_resize semantics, and
  // then clips the padded result to the original canvas.
  const cursorX = cursor[0] * imageSize;
  const cursorY = cursor[1] * imageSize;
  const x1 = cursorX - windowSize / 2;
  const y1 = cursorY - windowSize / 2;
  const x2 = cursorX + windowSize / 2;
  const y2 = cursorY + windowSize / 2;
  const xFloor = Math.floor(x1);
  const yFloor = Math.floor(y1);
  const xCeil = Math.ceil(x2);
  const yCeil = Math.ceil(y2);
  const supportWidth = Math.max(1, xCeil - xFloor);
  const supportHeight = Math.max(1, yCeil - yFloor);
  const cursorPatchX = (((xFloor + xCeil) / 2 - x1) / windowSize) * patchSize;
  const cursorPatchY = (((yFloor + yCeil) / 2 - y1) / windowSize) * patchSize;
  const supportPatchWidth = patchSize * supportWidth / windowSize;
  const supportPatchHeight = patchSize * supportHeight / windowSize;
  const sampleLeft = cursorPatchX - (supportPatchWidth - 1) / 2;
  const sampleTop = cursorPatchY - (supportPatchHeight - 1) / 2;

  for (let supportY = 0; supportY < supportHeight; supportY += 1) {
    const destinationY = yFloor + supportY;
    if (destinationY < 0 || destinationY >= imageSize) continue;
    const patchY = supportHeight > 1
      ? sampleTop + supportY * (supportPatchHeight - 1) / (supportHeight - 1)
      : sampleTop;
    for (let supportX = 0; supportX < supportWidth; supportX += 1) {
      const destinationX = xFloor + supportX;
      if (destinationX < 0 || destinationX >= imageSize) continue;
      const patchX = supportWidth > 1
        ? sampleLeft + supportX * (supportPatchWidth - 1) / (supportWidth - 1)
        : sampleLeft;
      const value = sampleBilinearPatch(patch, patchSize, patchX, patchY);
      const index = destinationY * imageSize + destinationX;
      canvas[index] = Math.min(1, canvas[index] + value);
    }
  }
}

function sampleBilinearPatch(patch, size, x, y) {
  if (x < 0 || y < 0 || x > size - 1 || y > size - 1) return 0;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(x0 + 1, size - 1);
  const y1 = Math.min(y0 + 1, size - 1);
  const wx = x - x0;
  const wy = y - y0;
  const top = patch[y0 * size + x0] * (1 - wx) + patch[y0 * size + x1] * wx;
  const bottom = patch[y1 * size + x0] * (1 - wx) + patch[y1 * size + x1] * wx;
  return top * (1 - wy) + bottom * wy;
}
