export function clamp(value, low, high) {
  return Math.max(low, Math.min(high, value));
}

export function normalizeM1To1(values) {
  const output = new Float32Array(values.length);
  for (let i = 0; i < values.length; i += 1) {
    output[i] = values[i] * 2 - 1;
  }
  return output;
}

function sourceIndex(x, y, width, channels) {
  return (y * width + x) * channels;
}

function sampleBilinear(source, width, height, channels, x, y, output, extrapolation = 0) {
  if (x < 0 || y < 0 || x > width - 1 || y > height - 1) {
    output.fill(extrapolation);
    return;
  }

  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(x0 + 1, width - 1);
  const y1 = Math.min(y0 + 1, height - 1);
  const wx = x - x0;
  const wy = y - y0;

  const topLeft = sourceIndex(x0, y0, width, channels);
  const topRight = sourceIndex(x1, y0, width, channels);
  const bottomLeft = sourceIndex(x0, y1, width, channels);
  const bottomRight = sourceIndex(x1, y1, width, channels);
  for (let c = 0; c < channels; c += 1) {
    const top = source[topLeft + c] * (1 - wx) + source[topRight + c] * wx;
    const bottom = source[bottomLeft + c] * (1 - wx) + source[bottomRight + c] * wx;
    output[c] = top * (1 - wy) + bottom * wy;
  }
}

export function resizeBilinear(image, outWidth, outHeight) {
  const { data, width, height, channels } = image;
  const output = new Float32Array(outWidth * outHeight * channels);
  const sample = new Float32Array(channels);

  for (let oy = 0; oy < outHeight; oy += 1) {
    const y = ((oy + 0.5) * height) / outHeight - 0.5;
    for (let ox = 0; ox < outWidth; ox += 1) {
      const x = ((ox + 0.5) * width) / outWidth - 0.5;
      sampleBilinear(data, width, height, channels, clamp(x, 0, width - 1), clamp(y, 0, height - 1), sample);
      const target = (oy * outWidth + ox) * channels;
      output.set(sample, target);
    }
  }
  return { data: output, width: outWidth, height: outHeight, channels };
}

// TensorFlow ResizeArea (align_corners=False): integrate pixel coverage.
// In particular, thin source lines must survive downsampling to 128x128.
export function resizeArea(image, outWidth, outHeight) {
  const { data, width, height, channels } = image;
  const output = new Float32Array(outWidth * outHeight * channels);
  const sx = width / outWidth;
  const sy = height / outHeight;
  for (let y = 0; y < outHeight; y += 1) {
    const top = y * sy;
    const bottom = (y + 1) * sy;
    for (let x = 0; x < outWidth; x += 1) {
      const left = x * sx;
      const right = (x + 1) * sx;
      const offset = (y * outWidth + x) * channels;
      for (let c = 0; c < channels; c += 1) {
        let sum = 0;
        for (let iy = Math.floor(top); iy < Math.ceil(bottom); iy += 1) {
          const wy = Math.min(bottom, iy + 1) - Math.max(top, iy);
          for (let ix = Math.floor(left); ix < Math.ceil(right); ix += 1) {
            const wx = Math.min(right, ix + 1) - Math.max(left, ix);
            sum += data[(Math.min(iy, height - 1) * width + Math.min(ix, width - 1)) * channels + c] * wx * wy;
          }
        }
        output[offset + c] = sum / (sx * sy);
      }
    }
  }
  return { data: output, width: outWidth, height: outHeight, channels };
}

export function cropAndResize(image, cursor, windowSize, outSize = 128, extrapolation = 0) {
  const { data, width, height, channels } = image;
  const output = new Float32Array(outSize * outSize * channels);
  const sample = new Float32Array(channels);
  const centerX = cursor[0] * width;
  const centerY = cursor[1] * height;
  const left = centerX - (windowSize - 1) / 2;
  const top = centerY - (windowSize - 1) / 2;
  const scaleX = outSize > 1 ? (windowSize - 1) / (outSize - 1) : 0;
  const scaleY = outSize > 1 ? (windowSize - 1) / (outSize - 1) : 0;

  for (let oy = 0; oy < outSize; oy += 1) {
    const y = top + oy * scaleY;
    for (let ox = 0; ox < outSize; ox += 1) {
      const x = left + ox * scaleX;
      sampleBilinear(data, width, height, channels, x, y, sample, extrapolation);
      output.set(sample, (oy * outSize + ox) * channels);
    }
  }
  return { data: output, width: outSize, height: outSize, channels };
}

export function imageFromCanvas(canvas, channels = 1) {
  const context = canvas.getContext('2d', { willReadFrequently: true });
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
  const data = new Float32Array(canvas.width * canvas.height * channels);
  for (let i = 0, p = 0; i < data.length; i += channels, p += 4) {
    if (channels === 1) {
      data[i] = pixels[p] / 255;
    } else {
      data[i] = pixels[p] / 255;
      data[i + 1] = pixels[p + 1] / 255;
      data[i + 2] = pixels[p + 2] / 255;
    }
  }
  return { data, width: canvas.width, height: canvas.height, channels };
}

export function makeSquareCanvas(image, background = 1) {
  const size = Math.max(image.width, image.height);
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext('2d');
  context.fillStyle = `rgb(${Math.round(background * 255)}, ${Math.round(background * 255)}, ${Math.round(background * 255)})`;
  context.fillRect(0, 0, size, size);
  context.drawImage(image, 0, 0);
  return canvas;
}

// Pillow's RGB BILINEAR resize is separable, widens the triangular filter
// when downsampling, and rounds to uint8 after each pass. Canvas drawImage
// filtering is browser-dependent, so use the same numeric policy here.
export function resizePhotoBilinear(image, outSize = 256) {
  const precision = 2 ** 22;
  function weights(inputSize, outputSize) {
    const scale = inputSize / outputSize;
    const support = Math.max(1, scale);
    return Array.from({ length: outputSize }, (_, i) => {
      const center = (i + .5) * scale;
      const start = Math.max(0, Math.floor(center - support + .5));
      const end = Math.min(inputSize, Math.floor(center + support + .5));
      const raw = Array.from({ length: end - start }, (_, j) => Math.max(0, 1 - Math.abs((start + j - center + .5) / support)));
      const sum = raw.reduce((a, b) => a + b, 0);
      return { start, values: raw.map(v => Math.floor(v / sum * precision + .5)) };
    });
  }
  const { width, height, channels } = image;
  const wx = weights(width, outSize), wy = weights(height, outSize);
  const horizontal = new Uint8Array(outSize * height * channels);
  for (let y = 0; y < height; y++) for (let x = 0; x < outSize; x++) for (let c = 0; c < channels; c++) {
    let sum = precision / 2;
    for (let j = 0; j < wx[x].values.length; j++) {
      sum += Math.round(image.data[(y * width + wx[x].start + j) * channels + c] * 255) * wx[x].values[j];
    }
    horizontal[(y * outSize + x) * channels + c] = clamp(Math.floor(sum / precision), 0, 255);
  }
  const data = new Float32Array(outSize * outSize * channels);
  for (let y = 0; y < outSize; y++) for (let x = 0; x < outSize; x++) for (let c = 0; c < channels; c++) {
    let sum = precision / 2;
    for (let j = 0; j < wy[y].values.length; j++) {
      sum += horizontal[((wy[y].start + j) * outSize + x) * channels + c] * wy[y].values[j];
    }
    data[(y * outSize + x) * channels + c] = clamp(Math.floor(sum / precision), 0, 255) / 255;
  }
  return { data, width: outSize, height: outSize, channels };
}

export function preparePhoto(image) {
  if (image.channels !== 3) throw new Error('Photo mode requires RGB input.');
  const size = Math.max(image.width, image.height);
  const data = new Float32Array(size * size * 3); // black, right/bottom padding
  for (let y = 0; y < image.height; y++) {
    data.set(image.data.subarray(y * image.width * 3, (y + 1) * image.width * 3), y * size * 3);
  }
  const square = { data, width: size, height: size, channels: 3 };
  return size === 256 ? square : resizePhotoBilinear(square);
}

// GeneralRawDataLoader's rough-sketch path preserves RGB and resolution.
// The source pixel ten pixels from the bottom/right supplies the padding.
// Clamp for tiny uploads, which the original loader does not handle safely.
export function prepareRoughSketch(image) {
  if (image.channels !== 3) throw new Error('Rough sketch mode requires RGB input.');
  const { width, height } = image;
  const size = Math.max(width, height);
  const offset = (Math.max(0, height - 10) * width + Math.max(0, width - 10)) * 3;
  const background = image.data.subarray(offset, offset + 3);
  const data = new Float32Array(size * size * 3);
  for (let i = 0; i < data.length; i += 3) data.set(background, i);
  for (let y = 0; y < height; y++) {
    data.set(image.data.subarray(y * width * 3, (y + 1) * width * 3), y * size * 3);
  }
  return { data, width: size, height: size, channels: 3 };
}

// test_rough_sketch_simplification.move_cursor_to_undrawn is a random move
// from the *last* cursor, not the clean sampler's canvas-coverage search.
export function chooseRoughCursor(image, cursor, patchSize = 128, random = Math.random) {
  const { data, width: size, channels } = image;
  const min = Math.floor(.3 / 2 * size);
  const max = Math.max(min + 1, Math.floor(.9 / 2 * size));
  const roundEven = value => {
    const floor = Math.floor(value);
    return value - floor === .5 ? floor + (floor % 2) : Math.round(value);
  };
  for (let attempt = 0; attempt < 20; attempt++) {
    const offsets = [0, 1].map(() => min + Math.floor(random() * (max - min)));
    const signs = [0, 1].map(() => Math.floor(random() * 2) === 0 ? -1 : 1);
    const next = cursor.map((v, c) => Math.fround(Math.fround(clamp(v * size + offsets[c] * signs[c], 0, size - 1)) / size));
    if (attempt === 19) return next;
    const [x, y] = next.map(v => roundEven(Math.fround(v * size)));
    const half = Math.floor(patchSize / 2);
    for (let iy = Math.max(0, y - half); iy < Math.min(size, y - half + patchSize); iy++) {
      for (let ix = Math.max(0, x - half); ix < Math.min(size, x - half + patchSize); ix++) {
        for (let c = 0; c < channels; c++) {
          if (data[(iy * size + ix) * channels + c] < 1) return next;
        }
      }
    }
  }
}

export function chooseInitialCursor(image, patchSize = 128, random = Math.random) {
  const { data, width, height, channels } = image;
  if (!data.some(v => v < 1)) return null;
  for (let attempt = 0; attempt < 10000; attempt++) {
    const x = Math.floor(random() * width);
    const y = Math.floor(random() * height);
    for (let iy = Math.max(0, y - patchSize / 2); iy < Math.min(height, y + patchSize / 2); iy++) {
      for (let ix = Math.max(0, x - patchSize / 2); ix < Math.min(width, x + patchSize / 2); ix++) {
        const offset = (iy * width + ix) * channels;
        for (let c = 0; c < channels; c++) {
          if (data[offset + c] < 1) return [x / width, y / height];
        }
      }
    }
  }
  const pixel = Math.floor(data.findIndex(v => v < 1) / channels);
  return [(pixel % width) / width, Math.floor(pixel / width) / height];
}

// test_vectorization.move_cursor_to_undrawn: binary nonzero ink coverage,
// x-major grid ordering, minimum-coverage escape, and a random point in the
// middle half of the selected cell (not its darkest source pixel).
export function chooseUndrawnCursor(image, canvas, patchSize = 128, history = {}, random = Math.random) {
  const { data, width, height, channels } = image;
  const gridCount = Math.ceil(width / patchSize);
  let best = null;
  let bestUndrawn = 0;
  let minAccuracy = 1;
  let minCell = null;

  for (let gridX = 0; gridX < gridCount; gridX += 1) {
    for (let gridY = 0; gridY < gridCount; gridY += 1) {
      const x0 = gridX * patchSize;
      const y0 = gridY * patchSize;
      const x1 = Math.min(x0 + patchSize, width);
      const y1 = Math.min(y0 + patchSize, height);
      let targetPixels = 0;
      let drawnPixels = 0;

      for (let y = y0; y < y1; y += 1) {
        for (let x = x0; x < x1; x += 1) {
          const imageOffset = (y * width + x) * channels;
          const target = data[imageOffset];
          if (target < 1) {
            targetPixels += 1;
            const canvasValue = canvas[y * width + x];
            if (canvasValue > 0) drawnPixels += 1;
          }
        }
      }

      const undrawn = targetPixels - drawnPixels;
      const accuracy = targetPixels === 0 || undrawn <= 5 ? 1 : drawnPixels / targetPixels;
      if (accuracy < minAccuracy) {
        minAccuracy = accuracy;
        minCell = [gridX, gridY];
      }
      if (undrawn > bestUndrawn) {
        bestUndrawn = undrawn;
        best = [gridX, gridY];
      }
    }
  }
  if (minAccuracy >= .95 || !best) return null;
  const minIndex = minCell[0] * gridCount + minCell[1];
  const same = history.minIndex === minIndex;
  if (same && history.times >= 2) {
    best = minCell;
    history.times = 1;
  } else {
    history.times = same ? history.times + 1 : 1;
  }
  history.minIndex = minIndex;
  const y = best[1] * patchSize + patchSize / 4 + Math.floor(random() * (patchSize / 2 + 1));
  const x = best[0] * patchSize + patchSize / 4 + Math.floor(random() * (patchSize / 2 + 1));
  return [Math.min(width - 1, x) / width, Math.min(height - 1, y) / height];
}
