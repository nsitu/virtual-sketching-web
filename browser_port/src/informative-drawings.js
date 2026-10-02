// RGB [0,1], NCHW, without mean/std normalization. Two stride-2 layers
// round output sizes upward: edge-pad to multiples of four and crop afterward.
// Tiny images need room for the network's reflection pads.
export function drawingTensorInput(image) {
  const { data, width, height, channels } = image;
  if (channels !== 3 || !Number.isInteger(width) || !Number.isInteger(height)
      || width < 1 || height < 1 || data.length !== width * height * 3) {
    throw new Error('Drawing model requires a nonempty RGB image.');
  }
  const paddedWidth = Math.max(32, Math.ceil(width / 4) * 4);
  const paddedHeight = Math.max(32, Math.ceil(height / 4) * 4);
  const plane = paddedWidth * paddedHeight;
  const tensor = new Float32Array(plane * 3);
  for (let y = 0; y < paddedHeight; y++) for (let x = 0; x < paddedWidth; x++) {
    const source = (Math.min(y, height - 1) * width + Math.min(x, width - 1)) * 3;
    for (let c = 0; c < 3; c++) tensor[c * plane + y * paddedWidth + x] = data[source + c];
  }
  return { data: tensor, dims: [1, 3, paddedHeight, paddedWidth] };
}

export function cropDrawingOutput(data, dims, width, height) {
  if (dims.length !== 4 || dims[0] !== 1 || dims[1] !== 1
      || dims[2] < height || dims[3] < width || data.length !== dims[2] * dims[3]) {
    throw new Error('Unexpected drawing model output dimensions.');
  }
  const cropped = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    cropped.set(data.subarray(y * dims[3], y * dims[3] + width), y * width);
  }
  return { data: cropped, width, height, channels: 1 };
}

// Keep the cached network result untouched, so the cutoff is cheap to preview.
// The clean sampler counts *any* value below 1 as ink; snapping faint gray to
// exactly white prevents it from spending rounds tracing the background.
export function cleanDrawing(image, cutoff = .98) {
  if (image.channels !== 1 || !Number.isFinite(cutoff) || cutoff < 0 || cutoff > 1) {
    throw new Error('Drawing cleanup requires grayscale input and a cutoff in [0,1].');
  }
  const data = Float32Array.from(image.data, value => value >= cutoff ? 1 : Math.max(0, Math.min(1, value)));
  return { ...image, data };
}

export function squareDrawing(image) {
  const size = Math.max(image.width, image.height);
  const data = new Float32Array(size * size).fill(1);
  for (let y = 0; y < image.height; y++) {
    data.set(image.data.subarray(y * image.width, (y + 1) * image.width), y * size);
  }
  return { data, width: size, height: size, channels: 1 };
}
