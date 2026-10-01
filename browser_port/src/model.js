import * as ort from 'onnxruntime-web';
import { cropAndResize, chooseInitialCursor, chooseUndrawnCursor, chooseRoughCursor, clamp, normalizeM1To1, resizeArea } from './preprocess.js';
import { pastePatch, renderStrokePatch } from './raster.js';
import { getMode } from './modes.js';

const MODEL_INPUTS = {
  patchPhoto: 'step_patch_photo:0',
  patchCanvas: 'step_patch_canvas:0',
  entirePhoto: 'step_entire_photo:0',
  entireCanvas: 'step_entire_canvas:0',
  cursor: 'step_cursor:0',
  imageSize: 'step_image_size:0',
  windowSize: 'step_window_size:0',
  previousWidth: 'step_prev_width:0',
  state: 'step_state_in:0',
};

const MODEL_OUTPUTS = {
  params: 'other_params:0',
  pen: 'pen_ras:0',
  state: 'state_out:0',
};

function tensor(name, type, data, dims) {
  return new ort.Tensor(type, data, dims);
}

function oneValue(value) {
  return new Float32Array([value]);
}

export class BrowserVectorizer {
  constructor({ mode = 'line', modelUrl, rasterSize = 128, minWindowSize = 32, minWidth = 0.01, maxScaling = 2 } = {}) {
    this.mode = mode;
    this.config = getMode(mode);
    this.modelUrl = modelUrl ?? this.config.modelUrl;
    this.rasterSize = rasterSize;
    this.minWindowSize = minWindowSize;
    this.minWidth = minWidth;
    this.maxScaling = maxScaling;
    this.session = null;
    this.image = null;
    this.fullPhotoSmall = null;
    this.canvas = null;
    this.state = null;
  }

  async load() {
    // Vite does not automatically publish package WASM files. The prepare
    // script copies them to this stable public path before dev/build.
    ort.env.wasm.wasmPaths = `${import.meta.env?.BASE_URL ?? '/'}ort-wasm/`;
    const response = await fetch(this.modelUrl);
    if (!response.ok || response.headers.get('content-type')?.includes('text/html')) {
      throw new Error(`ONNX asset unavailable: ${this.modelUrl} (HTTP ${response.status}).`);
    }
    this.session = await ort.InferenceSession.create(await response.arrayBuffer(), {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    return this.session;
  }

  setImage(image) {
    if (image.channels !== this.config.channels) throw new Error(`${this.config.label} expects ${this.config.channels} input channels.`);
    if (image.width !== image.height) throw new Error('The browser input must be square after padding.');
    if (this.mode === 'photo' && image.width !== 256) throw new Error('Photo input must be preprocessed to 256 × 256.');
    this.image = image;
    this.canvas = new Float32Array(image.width * image.height);
    this.strokes = [];
    const resized = resizeArea(image, this.rasterSize, this.rasterSize);
    // The exported graph normalizes the full-image tensors internally in
    // build_combined_encoder. Keep these tensors in the original [0, 1]
    // convention; only the local patch inputs are pre-normalized here.
    this.fullPhotoSmall = resized.data;
    this.resetState([0.5, 0.5]);
  }

  resetState(cursor = [0.5, 0.5]) {
    if (!this.image) throw new Error('Set an image before resetting the vectorizer.');
    this.state = new Float32Array(1024);
    this.cursor = [...cursor];
    this.previousWidth = this.minWidth;
    this.previousScaling = 1;
    this.previousWindowSize = this.rasterSize;
    this.startsNewPath = true;
  }

  async step() {
    if (!this.session) throw new Error('Load the ONNX session before running inference.');
    if (!this.image) throw new Error('Set an image before running inference.');

    const imageSize = this.image.width;
    const currentWindowSize = clamp(this.previousScaling * this.previousWindowSize, this.minWindowSize, imageSize);
    // Python crops inverted ink with zero extrapolation, then inverts back:
    // equivalent to cropping the white-background photo with extrapolation=1.
    const patchPhoto = normalizeM1To1(cropAndResize(this.image, this.cursor, currentWindowSize, this.rasterSize, 1).data);
    // The sampler stores the generated canvas as [0 = background, 1 = stroke],
    // while the trained model consumes canvas images as [0 = stroke, 1 =
    // background], followed by the model's own [-1, 1] normalization.
    const patchCanvasRaw = cropAndResize({ data: this.canvas, width: imageSize, height: imageSize, channels: 1 }, this.cursor, currentWindowSize, this.rasterSize).data;
    const patchCanvas = normalizeM1To1(patchCanvasRaw.map((value) => 1 - value));
    const fullCanvasRaw = resizeArea({ data: this.canvas, width: imageSize, height: imageSize, channels: 1 }, this.rasterSize, this.rasterSize).data;
    const fullCanvasSmall = fullCanvasRaw.map((value) => 1 - value);

    const feeds = {
      [MODEL_INPUTS.patchPhoto]: tensor(MODEL_INPUTS.patchPhoto, 'float32', patchPhoto, [1, this.rasterSize, this.rasterSize, this.config.channels]),
      [MODEL_INPUTS.patchCanvas]: tensor(MODEL_INPUTS.patchCanvas, 'float32', patchCanvas, [1, this.rasterSize, this.rasterSize, 1]),
      [MODEL_INPUTS.entirePhoto]: tensor(MODEL_INPUTS.entirePhoto, 'float32', this.fullPhotoSmall, [1, this.rasterSize, this.rasterSize, this.config.channels]),
      [MODEL_INPUTS.entireCanvas]: tensor(MODEL_INPUTS.entireCanvas, 'float32', fullCanvasSmall, [1, this.rasterSize, this.rasterSize, 1]),
      [MODEL_INPUTS.cursor]: tensor(MODEL_INPUTS.cursor, 'float32', new Float32Array(this.cursor), [1, 1, 2]),
      [MODEL_INPUTS.imageSize]: tensor(MODEL_INPUTS.imageSize, 'int32', new Int32Array([imageSize]), []),
      [MODEL_INPUTS.windowSize]: tensor(MODEL_INPUTS.windowSize, 'float32', oneValue(currentWindowSize), [1, 1, 1]),
      [MODEL_INPUTS.previousWidth]: tensor(MODEL_INPUTS.previousWidth, 'float32', oneValue(this.previousWidth), [1, 1, 1]),
      [MODEL_INPUTS.state]: tensor(MODEL_INPUTS.state, 'float32', this.state, [1, this.state.length]),
    };

    let result;
    let params, pen, nextState;
    try {
      result = await this.session.run(feeds);
      params = new Float32Array(result[MODEL_OUTPUTS.params].data);
      pen = new Float32Array(result[MODEL_OUTPUTS.pen].data);
      nextState = new Float32Array(result[MODEL_OUTPUTS.state].data);
    } finally {
      for (const value of [...Object.values(feeds), ...Object.values(result || {})]) value.dispose();
    }
    const penState = pen[1] > pen[0] ? 1 : 0;
    const stroke = penState === 0;

    if (stroke) {
      const patch = renderStrokePatch(params, this.previousWidth, this.rasterSize);
      pastePatch(this.canvas, patch, this.cursor, imageSize, currentWindowSize, this.rasterSize);
      this.strokes.push({ params: Array.from(params), previousWidth: this.previousWidth,
        cursor: [...this.cursor], imageSize, windowSize: currentWindowSize,
        startsNewPath: this.startsNewPath });
    }
    this.startsNewPath = !stroke;

    const nextScaling = Math.min(this.maxScaling, Math.max(0, params[5]));
    const nextWindowSize = clamp(nextScaling * currentWindowSize, this.minWindowSize, imageSize);
    this.previousWidth = params[4] * currentWindowSize / nextWindowSize;
    this.previousScaling = nextScaling;
    this.previousWindowSize = currentWindowSize;
    this.state = new Float32Array(nextState);

    const nextX = this.cursor[0] * imageSize + params[3] * currentWindowSize / 2;
    const nextY = this.cursor[1] * imageSize + params[2] * currentWindowSize / 2;
    this.cursor = [
      clamp(nextX, 0, imageSize - 1) / imageSize,
      clamp(nextY, 0, imageSize - 1) / imageSize,
    ];

    return {
      params: Array.from(params),
      pen: Array.from(pen),
      penState,
      stroke,
      cursor: [...this.cursor],
      windowSize: currentWindowSize,
      canvas: this.canvas,
    };
  }

  async vectorize({ maxRounds = this.config.rounds, maxStepsPerRound = this.config.steps, eosPatience = 12, stateResetInterval = 48, random = Math.random, onStep = null, onRound = null } = {}) {
    if (!this.image || !this.session) throw new Error('Load the model and set an image before vectorizing.');
    const steps = [];
    let totalSteps = 0;
    let strokeCount = 0;
    const isPhoto = this.mode === 'photo';
    // The original photo sampler always executes one fixed-length pass, even
    // through pen-up movements. Ink-coverage reseeding is only meaningful for lines.
    if (isPhoto) maxRounds = 1;
    let cursor = chooseInitialCursor(this.image, this.rasterSize, random) ?? (isPhoto ? [0, 0] : null);
    const cursorHistory = {};

    for (let round = 0; round < maxRounds && cursor; round += 1) {
      this.resetState(cursor);
      if (onRound) await onRound({ round, cursor });
      let noStrokeSteps = 0;

      for (let step = 0; step < maxStepsPerRound; step += 1) {
        // All three original evaluation paths use state_dependent=False and
        // reinitializes only the recurrent state every max_seq_len (48) steps.
        // Preserve the cursor/window/width trajectory while matching that
        // checkpoint's sampling cadence.
        if (stateResetInterval > 0 && step > 0 && step % stateResetInterval === 0) {
          this.state = new Float32Array(1024);
        }
        const result = await this.step();
        totalSteps += 1;
        if (result.stroke) {
          strokeCount += 1;
          noStrokeSteps = 0;
        } else {
          noStrokeSteps += 1;
        }
        const { canvas, ...metadata } = result;
        steps.push({ ...metadata, round, step });
        if (onStep) await onStep({ ...result, round, step, totalSteps, strokeCount });
        if (!isPhoto && noStrokeSteps >= eosPatience) break;
      }
      cursor = isPhoto ? null : this.mode === 'rough'
        ? chooseRoughCursor(this.image, this.cursor, this.rasterSize, random)
        : chooseUndrawnCursor(this.image, this.canvas, this.rasterSize, cursorHistory, random);
    }
    return { canvas: this.canvas, steps, totalSteps, strokeCount };
  }
}
