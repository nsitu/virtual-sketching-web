import './styles.css';
import { BrowserVectorizer } from './model.js';
import { makeSquareCanvas, imageFromCanvas, preparePhoto, prepareRoughSketch } from './preprocess.js';
import { getMode } from './modes.js';
import { createStrokePathElement, downloadStrokeSvg } from './svg.js';

const element = id => document.getElementById(id);
const input = element('image-input');
const modeInput = element('mode-input');
const sampleButton = element('sample-button');
const runButton = element('run-button');
const resetButton = element('reset-button');
const downloadSvgButton = element('download-svg-button');
const retryButton = element('retry-button');
const roundsInput = element('rounds-input');
const stepsInput = element('steps-input');
const status = element('status');
const inputCanvas = element('input-canvas');
const outputSvg = element('output-svg');
const strokeLayer = element('stroke-layer');
const modelMetric = element('model-metric');
const sizeMetric = element('size-metric');
const stepsMetric = element('steps-metric');
const strokesMetric = element('strokes-metric');

let vectorizer = null;
let loadedImage = null;
let sourceFile = null;
let sourceIsSample = false;
let busy = false;
let displayedStrokeCount = 0;

function setStatus(message, kind = '') {
  status.textContent = message;
  status.dataset.kind = kind;
}

function updateControls() {
  input.disabled = sampleButton.disabled = modeInput.disabled = stepsInput.disabled = retryButton.disabled = busy;
  roundsInput.disabled = busy || modeInput.value === 'photo';
  runButton.disabled = busy || !loadedImage || !vectorizer?.session;
  resetButton.disabled = busy || !loadedImage;
  downloadSvgButton.disabled = busy || !loadedImage || !vectorizer?.strokes?.length;
}

function drawInput(image) {
  inputCanvas.width = image.width;
  inputCanvas.height = image.height;
  const pixels = new ImageData(image.width, image.height);
  for (let i = 0; i < image.width * image.height; i++) {
    for (let c = 0; c < 3; c++) pixels.data[i * 4 + c] = Math.round(image.data[i * image.channels + (image.channels === 1 ? 0 : c)] * 255);
    pixels.data[i * 4 + 3] = 255;
  }
  inputCanvas.getContext('2d').putImageData(pixels, 0, 0);
}

function updateStrokePreview(size, fromIndex = 0) {
  outputSvg.setAttribute('viewBox', `0 0 ${size} ${size}`);
  for (let i = fromIndex; i < vectorizer.strokes.length; i++) {
    strokeLayer.append(createStrokePathElement(vectorizer.strokes[i]));
  }
  displayedStrokeCount = vectorizer.strokes.length;
}

function resetOutput() {
  if (!loadedImage) return;
  vectorizer.setImage(loadedImage);
  strokeLayer.replaceChildren();
  outputSvg.setAttribute('viewBox', `0 0 ${loadedImage.width} ${loadedImage.height}`);
  displayedStrokeCount = 0;
  stepsMetric.textContent = strokesMetric.textContent = '0';
}

async function prepareFile(file) {
  const bitmap = await createImageBitmap(file);
  try {
    if (modeInput.value !== 'line') {
      const canvas = document.createElement('canvas');
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      const context = canvas.getContext('2d');
      context.fillStyle = modeInput.value === 'photo' ? '#000' : '#fff';
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.drawImage(bitmap, 0, 0);
      const rgb = imageFromCanvas(canvas, 3);
      loadedImage = modeInput.value === 'photo' ? preparePhoto(rgb) : prepareRoughSketch(rgb);
    } else {
      loadedImage = imageFromCanvas(makeSquareCanvas(bitmap, 1), 1);
    }
  } finally {
    bitmap.close();
  }
  resetOutput();
  drawInput(loadedImage);
  sizeMetric.textContent = `${loadedImage.width} × ${loadedImage.height}`;
}

async function bundledFile() {
  const response = await fetch(getMode(modeInput.value).sampleUrl);
  if (!response.ok) throw new Error(`Could not load bundled sample (${response.status}).`);
  return response.blob();
}

async function perform(action) {
  if (busy) return;
  busy = true;
  updateControls();
  try {
    await action();
  } catch (error) {
    console.error(error);
    setStatus(error.message, 'error');
  } finally {
    busy = false;
    updateControls();
  }
}

async function loadMode() {
  const mode = modeInput.value;
  const config = getMode(mode);
  roundsInput.value = config.rounds;
  stepsInput.value = config.steps;
  element('input-label').textContent = { line: 'Input clean line drawing', rough: 'Input rough sketch', photo: 'Input photograph' }[mode];
  element('mode-description').textContent = config.description;
  sampleButton.textContent = config.sampleLabel;
  retryButton.hidden = true;
  modelMetric.textContent = 'loading…';
  setStatus(`Loading ${config.label.toLowerCase()} model…`);
  // Only one session is retained: switching modes does not accumulate models.
  const oldSession = vectorizer?.session;
  vectorizer = new BrowserVectorizer({ mode });
  loadedImage = null;
  strokeLayer.replaceChildren();
  displayedStrokeCount = 0;
  if (oldSession) await oldSession.release();
  try {
    await vectorizer.load();
  } catch (error) {
    modelMetric.textContent = 'load failed';
    retryButton.hidden = false;
    throw new Error(`Could not load ${config.label.toLowerCase()} model. Run npm run prepare-model and retry. ${error.message}`);
  }
  modelMetric.textContent = config.metric;
  if (sourceIsSample) sourceFile = await bundledFile();
  if (sourceFile) await prepareFile(sourceFile);
  setStatus(loadedImage ? 'Model and image ready.' : 'Model loaded. Choose an image or load the bundled sample.');
}

modeInput.addEventListener('change', () => perform(loadMode));
retryButton.addEventListener('click', () => perform(loadMode));
input.addEventListener('change', () => perform(async () => {
  const [file] = input.files;
  if (!file) return;
  await prepareFile(file);
  sourceFile = file;
  sourceIsSample = false;
  setStatus('Image ready.');
}));
sampleButton.addEventListener('click', () => perform(async () => {
  const file = await bundledFile();
  await prepareFile(file);
  sourceFile = file;
  sourceIsSample = true;
  input.value = '';
  setStatus('Bundled sample ready.');
}));
resetButton.addEventListener('click', () => {
  resetOutput();
  updateControls();
  setStatus('Reset.');
});

runButton.addEventListener('click', () => perform(async () => {
  if (!loadedImage || !vectorizer.session) return;
  resetOutput();
  updateStrokePreview(loadedImage.width);
  const config = getMode(modeInput.value);
  const rounds = Math.floor(Math.max(1, Math.min(12, Number(roundsInput.value) || config.rounds)));
  const steps = Math.floor(Math.max(1, Math.min(500, Number(stepsInput.value) || config.steps)));
  const start = performance.now();
  setStatus('Running ONNX Runtime Web…');
  const result = await vectorizer.vectorize({
    maxRounds: rounds, maxStepsPerRound: steps,
    onStep: async ({ totalSteps, strokeCount, round, step }) => {
      if (step !== 0 && step % 8 !== 0 && step !== steps - 1) return;
      updateStrokePreview(loadedImage.width, displayedStrokeCount);
      stepsMetric.textContent = String(totalSteps);
      strokesMetric.textContent = String(strokeCount);
      setStatus(`Round ${round + 1}, step ${step + 1}…`);
      await new Promise(resolve => setTimeout(resolve, 0));
    },
  });
  updateStrokePreview(loadedImage.width, displayedStrokeCount);
  stepsMetric.textContent = String(result.totalSteps);
  strokesMetric.textContent = String(result.strokeCount);
  setStatus(`Finished ${result.totalSteps} steps in ${((performance.now() - start) / 1000).toFixed(1)}s.`);
}));

downloadSvgButton.addEventListener('click', () => {
  if (!loadedImage || !vectorizer?.strokes.length) return;
  downloadStrokeSvg(vectorizer.strokes, loadedImage.width, loadedImage.height);
});

perform(loadMode);
