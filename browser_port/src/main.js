import './styles.css';
import { BrowserVectorizer } from './model.js';
import { makeSquareCanvas, imageFromCanvas, preparePhoto, prepareRoughSketch } from './preprocess.js';
import { getMode } from './modes.js';
import { appendStrokePathElement, downloadStrokeSvg, strokeToQuadratic, joinQuadraticSegments, removeRedundantSegments, splitIntersectingPaths, createPathElement, createQuadraticPathElement, previewPathColor, redrawQuadraticPaths } from './svg.js';

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
const joinDistance = element('join-distance');
const joinDistanceValue = element('join-distance-value');
const showOriginal = element('show-original');
const midpointJoining = element('midpoint-joining');
const removeRedundant = element('remove-redundant');
const redundancyTolerance = element('redundancy-tolerance');
const redundancyToleranceValue = element('redundancy-tolerance-value');
const splitIntersections = element('split-intersections');
const redrawCurves = element('redraw-curves');
const fitTolerance = element('fit-tolerance');
const fitToleranceValue = element('fit-tolerance-value');
const resetJoining = element('reset-joining');
const joiningSummary = element('joining-summary');
const originalLayer = element('original-layer');

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
  joinDistance.disabled = showOriginal.disabled = midpointJoining.disabled = resetJoining.disabled = removeRedundant.disabled = splitIntersections.disabled = downloadSvgButton.disabled;
  redundancyTolerance.disabled = downloadSvgButton.disabled || !removeRedundant.checked;
  redrawCurves.disabled = downloadSvgButton.disabled;
  fitTolerance.disabled = downloadSvgButton.disabled || !redrawCurves.checked;
}

function clearJoiningPreview() {
  originalLayer.replaceChildren();
  joiningSummary.textContent = 'Run the vectorizer to preview joining.';
}

function previewJoining() {
  joinDistanceValue.textContent = `${joinDistance.value} px`;
  redundancyToleranceValue.textContent = `${redundancyTolerance.value} px`;
  fitToleranceValue.textContent = `${fitTolerance.value} px²`;
  originalLayer.toggleAttribute('hidden', !showOriginal.checked);
  if (!loadedImage || !vectorizer?.strokes?.length) return;
  const segments = vectorizer.strokes.map(strokeToQuadratic);
  const baseline = joinQuadraticSegments(segments);
  const reducedSegments = removeRedundant.checked
    ? removeRedundantSegments(segments, Number(redundancyTolerance.value)) : segments;
  const exactReducedPaths = joinQuadraticSegments(reducedSegments);
  const joinedPaths = joinQuadraticSegments(reducedSegments, Number(joinDistance.value), { midpoint: midpointJoining.checked });
  const split = splitIntersections.checked ? splitIntersectingPaths(joinedPaths) : { paths: joinedPaths, intersections: 0 };
  const paths = redrawCurves.checked ? redrawQuadraticPaths(split.paths, Number(fitTolerance.value)) : split.paths;
  strokeLayer.replaceChildren(...paths.map((path, index) => createPathElement(path, { stroke: previewPathColor(index) })));
  originalLayer.replaceChildren(...(showOriginal.checked ? baseline.map(createQuadraticPathElement) : []));
  const removed = segments.length - reducedSegments.length;
  const joins = exactReducedPaths.length - joinedPaths.length;
  const segmentCount = pathSet => pathSet.reduce((count, path) => count + path.length, 0);
  const redrawSummary = redrawCurves.checked
    ? ` · ${segmentCount(joinedPaths)} → ${segmentCount(paths)} cubic segments`
    : ` · ${segmentCount(paths)} segments`;
  joiningSummary.textContent = `${baseline.length} → ${paths.length} paths · ${removed} redundant strokes removed · ${joins} nearby joins · ${split.intersections} intersections split${redrawSummary}`;
}

joinDistance.addEventListener('input', previewJoining);
showOriginal.addEventListener('change', previewJoining);
midpointJoining.addEventListener('change', previewJoining);
removeRedundant.addEventListener('change', () => { updateControls(); previewJoining(); });
redundancyTolerance.addEventListener('input', previewJoining);
splitIntersections.addEventListener('change', previewJoining);
redrawCurves.addEventListener('change', () => { updateControls(); previewJoining(); });
fitTolerance.addEventListener('input', previewJoining);
resetJoining.addEventListener('click', () => { joinDistance.value = '0'; previewJoining(); });

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
    appendStrokePathElement(strokeLayer, vectorizer.strokes[i], vectorizer.strokes[i - 1]);
  }
  displayedStrokeCount = vectorizer.strokes.length;
}

function resetOutput() {
  if (!loadedImage) return;
  vectorizer.setImage(loadedImage);
  strokeLayer.replaceChildren();
  clearJoiningPreview();
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
  clearJoiningPreview();
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
  previewJoining();
  setStatus(`Finished ${result.totalSteps} steps in ${((performance.now() - start) / 1000).toFixed(1)}s.`);
}));

downloadSvgButton.addEventListener('click', () => {
  if (!loadedImage || !vectorizer?.strokes.length) return;
  downloadStrokeSvg(vectorizer.strokes, loadedImage.width, loadedImage.height, 'virtual-sketching.svg', {
    joinDistance: Number(joinDistance.value), midpoint: midpointJoining.checked,
    removeRedundant: removeRedundant.checked, redundancyTolerance: Number(redundancyTolerance.value),
    splitIntersections: splitIntersections.checked,
    redraw: redrawCurves.checked, fitTolerance: Number(fitTolerance.value),
  });
});

perform(loadMode);
