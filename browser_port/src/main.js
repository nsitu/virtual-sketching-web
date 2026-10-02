import './styles.css';
import { BrowserVectorizer } from './model.js';
import { makeSquareCanvas, imageFromCanvas, preparePhoto, prepareRoughSketch } from './preprocess.js';
import { getMode } from './modes.js';
import { generateDrawing } from './generate-drawing.js';
import { cleanDrawing, squareDrawing } from './informative-drawings.js';
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
const generateButton = element('generate-drawing-button');
const drawingResolution = element('drawing-resolution');
const drawingCutoff = element('drawing-cutoff');
const drawingCanvas = element('drawing-canvas');
const downloadDrawingButton = element('download-drawing-button');

let vectorizer = null;
let loadedImage = null;
let sourceFile = null;
let sourceIsSample = false;
let busy = false;
let displayedStrokeCount = 0;
let photoImage = null;
let rawDrawing = null;

function setStatus(message, kind = '') {
  status.textContent = message;
  status.dataset.kind = kind;
}

function updateControls() {
  input.disabled = sampleButton.disabled = modeInput.disabled = stepsInput.disabled = retryButton.disabled = busy;
  roundsInput.disabled = busy || modeInput.value === 'photo';
  runButton.disabled = busy || !loadedImage || (modeInput.value !== 'drawing' && !vectorizer?.session);
  generateButton.disabled = busy || !photoImage;
  drawingResolution.disabled = busy;
  drawingCutoff.disabled = busy || !rawDrawing;
  downloadDrawingButton.disabled = busy || !rawDrawing;
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

function drawInput(image, canvas = inputCanvas) {
  canvas.width = image.width;
  canvas.height = image.height;
  const pixels = new ImageData(image.width, image.height);
  for (let i = 0; i < image.width * image.height; i++) {
    for (let c = 0; c < 3; c++) pixels.data[i * 4 + c] = Math.round(image.data[i * image.channels + (image.channels === 1 ? 0 : c)] * 255);
    pixels.data[i * 4 + 3] = 255;
  }
  canvas.getContext('2d').putImageData(pixels, 0, 0);
}

function updateStrokePreview(size, fromIndex = 0) {
  outputSvg.setAttribute('viewBox', `0 0 ${size} ${size}`);
  for (let i = fromIndex; i < vectorizer.strokes.length; i++) {
    appendStrokePathElement(strokeLayer, vectorizer.strokes[i], vectorizer.strokes[i - 1]);
  }
  displayedStrokeCount = vectorizer.strokes.length;
}

function resetOutput() {
  if (loadedImage) vectorizer.setImage(loadedImage);
  else if (vectorizer) vectorizer.strokes = [];
  strokeLayer.replaceChildren();
  clearJoiningPreview();
  outputSvg.setAttribute('viewBox', `0 0 ${loadedImage?.width ?? 512} ${loadedImage?.height ?? 512}`);
  displayedStrokeCount = 0;
  stepsMetric.textContent = strokesMetric.textContent = '0';
}

async function prepareFile(file) {
  const bitmap = await createImageBitmap(file);
  try {
    if (modeInput.value === 'drawing') {
      const limit = Number(drawingResolution.value);
      const scale = Math.min(1, limit / Math.max(bitmap.width, bitmap.height));
      const canvas = document.createElement('canvas');
      canvas.width = Math.max(1, Math.round(bitmap.width * scale));
      canvas.height = Math.max(1, Math.round(bitmap.height * scale));
      const context = canvas.getContext('2d');
      context.fillStyle = '#fff';
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
      photoImage = imageFromCanvas(canvas, 3);
      rawDrawing = loadedImage = null;
      drawingCanvas.getContext('2d').clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    } else if (modeInput.value !== 'line') {
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
  const preview = photoImage ?? loadedImage;
  drawInput(preview);
  sizeMetric.textContent = `${preview.width} × ${preview.height}`;
}

function updateDrawingPreview() {
  element('drawing-cutoff-value').textContent = Number(drawingCutoff.value).toFixed(3);
  if (!rawDrawing) return;
  const drawing = cleanDrawing(rawDrawing, Number(drawingCutoff.value));
  drawInput(drawing, drawingCanvas);
  loadedImage = squareDrawing(drawing);
  resetOutput();
  sizeMetric.textContent = `${loadedImage.width} × ${loadedImage.height}`;
  updateControls();
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
  element('input-label').textContent = { line: 'Input clean line drawing', rough: 'Input rough sketch', photo: 'Input photograph', drawing: 'Input photograph' }[mode];
  element('drawing-controls').hidden = element('drawing-figure').hidden = mode !== 'drawing';
  element('workbench').classList.toggle('with-drawing', mode === 'drawing');
  element('input-caption').textContent = mode === 'drawing' ? 'Photo' : 'Input';
  runButton.textContent = mode === 'drawing' ? 'Vectorize drawing' : 'Run vectorizer';
  element('mode-description').textContent = config.description;
  sampleButton.textContent = config.sampleLabel;
  retryButton.hidden = true;
  modelMetric.textContent = 'loading…';
  setStatus(`Loading ${config.label.toLowerCase()} model…`);
  // Only one session is retained: switching modes does not accumulate models.
  const oldSession = vectorizer?.session;
  vectorizer = new BrowserVectorizer({ mode: config.vectorMode ?? mode });
  loadedImage = null;
  photoImage = rawDrawing = null;
  inputCanvas.getContext('2d').clearRect(0, 0, inputCanvas.width, inputCanvas.height);
  drawingCanvas.getContext('2d').clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
  resetOutput();
  sizeMetric.textContent = '—';
  if (oldSession) await oldSession.release();
  try {
    // First-stage generation releases its worker before the vectorizer loads.
    if (mode !== 'drawing') await vectorizer.load();
  } catch (error) {
    modelMetric.textContent = 'load failed';
    retryButton.hidden = false;
    throw new Error(`Could not load ${config.label.toLowerCase()} model. Run npm run prepare-model and retry. ${error.message}`);
  }
  modelMetric.textContent = config.metric;
  if (sourceIsSample) sourceFile = await bundledFile();
  if (sourceFile) await prepareFile(sourceFile);
  if (mode === 'drawing') setStatus(photoImage ? 'Photo ready. Generate a drawing next.' : 'Choose a photo or load the bundled portrait.');
  else setStatus(loadedImage ? 'Model and image ready.' : 'Model loaded. Choose an image or load the bundled sample.');
}

drawingResolution.addEventListener('change', () => perform(async () => {
  if (sourceFile) await prepareFile(sourceFile);
  setStatus('Resolution changed. Generate the drawing again.');
}));
drawingCutoff.addEventListener('input', () => {
  updateDrawingPreview();
  setStatus('Drawing cleanup updated. Vectorize again to refresh the result.');
});
generateButton.addEventListener('click', () => perform(async () => {
  if (!photoImage) return;
  if (vectorizer.session) {
    await vectorizer.session.release();
    vectorizer.session = null;
  }
  const start = performance.now();
  const drawing = await generateDrawing(photoImage, setStatus);
  rawDrawing = drawing;
  updateDrawingPreview();
  setStatus(`Drawing generated in ${((performance.now() - start) / 1000).toFixed(1)}s. Adjust background cleanup, then vectorize.`);
}));
downloadDrawingButton.addEventListener('click', () => {
  drawingCanvas.toBlob(blob => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'informative-drawing.png';
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, 'image/png');
});

modeInput.addEventListener('change', () => perform(loadMode));
retryButton.addEventListener('click', () => perform(loadMode));
input.addEventListener('change', () => perform(async () => {
  const [file] = input.files;
  if (!file) return;
  await prepareFile(file);
  sourceFile = file;
  sourceIsSample = false;
  setStatus(modeInput.value === 'drawing' ? 'Photo ready. Generate a drawing next.' : 'Image ready.');
}));
sampleButton.addEventListener('click', () => perform(async () => {
  const file = await bundledFile();
  await prepareFile(file);
  sourceFile = file;
  sourceIsSample = true;
  input.value = '';
  setStatus(modeInput.value === 'drawing' ? 'Bundled photo ready. Generate a drawing next.' : 'Bundled sample ready.');
}));
resetButton.addEventListener('click', () => {
  resetOutput();
  updateControls();
  setStatus('Reset.');
});

runButton.addEventListener('click', () => perform(async () => {
  if (!loadedImage) return;
  if (!vectorizer.session) {
    setStatus('Loading clean-line vectorizer…');
    await vectorizer.load();
  }
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
