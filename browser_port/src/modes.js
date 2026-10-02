// Checkpoint-specific contracts from dataset_utils.py and the three test samplers.
const baseUrl = import.meta.env?.BASE_URL ?? '/';

export const MODES = {
  drawing: {
    label: 'Photo → drawing → vectors', metric: 'drawing → clean', channels: 1,
    vectorMode: 'line',
    modelUrl: `${baseUrl}models/virtual_sketching_step.onnx`,
    sampleUrl: `${baseUrl}samples/portrait.png`, sampleLabel: 'Load bundled portrait',
    rounds: 4, steps: 128,
    description: 'Turn a photograph into a line drawing with Informative Drawings, then trace it with the clean-line model. Generate and inspect the drawing first. This is experimental: fine details may be lost and contours can have gaps.',
  },
  line: {
    label: 'Line drawing', metric: 'clean / WASM', channels: 1,
    modelUrl: `${baseUrl}models/virtual_sketching_step.onnx`,
    sampleUrl: `${baseUrl}samples/duck.png`, sampleLabel: 'Load bundled duck',
    rounds: 10, steps: 500,
    description: 'Trace a clean, dark line drawing on a white background. Original resolution is preserved.',
  },
  rough: {
    label: 'Rough sketch', metric: 'rough / WASM', channels: 3,
    modelUrl: `${baseUrl}models/virtual_sketching_rough_step.onnx`,
    sampleUrl: `${baseUrl}samples/penguin1.png`, sampleLabel: 'Load bundled penguin',
    rounds: 10, steps: 128,
    description: 'Simplify pencil strokes and sketch noise into cleaner contours. RGB and original resolution are preserved; non-square inputs are padded with a sampled background color. Results vary between runs and may leave gaps.',
  },
  photo: {
    label: 'Photo (portraits)', metric: 'faces / WASM', channels: 3,
    modelUrl: `${baseUrl}models/virtual_sketching_faces_step.onnx`,
    sampleUrl: `${baseUrl}samples/portrait.png`, sampleLabel: 'Load bundled portrait',
    rounds: 1, steps: 100,
    description: 'Photo-to-sketch model trained on faces. Use a tightly framed portrait; other subjects may not work well. RGB input is padded black on the right/bottom and resized to 256 × 256. One continuous pass produces a stylized sketch, not an exact tracing.',
  },
};

export function getMode(mode) {
  if (!Object.hasOwn(MODES, mode)) throw new Error(`Unknown input mode: ${mode}`);
  return MODES[mode];
}
