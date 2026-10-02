import * as ort from 'onnxruntime-web';
import { drawingTensorInput, cropDrawingOutput } from './informative-drawings.js';

self.onmessage = async ({ data: { image, modelUrl, wasmUrl } }) => {
  let session, input, outputs;
  try {
    // One thread works on GitHub Pages without cross-origin isolation headers.
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = wasmUrl;
    self.postMessage({ status: 'Loading drawing model…' });
    const response = await fetch(modelUrl);
    if (!response.ok || response.headers.get('content-type')?.includes('text/html')) {
      throw new Error(`Drawing model unavailable (HTTP ${response.status}). Run npm run prepare-model.`);
    }
    session = await ort.InferenceSession.create(await response.arrayBuffer(), {
      executionProviders: ['wasm'], graphOptimizationLevel: 'all',
    });
    self.postMessage({ status: 'Generating drawing…' });
    const prepared = drawingTensorInput(image);
    input = new ort.Tensor('float32', prepared.data, prepared.dims);
    outputs = await session.run({ input });
    const drawing = cropDrawingOutput(outputs.output.data, outputs.output.dims, image.width, image.height);
    for (const tensor of Object.values(outputs)) tensor.dispose();
    outputs = null;
    input.dispose();
    input = null;
    await session.release();
    session = null;
    self.postMessage({ drawing }, [drawing.data.buffer]);
  } catch (error) {
    self.postMessage({ error: error.message || String(error) });
  } finally {
    input?.dispose();
    for (const tensor of Object.values(outputs || {})) tensor.dispose();
    if (session) await session.release();
  }
};
