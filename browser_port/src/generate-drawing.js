import { DRAWING_MODEL } from './informative-drawings-config.js';

export function generateDrawing(image, onStatus = () => {}) {
  const baseUrl = new URL(import.meta.env.BASE_URL, window.location.href);
  // A fresh worker releases the network's WASM heap after each generation.
  // The cached raster is enough for subsequent cleanup and vectorization.
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./drawing-worker.js', import.meta.url), { type: 'module' });
    const finish = (error, drawing) => {
      worker.terminate();
      if (error) reject(error); else resolve(drawing);
    };
    worker.onmessage = ({ data }) => {
      if (data.status) onStatus(data.status);
      else if (data.error) finish(new Error(data.error));
      else if (data.drawing) finish(null, data.drawing);
    };
    worker.onerror = event => finish(new Error(event.message || 'Drawing worker failed.'));
    worker.onmessageerror = () => finish(new Error('Could not read the generated drawing.'));
    worker.postMessage({ image, modelUrl: new URL(`models/${DRAWING_MODEL.filename}`, baseUrl).href,
      wasmUrl: new URL('ort-wasm/', baseUrl).href });
  });
}
