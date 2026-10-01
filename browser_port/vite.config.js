import { defineConfig } from 'vite';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const configDir = dirname(fileURLToPath(import.meta.url));
const ortWasmDir = resolve(configDir, 'public', 'ort-wasm');
const modelDir = resolve(configDir, 'public', 'models');

export default defineConfig({
  plugins: [{
    name: 'serve-inference-assets',
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        const pathname = (request.url || '').split('?')[0];
        // Model files are ignored by the watcher, so Vite's public-file cache
        // can miss a newly exported checkpoint until restart. Serve the
        // known artifacts directly and never return SPA HTML for missing ones.
        if (['/models/virtual_sketching_step.onnx', '/models/virtual_sketching_faces_step.onnx', '/models/virtual_sketching_rough_step.onnx'].includes(pathname)) {
          try {
            const source = readFileSync(resolve(modelDir, pathname.slice('/models/'.length)));
            response.setHeader('Content-Type', 'application/octet-stream');
            response.end(source);
          } catch {
            response.statusCode = 404;
            response.end('Model not found. Run npm run prepare-model.');
          }
          return;
        }
        if (!pathname.startsWith('/ort-wasm/') || !pathname.endsWith('.mjs')) {
          next();
          return;
        }
        const filename = pathname.slice('/ort-wasm/'.length);
        try {
          const source = readFileSync(resolve(ortWasmDir, filename));
          response.statusCode = 200;
          response.setHeader('Content-Type', 'application/javascript');
          response.end(source);
        } catch {
          next();
        }
      });
    },
  }],
  server: {
    watch: {
      ignored: [
        '**/public/models/**',
        '**/public/ort-wasm/**',
      ],
    },
  },
});
