// Reproduce the joining assessment for our single-segment, fixed-width export.
// This deliberately accepts that format only, not arbitrary external SVGs.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildQuadraticSvg, joinQuadraticSegments } from '../src/svg.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const original = await readFile(resolve(root, 'example-result.svg'), 'utf8');
const number = '-?\\d+(?:\\.\\d+)?';
const pattern = new RegExp(`^M(${number}) (${number})Q(${number}) (${number}) (${number}) (${number})$`);
const paths = [...original.matchAll(/<path\b[^>]*\/>/g)];
if (!paths.length || /\btransform=/.test(original)) throw new Error('Expected untransformed exported quadratic paths.');
const segments = paths.map(([path]) => {
  for (const attribute of ['fill="none"', 'stroke="#000"', 'stroke-width="3.5"', 'stroke-linecap="round"', 'stroke-linejoin="round"']) {
    if (!path.includes(attribute)) throw new Error(`Unexpected path style: ${attribute}`);
  }
  const match = path.match(/\bd="([^"]+)"/)?.[1].match(pattern);
  if (!match) throw new Error('Expected one absolute M/Q segment per path.');
  const values = match.slice(1).map(Number);
  return { start: values.slice(0, 2), control: values.slice(2, 4), end: values.slice(4, 6) };
});
const viewBox = original.match(/viewBox="0 0 ([\d.]+) ([\d.]+)"/);
if (!viewBox) throw new Error('Expected a zero-origin viewBox.');
const width = Number(viewBox[1]), height = Number(viewBox[2]);
const joined = buildQuadraticSvg(segments, width, height);
const groups = joinQuadraticSegments(segments);
const tangent = (a, b, fallbackA, fallbackB) => {
  const v = [b[0] - a[0], b[1] - a[1]];
  return Math.hypot(...v) > 1e-10 ? v : [fallbackB[0] - fallbackA[0], fallbackB[1] - fallbackA[1]];
};
const angles = groups.flatMap(group => group.slice(1).map((next, i) => {
  const prev = group[i];
  const a = tangent(prev.control, prev.end, prev.start, prev.end);
  const b = tangent(next.start, next.control, next.start, next.end);
  const cosine = (a[0] * b[0] + a[1] * b[1]) / (Math.hypot(...a) * Math.hypot(...b));
  return Math.acos(Math.max(-1, Math.min(1, cosine))) * 180 / Math.PI;
}));
// Compare every original control/end pair in order, independently of grouping.
const commands = svg => [...svg.matchAll(/Q[^MQ"<]+/g)].map(m => m[0]);
const controlsUnchanged = JSON.stringify(commands(original)) === JSON.stringify(commands(joined));
if (!controlsUnchanged) throw new Error('Joining changed quadratic commands.');
const metrics = {
  originalPaths: segments.length, joinedPaths: groups.length,
  quadraticSegments: segments.length, segmentCoordinatesUnchanged: controlsUnchanged,
  originalBytes: Buffer.byteLength(original), joinedBytes: Buffer.byteLength(joined),
  segmentsPerPath: groups.map(group => group.length),
  joinedVertices: angles.length,
  tangentChangesAbove30Degrees: angles.filter(a => a > 30).length,
  tangentChangesAbove60Degrees: angles.filter(a => a > 60).length,
};
const output = resolve(root, 'outputs/svg-joining');
await mkdir(output, { recursive: true });
await writeFile(resolve(root, 'example-result-joined.svg'), joined);
await writeFile(resolve(output, 'metrics.json'), JSON.stringify(metrics, null, 2));
const inline = svg => svg.replace(/<\?xml[^>]*\?>\s*/, '');
let colorIndex = 0;
const colored = inline(joined).replace(/stroke="#000"/g, () => `stroke="hsl(${colorIndex++ * 137.508 % 360} 75% 38%)"`);
const safeJson = value => JSON.stringify(value).replace(/</g, '\\u003c');
const html = `<!doctype html><html lang="en"><meta charset="utf-8"><title>SVG joining review</title>
<style>body{font:16px system-ui;margin:24px;color:#18232c;background:#f5f7f9}h1{font-size:24px}main{display:flex;gap:16px;flex-wrap:wrap}figure{margin:0;background:white;padding:16px;border:1px solid #d5dce2}figure svg{width:320px;height:320px}figcaption{font-weight:600}pre{white-space:pre-wrap;font-size:13px}#comparison{padding:16px;background:white}button{margin:8px 8px 8px 0}</style>
<h1>Join continuous strokes</h1><p>${metrics.originalPaths} paths → ${metrics.joinedPaths} paths. All ${metrics.quadraticSegments} quadratic segments retained. ${metrics.originalBytes} → ${metrics.joinedBytes} bytes.</p>
<main><figure><figcaption>Original</figcaption>${inline(original)}</figure><figure><figcaption>Joined</figcaption>${inline(joined)}</figure><figure><figcaption>Joined paths, colored</figcaption>${colored}</figure></main>
<p>Joining uses consecutive endpoint equality. This saved SVG has no pen-up or round metadata. Fresh model runs preserve those boundaries explicitly.</p>
<button id="normal">Normal size</button><button id="large">Enlarge</button>
<pre id="comparison">Measuring render differences at 1× and 4×…</pre><details><summary>Geometry measurements</summary><pre>${JSON.stringify(metrics, null, 2)}</pre></details>
<script type="module">
const originals = [${safeJson(original)}, ${safeJson(joined)}];
document.querySelector('#normal').onclick=()=>document.querySelectorAll('figure svg').forEach(s=>{s.style.width='320px';s.style.height='320px'});
document.querySelector('#large').onclick=()=>document.querySelectorAll('figure svg').forEach(s=>{s.style.width='640px';s.style.height='640px'});
async function render(svg,scale){const url=URL.createObjectURL(new Blob([svg],{type:'image/svg+xml'}));try{const img=new Image();img.src=url;await img.decode();const canvas=document.createElement('canvas');canvas.width=${width}*scale;canvas.height=${height}*scale;const ctx=canvas.getContext('2d');ctx.fillStyle='#fff';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.drawImage(img,0,0,canvas.width,canvas.height);return ctx.getImageData(0,0,canvas.width,canvas.height).data}finally{URL.revokeObjectURL(url)}}
const results=[];
for(const scale of [1,4]){const [a,b]=await Promise.all(originals.map(s=>render(s,scale)));let changed=0,totalError=0,ink=0,lost=0,added=0;for(let i=0;i<a.length;i+=4){const d=Math.abs(a[i]-b[i]);if(d)changed++;totalError+=d;ink+=255-a[i];lost+=Math.max(0,b[i]-a[i]);added+=Math.max(0,a[i]-b[i]);}results.push({scale,changedPixels:changed,totalPixels:a.length/4,absoluteDifferenceAsPercentOfOriginalInk:100*totalError/ink,lostInkPercent:100*lost/ink,addedInkPercent:100*added/ink});}
document.querySelector('#comparison').textContent=JSON.stringify(results,null,2);
</script></html>`;
await writeFile(resolve(output, 'review.html'), html);
console.log(JSON.stringify(metrics, null, 2));
console.log(`Review: ${resolve(output, 'review.html')}`);
