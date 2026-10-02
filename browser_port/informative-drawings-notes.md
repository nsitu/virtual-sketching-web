# Photo → Informative Drawings → Virtual Sketching

Investigation date: 2026-10-01. The chain is feasible with the existing ONNX
Runtime Web dependency. A local experiment has run both models in sequence and
generated SVGs. This is an experiment, not a new browser input mode yet.

## Verified model sources

- [Original project](https://github.com/carolineec/informative-drawings).
- [Author's Hugging Face Space](https://huggingface.co/spaces/carolineec/informativedrawings),
  whose [app.py](https://huggingface.co/spaces/carolineec/informativedrawings/blob/main/app.py)
  loads two drawing styles as PyTorch checkpoints.
- [Published ONNX export](https://huggingface.co/rocca/informative-drawings-line-art-onnx/tree/main):
  a single 17,193,338-byte `model.onnx`, with no external weights required.
- [JavaScript/WASM reference](https://github.com/josephrocca/image-to-line-art-js)
  and its [conversion notebook](https://github.com/josephrocca/image-to-line-art-js/blob/main/Informative_Drawings_Line_Art_Generator_ONNX_conversion.ipynb).
  The notebook exports `model.pth`, corresponding to the Space's first style;
  the published file does not implement the Space's two-style selector.

The probe pins the model repository to revision
`d38eccbd448cdcd228fb81d708506e5e60b41ccb` and checks SHA-256
`1fef40b8f7126d827e30fbebccf95ae9b0b391795df926bf9366a821bad4f498`.
If a second style is wanted, verify another published export or convert the
Space's second checkpoint separately.

## Tensor handoff

Informative Drawings accepts a float32 tensor named `input`, shaped
`[1, 3, H, W]`. Supply RGB channel planes in `[0, 1]`; do not apply the Virtual
Sketching patch normalization to this input. Its `output` tensor is
`[1, 1, Hout, Wout]`, with dark ink near zero and white background near one.
The conversion notebook uses ONNX opset 12 and dynamic spatial axes.

Our local WASM probe confirmed this contract at 256×256. A 64×96 tensor also
returned 64×96, while a 65×97 tensor returned 68×100. Use dimensions divisible
by four (or explicitly pad and crop) rather than assuming arbitrary input and
output sizes match. This is consistent with the network's two downsampling
and upsampling stages.

The one-channel output can become our existing image record directly:
`{ data: Float32Array, width, height, channels: 1 }`. After cleanup and white
square padding, pass it to `BrowserVectorizer({ mode: 'line' }).setImage(...)`.
There is no need for a JPEG round trip, an inversion, or RGB duplication.
The vectorizer already normalizes its own local patches and maintains its
variable-width raster feedback. Its clean-line checkpoint is the first choice
for this handoff; the existing face-trained `photo` checkpoint is a separate
approach to direct photo vectorization.

## Important adaptation: near-white background

`chooseInitialCursor` and `chooseUndrawnCursor` count every input pixel below
exactly one as ink. The generated floating-point drawing contains faint gray
values across nearly its entire background. Feeding it directly can spend
tracing rounds on nearly invisible marks.

Use a configurable white cutoff: snap values at or above the cutoff to exactly
one, keeping darker values and their soft edges. The probe compares raw output
with cutoffs 0.995 and 0.98. This is background cleanup, not conversion to a
binary image. Keep the original drawing separately and recompute from it when
the cutoff changes. A cutoff slider should preview the actual tracing input;
lower values discard more faint detail. These cutoffs are experimental settings,
not validated defaults across photographs.

## Local feasibility results

`node scripts/probe-informative-drawings.mjs` runs from `browser_port` using the
existing 256×256 RGB fixture at `outputs/photo-parity/photo.f32` and the existing
clean-line checkpoint. It downloads only the public weights and runs locally;
no image is uploaded. Results and cached weights go into the ignored
`outputs/informative-drawings-probe` directory.

The successful run used `onnxruntime-web` 1.23.2, its WASM execution provider,
Node.js, and one thread. Model session creation took 0.39 seconds; the drawing
pass took 2.60 seconds. The clean-line vectorizer used seed 42, up to four
rounds and 128 steps per round, with exact joining and no additional geometry
cleanup, to isolate the two-model handoff:

| White cutoff | Pixels counted as ink | Model steps | Stroke segments | Exact joined paths | Tracing time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw (1) | 99.06% | 356 | 192 | 96 | 20.73 s |
| 0.995 | 36.77% | 345 | 178 | 84 | 12.26 s |
| 0.98 | 25.82% | 293 | 158 | 76 | 10.37 s |

Timing includes effects such as runtime warm-up and is not a browser benchmark.
The resulting SVGs are recognizable but visibly simplify the drawing, leave
gaps, and lose small details. The uniform 3.5-pixel export width also appears
heavy at this low resolution. This confirms compatibility, not preservation
of every contour. The side-by-side diagnostic PNG uses sampled SVG curves;
the SVG files themselves contain the exact exported quadratic geometry.

One portrait is not sufficient to evaluate general photographs. Compare
objects, buildings, scenes, and portraits at 256 and 512 pixels, with fixed
random seeds and tracing budgets. Evaluate the intermediate drawing as well
as the vector result: details discarded in the first stage cannot be recovered
by the second. Tune background cleanup and tracing coverage before enabling
aggressive pruning, joining, or redraw for this new input source.

## Browser integration

Implemented in the **Photo → drawing → vectors** mode. It includes a dedicated
WASM worker, 256/512 maximum-size selection, a cached grayscale preview,
background cutoff (0.980 default), PNG download and the existing clean-line
vectorizer / SVG controls. The worker terminates before the clean-line session
loads. `prepare-model` fetches the pinned export, verifies its SHA-256 and hosts
it alongside the existing same-origin models. The upstream MIT notice ships
with the site. Photos are never enlarged; output remains on the vectorizer's
square white-padded coordinate system. Mobile performance, other photographic
subjects, alternate drawing styles and WebGPU still need evaluation.

The original integration plan is retained below for context and follow-up:

Browser smoke checks on the production WASM build also succeeded at both `/`
and `/virtual-sketching-web/`. One 256×256 portrait run took 2.1 s for drawing
generation and 8.5 s for 352 tracing steps / 188 strokes. A rectangular fixture
resized to 512×343 (network padding cropped back to that size) took 3.0 s for
drawing generation and 11.0 s for 434 tracing steps / 244 strokes, with square
white padding applied only for vectorization. These are local desktop timings,
not mobile benchmarks; stochastic tracing changes the counts between runs.
The cutoff cleared stale vectors without network inference, changing resolution
invalidated the cached drawing, and the desktop/mobile preview layouts worked.

1. Add a separate **Photo → drawing → vectors** input option using the clean-line
   checkpoint. Keep the existing portrait mode available for comparison.
2. Decode RGB, composite transparency on white, and preserve aspect ratio.
   Start with a configurable working resolution of 512 pixels on the long
   side, with 256 as a lower-memory option. Pad to multiples of four for the
   drawing network and crop that padding after inference. Do square white
   padding for Virtual Sketching after the drawing pass, rather than feeding
   the drawing network the current portrait mode's black padding.
3. Implement a small drawing-session wrapper, NCHW packing, output validation,
   and tensor disposal. Load the drawing model lazily, then release its session
   before loading the 40.7 MB clean-line checkpoint. WASM activation memory is
   larger than the weight download; benchmark mobile memory and 512-pixel cost.
4. Show **Photo / Generated drawing / Vector result**, with a white-cutoff
   control and separate **Generate drawing** and **Vectorize** actions. Cache
   the intermediate drawing so cleanup changes do not rerun its network.
5. Feed the cleaned drawing through the current vectorizer and SVG cleanup
   controls. Preserve working-image transforms so SVG coordinates can later
   be mapped to the photo's original aspect ratio and square padding cropped.
6. Verify the actual browser WASM path, error recovery and session transitions.
   Start with WASM; WebGPU compatibility and speed remain untested. Keep long
   inference off the UI thread and show progress for both stages.

For GitHub Pages, host a pinned, verified model asset alongside our current
models or fetch it lazily from the pinned Hugging Face URL. Extend asset
preparation and deployment if hosting locally; cache the download either way.
The initial prototype requires no retraining and no new inference framework.
The main work is preprocessing, intermediate preview, resource management,
and evaluating the two stages' combined fidelity.

The [upstream code license](https://github.com/carolineec/informative-drawings/blob/main/LICENSE)
is MIT. The ONNX repository has no separate license declaration. Preserve
upstream attribution and the license text with redistributed implementation
code, and record checkpoint provenance when adding the model asset.
