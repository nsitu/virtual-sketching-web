# Browser ONNX demo

This directory contains a Vite + ONNX Runtime Web integration for the exported
clean-line-drawing, rough-sketch and portrait photo-to-line models. Each ONNX
graph is batch-one and one-step; JavaScript owns the autoregressive sampling
loop and keeps the recurrent state, raster feedback buffer, cursor, window size,
previous stroke width and vector stroke records.

## Run locally

From this directory:

```powershell
npm install
npm run dev
```

`prepare-model` copies the three ONNX models into the Vite public directory.
All three must be exported first (see below). The generated models are ignored
by Git and are about 40 MB each. `npm run build` performs the same copy before
building.

Open the local URL printed by Vite, select **Line drawing**, **Rough sketch**, or
**Photo (portraits)**, upload an image, and click **Run vectorizer**. The sample
button loads a duck, rough penguin or portrait for the selected mode.
The demo uses WASM execution through ONNX
Runtime Web. Images stay in the browser; there is no image-upload endpoint.
Only the selected model is loaded, and switching releases the previous session.
Uploaded originals are retained and reprocessed on mode changes; bundled samples
switch to the appropriate sample. Each run starts from a fresh canvas.

## Rough-sketch mode

Uses `pretrain_rough_sketches` to simplify pencil marks and paper noise into
cleaner contours. It is a separate learned model, not a thresholding filter or
the clean-line checkpoint with different settings. The bundled example is
`sample_inputs/rough_sketches/penguin1.png`.

- RGB and native resolution are preserved. Non-square images are padded on the
  right/bottom with the RGB pixel at `(width - 10, height - 10)`, matching the
  original loader. For uploads smaller than ten pixels, that location is clamped.
- Defaults match `test_rough_sketch_simplification.py`: 10 rounds, up to 128
  steps each, ending a round after 12 consecutive pen-up steps. Recurrent state
  resets every 48 steps and cursor/window/width state resets at each new round.
- Between rounds the cursor moves from its previous endpoint by random signed
  offsets of 15–45% of the image size on each axis, clamped to the image. It
  checks for nonwhite RGB pixels nearby, retrying up to 20 times. This is not
  the clean-line mode's canvas-coverage selection.
- Runs are stochastic and can leave gaps or omit details. Another run may give
  a better simplification. Entirely white inputs skip inference as a browser
  safety/efficiency improvement over the Python sampler.

## Photo mode

This is the repository's **pretrain_faces** checkpoint, not an edge filter applied
to the clean-line model. It was trained on face photographs and produces sparse,
stylized portrait sketches. Use a tightly framed face. Landscapes, products,
animals and other out-of-domain photographs may produce poor results; this is
not a general-purpose photographic scene vectorizer.

The port follows `dataset_utils.GeneralRawDataLoader` and
`test_photograph_to_line.sample`:

- Preserve RGB, pad the right/bottom black to square, and resize to 256×256.
  Numeric antialiased bilinear resizing matches Pillow's uint8 two-pass behavior.
- Use the RGB photo checkpoint with 3-channel photo and 1-channel canvas tensors.
  Local out-of-bounds crops still extrapolate **white**, as in the original graph;
  this is distinct from the black padding of the source image.
- Run one fixed-length pass of 100 steps by default; reset recurrent state every
  48 steps. There is no clean-line ink-coverage reseeding or pen-up early stopping.
- The rounds control is fixed at one; steps can be adjusted up to 500, though
  extra steps do not guarantee a better sketch. Reset clears the generated canvas.

The reusable runtime is `src/model.js`; `src/preprocess.js` implements the
model's crop/resize and cursor selection, while `src/raster.js` implements
quadratic stroke rasterization and canvas pasting.

## Model inputs

All image tensors are NHWC and have spatial size `128 x 128`.

Both photo and canvas tensors use **dark strokes on a white background**:
raw 0 = stroke / 1 = background, normalized -1 = stroke / +1 = background.
The host's accumulating ink canvas uses the opposite polarity and is inverted
before inference. Full images use AREA resizing; photo crops extrapolate white.

| Input | Shape | Values |
| --- | --- | --- |
| `step_patch_photo` | `[1, 128, 128, C]` | normalized to `[-1, 1]` |
| `step_patch_canvas` | `[1, 128, 128, 1]` | normalized to `[-1, 1]` |
| `step_entire_photo` | `[1, 128, 128, C]` | raw `[0, 1]` |
| `step_entire_canvas` | `[1, 128, 128, 1]` | raw `[0, 1]` |
| `step_cursor` | `[1, 1, 2]` | normalized `(x, y)` |
| `step_image_size` | `[]` | original image size, integer |
| `step_window_size` | `[1, 1, 1]` | current window in original-image pixels |
| `step_prev_width` | `[1, 1, 1]` | previous stroke width in patch coordinates |
| `step_state_in` | `[1, 1024]` | HyperLSTM state |

`C` is 1 for clean line vectorization and 3 for the rough-sketch and
photograph models.

## Model outputs

| Output | Shape | Meaning |
| --- | --- | --- |
| `other_params` | `[1, 6]` | control point, endpoint, width, and scale parameters |
| `pen_ras` | `[1, 2]` | pen-state probabilities |
| `pen_state_soft` | `[1, 1]` | differentiable pen-state value |
| `state_out` | `[1, 1024]` | next HyperLSTM state |

The browser crops the patch from the full image/canvas, resizes it to 128×128,
runs the model, renders the returned quadratic stroke, updates the cursor and
window state, and repeats until EOS or the configured step limit. The clean
evaluation checkpoint is sampled with a fresh recurrent state every 48 steps,
so the browser loop mirrors that `state_dependent=False` cadence while keeping
the cursor/window/width trajectory continuous. After a round ends, the demo
selects another grid cell containing undrawn source pixels and starts a fresh
recurrent state, matching the original vectorization sampler's multi-round
behavior.

## SVG output

Each pen-down model step is stored as its quadratic control/end parameters,
starting cursor, patch window, image size, and previous/current width. The
preview and **Download SVG** button join consecutive pen-down segments into
multi-segment quadratic Bézier paths (`M … Q … Q …`), with no fill, a uniform `stroke-width="3.5"`,
and round caps and joins. The SVG has a transparent background and scales with
its `viewBox`; it does not embed the source image.

Coordinates follow `tools/svg_conversion.py`: start at the recorded cursor in
image coordinates, add the predicted column/row offsets times half the window
size for the endpoint, and interpolate each control coordinate using its model
parameter. Endpoints are not quantized to raster samples or clamped to the image.
This follows Python's `cluster` path structure and fixed width. Pen-up steps
and new rounds explicitly start new paths. A mismatch between consecutive
endpoints at the export's three-decimal precision also starts a new path (for
example, after cursor clamping). Older records without boundary metadata use
endpoint continuity alone. Joining preserves every quadratic segment and control
point; it does not snap gaps, reorder strokes, or merge overlaps.
The UI's stroke counter still counts model segments, not joined SVG paths.

After a run, **Join distance** previews more aggressive joining from 0 to 20
image pixels in 0.5-pixel steps. Zero restores the exact-joining baseline above.
Positive values match free endpoints across separate paths, including pen-up
and round boundaries and paths drawn in another order or direction. The closest
eligible pairs are joined first; each endpoint is used once, existing closed
contours are left alone, and new cycles are not forced. Paths may be reversed.
**Join at midpoint** is enabled by default: each matched endpoint pair is
replaced by its arithmetic midpoint, shared by the adjoining segments. Existing
quadratic control points remain fixed, so the curves change locally as their
endpoints move. Uncheck it to retain the original endpoints and bridge gaps with
straight `L` segments. Neither mode refits curves or consolidates overlapping ink.

The slider recomputes from the original stroke records, so lowering it or using
**Reset joining** reverses the transformation. Each generated preview path gets
a stable contrasting color so adjoining paths are easy to distinguish while
the controls are adjusted. **Overlay original in orange** compares against the
baseline, and the preview reports baseline/result path counts and additional
joins. The download uses the current distance and midpoint setting, with no
overlay or background rectangle; exported paths remain black. These controls are available after
inference completes; changing distance does not rerun or modify model inference.
Distance alone cannot infer intended contours; inspect larger thresholds for
connections between distinct details, especially around junctions.

**Remove redundant strokes** is enabled by default and runs before joining.
Its 1–12 pixel **Redundancy tolerance** slider starts at a conservative 3
pixels. The pass samples each quadratic centerline and removes a shorter
segment only when most of it is already covered by an aligned segment from a
different continuous path. It skips same-path turns and sharp curves, then
recomputes from the untouched model records whenever the checkbox or slider
changes. Larger tolerances allow more aggressive pruning and may remove small
details. The download uses the same survivor set as the preview.

**Split intersecting paths** is enabled by default as the final topology pass.
After joining and pruning, it detects interior crossings between sampled
quadratic segments, including non-adjacent segments of one path, and splits
the second participating path at each crossing. This preserves the centerline
while making the crossing an explicit pair of path endpoints. Endpoint
contacts and collinear overlaps are ignored. The optional redraw then fits
each resulting piece independently.

**Redraw with fitted curves** is an optional final pass after joining and
intersection splitting. It samples each resulting path and uses the
MIT-licensed [`fit-curve`](https://github.com/soswow/fit-curve)
implementation of Schneider's curve-fitting algorithm to produce cubic Bézier
segments. The **Fidelity / Smoothness** tolerance is expressed as squared image
pixels: lower values retain more of the input geometry and usually produce more
segments; higher values smooth irregularities and usually produce fewer. This
single tolerance is preferable to separate fidelity and smoothing controls
because it is the actual fitting error budget. The default is opt-in, with a
1 px² tolerance when enabled. The original overlay remains available, and the
download uses the current redraw setting. Redraw never changes the model's
raster feedback or the source stroke records.

Predicted widths and within-segment taper are intentionally discarded for SVG
display and export. Consequently these SVGs differ visually from the model's
variable-width raster output and the paper's results; disclose this difference
when using the SVGs for qualitative comparisons. Older exports used 100 filled
circle subpaths per segment; new exports contain only the quadratic centerline.

The model still needs the accumulated raster canvas at every inference step as
part of its input. That feedback remains in a numeric buffer. The output preview
is SVG, so no output canvas image is needed or embedded in the downloaded file.
The inference rasterizer still uses the predicted widths and sampled circles.

Future consolidation research is recorded in
[stroke-consolidation-notes.md](stroke-consolidation-notes.md).

To reproduce the joining review of the saved single-segment example, run
`node scripts/review-svg-joining.mjs` from this directory. It preserves
`example-result.svg`, writes `example-result-joined.svg`, and generates
`outputs/svg-joining/review.html` and `metrics.json`. Open the review through
the local Vite server at `/outputs/svg-joining/review.html` for original/joined
views, path coloring, and browser-rendered difference measurements at 1× and 4×.
This review importer accepts only this demo's uniform-width, single-M/Q export.

The demo defaults to the original clean sampler's 10 rounds of up to 500 steps,
with a round ending after 12 consecutive pen-up steps. Cursor selection uses
the original nonzero-ink coverage, minimum-coverage escape rule, and random
positions within grid cells. Blank inputs terminate without inference.

Stroke parameters are in **(row, column)** order. Every stroke starts at
`(0.5, 0.5)` in its patch, not `(0, 0)`. The numeric renderer matches Python's
100 OpenCV filled-circle stamps, integer coordinate/radius rounding and 2:1
bilinear downsample. It avoids allocating browser canvases for each stroke.

## Reproducible parity checks

`npm test` includes independently generated OpenCV stroke hashes and tests for
thin-line downsampling, crop boundaries, and cursor selection.

For model and full-loop validation, from the repository root:

```powershell
conda run -n virtual-sketching-tf1 python browser_port/scripts/reference-parity.py
cd browser_port
npm run test:parity
```

The reference generator uses the original TF1 model/checkpoint, `utils.draw`,
`DiffPastingV3`, and Python sampler. The checker compares original TF1 to the
frozen graph and ONNX Runtime Web WASM (running under Node), tests JavaScript
preprocessing and compositing against those references, replays the Python
trajectory, then runs the production JavaScript sampling loop autonomously.
Generated tensors, metrics and images live in ignored `outputs/parity`.

Model-output tolerance is 1e-4; complete preprocessing-plus-inference tolerance
is 1e-3. The recorded trajectory's average canvas error must be below .002.
The autonomous run's raw pixel coverage and precision must be within five
percentage points of the Python baseline. Raw pixel recall is sensitive to
line thickness and antialiasing, so it is compared to Python rather than 100%.
Full autonomous runs are not bitwise identical: random cursor starts and
floating-point differences can alter later strokes in this feedback loop.

Photo references and verification (from the repository root):

```powershell
conda run -n virtual-sketching-tf1 python browser_port/scripts/reference-photo-parity.py
cd browser_port
npm run test:photo-parity
```

This independently checks four Pillow RGB preprocessing cases (including
non-square, upsampled and downsampled images), three original TF1/frozen/ONNX
step cases, Python trajectory replay, and a 100-step production photo run.
Artifacts are in ignored `outputs/photo-parity`. The single-step model tolerance
is 1e-4; Pillow pixel comparisons are exact. Complete sequences can diverge
through rounding and recurrent raster feedback, even with the same starting
cursor. The end-to-end photo canvas MAE check is a smoke test, not a perceptual
quality benchmark. The original clean-line parity suite remains separate.

## Current scope

`src/modes.js` defines all three checkpoint contracts.
General photo-to-line conversion
would need a differently trained model or a separate contour-extraction pipeline.

## Export

Run `export_onnx_step.py` in the repository's TensorFlow 1.x environment after
downloading a trained checkpoint. It writes a frozen GraphDef. Convert that
GraphDef to ONNX in a separate modern environment with the command printed by
the exporter, then validate the ONNX model with ONNX Runtime before using
`onnxruntime-web`.

For the photo checkpoint, run these from the repository root (the weights must
already be in `outputs/snapshot/pretrain_faces`):

```powershell
conda run -n virtual-sketching-tf1 python export_onnx_step.py --model-name pretrain_faces --infer-dataset faces --output-graphdef outputs/onnx/virtual_sketching_faces_step.pb
conda run -n virtual-sketching-onnx python -m tf2onnx.convert --graphdef outputs/onnx/virtual_sketching_faces_step.pb --inputs step_patch_photo:0,step_patch_canvas:0,step_entire_photo:0,step_entire_canvas:0,step_cursor:0,step_image_size:0,step_window_size:0,step_prev_width:0,step_state_in:0 --outputs other_params:0,pen_ras:0,pen_state_soft:0,state_out:0 --opset 18 --output outputs/onnx/virtual_sketching_faces_step.onnx
cd browser_port
npm run prepare-model
```

The exporter selects the normal/photo hyperparameter defaults for `faces` and
restores the original checkpoint without retraining or changing its weights.

For rough sketches, use the matching hyperparameter defaults and checkpoint:

```powershell
conda run -n virtual-sketching-tf1 python export_onnx_step.py --model-name pretrain_rough_sketches --infer-dataset rough_sketches --output-graphdef outputs/onnx/virtual_sketching_rough_step.pb
conda run -n virtual-sketching-onnx python -m tf2onnx.convert --graphdef outputs/onnx/virtual_sketching_rough_step.pb --inputs step_patch_photo:0,step_patch_canvas:0,step_entire_photo:0,step_entire_canvas:0,step_cursor:0,step_image_size:0,step_window_size:0,step_prev_width:0,step_state_in:0 --outputs other_params:0,pen_ras:0,pen_state_soft:0,state_out:0 --opset 18 --output outputs/onnx/virtual_sketching_rough_step.onnx
conda run -n virtual-sketching-tf1 python browser_port/scripts/reference-photo-parity.py --rough
cd browser_port
npm run prepare-model
npm run test:rough-parity
```

The shared RGB parity scripts accept `--rough` to independently compare against
the original rough-sketch model/sampler. They verify exact RGB padding, cursor
movement with captured Python RNG choices (including retry exhaustion), model
outputs, full Python trajectory replay, and a ten-round browser-runtime run.
Artifacts go to `outputs/rough-parity`. The canvas MAE check is only a smoke
test, not a perceptual-quality guarantee; full random runs are not identical.
