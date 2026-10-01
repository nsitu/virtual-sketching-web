# Future work: stroke consolidation

Deferred while the SVG exporter is simplified to one quadratic centerline per
model segment. The later goal is to replace redundant overlapping strokes and
compatible short segments with fewer meaningful curves at low visual error.

## References

- [StrokeStrip: Joint Parameterization and Fitting of Stroke Clusters (2021)](https://github.com/davepagurek/StrokeStrip)
  fits an intended curve to an already identified cluster of overlapping strokes.
  Its `.scap` input contains sampled polylines and cluster IDs. It can also fit
  widths, although widths are optional for our use case. It does not supply the
  automatic clustering needed for arbitrary model output.
- [StripMaker: Perception-driven Learned Vector Sketch Consolidation (2023)](https://www.cs.ubc.ca/labs/imager/tr/2023/stripmaker/)
  identifies stroke groups using local and contextual classifiers, then fits
  consolidated curves with a modified StrokeStrip implementation.
  [Source](https://github.com/squidrice21/StripMaker). This is the closer match
  for automatically consolidating model-generated strokes, though its behavior
  on these strokes needs evaluation against its artist-sketch use case.
- [SceneSketch / CLIPascene](https://github.com/yael-vinker/SceneSketch)
  is related work on fidelity and simplicity: it optimizes stroke positions and
  learns which strokes to remove. Its abstraction objective differs from merging
  redundant strokes while retaining the current drawing.

The published StrokeStrip and StripMaker implementations depend on native C++
and an installed, licensed Gurobi optimizer. They are research references and
possible offline benchmarks, not drop-in ONNX Runtime Web components.

## When revisiting

Work from `BrowserVectorizer.strokes` before SVG serialization, where quadratic
parameters, cursor positions, window sizes, and model widths are available.
Preserve pen-up and round boundaries explicitly if using drawing continuity or
stroke order: the current stroke array stores only pen-down segments.

Compare endpoint joining, overlap consolidation, and curve refitting while
protecting corners, crossings, nearby distinct contours, and small details.
Measure actual segments/control points and file size as well as path count;
combining subpaths into one SVG element alone does not simplify geometry.
Validate visual error at normal and enlarged scales before choosing tolerances.

Keep the model's variable-width raster feedback independent of any display or
export simplification. Joining existing continuous segments into multi-segment
paths (Python's `cluster` mode) is also distinct from consolidating overlaps.
