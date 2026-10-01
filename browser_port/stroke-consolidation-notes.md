# Future work: stroke consolidation

The SVG exporter now joins consecutive quadratic centerlines into continuous
paths, preserving all original segments. The later goal is to replace redundant overlapping strokes and
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
The stroke array stores only pen-down segments; `startsNewPath` now records
pen-up and round boundaries. Legacy SVG files do not retain this metadata, so
joining those files can only infer continuity from consecutive endpoints.

Compare endpoint joining, overlap consolidation, and curve refitting while
protecting corners, crossings, nearby distinct contours, and small details.
Measure actual segments/control points and file size as well as path count;
combining subpaths into one SVG element alone does not simplify geometry.
Validate visual error at normal and enlarged scales before choosing tolerances.

Keep the model's variable-width raster feedback independent of any display or
export simplification. Joining existing continuous segments into multi-segment
paths (Python's `cluster` mode) is also distinct from consolidating overlaps.

## Joining assessment (2026-10-01)

`node scripts/review-svg-joining.mjs` reproduces a review using the saved
`example-result.svg`, with a separate `example-result-joined.svg` as output.
The input contains no pen-boundary metadata; its consecutive endpoint matches
yield 18 paths from 115 individual paths. All 115 quadratic commands and their
coordinates remain unchanged. Size falls from 17,584 to 5,967 bytes (66.1%).
New model runs also respect explicit pen-up and round boundaries.

In the browser comparison, the sum of absolute grayscale differences, divided
by the original total ink, was 2.390% at 640×640 and 0.984% at 2560×2560.
At native size, lost ink was 1.465% and added ink 0.925%; at 4×, they were
0.533% and 0.452%. These are rendering measurements, not geometric errors or
perceptual quality scores. Combining separately antialiased strokes into joined
paths changes caps/joins and compositing; pixel-identical output is not promised.
Visual inspection of the side-by-side review showed consistent overall contours.

The five longest paths contain 23, 23, 16, 14, and 10 segments: 86 of the 115
segments. Seven paths contain only one segment. Of the 97 joined vertices,
28 have a tangent direction change greater than 30 degrees, including 14 greater
than 60 degrees. Some represent intentional corners; these statistics are not
an automatic corner classification.

The next useful experiment is corner-preserving refitting of smooth sections
within those long paths, measured against this uniform-width joined SVG. Keep
short features, gaps, and junctions explicit. Try independently validated error
budgets before considering overlap consolidation; joining alone does not show
which overlaps are redundant.

## Optional redraw fitting

The browser now has an opt-in **Redraw with fitted curves** pass after joining.
It samples each joined quadratic path and uses the MIT-licensed `fit-curve`
package, a JavaScript implementation of Schneider's automatic digitized-curve
fitting algorithm, to emit cubic Bézier paths. One squared-pixel tolerance is
used as the user-facing **Fidelity / Smoothness** control: lower tolerance is a
tighter redraw with more detail, while higher tolerance permits more smoothing
and fewer cubic segments. A single error budget maps directly to the library's
`maxError` parameter and is easier to interpret than independent controls that
could disagree. The default is opt-in at 1 px².

The pass preserves each joined path's endpoints and leaves the model stroke
records and raster feedback untouched. It does not infer overlaps or corners;
the fitter can smooth a true corner if the tolerance is too high. Keep the orange
original overlay enabled while exploring thresholds and inspect small details.
Redraw is applied after nearby joining and the same settings are used for the
preview and SVG download.

## Configurable nearby joining

The browser now offers a 0–20 image-pixel **Join distance** slider with live
preview after inference. At zero it retains the exact-joining baseline. Positive
values greedily match the closest free endpoints of separate paths, regardless
of draw order or orientation. Paths can be reversed. With midpoint mode enabled
by default, each matched pair is replaced by its shared midpoint while retaining
the original quadratic controls. With midpoint mode disabled, the original ends
remain and a straight connector is inserted. Each endpoint is matched at most
once; closed contours are excluded, and new cycles are not forced. This is
endpoint joining, not refitting or overlap consolidation.

For the saved 640×640 example, measured path counts are:

| Distance (image pixels) | Paths | Added connectors |
| --- | --- | --- |
| 0 | 18 | 0 |
| 2 | 18 | 0 |
| 4 | 14 | 4 |
| 8 | 14 | 4 |
| 12 | 13 | 5 |
| 20 | 7 | 11 |

These counts are structural measurements, not fidelity ratings. Higher values
can connect separate details. Use the original overlay and inspect the result.
Lowering the slider recomputes from source strokes; it never accumulates edits.
The download uses the current threshold and midpoint setting but excludes the
orange overlay.

## Redundancy pruning assessment (2026-10-01)

Redundancy removal should operate on the model's individual centerline
segments before nearby joining and curve redraw. Joining first would hide the
identity of the candidate segment and could make a deleted segment look like a
new, unintended connector. The final order should be:

1. score and conservatively remove redundant source segments;
2. join the surviving segments using the existing endpoint and midpoint rules;
3. split interior intersections so crossings become explicit endpoints;
4. optionally redraw the resulting paths with the fitted-curve pass.

A one-off geometric scan of the saved `example-result.svg` used the uniform
3.5-pixel display width as a proximity scale. It found one convincing
cross-path near-duplicate: zero-based segments 7 and 114 lie within about 1.5
pixels on average and cover nearly the same short mark. The next closest pairs are
adjacent segments that turn back along an existing contour; their centerline
proximity does not make either one redundant. This means a global
"delete anything nearby" rule would remove corners, loops, or crossings.

The safer first implementation should use a candidate score rather than a
single distance threshold. For each segment, sample its centerline and look
for another segment (or already accepted path) that covers most samples within
an epsilon, has a compatible tangent direction, and is not merely sharing an
endpoint. Reject candidates at sharp turns, near junctions, or when removal
would create a new bridge across a pen-up or round boundary. Then validate a
greedy deletion against a rasterized union: keep the deletion only when the
lost ink stays below a small user-facing error budget. This directly measures
the visible result and protects small details better than centerline distance
alone.

The browser now exposes this as a **Remove redundant strokes** preview, checked
by default with a conservative 3-pixel tolerance slider. It reports the
removed-segment count and uses exactly the same survivor set for download. The
orange original overlay remains available for inspection. If the example
produces too few deletions under these safeguards, the next step should be
curve consolidation of parallel partial overlaps rather than a more aggressive
deletion rule.

## Intersection splitting (2026-10-01)

The browser now runs an enabled-by-default **Split intersecting paths** pass
after pruning and nearby joining, before optional curve redraw. It samples each
quadratic segment into a polyline, detects interior crossings between
non-adjacent segments or separate paths, and splits only the second
participating path at each crossing using the quadratic de Casteljau
construction. The centerline is unchanged, but the crossing becomes an
explicit endpoint and the preview reports how many intersections were split.
Endpoint contacts and collinear overlaps are intentionally ignored. The pass
is deterministic and can be unchecked to compare the original joined topology.
On the saved example, the default 3-pixel pruning removes one segment before
joining; the resulting 19 paths contain one detected interior crossing and
produce 20 explicit path pieces after splitting.
