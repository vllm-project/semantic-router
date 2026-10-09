# System One Auto — editable figure sources

These figures use the frozen public evidence accompanying the article. The
charts are data-driven Matplotlib output; the cover, cascade call-flow and
routing comparison are repository-native SVG. These system diagrams depict
request flow, not neural model internals. `diagram_assets.py` owns their
editable shapes, typography and connections; `generate_blog_assets.py` owns
the measured charts and shared export checks.

From the Semantic Router repository root:

```bash
python3 -m venv .venv-figures
.venv-figures/bin/pip install matplotlib==3.11.2 Pillow==11.3.0
.venv-figures/bin/python website/static/img/blog/system-one-auto/source/generate_blog_assets.py \
  --data website/static/img/blog/system-one-auto/source/figure-data.json \
  --logo website/static/img/vllm-sr-logo.white.png \
  --output /tmp/system-one-auto-figures \
  --chromium /path/to/chrome-headless-shell
```

The measured rendering used Chrome for Testing 153.0.8010.12. Chromium must have
its usual platform libraries installed. The generator is offline: it makes no
model or network requests. Data hash verification prevents accidental use of a
different experimental result. The numerical charts export editable SVG,
embedded-font PDF and 300-DPI PNG; the SVG diagrams export vector PDF
at their declared canvas size. The call-flow and comparison rasters are PNG; the cover raster
is a progressive JPEG at quality 98 with 4:4:4 chroma sampling, below 500 KiB.
The original cover SVG remains editable. DejaVu Sans is used throughout.

The banner preserves the original vLLM-SR logo's alpha geometry and uses an SVG
color filter to render it in pure white. The source logo PNG is unchanged. It
contains no KR Labs mark and makes no novel-algorithm claim.

Outputs:

- `quality`: direct Kai, Auto and direct Vega on the same 231 public tasks,
  with paired source-group bootstrap intervals for the quality differences.
- `latency`: warm mean and p95 together, retaining both passes separately.
- `paths`: request outcomes beside actual native API-call counts.
- `cascade`: one original Kai answer, an explicit gate, optional Vega.
- `ecosystem`: LLM routing and decision-model routing side by side. Messages
  produce generated text; System One questions produce typed answers.
- `hero`: the article's brand cover.

`generation-receipt.json` records file hashes, image dimensions and automated
text-boundary checks. The renderer checks real browser text bounds,
text-to-text overlap and declared text containers for the system diagrams.
`layout-review.json` records the final visual and PDF checks. Visual review
remains necessary: geometry checks do not validate every connector or model
claim. The article evidence
archive includes the unchanged upstream scorer replay and underlying safe
observations; these figure sources do not independently validate a serving
build. See the article for quality, tail-latency and capacity limitations.

## Structure and captions

The call-flow follows `src/semantic-router/pkg/systemone/execution.go`:
`invokeStage` passes the original native request to each model; `accept`
checks complete answers and configured probability thresholds. A failed Kai
gate advances to Vega, rather than running a second mandatory classifier.
The public result preserves the native answer schema. If no stage returns an
acceptable response, the request is unresolved. The figure's two-call limit
belongs to the illustrated Kai/Vega configuration.

The comparison illustrates the existing recipe/decision/algorithm boundary,
not a benchmark of every backend or provider. Kai and Vega are independent
Decision 2.0 checkpoints. Generic LLM sizes on the left are conceptual. The
article measures the Kai-to-Vega cascade; evaluating broader provider pools
is future work, while external native backend binding is already supported.

Suggested captions:

- **Cascade:** Kai answers the original bundle. A probability gate either
  returns that answer or sends the same input and questions to Vega. Both
  successful paths return the System One answer schema.
- **Comparison:** Model orchestration extends from generated language to
  structured decisions. The API contracts stay distinct: Chat/Responses
  produces text, while System One produces typed answers and probabilities.

The three measured chart families (`quality`, `latency`, `paths`) and
`figure-data.json` retain their original bytes in this visual revision.
