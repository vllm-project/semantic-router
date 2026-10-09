# System One Auto — editable figure sources

These figures use the frozen public evidence accompanying the article. The
charts are data-driven Matplotlib output; the cover and cascade call-flow are
repository-native SVG. The call-flow does not depict neural model internals.

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
embedded-font PDF and 300-DPI PNG; the SVG cover and call-flow export vector PDF
at their declared canvas size. The call-flow raster is PNG; the cover raster
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
- `hero`: the article's brand cover.

`generation-receipt.json` records file hashes, image dimensions and automated
text-boundary checks. Visual review remains necessary. The article evidence
archive includes the unchanged upstream scorer replay and underlying safe
observations; these figure sources do not independently validate a serving
build. See the article for quality, tail-latency and capacity limitations.
