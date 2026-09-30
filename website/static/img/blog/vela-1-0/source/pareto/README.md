# Omni complete-panel and selected-task figures

These are the original Matplotlib SVGs and complete observation data from the
[Nano revision `0496b39a`](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Nano/tree/0496b39a51c8199592e58cbff81c250f056bd94b)
and [Mini revision `f7fafd36`](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Mini/tree/f7fafd36abf49adf88b1b2ec0186c68b008eeb07).
Every input has a pinned URL and SHA-256 in `provenance.json`; no peer observation
is selected or dropped locally. Shared numeric data agrees across these pins.

All figures count whole-model parameters across text, image and audio:
**163,771,288 for Nano** and **1,361,475,288 for Mini**. Registry peer sizes are
reported totals and can be rounded. The reference registry snapshot is
17 September 2026; current Vela measurements and their applicability evidence
are retained separately.

The primary gallery contains four complete-panel figures: English-v2's 41 tasks
and the audio-only MAEB panel's 19 tasks, each with a Nano or Mini highlight and
its size-constrained comparison table. The plotted primary metric is
**Mean(TaskType)**; the complete data and ranks also retain **Mean(Task)**.
English uses default shared text for Nano and fixed official MTEB task
instructions for Mini. Audio uses default shared audio for both models.
Neither model is on either complete-panel frontier under either aggregation.
There is no combined text–image–audio ranking.

The separate selected-task gallery shows Nano on IMDb and NMSQA, and Mini on
Mridingham tonic and SIB-FLEURS spoken-topic classification. These task-level
results are not overall benchmark leadership. Original ArXiv and Vehicle
observations remain in the complete source data even though the old figures
have been retired upstream and removed from the article.

`export.mjs` exports the unmodified SVGs with Sharp 0.35.4 at 170 dpi: the four
complete-panel PNGs are 2924 × 1700 and the four selected-task PNGs are
2448 × 1496. Make DejaVu Sans regular and bold available to fontconfig, then run:

```sh
node export.mjs
python3 verify.py
```

`SHARP_MODULE` can specify an already installed Sharp package. The exporter does
not move points, rewrite labels or recompute chart layout. SVGs remain editable
vector source. `verify.py` verifies source hashes, independently recomputes both
aggregations from all task scores, checks global and size-constrained ranks and
gaps, and calculates all Pareto memberships with pairwise dominance and a
sorted sweep. It also checks all four figure metadata records and preserves
its results in `audit.json`.

The audit covers 719 retained task observations and 252 complete-panel
observations. Models without a known positive size remain eligible for global
rankings but are excluded from size-constrained ranks and scatter plots.
Training exposure is source-declared, not independently audited; an empty
training declaration does not prove zero exposure. Peer protocols differ.

See the pinned [methodology](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Nano/blob/0496b39a51c8199592e58cbff81c250f056bd94b/benchmarks/pareto-methodology.md)
for evaluation modes, source identities, aggregate definitions and limitations.
