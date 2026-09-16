# Grounding-Aware Fusion: Filter-Policy Findings

This document records a June 2026 exploratory evaluation of hard filtering in
the Fusion looper. It is historical, directional evidence rather than a current
release benchmark: the run did not capture immutable source, dataset, or model
revisions, and its sample was small.

## Evaluation identity

| Dimension | Value recorded by the run |
| --- | --- |
| Dataset | DRACO, Medicine and Law domains |
| Sample size | 12 items, with 4 to 9 contested items depending on threshold |
| Panel | `qwen3:8b`, `llama3.1:8b`, `gemma3:12b` |
| Fusion judge | `qwen3:14b` |
| Rubric grader | `qwen3:14b` |
| Runtime | Local Ollama on Apple Silicon, through the no-thinking proxy |
| Grounding mode | Cross-model `panel` NLI |
| Policy | Hard `filter`, with `min_keep: 1` |
| Thresholds | `min_score` 0.34, 0.55, and 0.60 |
| Sampling | `temperature: 0`; panel regenerated independently for each arm |

DRACO grades an answer against weighted positive and negative criteria. The
negative criteria can carry large penalties for incorrect, unsafe, or poorly
sourced claims. DRACO does not provide source passages, so this evaluation did
not exercise `context` or `hybrid` grounding.

## Method

The evaluation considered two levels:

- **Intrinsic:** grade each available panel response and correlate its rubric
  quality with the panel-NLI grounding score.
- **Final answer:** compare plain Fusion with grounding-aware Fusion and compute
  paired bootstrap confidence intervals for the normalized DRACO score.

The contested subset contains items where the filter dropped at least one panel
response. It is the subset on which hard filtering can directly change the
synthesis input.

The scorer used for this run retained the complete three-class NLI probability
distribution and evaluated sentence-level pairs rather than truncating two long
answers into one model input. Under those conditions, observed scores ranged
from 0.52 to 0.69 and the Level-1 Spearman correlation with panel rubric quality
was +0.21. This indicates some discrimination, but it does not by itself show
that the score is useful as a synthesis weight.

## Results

Delta is grounding-on minus grounding-off for final-answer normalized DRACO
score. A negative value favors the plain-fusion arm. `ns` means the paired
bootstrap confidence interval included zero; `sig` means it excluded zero.

| `min_score` | Contested items | Overall delta | Contested delta | Contested negative-penalty delta |
| ---: | ---: | ---: | ---: | ---: |
| 0.34 | 0 / 12 | -0.035 (`ns`) | Not applicable | Not applicable |
| 0.55 | 4 / 11 | -0.058 (`ns`) | **-0.113 (`sig`)** | -2.50 (`ns`) |
| 0.60 | 9 / 12 | -0.090 (`ns`) | **-0.132 (`sig`)** | +4.44 (`ns`) |

At 0.60, eight of nine contested items scored worse with filtering. Across that
threshold's panel responses, `gemma3:12b` was dropped six times, `qwen3:8b` five
times, and `llama3.1:8b` four times. The negative-criteria penalty did not show a
consistent significant improvement.

## Interpretation

Within this run, more aggressive hard filtering was associated with lower
answer quality on contested factual questions. Cross-model agreement is not the
same as factual correctness: several models can share an error while a
dissenting response contains the useful minority view. A policy that deletes
the least consistent response can therefore remove evidence that the synthesis
judge needs.

The evidence supports keeping `filter` opt-in. The router currently defaults to
`weight`, which preserves every panel response. These results do not establish
that `weight` improves on plain fusion; they only show that the evaluated hard
filtering setup was harmful on its contested subset.

## Limitations

- The run used 12 items, and only 4 to 9 items were contested. Overall deltas
  were not statistically significant.
- Each arm generated a new panel, so paired deltas include response-sampling
  variation even with a zero temperature setting.
- The 8B to 12B local panel may not represent larger or more diverse model
  fleets.
- Only panel grounding was evaluated. Context-grounded factuality requires a
  dataset with source documents.
- Immutable source, dataset, and model revisions were not recorded. The exact
  magnitudes cannot be treated as a reproducible baseline.
- The run did not compare `weight`, `annotate`, or a random-weight placebo
  against plain fusion.

## Evidence still needed

A policy decision for soft grounding requires the cached-panel design described
in [README.md](README.md): generate one panel per item, reuse its exact bytes for
all arms, and compare judge-only, plain-fusion, grounded-weight, and seeded
random-weight arms. A larger sample, pinned revisions, model digests, and a
context-grounded dataset would make the result suitable for release-level
claims.
