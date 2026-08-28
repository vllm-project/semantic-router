---
title: MoM Evaluation
---

# MoM Evaluation

Evaluate a published Mixture-of-Models (MoM) as a complete system identity
under the versioned `vllm-sr/mom-evaluation/v1` contract. This complements
decision-level routing evaluation in `probes.yaml`.

## When to use

| Question | Use MoM evaluation |
| --- | --- |
| Does the complete MoM beat a qualified standalone model? | Yes |
| Did a routing decision select the expected algorithm? | No — use `probes.yaml` / `vllm-sr eval` |
| Did a code path regress allocations? | No — use `perf/` |

## Quick start

```bash
make mom-eval-validate
make mom-eval-smoke MOM_EVAL_ENTRYPOINT=vllm-sr/mom-v1-blend
```

Serve the entrypoint first when running against live backends:

```bash
vllm-sr serve vllm-sr/mom-v1-blend
make mom-eval-rc MOM_EVAL_ENTRYPOINT=vllm-sr/mom-v1-blend
```

## Contract layers

1. **Core suite (mandatory)** — general quality, instruction following,
   robustness, safety baseline, operational metrics, regression
2. **Extension packs (additive)** — cost, latency, security, orchestration
3. **Decision-level probes (complementary)** — routing correctness without
   backend answer quality

Extension pack regressions never override a core-suite failure.

## Artifacts

Each run produces:

- `mom_eval_result.json` — machine-readable bundle
- `scorecard.json` / `scorecard.md` — publication artifacts
- `regression_report.json` — deltas vs previous release
- `failure_slices.json` — diagnostic breakdown

Historical scorecards: `config/evaluation/scorecards/`

## MoM V1 reference scorecards

| Entrypoint | Scorecard |
| --- | --- |
| `vllm-sr/mom-v1-blend` | [blend/1.0.0](../../../config/evaluation/scorecards/mom-v1/mom-v1-blend/1.0.0/scorecard.md) |
| `vllm-sr/mom-v1-lite` | [lite/1.0.0](../../../config/evaluation/scorecards/mom-v1/mom-v1-lite/1.0.0/scorecard.md) |
| `vllm-sr/mom-v1-flash` | [flash/1.0.0](../../../config/evaluation/scorecards/mom-v1/mom-v1-flash/1.0.0/scorecard.md) |
| `vllm-sr/mom-v1-ultra` | [ultra/1.0.0](../../../config/evaluation/scorecards/mom-v1/mom-v1-ultra/1.0.0/scorecard.md) |
| `vllm-sr/mom-v1-vault` | [vault/1.0.0](../../../config/evaluation/scorecards/mom-v1/mom-v1-vault/1.0.0/scorecard.md) |

## Maintainer workflow

See [bench/mom_eval/README.md](../../../bench/mom_eval/README.md) and the
maintainer skill at
[`tools/agent/skills/maintainer/mom-evaluation/SKILL.md`](../../../tools/agent/skills/maintainer/mom-evaluation/SKILL.md).
