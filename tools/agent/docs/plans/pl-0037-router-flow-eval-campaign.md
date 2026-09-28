# PL-0037: Router Flow Evaluation Campaign

## Goal

Produce reproducible evidence for benchmark-specific recipes exposed through
the public `vllm-sr/auto` model name. The campaign measures whether routing and
multi-model orchestration improve quality or efficiency over declared
single-model baselines.

## Scope

- Evaluate HLE text, SWE-Bench Pro, SciCode, and Terminal-Bench 2.1.
- Use one saved recipe per benchmark and disclose every backend model in it.
- Keep two result tracks distinct:
  - a same-model diagnostic track for measuring router-side uplift;
  - a mixed-model track for measuring the best practical routed system.
- Prefer EvalScope adapters. Add a minimal adapter only when the benchmark
  cannot be reproduced through EvalScope.
- Store commands, configuration, raw results, summaries, environment metadata,
  and scorecards together under the benchmark result directory.

## Non-Goals

- Treating prompt-level smoke tests as benchmark results.
- Comparing scores produced with different benchmark versions or judge rules.
- Publishing provider credentials, private infrastructure details, or local
  execution logs.
- Using one global recipe for workloads with materially different tool and
  verification requirements.

## Exit Criteria

Each benchmark has a reviewed result set containing:

- the exact router recipe and model pool;
- the dataset and harness version;
- a reproducible command and environment description;
- raw outputs and a machine-readable summary;
- comparable, cited baselines;
- analysis of quality, cost, latency, failures, and limitations;
- generated scorecard data and images.

Results that do not meet those requirements remain internal engineering data.

## Task List

- [x] `EVAL-01` Define the artifact contract and benchmark-specific recipe
  layout.
- [x] `EVAL-02` Establish HLE diagnostic and mixed-model recipes.
- [ ] `EVAL-03` Freeze and review the final HLE text result set.
- [ ] `EVAL-04` Run and review SWE-Bench Pro.
- [ ] `EVAL-05` Run and review SciCode.
- [ ] `EVAL-06` Run and review Terminal-Bench 2.1.
- [ ] `EVAL-07` Generate aligned scorecards and final cross-benchmark analysis.

## Next Action

Finish the HLE artifact review: verify the model-pool disclosure, dataset and
judge alignment, commands, raw outputs, and source metadata before accepting a
score. Then apply the same artifact contract to SWE-Bench Pro.

## Operating Rules

- A score without its recipe, raw output, and command is not publishable.
- Keep benchmark receipts beside result artifacts, not in this plan.
- Record concurrency, timeout, provider, sandbox, and judge settings in the run
  manifest.
- Preserve failed runs only when they explain a reproducible harness or routing
  failure; otherwise keep them out of the curated result set.
- Treat third-party numbers as claims until the benchmark settings are proven
  equivalent.

## Related Docs

- [Router Flow real-eval guide](../../../../bench/router_flow/real_eval/README.md)
- [Signal and decision evaluation plan](../../../../bench/router_flow/SIGNAL_DECISION_EVAL_PLAN.md)
