# PL-0043: Fusion Adaptive Execution

## Goal

Reduce Fusion cost and latency through explicit, evidence-backed execution
controls while preserving panel quorum, recipe-owned budgets, fallback,
provenance, and aggregate accounting.

## Scope

- Land explicit `separate`, `one_call`, and `none` analysis execution modes.
- Evaluate direct, reduced-call, and full Fusion under matched budgets.
- Add repeated same-model samples after shared panel and budget contracts land.
- Graduate a deterministic adaptive gate only after calibration evidence passes.
- Defer public mode-trace transport and final-stage protocol behavior until
  issue #3378 settles transport ownership.

## Non-Goals

- Changing the default mode before matched-budget quality evidence exists.
- Weakening recipe-owned quorum, call, token, cost, timeout, or fallback policy
  from request input.
- Best-of-N selection, recursive Loopers, tree search, arbitrary tool
  trajectories, or grounding-default changes.
- Duplicating the evaluation plane owned by issue #2858.

## Exit Criteria

- [ ] Every execution control is explicit, validated, bounded, and covered by
  deterministic accounting and failure tests.
- [ ] Matched-budget evaluation reports quality, reliability, latency, token
  use, cost, and failure behavior for every candidate default.
- [ ] Repeated samples preserve distinct provenance and aggregate accounting.
- [ ] Any adaptive gate has deterministic features, calibration evidence, and
  a documented fallback.
- [ ] Public mode-trace transport and final protocol behavior are explicit and
  covered after issue #3378 settles transport ownership.

## Task List

- [ ] `MODE-01` Land recipe-owned `analysis_mode` with a `separate`
  compatibility default. Prerequisite: none. Budget: configured panel attempts
  plus two judge calls for `separate`, or one terminal judge call for
  `one_call`/`none`. Failure contract: separate-analysis failure falls back;
  terminal synthesis failure remains terminal. Evaluation gate: deterministic
  config, prompt, call-count, token, trace, tool, failure, and Looper E2E tests.
- [ ] `EVAL-02` Compare direct, reduced-call, and full Fusion using issue
  #2858's byte-identical cached panels and matched budgets. Prerequisite:
  `MODE-01` and the #2858 benchmark contract. Budget: both fixed-call and
  fixed-token envelopes. Failure contract: report failures by stage without
  dropping unsuccessful cases. Evaluation gate: quality, reliability, latency,
  tokens, cost, and failure results reviewed before any default changes.
- [ ] `SAMPLE-03` Add bounded repeated same-model sampling. Prerequisite:
  issues #2856 and #2861. Budget: explicit aggregate call/token/cost/time limits.
  Failure contract: preserve usable-response quorum and per-attempt evidence.
  Evaluation gate: distinct sample provenance and aggregate accounting tests
  plus matched-budget results.
- [ ] `GATE-04` Add adaptive mode selection. Prerequisite: `EVAL-02` and stable
  calibration data. Budget: never exceed the selected recipe's declared
  limits. Failure contract: deterministically fall back to `separate` or a
  configured conservative mode. Evaluation gate: held-out calibration,
  deterministic feature tests, and regression thresholds.
- [ ] `PROTO-05` Settle public mode-trace transport and final-stage streaming
  and protocol behavior. Prerequisite: issue #3378. Budget: no hidden extra
  model calls. Failure contract: preserve the OpenAI-compatible terminal error
  and tool-call contracts. Evaluation gate: protocol and streaming E2E
  coverage.

## Next Action

Complete `MODE-01`, then run `EVAL-02` through the shared #2858 evaluation
contract before proposing any default or adaptive-policy change.

## Operating Rules

- Keep each task independently reviewable and safety-complete.
- Keep `analysis_mode`, quorum, aggregate budgets, and fallback recipe-owned.
- Keep `include_analysis` limited to trace visibility.
- Keep mode trace internal until issue #3378 owns its public wire transport.
- Treat deterministic fixtures as contract evidence, not quality evidence.
- Do not begin a blocked task until its named prerequisite is available.

## Related Docs

- [Issue #2865](https://github.com/vllm-project/semantic-router/issues/2865)
- [Parent hardening issue #2336](https://github.com/vllm-project/semantic-router/issues/2336)
- [Evaluation contract issue #2858](https://github.com/vllm-project/semantic-router/issues/2858)
- [Panel contract issue #2856](https://github.com/vllm-project/semantic-router/issues/2856)
- [Aggregate budget issue #2861](https://github.com/vllm-project/semantic-router/issues/2861)
- [Protocol ownership issue #3378](https://github.com/vllm-project/semantic-router/issues/3378)
- [Fusion guide](../../../../website/docs/tutorials/algorithm/looper/fusion.md)
