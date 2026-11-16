---
name: vllm-sr-compare-configurations
description: Compare compatible vLLM Semantic Router configurations against a single-model reference using paired sr-bench runs. Use when a user wants quality, latency, cost or reliability evidence with frozen cases, run IDs and uncertainty.
---

# Compare Router configurations

Start with a completed live single-model reference run and one live Router run
for each configuration. Fix the objective before evaluating the candidates.
sr-bench rejects a Router-only run as the comparison baseline. The [benchmark skill](https://vllm-sr.ai/install/agent/vllm-sr-benchmark/SKILL.md)
freezes and runs the evidence; this skill interprets sr-bench's own
[comparison contract](https://vllm-sr.ai/docs/benchmarking/sr-bench/iterate#replay-and-compare)
and [result definitions](https://vllm-sr.ai/docs/benchmarking/sr-bench/results#paired-comparisons).
Do not compute a parallel verdict from unrelated summary averages.

## 1. Verify the runs before judging them

Read each frozen manifest, report and run status. Check exact case IDs and
input fingerprints, benchmark/profile/seed/split, final-answer grader and
judge versions, target capabilities, effective sampling and request protocol,
limits, prices and cost policy. Target overrides beat run defaults. All runs
must be live for measured quality, cost or latency claims; preview and replay
can diagnose routes or estimate eligible reuse but cannot prove live savings.

For each Router run, verify its frozen target `config_hash` matched the Router's
active hash in its saved receipts. The configuration hashes may
differ by design; drift *within* a run or a hash that does not identify the
claimed configuration invalidates attribution. Record model artifact
revisions, plugin and Learning/session state where they affect outcomes. A
config hash alone does not freeze evolving Learning state.

Use the server's read-only compatibility discovery with the single-model run
as baseline, including pagination:

```bash
vllm-sr benchmark show SINGLE_ID
vllm-sr benchmark show CONFIG_A_ID
vllm-sr benchmark show CONFIG_B_ID
vllm-sr benchmark comparison-options SINGLE_ID --limit 10
```

An empty page with `has_more: true`, or `scan_limited: true`, is not a final
negative result. Follow the cursor or report incomplete discovery. The
service rechecks compatibility on submission. A failed run can be compared
only if every planned cell has an explicit terminal outcome; failures count
as incorrect, while missing or ungraded cells block comparison. Do not drop
failed cases or regrade against a new answer key to force comparability.

## 2. Use the paired service result

```bash
vllm-sr benchmark compare SINGLE_ID CONFIG_A_ID
vllm-sr benchmark compare SINGLE_ID CONFIG_B_ID
vllm-sr benchmark report SINGLE_ID --output single-report.json
vllm-sr benchmark report CONFIG_A_ID --output config-a-report.json
vllm-sr benchmark report CONFIG_B_ID --output config-b-report.json
```

Keep both comparison responses and all three reports. Each service result
compares one Router configuration with the same strongest single-model
reference; it is not a direct paired interval for A versus B. Reconcile planned versus
paired cases and failures, skipped/ungraded cells, and coverage by benchmark
and stratum. Compare final-channel quality and the service's conservative
paired interval; latency/TTFT, reliability, routing/decision distribution,
unknown/fallbacks and guard outcomes where recorded; four-bucket token usage;
total cost including judges/simulators and subject cost separately. Unknown
usage or prices do not become zero. `capability_only` cannot support a
savings verdict. The strongest observed single is selected over the same
aggregate, not separately for each question.

## 3. State the prespecified decision

Map the service results to the objective and guard floors recorded before the
run. State **pass** only when the paired evidence, coverage and uncertainty
meet every required margin; **fail** when a prespecified floor is clearly
violated; otherwise **insufficient evidence**. A small development sample or
zero observed quality difference does not establish equivalence. A
non-inferiority claim needs a prespecified margin and an independent holdout
confidence interval. A retest or previously inspected task is not an unseen
holdout merely because its seed or split label changed.

Do not infer that A beats B by subtracting their separate intervals against
the reference. A direct A/B verdict needs its own prespecified paired
analysis; otherwise state that the service comparisons do not settle it.

Name confounders: changed backend model or grader, unequal output budget,
incomplete price accounting, different request protocol, runtime hash drift,
Learning or telemetry state, cache warming, failed deliveries, and exposure of
holdout cases. Preserve raw result and call links next to summary numbers.
Do not automatically promote or roll back a config. Hand a promising
development result to the existing
[recipe-lab workflow](https://github.com/vllm-project/semantic-router/issues/5497)
or [routing calibration](https://github.com/vllm-project/semantic-router/blob/main/tools/agent/skills/routing-calibration/SKILL.md)
with the frozen cases, guard floors and holdout still separate.

Use the [paired evidence example](https://vllm-sr.ai/install/agent/vllm-sr-compare-configurations/references/paired-evidence.md) for
anonymized compatible runs and an inconclusive case. It illustrates report
classification, not measured project performance.
