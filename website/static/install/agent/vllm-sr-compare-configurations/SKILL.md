---
name: vllm-sr-compare-configurations
description: Compare two compatible vLLM Semantic Router configurations using paired sr-bench runs. Use when a user wants an A/B quality, latency, cost or reliability verdict with frozen cases, run IDs and uncertainty.
---

# Compare Router configurations

Start with two durable sr-bench run IDs and the objective fixed before the
candidate was evaluated. The [benchmark skill](https://vllm-sr.ai/install/agent/vllm-sr-benchmark/SKILL.md)
freezes and runs the evidence; this skill interprets sr-bench's own
[comparison contract](https://vllm-sr.ai/docs/benchmarking/sr-bench/iterate#replay-and-compare)
and [result definitions](https://vllm-sr.ai/docs/benchmarking/sr-bench/results#paired-comparisons).
Do not compute a parallel verdict from unrelated summary averages.

## 1. Verify the pair before judging it

Read each frozen manifest, report and run status. Check exact case IDs and
input fingerprints, benchmark/profile/seed/split, final-answer grader and
judge versions, target capabilities, effective sampling and request protocol,
limits, prices and cost policy. Target overrides beat run defaults. Both runs
must be live for measured quality, cost or latency claims; preview and replay
can diagnose routes or estimate eligible reuse but cannot prove live savings.

For each MoM run, verify its frozen target `config_hash` matched the Router's
active hash in its saved receipts. The baseline and candidate hashes may
differ by design; drift *within* a run or a hash that does not identify the
claimed configuration invalidates attribution. Record model artifact
revisions, plugin and Learning/session state where they affect outcomes. A
config hash alone does not freeze evolving Learning state.

Use the server's read-only compatibility discovery, including pagination:

```bash
vllm-sr benchmark show BASELINE_ID
vllm-sr benchmark show CANDIDATE_ID
vllm-sr benchmark comparison-options BASELINE_ID --limit 10
```

An empty page with `has_more: true`, or `scan_limited: true`, is not a final
negative result. Follow the cursor or report incomplete discovery. The
service rechecks compatibility on submission. A failed run can be compared
only if every planned cell has an explicit terminal outcome; failures count
as incorrect, while missing or ungraded cells block comparison. Do not drop
failed cases or regrade against a new answer key to force comparability.

## 2. Use the paired service result

```bash
vllm-sr benchmark compare BASELINE_ID CANDIDATE_ID
vllm-sr benchmark report BASELINE_ID --output baseline-report.json
vllm-sr benchmark report CANDIDATE_ID --output candidate-report.json
```

Keep the comparison response and both reports. Reconcile planned versus
paired cases and failures, skipped/ungraded cells, and coverage by benchmark
and stratum. Compare final-channel quality and the service's conservative
paired interval; latency/TTFT, reliability, routing/decision distribution,
unknown/fallbacks and guard outcomes where recorded; four-bucket token usage;
total cost including judges/simulators and subject cost separately. Unknown
usage or prices do not become zero. `capability_only` cannot support a
savings verdict. The strongest observed single is selected over the same
aggregate, not separately for each question.

## 3. State the prespecified decision

Map the service result to the objective and guard floors recorded before the
run. State **pass** only when the paired evidence, coverage and uncertainty
meet every required margin; **fail** when a prespecified floor is clearly
violated; otherwise **insufficient evidence**. A small development sample or
zero observed quality difference does not establish equivalence. A
non-inferiority claim needs a prespecified margin and an independent holdout
confidence interval. A retest or previously inspected task is not an unseen
holdout merely because its seed or split label changed.

Name confounders: changed backend model or grader, unequal output budget,
incomplete price accounting, different request protocol, runtime hash drift,
Learning or telemetry state, cache warming, failed deliveries, and exposure of
holdout cases. Preserve raw result and call links next to summary numbers.
Do not automatically promote or roll back a config. Hand a promising
development result to the existing
[recipe-lab workflow](https://github.com/vllm-project/semantic-router/issues/5497)
or [routing calibration](https://github.com/vllm-project/semantic-router/blob/main/tools/agent/skills/routing-calibration/SKILL.md)
with the frozen cases, guard floors and holdout still separate.

Use the [paired evidence example](https://vllm-sr.ai/install/agent/vllm-sr-compare-configurations/references/paired-evidence.md) for an
anonymized compatible pair and an inconclusive pair. It illustrates report
classification, not measured project performance.
