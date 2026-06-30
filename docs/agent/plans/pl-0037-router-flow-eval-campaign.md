# PL-0037 Router Flow Eval Campaign

## Goal

Produce reproducible benchmark evidence for vLLM Semantic Router's unified
VSR score identity: one public model API, `vllm-sr/auto`, with one
benchmark-specific auto recipe per benchmark.

The campaign should show router-side scaling value against GLM-5.2 native,
closed single-model baselines, and the Fugu/Fugu Ultra reference rows when
benchmark settings are aligned. The public API stays `vllm-sr/auto`, but the
model-pool policy is deliberately split into two tracks:

- GLM-only diagnostic track: every backend model ref maps to OpenRouter
  `z-ai/glm-5.2`. This proves whether loop routing, planning, verification,
  repair, and synthesis improve the same base model over its official native
  score. This track must beat GLM-5.2 native before it becomes public evidence.
- Competitive hybrid track: benchmark-specific recipes may combine GLM-5.2 with
  stronger closed models for high-risk escalation, verification, or final
  synthesis. This is the track for beating closed single-model baselines and
  approaching or beating Fugu/Fugu Ultra. It must disclose its model pool and
  must not be described as GLM-only.

The next campaign rows run in this order:

1. HLE text representative subset
2. SWE-Bench Pro
3. SciCode
4. Terminal-Bench 2.1

## Scope

- Run every campaign benchmark once through `vllm-sr/auto`, using that
  benchmark's saved auto recipe.
- Prefer EvalScope adapters. Use another harness or a minimal adapter only when
  EvalScope is unavailable, unstable, or cannot match the target benchmark
  setting.
- Allow each benchmark to have its own router config. The config may tune
  signals, decisions, `modelRefs`, `output_contract`, timeout, concurrency,
  fallback, planner, and algorithm parameters for that benchmark's task shape.
- Save each benchmark config with the result artifacts so every score can be
  reproduced.
- Use the official GLM-5.2 Hugging Face benchmark table as the native baseline
  source for HLE and SWE-Bench Pro. Do not rerun native GLM-5.2 for this
  campaign in the diagnostic track.
- Use the Fugu technical report as the strict comparison source when benchmark
  settings are aligned. Keep product-page and third-party leaderboard numbers
  separate unless the harness settings are proven aligned.
- Generate scorecards with the local `vllm-sr-scorecard-generator` skill.

## Model-Pool And Algorithm Fit

- ReMoM with one repeated model is useful for self-consistency, stochastic
  exploration, answer-contract repair, and light synthesis. It is weak against
  correlated knowledge gaps because every candidate shares the same model prior.
- Fusion with one repeated model is useful when candidate diversity comes from
  temperature, prompt framing, or local grounding. It is not a reliable way to
  have the same model judge away its own blind spots.
- Flow/workflows with one repeated model can still be valuable on agentic tasks
  when tool execution, tests, or sandbox output provide external feedback. It is
  much less convincing on pure closed-book reasoning without an external
  verifier.
- Confidence, ratings, or direct model-selection algorithms add little value
  when all candidates are aliases of the same model, unless they consume
  independent external signals.
- For HLE-style hard reasoning, a competitive recipe should usually use GLM-5.2
  for breadth/planning and a stronger closed model as verifier/finalizer or
  high-risk escalation.
- For SWE-Bench Pro, GLM-5.2-only role separation is more defensible because
  the harness can supply external tool/test feedback, but a competitive recipe
  should still evaluate a strong closed verifier/finalizer path.

## Non-Goals

- Do not treat one-prompt proxy scores as publishable benchmark rows.
- Do not commit provider API keys, private AMD host details, or private remote
  validation logs.
- Do not relabel public baseline numbers as locally reproduced.
- Do not require one global router config to perform well on every benchmark;
  benchmark-specific configs are allowed and must be tracked.

## Exit Criteria

- HLE text representative subset, SWE-Bench Pro, SciCode, and Terminal-Bench
  2.1 each have one `vllm-sr/auto` VSR result from a benchmark-specific recipe.
- Each benchmark row has:
  - the exact router config used for that benchmark;
  - raw EvalScope or benchmark-native result files;
  - summary metrics;
  - reproducible commands;
  - scorecard JSON, SVG, and PNG;
  - result analysis against single-model baselines, Fugu, and Fugu Ultra.
- Any diagnostic row that does not beat official GLM-5.2 native has a failure
  analysis, attempted optimizations, invalidated optimizations, and next action.
- Any competitive row that does not beat closed single-model baselines and
  Fugu/Fugu Ultra has the same analysis trail.
- The final scorecards include VSR 1.0, GLM-5.2 native reference, Fugu, Fugu
  Ultra, and single-model baselines with traceable sources.

## Task List

- [x] Preserve existing GPQA-Diamond and LiveCodeBench historical artifacts and
  keep weak Kimi-only LiveCodeBench runs internal.
- [x] Validate EvalScope dry-run command generation for single-model-key
  `vllm-sr/auto` VSR runs.
- [x] Build HLE text representative subset recipe and suite wiring for
  GLM-5.2-only `VSR 1.0`.
- [x] Build SWE-Bench Pro agentic recipe and suite wiring for GLM-5.2-only
  `VSR 1.0`.
  - The suite now distinguishes the lower-cost
    `swe_bench_verified_mini_agentic` shakeout row from the real
    `swe_bench_pro` adapter. Mini Agentic results are useful for protocol and
    patch-quality debugging, but must not be reported as the official
    SWE-Bench Pro score.
- [x] Run HLE text smoke/slice with `amd_auto_hle_glm52.yaml`.
  - HLE Math smoke, 24 examples, EvalScope batch 24:
    `VSR 1.0` GLM-only scored 45.83.
  - This beats the official GLM-5.2 native reference 40.5 by +5.33, but does
    not beat Fugu 48.5 or Fugu Ultra 50.0.
  - Artifacts:
    `bench/router_flow/results/hle-glm52-vsr1-smoke-b24-v1/` and
    `bench/router_flow/results/hle-glm52-vsr1-smoke-b24-report-v1/`.
- [x] Classify HLE diagnostic result.
  - GLM-only HLE has useful self-uplift but is not enough evidence for the
    competitive target.
  - Tail latency is high: the 24-example smoke took about 22 minutes, with
    average latency 439.59s and long tail after 20/24 examples.
- [x] Design initial competitive hybrid HLE recipe with explicit model-pool
  disclosure.
- [x] Run HLE hybrid smoke using `amd_auto_hle_hybrid.yaml`.
  - HLE Math smoke, 24 examples, EvalScope batch 24:
    `VSR Hybrid` scored 79.17.
  - This beats GLM-5.2 40.5, Fugu 48.5, and Fugu Ultra 50.0 on the smoke
    subset, but it is not publishable as full HLE evidence because the slice is
    small and Math-only.
  - Latency improved relative to GLM-only: about 8 minutes wall time and
    172.87s average latency, versus about 22 minutes and 439.59s average
    latency for GLM-only.
  - Artifacts:
    `bench/router_flow/results/hle-hybrid-smoke-b24-v1/` and
    `bench/router_flow/results/hle-hybrid-smoke-b24-report-v1/`.
- [x] Run HLE hybrid cross-subject text slice using formal mode with a bounded
  limit, so smoke success is tested outside the Math-only subset.
  - HLE text cross-subject slice, 8 subsets x 6 examples = 48 examples,
    EvalScope batch 24:
    `VSR Hybrid` scored 56.25.
  - This beats GLM-5.2 40.5 by +15.75, Fugu 48.5 by +7.75, Fugu Ultra 50.0
    by +6.25, Opus 4.8 45.7 by +10.55, Gemini 3.1 Pro 44.7 by +11.55, and
    GPT 5.5 44.3 by +11.95 on this bounded cross-subject slice.
  - Subset scores: Biology/Medicine 33.33, Chemistry 50.00, Computer
    Science/AI 33.33, Engineering 50.00, Humanities/Social Science 50.00,
    Math 83.33, Physics 83.33, Other 66.67.
  - Performance: 48 examples completed in about 15m08s with average latency
    153.07s, average input 2237.5 tokens, and average output 17603.5 tokens.
    The final sample was a major tail-latency outlier; next HLE tuning should
    add tighter per-round timeout/fallback protection before scaling.
  - Artifacts:
    `bench/router_flow/results/hle-hybrid-formal8x6-b24-v2/` and
    `bench/router_flow/results/hle-hybrid-formal8x6-b24-report-v2/`.
- [x] If HLE hybrid does not beat Fugu/Fugu Ultra on the cross-subject slice,
  optimize the hybrid recipe or failure-classify the gap.
  - Not needed for this slice; the hybrid recipe cleared Fugu and Fugu Ultra.
    Remaining HLE work is confidence scaling, source alignment, and long-tail
    timeout optimization, not gap closure on this bounded slice.
- [x] Run full HLE text-only eval using the competitive HLE hybrid recipe.
  - Full text-only EvalScope HLE resolved to 2158 items, batch 24, with judge
    routed through local `gpt55-verifier`.
  - `VSR Hybrid` scored 47.08. This beats GLM-5.2 40.5 by +6.58 and the
    listed closed single-model rows in the joined report, but does not beat
    Fugu 48.5 or Fugu Ultra 50.0.
  - Artifacts:
    `bench/router_flow/results/hle-hybrid-full-text-b24-v2/` and
    `bench/router_flow/results/hle-hybrid-full-text-b24-report-v2/`.
- [ ] Run full HLE text-only eval using an Opus/Gemini-only closed model pool.
  - Started as remote durable `nohup` job `hle-closed-full-text-b24-v1` on
    `root@134.199.199.149`; local laptop sleep or SSH/network loss will not
    stop the eval.
  - Router mounted config was switched to
    `/root/vllm-sr-flow-eval/bench/router_flow/configs/amd_auto_hle_closed.yaml`
    and contains only `anthropic/claude-opus-4.8` and
    `google/gemini-3.1-pro-preview`.
  - HLE judge is `gemini31-worker` through the same local router so GPT is not
    introduced into this closed-pool comparison.
  - Remote PID file:
    `/root/router-flow-eval/results/hle-closed-full-text-b24-v1/_run_logs/nohup.pid`.
  - Remote live log:
    `/root/router-flow-eval/results/hle-closed-full-text-b24-v1/_run_logs/nohup.log`.
  - Output root:
    `bench/router_flow/results/hle-closed-full-text-b24-v1/` after pull.
  - Report root:
    `bench/router_flow/results/hle-closed-full-text-b24-report-v1/` after pull.
  - EvalScope resolved HLE text-only to 2158 items, batch 24, with judge routed
    through local `gemini31-worker`.
- [ ] If GLM-only HLE cannot plausibly beat closed baselines/Fugu, keep the
  competitive hybrid HLE recipe with explicit model-pool disclosure.
- [ ] Run SWE-bench Pro smoke/slice, then public-full campaign using one SWE
  auto recipe.
  - Prepared AMD host `root@165.245.131.56` with EvalScope 1.8.1 and
    `swebench==4.1.0`.
  - Initial durable smoke `swe-glm52-vsr1-smoke-l1-165-v3` was stopped as a
    protocol-failure artifact: the backticks agent expected exactly one
    `mswea_bash_command` fenced block, while the recipe allowed normal
    `bash` fences. The run completed many router loop turns but produced no
    sandbox execution or prediction artifact.
  - Updated `amd_auto_swe_glm52.yaml` to normalize every SWE action into a
    single EvalScope `mswea_bash_command` fenced block and verified the real
    first SWE prompt now returns a parser-valid block.
  - Started durable smoke `swe-glm52-vsr1-smoke-l1-165-v4` on host
    `165.245.131.56`; PID file:
    `/root/router-flow-eval/results/swe-glm52-vsr1-smoke-l1-165-v4/_run_logs/nohup.pid`.
  - Remote live log:
    `/root/router-flow-eval/results/swe-glm52-vsr1-smoke-l1-165-v4/_run_logs/nohup.log`.
  - V4 smoke completed with valid prediction, review, SWE test log, manifest,
    and report artifacts, but scored 0/1. The failed patch applied cleanly but
    failed both hidden `FAIL_TO_PASS` tests because it set the `maxlength`
    widget attribute as a string while the tests expected an integer. Treat this
    as a patch-quality failure, not a harness/protocol failure.
  - Started durable full Verified Mini Agentic run
    `swe-glm52-vsr1-full-mini50-165-v1` as internal failure-analysis data, not
    publishable evidence. PID file:
    `/root/router-flow-eval/results/swe-glm52-vsr1-full-mini50-165-v1/_run_logs/nohup.pid`.
  - Remote live log:
    `/root/router-flow-eval/results/swe-glm52-vsr1-full-mini50-165-v1/_run_logs/nohup.log`.
  - Real `swe_bench_pro` smoke should start only after the Mini Agentic run
    gives enough failure distribution to avoid immediately repeating obvious
    protocol or patch-quality failures on the more expensive official adapter.
  - Early Mini Agentic v1 partial: first two completed samples were 1 resolved
    and 1 unresolved. The unresolved sample exposed a patch-quality issue
    (`maxlength` value type); the next sample exposed repeated repo-local
    Python import failures from running `/tmp` scripts without setting
    `PYTHONPATH` to the worktree. The local SWE recipe now records a
    PYTHONPATH/worktree-root rule for the next run, but that change is not
    hot-loaded into the active v1 run to keep v1 artifacts reproducible.
- [ ] If GLM-only SWE cannot plausibly beat closed baselines/Fugu, design a
  competitive hybrid SWE recipe with explicit model-pool disclosure.
- [ ] Create one SciCode auto recipe, run smoke, classify failures, and adjust
  the SciCode-specific config.
- [ ] Run SciCode report-aligned run using the frozen SciCode auto recipe.
- [ ] Run Terminal-Bench 2.1 smoke, then slice, then full 89 tasks for all four
  stages using one TerminalBench auto recipe.
- [ ] Collect public baselines from traceable sources and keep source metadata
  next to scorecard specs.
- [ ] Generate scorecard JSON/SVG/PNG for every campaign benchmark using the
  vLLM-SR scorecard generator.
- [ ] Write final analysis covering score gaps, effective optimizations,
  ineffective optimizations, and routing/eval bugs found.

## Next Action

Let the durable Opus/Gemini-only HLE full text job continue. Pull HLE closed
artifacts after completion, compare it against the completed HLE hybrid report,
verify no secrets are present, and publish generated score charts only from
stable, defensible results. Treat the current SWE mini run as failure-analysis
data. Use its failure distribution to decide whether to tune the GLM-only recipe
before starting the real `swe_bench_pro` smoke.

## Operating Rules

- Keep benchmark configs and result manifests together. A score without its
  config is not publishable evidence.
- Increase concurrency only when the runner, provider, and sandbox can safely
  support it. Record `eval_batch_size`, router concurrency, provider settings,
  and sandbox parallelism in the run manifest.
- Long-running remote jobs should use background execution, durable logs,
  polling, and resumable collection.
- If a benchmark needs another harness, add the smallest adapter that emits the
  same raw-result, summary, command, config, and scorecard artifacts.
- Use subagents or parallel workstreams for independent investigation, but every
  subagent result must list changed files, commands, result files, conclusions,
  and blockers.

## Related Docs

- [Router Flow real eval README](../../../bench/router_flow/real_eval/README.md)
- [Signal decision eval plan](../../../bench/router_flow/SIGNAL_DECISION_EVAL_PLAN.md)
- [Router Flow workflows plan](pl-0035-router-flow-workflows.md)
- [Fusion API plan](pl-0034-fusion-api.md)
