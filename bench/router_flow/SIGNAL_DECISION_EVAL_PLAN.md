# Signal Decision Design And Eval Plan

This note is the working contract for Router Flow benchmark configs. It is
intended for config authors, benchmark operators, and planner-prompt iteration.

## Signal Decision Design Prompt

Design vLLM Semantic Router signals and decisions for one public model API mode,
`vllm-sr/auto`, reported externally as `VSR 1.0` for GLM-5.2-only recipes and
`VSR 1.0 Pro` for closed-model recipes. Each benchmark owns one
benchmark-specific auto recipe; scorecards expose one VSR row per recipe
identity rather than separate ReMoM/Fusion/Flow/auto rows.

Core rule:

- Signals describe task shape, not model names: benchmark family, answer
  format, tool need, code/terminal need, long-context need, ambiguity, risk,
  latency budget, and strict output contract.
- Decisions bind signals to a concrete algorithm, `modelRefs`,
  `output_contract`, timeout, concurrency, fallback policy, and observability
  labels.
- Every config must include a low-priority fallback decision so unmatched
  traffic never silently falls through.

Recipe requirements:

1. `vllm-sr/auto`
   - Treat auto as a benchmark-specific meta-policy, not one fixed algorithm.
   - It may choose single-model routing, Fusion, ReMoM, or Flow based on signals.
   - Optimize for the benchmark row's quality target first, while keeping cost,
     latency, and provider reliability observable.
   - Publishable VSR recipes should use loop algorithms when the row is meant to
     prove router-side scaling. GLM-5.2-only VSR 1.0 recipes must not include a
     direct single-model decision.
   - Example: easy short MCQ -> single strong model; hard MCQ -> Fusion/ReMoM;
     agentic code or terminal task -> Flow.

2. Fusion decisions inside auto
   - Use when parallel independent answers plus synthesis or judging is likely
     to improve reliability.
   - Configure multiple fusion decisions for different signals, not one global
     fusion.
   - Each decision can vary worker panel, synthesis model, grounding reference,
     grounding policy, `min_successful_responses`, and `output_contract`.
   - Best default for strict QA, GPQA/HLE-style reasoning, and code-answer
     synthesis.

3. ReMoM decisions inside auto
   - Use when iterative or multi-round reasoning can improve hard tasks.
   - Configure `breadth_schedule`, `synthesis_model`, compaction, concurrency,
     timeout, and response budget per signal.
   - ReMoM is an algorithm selected by an auto decision for score reporting.
   - Prefer it for high-ambiguity reasoning, disagreement-heavy tasks, and tasks
     where critique/refinement helps.

4. Flow decisions inside auto
   - Use static flow when the topology is known: solver panel, verifier,
     finalizer, tool role, patch role.
   - Use dynamic flow when task shape is unknown or agentic planning is needed.
   - Dynamic flow must validate planner output against `modelRefs`, `max_steps`,
     `max_parallel`, `access_list`, tool policy, and fallback policy.
   - If dynamic planning fails or times out, fallback to static flow or auto;
     never fail open.

Fallback:

- Add one catch-all fallback decision with lowest priority.
- Fallback should route to auto-balanced or a safe single-model route.
- It must preserve `output_contract` and emit trace fields:
  `matched_signals`, `selected_decision`, `selected_algorithm`, and
  `fallback_reason`.

## Eval Order

Use staged promotion. Do not tune on the final run after a config is frozen.

| Stage | Goal | Benchmarks | Completion signal |
| --- | --- | --- | --- |
| 0 | Preserve completed baseline | GPQA-Diamond | `amd_auto_gpqa_omni.yaml` records the completed VSR 1.0 Pro reasoning recipe. |
| 1 | Code generation proof | LiveCodeBench Jan-Apr 2025 | `amd_auto_livecode_omni.yaml` runs against the Fugu technical-report 175-question window with five retries. |
| 2 | Scientific code proof | SciCode | One SciCode-specific auto recipe with background-provided scoring aligned to the Fugu technical report. |
| 3 | Representative hard reasoning | HLE text representative subset | One HLE-specific auto recipe, text-only, using a representative subject mix instead of Math-only smoke. |
| 4 | Agentic terminal proof | TerminalBench 2.1 | One TerminalBench-specific auto recipe, smoke -> slice -> full 89 after tool-loop stability is proven. |
| 5 | Agentic coding proof | SWE-bench Pro | One SWE-specific auto recipe, smoke/slice before the public full campaign. |
| 6 | Final freeze | Selected publishable rows | Re-run from saved benchmark-specific configs with no prompt/config edits during final collection. |

## Public Baselines For Next Rows

Keep benchmark-aligned references separate from broader market-context
leaderboards. The Fugu technical report is the strict comparison source for the
January-April 2025 LiveCodeBench v6 setup because it states the same 175-question
split and five-retry baseline setting. The public Fugu product page may show
newer or differently labeled marketing-table values; do not mix those values
with the 175-question report row unless the harness is confirmed identical.

| Benchmark row | Source-aligned reference scores |
| --- | --- |
| LiveCodeBench v6, Jan-Apr 2025, 175 questions | Fugu Ultra 92.0, Fugu 90.3, Claude Opus 4.8 90.3, Gemini 3.1 Pro 88.9, GPT-5.5 90.7. Source: Sakana Fugu technical report, Table 1 and Appendix A. |
| SciCode, background provided, canonical 288 subproblems | Fugu Ultra 58.7, Fugu 60.1, Claude Opus 4.8 53.5, Gemini 3.1 Pro 58.9, GPT-5.5 56.1. Source: Sakana Fugu technical report, Table 1 and Appendix A. |
| HLE text | Fugu Ultra 50.0, Fugu 48.5, Claude Opus 4.8 45.7, Gemini 3.1 Pro 44.7, GPT-5.5 44.3. Source: Sakana Fugu technical report text-only HLE row. |
| LiveCodeBench broader market context | Artificial Analysis currently reports Gemini 3 Pro Preview (high) 91.7, Gemini 3 Flash Preview (Reasoning) 90.8, and DeepSeek V3.2 Speciale 89.6. Use only as market context unless the same date window and retries are reproduced. |
| SciCode broader market context | Artificial Analysis currently reports Claude Fable 5 adaptive reasoning 60.2, Gemini 3.1 Pro Preview 58.9, and GPT-5.4 xhigh 56.6. Use only as market context unless the same scoring setup is reproduced. |

## Runtime Estimation

Estimate wall time before launching a run:

```text
wall_time ~= ceil(num_items / effective_parallelism)
             * p95_request_seconds
             + setup_time
             + collection_time
```

Use `effective_parallelism = min(eval_batch_size, router_safe_concurrency,
provider_rate_limit, sandbox_parallelism)`. For loopers, one request fans out to
multiple workers, so provider rate limits usually dominate before local CPU does.

Rough planning bands:

| Benchmark | Publishable size | Recommended first run | Per-request band | VSR wall-time band |
| --- | ---: | ---: | ---: | ---: |
| GPQA-Diamond | 198 | 50, then 198 | 45-180s | 4-12h |
| HLE text-only | 2,500 | 200 stratified | 45-180s plus judge | 6-18h for 200; 2-5d for full text |
| LiveCodeBench Jan-Apr 2025 | 175 | 20, then 175 | 2-10m including execution/retries | 8-36h |
| LiveCodeBench full release | about 1,055 | after Jan-Apr slice | 2-10m | multi-day campaign |
| SciCode | 65 | 10, then 65 | 3-12m including sandbox | 8-24h |
| MRCRv2 / LCR | 100-class slices | 5-20 | 2-8m due long prompts | 8-36h |
| TerminalBench 2.1 | 89 | 1, then 5, then 20 | 10-60m terminal loop | 1-3d for 20; 3-10d for full on one host |
| SWE-bench Verified agentic | 500-class public set | 3, then 50 | 20-90m Docker/agent loop | 1-4d for 50 on one host |
| SWE-bench Pro public | 731 | 10, then 50, then 731 | 20-120m Docker/agent loop | multi-day to multi-week campaign |

These bands now assume one public API mode, `vllm-sr/auto`, with a
benchmark-specific auto recipe. Older four-mode campaigns are historical probes,
not the current scorecard shape.

## Campaign Config Tracking

Each benchmark row owns exactly one saved `vllm-sr/auto` recipe under
`bench/router_flow/configs/`. Before a score is treated as publishable, copy the
exact router config used for that row into the result artifact directory or
reference an immutable config commit plus command-line arguments. A score
without its benchmark-specific config snapshot is an engineering probe, not
final evidence.

## Optimization Loop During Eval

Use the same loop for every benchmark family:

1. Run a smoke subset with trace collection enabled.
2. Classify failures as routing miss, output contract loss, planner schema
   failure, worker timeout, sandbox failure, judge/scorer issue, or true model
   error.
3. Fix the smallest layer:
   - signal/decision config for routing misses;
   - `output_contract` for format failures;
   - planner prompt/schema bounds for dynamic flow;
   - timeout/concurrency only when the model answer is otherwise usable.
4. Re-run only the failed subset.
5. Promote to the next subset size only after the previous subset is stable.
6. Freeze config before the publishable final run.
