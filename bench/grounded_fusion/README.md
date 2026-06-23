# Grounding-Aware Fusion Benchmark (DRACO)

A/B evaluation of the Fusion looper's **grounding-aware synthesis** stage against
the [DRACO](https://huggingface.co/datasets/perplexity-ai/draco) rubric-graded
deep-research benchmark. Runs fully locally on macOS (Apple Silicon) with Ollama.

## What it measures

DRACO grades a free-text answer against a **weighted rubric** (positive criteria
reward correctness/coverage; negative criteria, down to −500, penalize
confident-but-wrong / unsafe / badly-sourced claims). We score two levels:

- **Level 2 (extrinsic)** — does grounding-aware fusion beat plain fusion on the
  final answer? Headline: the **negative-criteria penalty** should drop. `compare.py`
  reports paired bootstrap CIs, overall + per-domain + "contested slice" (items where
  the `filter` policy actually dropped a response — meaningful only under `filter`).
- **Level 1 (intrinsic)** — is the grounding *scorer* any good? `--grade-panel`
  grades each panel response too, and `evaluate.py` reports the Spearman
  correlation between the cross-model NLI grounding score and panel-response
  quality, plus discard precision/recall.

## Grounding policy (what we use the score for)

The grounding stage now defaults to **`policy: weight`** — keep every panel response
and let the judge soft-weight by groundedness score (protecting a correct lone
dissenter) — instead of hard-dropping. `make_configs.py --policy {weight,annotate,filter}`
selects the lever for the `on` arm so follow-up CRs can A/B them.

**Status / open questions (tracked, not yet answered here):**

- ✅ **Hard-drop (`filter`) hurts.** The first DRACO A/B showed dropping the least
  mutually-consistent response significantly *lowers* answer quality on contested
  factual items (the dissenter is often the right minority view). So the production
  default no longer drops facts — see `FINDINGS.md`. This is why `weight` is default.
- ❓ **Does `annotate`/`weight` actually *help*?** Unknown. Keeping all responses and
  passing scores to the judge as weights/notes may help, hurt, or be neutral vs plain
  fusion. To be measured in a follow-up CR (`--policy weight` and `--policy annotate`
  arms vs the `off` baseline).
- ❓ **Is the *scorer* trustworthy enough to weight on?** Level-1 Spearman was +0.21
  (it discriminates), but whether that signal is strong enough to improve synthesis
  when used as a soft weight is not yet established. Follow-up experiment.

## Architecture (local stack)

```
harness → Envoy(:8801) → router extprocofr_(:50051) → Fusion looper
                                                     ├─ panel/judge → no-think proxy(:11435) → Ollama(:11434)
                                                     └─ grounding: candle NLI (models/mom-halugate-explainer)
grader (DRACO rubric) ─────────────────────────────→ no-think proxy(:11435) → Ollama
```

- **Panel** (cross-family diversity → real NLI signal): `qwen3:8b`, `llama3.1:8b`,
  `gemma3:12b`. **Fusion judge**: `qwen3:14b`. **Rubric grader**: `qwen3:14b`/`32b`.
- **No-think proxy** (`ollama_proxy.py`): Ollama's OpenAI endpoint ignores the
  `think` flag, so Qwen3 burns the whole token budget on reasoning (5+ min/req,
  truncated answers). The proxy forwards to Ollama's *native* `/api/chat` with
  `think:false` → ~30 s/req, complete answers. Point the looper + grader at it.
- **Grounding runs in `panel` mode** (cross-model NLI) because DRACO ships no
  source documents (so `context`/`hybrid` modes have nothing to score against).

## Setup

```bash
# 1. Models (once)
brew install ollama && brew services start ollama
ollama pull qwen3:8b && ollama pull llama3.1:8b && ollama pull gemma3:12b && ollama pull qwen3:14b

# 2. Python env (once)
python3 -m venv .venv-bench
.venv-bench/bin/pip install openai requests numpy scipy tqdm pandas pyyaml pytest \
  --trusted-host pypi.org --trusted-host files.pythonhosted.org

# 3. Generate the two router configs (grounding on/off).
#    --policy {weight,annotate,filter} picks the lever for the 'on' arm
#    (default weight = no hard-drop).
.venv-bench/bin/python -m bench.grounded_fusion.make_configs --policy weight
```

## Run

```bash
# Start supporting services (each in its own shell / backgrounded):
.venv-bench/bin/python -m bench.grounded_fusion.ollama_proxy --port 11435
tools/bin/func-e run --config-path /tmp/envoy-bench.yaml      # see run_ab.sh for envoy prep

# Full A/B (brings the router up per arm, runs both, compares):
bench/grounded_fusion/run_ab.sh --max-samples 100 --grade-panel

# Or a quick pilot (recommended first):
bench/grounded_fusion/run_ab.sh --domains Medicine,Law --max-samples 8 --grade-panel
```

`run_ab.sh` runs arm `on` against `config-fusion-on.yaml`, restarts the router with
`config-fusion-off.yaml`, runs arm `off`, then `compare.py`. Results land in
`results/` (`samples_{on,off}.jsonl`, `summary_{on,off}.json`, `ab_report.json`).

### Runtime
~30 s per fusion request + rubric grading. Full 100×2 arms + panel grading is an
overnight-scale job; start with a pilot. Runs are resumable (`--resume`).

## Cached-panel, paired, multi-arm evaluation (the one that settles it)

The `run_ab.sh` design above regenerates the panel **per arm**, so each delta mixes
the intervention with fresh-sample noise — at small N the result is uninterpretable
(see `FINDINGS.md`, and why `weight` shipped without efficacy evidence). The
cached-panel evaluator fixes this: generate the panel **once** per item, then
synthesize **every arm from the byte-identical panel**, so deltas isolate the
intervention. It also adds the baselines the A/B lacked.

Arms:

| Arm | What | Answers |
|-----|------|---------|
| `A` | judge-solo (one model, no fusion) | does fusion beat one model at all? |
| `B` | plain fusion (grounding off) | does grounding add anything over plain synthesis? |
| `C` | `weight` (the shipped default) | the actual claim |
| `D` | random-weight placebo (real synthesis, NLI replaced with seeded-random scores) | is it the *score* or just *any* weighting? |
| `annotate`, `filter` | optional, via `--arms` | the other policies |

Pre-registered decision rule (gated on the normalized DRACO score, paired bootstrap
CI excluding 0): **KEEP_GROUNDING** if `C>B` and `C>D` and `B>=A`;
**KILL_GROUNDING_ADDON** if `C≈B` or `C≈D`; **KILL_FUSION** if `A` beats `B`;
else **INCONCLUSIVE**.

This path drives the looper **in-process** (the `fusioneval` Go binary, via the
`Request.CachedPanel` seam) with the **real candle NLI** — no Envoy/extproc — so it
needs the candle lib + NLI model + an Ollama endpoint, not the router config dance.

```bash
# 0. Build the driver (needs the candle lib on the linker path).
cd src/semantic-router && \
  CGO_LDFLAGS="-L$PWD/../../candle-binding/target/release" go build -o ../../bin/fusioneval ./cmd/fusioneval
cd ../..

# 1. Dump items + start the no-think Ollama proxy.
.venv-bench/bin/python -m bench.grounded_fusion.items \
  --draco-path ~/Downloads/draco.json --domains Medicine,Law --max-samples 100 \
  --out bench/grounded_fusion/results/items.jsonl
.venv-bench/bin/python -m bench.grounded_fusion.ollama_proxy --port 11435 &

# 2. Generate the panel ONCE, then run arms A–D from the identical cached panel.
LD_LIBRARY_PATH=candle-binding/target/release bin/fusioneval \
  --items bench/grounded_fusion/results/items.jsonl \
  --nli-model models/mom-halugate-explainer \
  --endpoint http://localhost:11435/v1/chat/completions \
  --judge qwen3:14b --panel qwen3:8b,llama3.1:8b,gemma3:12b \
  --arms A,B,C,D --out-dir bench/grounded_fusion/results --max-items 4   # drop --max-items for the full run

# 3. Grade each arm with the SAME rubric grader (reuses evaluate.grade_sample).
for arm in A B C D; do
  .venv-bench/bin/python -m bench.grounded_fusion.grade_only \
    --answers bench/grounded_fusion/results/answers_$arm.jsonl --arm $arm \
    --draco-path ~/Downloads/draco.json --grader-model qwen3:14b --resume
done

# 4. Multi-arm verdict (writes verdict.json with the KEEP/KILL decision).
.venv-bench/bin/python -m bench.grounded_fusion.compare_multiarm \
  --results-dir bench/grounded_fusion/results --arms A,B,C,D \
  --json-out bench/grounded_fusion/results/verdict.json
```

**Smoke first** (`--max-items 4`): re-run step 2 and confirm `panel_cache.jsonl` still
has 4 rows (resume didn't regenerate); every `answers_*.jsonl` row shares the same
`panel_sha256` per `id` (byte-identical panel across arms). The in-package Go tests
(`fusion_cached_panel_test.go`) already assert B/C/D arm isolation without the stack.

**Context mode (next step, sequenced after the DRACO smoke passes):** the dataset
seam is in place — `get_dataset("jsonl", ...)` reads gold passages into
`metadata["context"]`, `items.py` threads them, and the driver injects context as a
system message. The remaining wiring is the **hallucination-detector** backend for
`--grounding-reference context` (the driver currently wires panel-mode NLI only);
DRACO ships no source docs, so context mode needs a context-grounded dataset.

## Gotchas (learned the hard way)

- **Silent grounding no-op:** `panel`-mode NLI only fires if
  `hallucination_mitigation` is enabled *with* an NLI model — here it's wired via
  the `explainer` block (`models/mom-halugate-explainer`). `evaluate.py
  --assert-grounding` aborts if the `on` arm lacks a grounding trace, so you never
  measure plain fusion twice.
- **Don't base configs on `config/config.yaml` naively** — it enables milvus/redis/
  postgres/llama_stack stores and embedding-backed signals that fatal locally.
  `make_configs.py` disables them and strips routing to a single sentinel-keyword
  fusion decision.
- **Qwen3 thinking** ruins latency and truncates answers via the OpenAI endpoint —
  always go through `ollama_proxy.py`.

## Files

| File | Role |
|------|------|
| `datasets.py` | DRACO loader + generic rubric-graded `jsonl` loader (context-mode seam) |
| `rubric_judge.py` | DRACO-style grader: per-criterion LLM check → weighted score |
| `runner.py` | drives the router fusion endpoint, parses the `fusion` trace |
| `metrics.py` | Spearman, discard precision/recall, paired bootstrap CI |
| `evaluate.py` | one-arm orchestrator (run → grade → Level-1/2 summary) |
| `compare.py` | paired A/B report (on − off) with CIs |
| `make_configs.py` | generates the two router configs |
| `ollama_proxy.py` | OpenAI→Ollama-native shim that disables Qwen3 thinking |
| `run_ab.sh` | end-to-end A/B driver |
| `run_sweep.sh` | `min_score` threshold-sweep driver (`filter` policy only; off arm + on arms, resumable) |
| `sanitize_results.py` | strip local paths from result JSON before sharing |
| `test_grounded_fusion.py` | unit tests (loader + grader math, no model needed) |
| `FINDINGS.md` | evaluation write-up: bugs fixed, results, next experiments |
| **Cached-panel multi-arm** | (the paired evaluator that settles `weight` efficacy) |
| `items.py` | dump DRACO items as JSONL for the `fusioneval` driver |
| `../../src/semantic-router/cmd/fusioneval` | Go driver: cache panel once, run arms A–D from the identical panel via the real candle NLI |
| `grade_only.py` | grade the driver's `answers_{arm}.jsonl` (reuses `evaluate.grade_sample`) |
| `compare_multiarm.py` | N-arm paired comparison + pre-registered KEEP/KILL `verdict.json` |
| `test_compare_multiarm.py` | unit tests for the verdict logic (no model needed) |
