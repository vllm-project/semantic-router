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
| `datasets.py` | DRACO loader (parses the weighted rubric) |
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
