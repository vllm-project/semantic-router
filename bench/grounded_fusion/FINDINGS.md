# Grounding-Aware Fusion — Evaluation Findings

A first end-to-end evaluation of the Fusion looper's **grounding-aware synthesis**
stage against the [DRACO](https://huggingface.co/datasets/perplexity-ai/draco)
rubric-graded deep-research benchmark, run fully locally (Ollama, Apple Silicon).
This documents the design, two scorer bugs it surfaced and fixed, the headline
result, and how to improve the experiment next.

## TL;DR

- **The panel-NLI grounding scorer was broken** and silently returned ~0.500 for
  every response. Two root causes, both fixed in this change:
  1. `classify_nli` reconstructed the 3-class distribution from the argmax
     confidence, so any "neutral" prediction forced `entailment == contradiction`
     → the consistency score collapsed to exactly 0.5.
  2. The NLI input was a single `premise [SEP] hypothesis` string truncated to 512
     tokens; panel answers run 600–1000+ tokens, so the hypothesis was truncated
     away entirely → the model saw half of one answer → predicted neutral.
- **After the fix the scorer discriminates** (live scores spread 0.52–0.69;
  Level-1 Spearman vs panel rubric quality **+0.21**).
- **But the *intervention* — dropping low-consistency panel responses before
  synthesis — does not help and significantly hurts on hard factual questions.**
  On the slice where grounding actually drops a response, the final-answer DRACO
  score drops with a bootstrap CI excluding 0, and the harm grows monotonically as
  more responses are dropped.
- **Mechanism:** cross-model consistency drops the *dissenter*, but on
  graduate-level questions the dissenter (often the strongest model) carries the
  correct minority view. **Consistency ≠ correctness.**

## Status / production decision (2026-06)

Acting on the headline result, the grounding stage now exposes a `grounding.policy`
lever and **defaults to `weight` (soft down-weighting, no hard-drop)**:

- `weight` (default) — keep every panel response; the judge is told to weight each
  answer by its groundedness score and to protect a correct lone dissenter.
- `annotate` — keep every response; pass the scores to the judge as notes only.
- `filter` — the prior hard-drop (below `min_score`); now opt-in.

**Decided:** we do **not** hard-drop facts by default — `filter` is known to regress
quality on contested factual items (this document).

**Still open (to validate in follow-up CRs, not in this change):**

1. Does `weight`/`annotate` actually *beat* plain fusion, or is it merely
   non-harmful? Run `make_configs.py --policy weight` / `--policy annotate` arms
   against the `off` baseline.
2. Is the scorer (Level-1 Spearman +0.21) strong enough that soft-weighting on it
   improves synthesis? Discrimination ≠ usefulness-as-a-weight.

This change ships the policy + default; the efficacy A/B above is deferred.

## What was measured

Two levels (most eval efforts only do level 2 and then can't explain a null/noisy
result):

- **Level 1 — intrinsic:** is the scorer any good? Grade each panel response with
  the DRACO rubric and correlate (Spearman) the grounding score with panel-response
  quality.
- **Level 2 — extrinsic:** does the *final answer* improve? Same DRACO items
  through plain Fusion (`grounding.enabled: false`) vs grounding-aware Fusion,
  both graded by the DRACO rubric; paired bootstrap CIs on the deltas, reported
  overall + per-domain + on the **contested slice** (items where grounding actually
  dropped ≥1 response — where it can do anything at all).

DRACO grades a free-text answer against a weighted rubric (positive criteria reward
correctness/coverage; negative criteria down to −500 penalize confident-wrong /
unsafe / badly-sourced claims). Grounding ran in **`panel` mode** (cross-model NLI)
because DRACO ships no source documents, so `context`/`hybrid` modes have nothing
to score against.

## Setup

- Domains: **Medicine + Law** (12 items) — where the −100…−500 penalties live and
  models most disagree.
- Panel: `qwen3:8b`, `llama3.1:8b`, `gemma3:12b` (cross-family for NLI diversity).
  Judge + rubric grader: `qwen3:14b`. All local via Ollama behind a no-think proxy.
- Threshold sweep on `grounding.min_score`: **0.34 / 0.55 / 0.60** (`min_keep: 1`).
- MVP two-config A/B at `temperature: 0` (see Limitations re: per-arm panel
  regeneration).

## Headline result

Δ = grounding-on − off on the final-answer normalized DRACO score (negative = worse):

| min_score | contested items | overall Δnorm | **contested Δnorm** | contested Δneg-penalty |
|-----------|-----------------|---------------|---------------------|------------------------|
| 0.34      | 0 / 12          | −0.035 (ns)   | — (nothing dropped) | —                      |
| 0.55      | 4 / 11          | −0.058 (ns)   | **−0.113 (sig)**    | −2.50 (ns)             |
| 0.60      | 9 / 12          | −0.090 (ns)   | **−0.132 (sig)**    | +4.44 (ns)             |

- Monotonic: the more grounding drops, the more it hurts overall quality.
- On the contested slice the harm is statistically significant (CI excludes 0) at
  both 0.55 and 0.60. At 0.60, 8 of 9 contested items got worse.
- The dropped-model frequency at 0.60 was `gemma3:12b` 6×, `qwen3:8b` 5×,
  `llama3.1:8b` 4× — the largest model is dropped most, consistent with
  consistency filtering removing the strongest dissenter.
- The intended benefit (cutting the negative-criteria penalty) was **not robustly
  realized** (neg-penalty deltas not significant, mixed sign).

## Interpretation

The scorer is now sound, but the **policy of hard-dropping** the least mutually
consistent response is the wrong lever for hard factual QA. Three smaller models
can be confidently wrong *together* (high mutual entailment) while the model that
disagrees is right; consistency filtering deletes exactly that signal. With
`min_keep: 1`, high thresholds collapse synthesis onto the single most-agreed
response — the most consensus-biased answer.

## Limitations (read before trusting the magnitudes)

- **Small N** (12 items; contested 4–9). Overall deltas are not significant; only
  the contested slices are.
- **MVP two-config design** regenerates the panel per arm, so deltas carry
  sampling noise (one off-arm answer scored worse than its on-arm counterpart).
  The cached-panel paired design (below) removes this.
- **Weak panel** (8–12B local models). The "dissenter is right" effect may differ
  with stronger / more diverse models or a larger panel.
- **`panel` mode only.** The `context`-mode faithfulness lever (hallucination
  detector vs real RAG sources) is untested here and is the mode most likely to
  help — DRACO just can't exercise it (no source docs).

## How to improve the experiment (next experiments)

1. **Cached-panel paired design** — generate the panel **once** per item, then
   replay judge+synthesis for each lever/threshold so all cells share identical
   panel outputs. Biggest single noise reducer; turns the sweep into a clean paired
   test. Needs the looper to support replaying a fixed panel (or a harness-side
   refactor that separates panel generation from synthesis).
2. **Ablate the two levers** — grounding both *filters* the panel and *annotates*
   the judge prompt. Run `off` / `notes-only` / `filter-only` / `both`. The sweep
   conflated them; notes-only (information without deletion) may help while
   filtering hurts.
3. **Down-weight instead of drop** — keep all responses and pass grounding scores
   to the judge as soft weights/annotations rather than removing responses. Test
   whether soft beats hard-drop.
4. **Larger N / factuality subset** — 12 items is underpowered. Run the full DRACO
   set (or a factuality-weighted subset) for tighter CIs.
5. **Test `context` mode on a RAG benchmark** with real source docs — the lever
   this dataset cannot exercise, and the one with a genuine faithfulness signal.
6. **Stronger / larger / more diverse panel** — re-test the "dissenter is right"
   effect with stronger models and panel size > 3.
7. **Grade dropped responses too (Level-1 completeness)** — emit dropped responses'
   content in the fusion trace so Level-1 can correlate grounding score with quality
   across the *full* panel, not just the kept responses.
8. **Per-section breakdown** — report which rubric sections (Factual Accuracy vs
   Citation Quality) move, to localize where grounding helps/hurts. (Citation
   Quality is confounded: vSR Fusion has no server-side web tools, so both arms
   cite from parametric memory.)

## Reproducing

See `README.md`. The two scorer fixes are in `candle-binding/src/ffi/classify.rs`
(real softmax) and `src/semantic-router/pkg/looper/grounding.go` (sentence-level
NLI). The harness writes per-sample JSONL + summary JSON per arm; `compare.py`
produces the paired A/B report. Run artifacts are git-ignored.
