# Deliberation Algorithms for vLLM Semantic Router

**Version:** 1.0
**Authors:** vLLM Semantic Router Team
**Status:** Proposal

## Abstract

The `fusion` looper (panel → judge → synthesis) gives vLLM Semantic Router an
OpenRouter-equivalent multi-model deliberation mode. This proposal surveys the next
generation of *original* deliberation algorithms in the spirit of ReMoM, identifies
where vSR can structurally outperform OpenRouter's Fusion, and recommends building
**grounding-aware synthesis** first — a factuality lever OpenRouter has no equivalent
for, because vSR is a classifying gateway with a built-in groundedness detector.

## 1. Problem

Fusion deliberation works, but it has three structural limits:

- **It always pays the full cost.** Every request fans out to the whole panel plus a
  judge and a synthesis call (N+2), regardless of how easy the question is.
- **Its judge has no grounding oracle.** The judge is a bare LLM reading raw panel
  text. OpenRouter's own deep-research benchmark (DRACO) explicitly *penalizes
  confident-but-wrong answers*, and judge choice alone swings scores 10–25 points.
- **The spend/save decision is static.** Operators pick "single model" or "fusion"
  per route; nothing decides per request whether deliberation is worth it.

The underlying tension is two-fold: **save tokens** (route to a single cheap model)
vs **spend tokens for accuracy** (deliberate). These are two ends of one adaptive
spectrum, not two competing products.

## 2. How OpenRouter Fusion works

Fan a prompt to a panel of models in parallel (each with server-side web search /
fetch), have a judge produce structured analysis (consensus, contradictions, partial
coverage, unique insights, blind spots), then have the calling model write the final
answer grounded in that analysis. Reported findings:

- **Diversity + synthesis beats any single frontier model**, and a *budget* panel can
  beat a frontier solo model at ~50% cost.
- **Self-fusion** (a model paired with itself) still gains ~+6.7 points — a meaningful
  share of the lift comes from the synthesis step, not just architecture diversity.

vSR's Fusion matches the pipeline shape but lacks the server-side web tools and
exclude-lists (an honest gap where OpenRouter leads).

## 3. Why fusion works — and what the consistency score is *not*

It is easy to misread fusion as "average several models and trust where they agree."
That is the wrong mental model, and it leads directly to the regression that motivated
the `weight` default. Spelling out the actual mechanism keeps the design honest.

### 3.1 The three mechanisms — only one carries the lift

- **Coverage (raises the ceiling).** Models have different blind spots, so the *union*
  of what a panel knows is larger than any single member. On a hard question the correct
  fact, framing, or citation often appears *somewhere* in the panel even if no member
  got the whole answer right. This raises the best *achievable* answer; it does not, by
  itself, produce it.
- **Self-consistency / marginalization.** Sampling many answers and taking the consensus
  cancels reasoning-path noise — but only for tasks with a checkable answer (e.g. math).
  It does little for the open-ended factual questions that dominate real traffic.
- **Verification is easier than generation — this is where the lift comes from.** A judge
  reading several candidate answers *with the question in front of it* operates in
  verify-mode: cross-check this claim against that one, notice the contradiction, keep the
  supported piece. That is cheaper and more reliable than generating from a blank page.

The decisive evidence for the third mechanism is **self-fusion**: pairing a model with
*itself* (zero panel diversity) still gains ~+6.7 points (§2). There is no ensemble magic
there — the gain is the synthesis pass reconsidering candidates. **The intelligence in
fusion lives in the judge, not in the panel and not in the consistency score.**

### 3.2 What the consistency score is — and is not

The cross-model NLI / groundedness score measures *agreement and faithfulness*, not
truth. On factual questions agreement can be actively misleading: several mediocre models
can be confidently, consistently wrong together while the strongest model is the lone
dissenter *because* it knows something they do not. Empirically the score discriminates
weakly (DRACO Level-1 Spearman ≈ +0.21) — useful as a hint, useless as an oracle. Treated
as a hard filter it deletes exactly the correct-minority signal, which is the regression
documented in `bench/grounded_fusion/FINDINGS.md`. So the score is a **soft attention hint
handed to the judge**, never a gate (this is the `weight` policy; see §6).

### 3.3 Does fusing a frontier model with weaker ones beat the frontier model solo?

Often it does **not**, and the design should not assume it does:

- On questions the frontier model already nails, weaker panelists are noise and cost. A
  confidently wrong panelist can even *anchor* the judge and make a frontier answer worse.
- When it *does* help, the win is the strongest model doing verify-mode synthesis over a
  *superset* of content — a weaker model occasionally surfaces a fact or angle the frontier
  model omitted, and the judge folds it in. You are not averaging models; you are widening
  the candidate pool the verifier reads.
- The gate that decides the outcome is **judge competence**. A weak judge regresses to the
  consistent-but-wrong majority. The judge should be the strongest available model.

### 3.4 Consequences for the design

1. **Synthesize with a strong judge** — that is where the value is; do not synthesize with
   a weak calling model.
2. **Treat panel answers as evidence to verify, not authorities to average**, with the
   grounding score as a soft annotation (the `weight` policy).
3. **Gate the spend** — pay the N+2 calls only when the question is hard/contested or the
   lead model is low-confidence (the adaptive-gating follow-up in §8).
4. **Prefer ground truth over mutual agreement** — retrieval/tool context lets the panel
   anchor on *sources* and the judge synthesize against citations. `context`-mode grounding
   beats panel-mode consistency because it scores against truth-ish references rather than
   agreement; this is also the real capability gap vs OpenRouter (§4).

## 4. Candidate algorithms

Each maps onto the existing `BaseLooper` substrate (`client.CallModel`, `SumUsage`,
response formatting) and registers as a looper algorithm.

| Algorithm | Lineage | Idea | Distinct from Fusion/ReMoM |
|-----------|---------|------|----------------------------|
| **Grounding-aware Fusion** (recommended) | Finch-Zk / SelfCheckGPT / NLI | Score panel responses for faithfulness, then rank/filter before the judge | Adds a groundedness oracle to the judge step |
| Multi-Agent Debate | Du et al. 2024 | Iterative cross-critique + revision, convergence early-stop, then synthesis | Multi-round mutual revision vs Fusion's single round |
| Cross-model self-consistency | SelfCheckGPT | Cluster semantically-equivalent answers, return the consensus | No judge; statistical consensus |
| Confidence-gated / adaptive deliberation | AutoMix | Cheap model first; deliberate only when low-confidence | Resolves the spend/save tension at the gateway |

## 5. Where vSR beats OpenRouter

OpenRouter is a pass-through API, so its Fusion must be static and model-driven. vSR
is a classifying gateway, so its Fusion can be adaptive and signal-driven.

| OpenRouter approach | vSR structural advantage | Improvement |
|---|---|---|
| Model decides when to invoke Fusion, or always pays | Confidence + difficulty/domain signals at the gateway | Adaptive-gated deliberation |
| Hand-picked / static panels | Per-model pricing, `param_size`, `CostQualityTradeoff`, selection pkg | Cost+diversity auto-panel |
| Judge is a bare LLM | Built-in hallucination detector + NLI entailment model | **Grounding-aware synthesis** |
| Always full panel + judge | Loop control (ReMoM breadth scheduling) | Adaptive compute / early-stop |
| No per-deployment learning | Runs in your env; `rl_driven` + selection registry | Learned routing from Fusion traces |

Honest gaps where OpenRouter leads: server-side web tools + exclude-lists, four
polished entry modes, and the DRACO eval harness (worth borrowing to *prove* the
grounding lift).

## 6. The ground-truth reality

The detector measures **groundedness against a provided reference**, not truth. So
the design choice is *what serves as the reference*:

- **Context** (RAG/tool output) — strongest, available only when the request carries it.
- **Panel** (cross-model NLI) — the panel as its own mutual reference; no external
  dependency; works on any query.
- **External verifier** — strong but an operational dependency.

Reliability hierarchy of signals: grounded > peer-supported > confident >
self-consistent > relevant. None is truth, but stacked they give a robust *relative*
score — enough to down-weight the least-supported responses before synthesis.

**Use the score as a soft weight, not a hard filter.** The first evaluation
(`bench/grounded_fusion/FINDINGS.md`) found that *hard-dropping* the least
mutually-consistent panel response regresses quality on contested factual questions:
three weaker models can be confidently wrong together (high mutual entailment) while
the lone dissenter — often the strongest model — is right, and consistency filtering
deletes exactly that signal. The grounding stage therefore defaults to the `weight`
policy (keep every response, let the judge weight by score and protect a correct
dissenter); `filter` remains available but is opt-in.

## 7. Recommendation — Grounding-Aware Fusion (hybrid reference)

Extend the existing `FusionLooper` with an optional grounding stage (off by default):
after the panel returns and before the judge runs, score each response for
groundedness, then guide synthesis with the scores (soft-weight by default; hard
filter is opt-in). Hybrid reference: detector against context when present, otherwise
cross-model NLI.

This is the highest-leverage first build because:

- It reuses two already-built subsystems — the hallucination detector
  (`pkg/classification`) and the NLI binding (`candle-binding`) — plus the merged
  usage substrate.
- It is the differentiator OpenRouter structurally cannot match (its judge has no
  grounding oracle).
- It directly attacks DRACO's factual-accuracy / negative-criteria axis.

See the implementation in `tutorials/algorithm/looper/fusion.md` (Grounding-Aware
Synthesis). It makes no extra LLM calls and degrades gracefully (`on_error: skip`
falls back to plain Fusion when the detectors are unavailable).

## 8. Evaluation

Borrow DRACO-style scoring (with negative criteria) to prove the lift: A/B plain
Fusion vs grounding-aware Fusion on a factuality slice, measuring resolve quality and
the rate at which contradicted/ungrounded panel responses are kept out of synthesis.

## 9. Follow-ups

- Multi-Agent Debate as a high-accuracy escalation engine.
- Adaptive gating (cheap-first, deliberate-on-low-confidence) to fix Fusion's
  always-N+2 cost profile.
- Cost+diversity auto-panel composition.

## References

1. OpenRouter, *Surpassing Frontier Performance with Fusion* (2026).
2. Goel et al., *Finch-Zk: cross-model consistency for hallucination detection* (arXiv:2508.14314).
3. Manakul et al., *SelfCheckGPT* (arXiv:2303.08896).
4. Du et al., *Improving Factuality and Reasoning via Multiagent Debate* (2024).
5. See also `proposals/hallucination-mitigation-milestone.md` (TruthLens).
