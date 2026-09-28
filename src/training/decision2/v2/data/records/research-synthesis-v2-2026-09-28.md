# Decision 2.0 research synthesis v2 (2026-09-28)

Concise successor to `research-synthesis.md`, `decision-2-research.md`,
`kev-decider-data-lessons-2026-09-27.md`, `decision-2-architecture-gap-2026-09-28.md`,
the Jev inventory and the catalog-gap audit (all at pinned commit `be472b957`). It
feeds `experiment-matrix-v1-2026-09-28.md`. **[A]** = author, card or leaderboard
claim, not reproduced; **[M]** = our own measurement (post-key same-panel JevArena
v3 unless marked development).

## 1. Where the family stands [M]

| Size | Best eligible 2.0 so far | Own 1.0 | Strongest same-panel peer | Main regressions |
| --- | --- | --- | --- | --- |
| 0.6B | official Qwen3-0.6B-Base on A0: v3 38.520, public231 143 | Kai1 35.938 / 114 | Bosun 38.524 / 133 | Choice 109 vs 277/800; Score 80 vs 98/400 (Noul 458 vs 404) |
| 0.8B | none qualified (Eos continuation dev +0.163) | Eos1 | — | Score 85/400 (dev) |
| 2B | Sol continuation 43.960 / 162 | Sol1 45.580 / 161 | Decider2B 49.499 / 175 | below own 1.0 |
| 4B | official Base 53.218 / 171 | Nox1 56.470 / 173 | Decider4B 61.882 / 192 | below own 1.0 |
| 9B | none qualified (dev proxy 61.255 vs Lux1 DEV 70.326) | Lux1 | JPT-9B (peer record) | Score, transfer |
| ~27B | none qualified (dev proxy 68.53) | — | AutoJev27 72.310 / 200 | Score |

Pattern: continuing own 1.0 on the 7,455-row rights-clean control (A0) does not beat
1.0; official bases start lower; Score and Choice are the recurring losses; readout
changes at a fixed small budget (type-separated head 553/700, candidate interaction
333/700, projection 547/700 vs control 562/700 SELECT) did not help. The common
factor across sizes is **data**: A0 has ~4.1M tokens (Qwen3 tokenizer), 76.3% of them
from one programmatic composition source, 516 Score rows (102 three-level), English
and Chinese only.

## 2. Focus areas

**Data breadth.** [A] Decider-4B stage 1: 1.89M items / 742M tokens (60% public mix,
8% 26-source expansion, 32% ten verifiable generated families), then 29k hard/replay
rows. [A] Kev: 10k rows from ten English public sources + 896 policy minimal pairs +
1,680 generated rules, later CFPB narratives and solver-labelled hard cases.
[A] Decision 1.0 Lux: 24k-row full-parameter adaptation; its control/treatment
mixture changed tokens (30.21M vs 22.01M), Chinese and Score exposure together, so no
single source effect is identified. Implication: A0 is two orders of magnitude
smaller and far narrower than the strongest open peer; breadth has to be added as
separately ablatable arms (A1–A6), token-matched (matrix D1–D6, D8, D10).

**Decision readout.** [A] 1.0 paper: Kai = bidirectional encoder with separate
Choice/Noul/Score paths (1,024-token cap); Eos–Lux = causal candidate endpoints + a
shared FP32 bilinear/MLP head (16,384 cap). [A] This-That-Model: one-forward slot
readout with CE+Brier. [A] CLM: frozen backbone, state/action projection heads,
relative candidate similarity (details in §3). [M] 0.6B readout variants at the
frozen budget were negative; the causal option path is order-asymmetric (69/400
jointly correct original/reordered Choice pairs); a cached option-isolation path
failed hidden-state parity (0.218–0.480 vs 0.01). Implication: test readouts on
fixed, better data (M1 after wave-1 arms) and attack order sensitivity with data too
(D9, A0p).

**Contrastive learning and hard negatives.** [A] CLM trains state/action projections
with InfoNCE and hard negatives; [A] Kev uses policy minimal pairs; contrast sets
(Gardner et al., 2020) and counterfactual augmentation (Kaushik et al., 2020) are the
standard remedy for annotation artifacts. [M] Our sources repeatedly failed
shortcut gates: policy-conflict v1/v2 state-removed 67.19% / 57.81% vs 50% majority;
SNLI/OCNLI strong hypothesis-only; ConTRoL +7.42 points; ContractNLI
hypothesis-identity 69.59% vs 52.32%; Score v1–v5 count/weight cues. Implication:
generated arms are built as **counterfactual groups** — byte-identical question and
options, every gold realized once — so state-free baselines are at chance by
construction (A2, A6); hard negatives are tested as a paired contrast with identical
states and golds (A4h vs A4r, D4); contrastive objectives are a separate readout
factor (M2, M3).

**Replay.** [A] Decider v2.1: hard replay sharpened probabilities; KL to v1's soft
distribution plus per-type temperature partly restored them, but v2.1 stayed below v1
on its regression held-out (0.784 vs 0.788). [M] Eos own-source soft replay (KL 0.5,
512 repeated rows) failed its dev gate; Sol soft replay stopped at BF16 zero-step
parity; own-Kai TRAIN-only teacher was a negative feasibility result. Implication: a
frozen replay prompt set (RP-v1) with **attested** own-1.0 targets per tier and a
Decider-like design where new data excludes A0 (D7b); a zero-step parity preflight
between the student runtime and the teacher's native output is mandatory.

**Calibration.** [A] proper-scoring and confidence-aware RL papers: binary-only
objectives drift overconfident; CE+Brier helps in studied settings. [A] Lux1 card:
transfer Brier 0.2911 vs Jev 0.1912; [A] social-science study: Jev median ECE 0.157
and high-confidence failures. [M] DEV2.0-0.6B typed Brier 0.3535, ECE10 0.1785, Score
expected-level MAE 1.18. Implication: per-type CAL temperatures on every arm, CE+Brier
as M4, calibration as a retention floor on every data factor.

**Cross-source transfer.** [A] Kev hard-v1: +26.3 pp in-family, +0.3 pp on its locked
out-of-domain panel; [A] This-That-Model: seen synthetic families did not transfer to
unseen ones. [M] 9B Score-cardinality residual raised Score (174→237/400) while all
three human pilot tasks lost macro-F1. Implication: report in-family (arm held-out
AHO) and transfer (typed DEV mechanism-disjoint, CSS pilot, later v3) separately;
keep generated families mechanism-disjoint from the typed DEV/FINAL families and
exclude every Decision Bench/CSS/JevBench source; test source diversity at fixed
tokens (D8).

**Score (recurring failure).** [M] 0.6B predicted level 0 for 400/400 typed-DEV Score
items; RPS objective alone did not help (SELECT 276/700); the v6 English pilot moved
the middle level 20→43 but the top level 52→41; human ordinal sources screened so far
(HelpSteer2 17/24 within one grade, ANLI 9/24, TREC DL 2023, VitaminC, FEVEROUS,
F1000RD, PeerRead) remain HOLD. [A] Jev guide lists score interpolation as a weak case.
Implication: A6 balances level counts 2–10 and grades exactly, uses explicit
verifiable rubrics (band, checklist, interpolation, rank, evidence status), and pairs
with M5 only after D6.

## 3. Sources and what we take from them

| Source | Claim relevant to data/training [A] | What we use |
| --- | --- | --- |
| [Jev](https://hanxiao.io/all-about-jev/) / TypeSafe docs | Hosted typed-probability model (RLCD, "probabilities trained against outcomes"; data undisclosed; `jev-1.13.0`; 32k-token state); documented weak cases: counting/arithmetic, dates, multi-hop, long distracting state, prompt injection, score interpolation; its limitations page says Score levels are weak in numerical calibration | A2 families target those skills. **Not a teacher:** the [Master Customer Agreement](https://typesafe.ai/legal/mca) §2.3(b) forbids using outputs to distill or train a model; §2.3(f) restricts publishing benchmark results (legal review before publishing Jev comparisons) |
| [Decision 1.0 paper](https://vllm-sr.ai/decision-paper) and [cards](https://huggingface.co/collections/llm-semantic-router/decision-10) | Kai ≈21k examples from 15 source families (Cosmos QA, SNLI and QASC also form 1.0's own Inference panel); Lex 6k rows with equal soft/hard weighting; decoders 24k mixes; one scalar temperature (Lux 2.005 for all types); no hard negatives; option-order flips on 36 pairs Jev 0%, Kev-9B 2.8%, Lux 8.3%, Sol 38.9%; evidence-removal confidence drop Kev-9B 53 pts, Jev 30, Lux 27, Sol 6, Kai 0.6; evaluation weights chosen after outcomes | 1.0 is a start and a regression control, never a blind test; per-type temperature (M4); order (D9) and evidence-removal (D11) factors |
| [Decider-4B](https://huggingface.co/Mapika/decider-4b) | Large public + generated mix; soft replay + per-type temperature (card unchanged since 09-24) | D1/D2 scale ambition; D7 |
| [Kev](https://github.com/jaredpalmer/kev) | Small source-tracked mix; minimal pairs; in-family gains ≠ OOD; 09-28 dropped an external test set after finding contradicting labels | D2 transfer vs in-family split; D4; label audits on any gate set |
| [CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) / [code](https://github.com/Contrastive-LM/CLM) | Frozen Qwen3-8B, 4096-d last-token L2 embeddings; separate state/action MLPs (width 1536, depth 3, GELU, LayerNorm → 512, L2); logit = exp(scale)·cosine (init 1/0.07, cap 100); symmetric in-batch InfoNCE; ~30M LLM-*generated* hard negatives on the state→action side only; stages ≈60M QA pairs → hard-negative mid-training → ≈1M agent steps with 40% replay (replay keeps hard-negative top-1 68.5% vs 56.2% without; late hard negatives 69.2% vs 62.4% when mixed from the start); probabilities are relative to the supplied set; Noul/Score via text templates; no absolute-probability head, ordinal loss or fitted calibration; Decision Index pooled ECE 0.32 [L]; weights + code Apache-2.0; pair text, negatives, post-training mix and stage scripts not released | M1–M3 at 9B on our data; D4b staged hard negatives; replay dose in D7; absolute Choice/Noul/Score readouts must be built and validated separately |
| Open peers (Decision Index cards) | JPT-9B: multi-class Brier loss, 49k questions, shuffled-option copies (CC BY-NC); Jebadiah-27B: per-type temperatures; JevK5: teacher-written hard questions; Bespoke Nimble: 2,676 curated contrastive examples; AutoJev-27B: 73k examples, one temperature | M4 (Brier, per-type T), D9 (shuffled options), D4 |
| Replication notes collected by the Jev dataroom | Adding 1,600 Japanese items raised new tasks 45.2→68.0% but old tasks fell 92.2→90.7% and ECE rose without replay; a DeBERTa replica fell from 0.854 in-domain to 0.690 on unseen questions; 94/400 typed-decisions test records near-duplicate training states | D5 with retention floors; D7; overlap scans on every gate set |
| This-That-Model | One-forward slot readout, CE+Brier, abstention construction; seen families do not transfer | M4; family-disjoint splits |
| Option/rubric reversal study (2609.26758) | Naming options yes/no instead of 0/1 raises decision flips by up to 70.4 points; AUC 0.938→0.232 for one peer | D9, A0p; order/name counterfactuals |
| Methods papers | Instruction Diversity (task count beats examples per task); NV-Retriever (positive-aware false-negative filtering for mined negatives); Thermometer (per-task temperature); ORCU (unimodal soft ordinal targets) | D8; D4 mining filter; M4; M5 |

## 4. What changed in the v2 data design

1. Counterfactual-group construction for every generated arm; shortcut gates on
   large samples (thousands of rows, group-disjoint 5-fold), so the ±6-point noise of
   the earlier 128-group screens does not decide admission.
2. Source isolation now includes every Decision Bench v4 record source (ContractNLI,
   CUAD, CFPB, civil_comments, HAGRID, …) and the CSS tasks; FLUTE is excluded from all
   new arms and from replay.
3. The protected overlap scan covers all 39 protected roles (40,857 input rows)
   including the long leaves the earlier scanner skipped, plus an embedding scan.
4. Replay targets are produced by each 1.0 model's own validated runtime at an
   attested revision, with per-type teacher-quality reports, so size tracks can see
   what R2 would distill. [M] On RP-v1 (an A0 slice, in-distribution for 1.0) Sol,
   Nox and Lux are strong teachers (Score argmax 0.776 / 0.807 / 0.878), while Kai,
   Lex and Eos give near-uniform Score distributions (normalized entropy 0.86–1.0,
   Score argmax 0.20–0.29): at 0.6B/0.8B R2 cannot transfer Score behavior.
5. Gates stay proportionate: they block leakage, shortcuts, rights problems and
   unfair comparisons; they do not require AI-reviewer agreement thresholds for
   verifiable arms whose labels are computed and independently re-derived.
