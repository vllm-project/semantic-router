# Amendment 3 to the PN1 preregistration (research & data Milestone 4, 2026-09-29)

Amends [m4-pn1-prereg-2026-09-29.md](m4-pn1-prereg-2026-09-29.md) after amendments
[1](m4-pn1-prereg-amendment-1-2026-09-29.md) and [2](m4-pn1-prereg-amendment-2-2026-09-29.md). Committed and
pushed before any GPU step it describes. No further prompt iteration is allowed after this amendment.

## 1. Reason: the judge prompts are miscalibrated

The judge code is unchanged. The v1 prompts ("exactly the same meaning"; "natural and grammatical") make
Qwen3.8-27B with thinking disabled reject register, politeness and wording variants. Measured on both v1
passes (89,128 items):

| Construction (should be) | v1 label disagreement per language | Median P(yes) |
| --- | --- | --- |
| `pn-hop` (yes) | 70–91% | 0.11 |
| name coordination swap (yes) | 15–49% (pass 1) | 0.59 |
| twin "same" (yes) | 41–87% (pass 1) | 0.32 |
| `pn-near`, role swap, twin "different" (no) | 0.3–11% | ≈ 0 |

- v1 fluency failed 57–79% of twin edits and 31–94% of name swaps.
- Only the most literal hop pairs survived. That produced the overlap shortcut in the self-check of build
  `e8814a305fbd`: overlap-only logistic regression reached majority + 0.056 pooled, and +0.13 in ja and zh.
  The yes rows had a higher character-multiset Jaccard than the no rows within the same bigram bin
  (ja .744 vs .683, zh .687 vs .604).

## 2. Replacement prompts (v2, fixed)

The A/B order stays random, as in v1 (`sha256("pn1-judge-order:" + cid)`).

- **Label v2:** "Here are two sentences in <Language>.\nSentence A: <a>\nSentence B: <b>\nDo the two sentences
  mean the same thing? Differences in wording, word order, politeness, formality or punctuation do not matter.
  Answer No if they differ in who does what to whom, in a person, place, object, time or quantity, or in any
  other fact. Answer Yes or No."
- **Fluency v2:** "Here is a sentence in <Language>:\n<s>\nIs this a grammatical sentence that a native speaker
  could write? Minor awkwardness or unusual word order is fine; answer No only for a clear grammatical error or
  a sentence that makes no sense. Answer Yes or No."
- **Thresholds are unchanged:** a yes row is kept at P ≥ 0.5, a no row at P < 0.5, and an edited sentence
  needs fluency P ≥ 0.5.

## 3. Control probe (≤ 0.03 GPU-h, before the re-judge)

- **Items,** 25 per language each, in hash order (`pn1_build probe-items`), judged under both v1 and v2 in the
  same run as before/after evidence:
  - 200 identical pairs (s, s) of unedited pool sentences;
  - 200 `pn-name` coordination-swap pairs;
  - fluency on 200 original twin seeds (unedited Tatoeba text).
- **PASS iff, under v2:**
  - median P(yes) on identical pairs ≥ 0.90;
  - median P(yes) on coordination swaps ≥ 0.80;
  - at least 80% of seeds pass fluency (P ≥ 0.5).
- **If the probe fails:** stop, record and report. No other prompt is tried.

## 4. One full re-judge of the existing pool

- **Scope:** every row of the existing candidate pool (103,114 rows, all families), with label v2 for every pair
  and fluency v2 for every edited sentence of `pn-name` and `pn-twin`. There is no new generation.
- **Order:** rows are ordered by need (seed-hash position over the planned units per stratum and label, as in the
  first pass).
- **Budget:** the jobs stop at a cumulative 1.05 GPU-h for the milestone (0.504 before this amendment).
  - The volume is estimated from the probe's throughput.
  - If the budget is short, the lowest-need rows go unjudged. No check is skipped for a judged row.
- **Superseded:** amendment 2's second-pass judgments, and all v1 judgments, are superseded for selection.
  Their receipts are kept.

## 5. Matching replaces the coarse overlap bins

This is the preregistered A4 "rebalance once" remedy, applied now because the self-check already shows the
failure.

- **Matching rule:** within language × construction group (natural / swap) × same-multiset flag, yes and no rows
  are matched 1:1 by greedy nearest neighbour.
  - Coordinates: character-bigram Jaccard, character-unigram-multiset Jaccard and length ratio, with a caliper
    of 0.03 on each coordinate.
  - Order: no rows are processed in seed-hash order. Each takes the nearest (Euclidean) unmatched yes row
    inside the caliper; ties go to seed-hash order.
  - Unmatched rows are dropped.
- **Unchanged:** dev-first isolation, the per-language 50/50, the dev quotas (300 × de es fr ja ko zh, 100 × ru
  ar), the TRAIN targets and the swap-share goals. Shortfalls are reported and not redistributed.
- **TRAIN:** a group's pairs are split over its two same-multiset strata in proportion to their pair counts,
  first pairs first.
- **Dev:** matched pairs are drawn in `sha256("pn1-dev:" + no-row group key)` order. Stratum units and
  family-pair quotas are proportional to the planned TRAIN selection.
- **`--drop-groups`:** the re-run re-matches the remaining rows of each language and split with the same rule
  and keeps every matched pair. An empty list reproduces the selection.
- **Kept for reporting:** the bigram bins stay as reported strata only.

## 6. Superseded builds

Builds `aaad8cd7a5f3` and `e8814a305fbd` are moved to `superseded/`. Neither was audited.
