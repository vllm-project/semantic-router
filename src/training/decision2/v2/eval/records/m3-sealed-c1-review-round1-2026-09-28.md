# JevArena-C1 blind quality review — round 1 (2026-09-28)

Rules: [amendment 2](m3-prereg-amendment-2-sealed-c1-2026-09-28.md) §4. Six independent
reviewer subagents took part: three per packet, with reviewer ids r1–r3 spanning the short
and long packets. They saw gold-free prompts only, over SSH on node A. Sample: 232 items in
16 tasks, 16 per task or 8 for mostly-long tasks, drawn from build-3 (prompts
`02fd5132…`, gold `60a4cdf2…`). Receipt `review-receipt.json` SHA-256 `eee82526…` (node A,
private). The table has counts and rates only.

| Task | n | Flagged by ≥ 2 | Majority = gold | Chance | Fleiss κ | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| delichess/communicative_function | 16 | 0 | 0.44 | 0.11 | 0.85 | PASS |
| delichess/epistemic_stance | 16 | 0 | 0.50 | 0.33 | 0.85 | PASS |
| hallutruthqa/find_truth | 16 | 1 | 0.94 | 0.17 | 1.00 | PASS |
| hallutruthqa/hallucination | 16 | 0 | 1.00 | 0.50 | 1.00 | PASS |
| implicaturex/likelihood | 16 | 0 | 0.31 (within 1 level: 0.88) | 0.14 | 0.76 | PASS |
| innoduel/preferred_idea | 16 | 1 | 0.62 | 0.50 | 0.67 | PASS |
| legal_case_law/argument_function | 8 | 1 | 1.00 | 0.20 | 1.00 | PASS |
| mcjudgebench/constraint | 16 | **4 (25%)** | 0.81 | 0.33 | 0.75 | FAIL: quality flags |
| narrative_gold/event_causality | 16 | 0 | 0.69 | 0.33 | 0.86 | PASS |
| narrative_gold/setting_concreteness | 16 | 0 | 0.25 (within 1 level: 0.88) | 0.20 | 0.58 | PASS |
| narrative_gold/setting_temporal_grounding | 16 | 0 | 0.31 (within 1 level: 0.88) | 0.20 | 0.77 | PASS |
| narrative_gold/span_is_event | 16 | 0 | 0.62 | 0.50 | 1.00 | PASS |
| stance_it/stance | 16 | **4 (25%)** | 0.94 | 0.33 | 0.78 | FAIL: quality flags |
| tutormoments/is_rapport | 8 | **2 (25%)** | 0.88 | 0.50 | 1.00 | FAIL: quality flags |
| tutormoments/moment_type | 8 | 1 | 1.00 | 0.50 | 1.00 | PASS |
| wb_reviews/star_rating | 16 | 1 | 0.44 (within 1 level: 0.88) | 0.20 | 0.66 | PASS |

Reviewer–gold agreement is the AI-agreement estimate. It is disclosed and never used to
select items. Items are not dropped for disagreeing with the human gold.

## Decisions (one fix allowed per failing task)

- **stance-it: excluded.** Flags were for missing post context, garbled or very short
  comments, and one crude remark about a named politician. With 13 source groups and a
  single annotator, a template fix cannot repair the task. `c1-config.json` v2 drops it.
- **MCJudgeBench: template fixed once.** Flags were for the boundary of "partially
  satisfied" and for responses that end abruptly in the source. The instructions now use
  the paper's own annotation criteria (arXiv 2605.03858, appendix: partial "only when the
  constraint is met in part with clear evidence"; annotators judge the response as shown).
  They now say to judge the response exactly as shown and give examples of partial
  satisfaction. Labels are unchanged.
- **tutormoments (both tasks, which share one template): template fixed once.** Flags were
  for sessions that are not all maths, speaker-label errors, and an asymmetric rapport
  definition. The instructions now say the sessions are mostly mathematics and that
  speaker labels may contain transcription errors, and they ask which kind the span is
  primarily about. The rapport definition now also covers missed chances to respond, as
  the scaffolding definition already did.
- The three fixed tasks are re-reviewed once by three fresh reviewer subagents on the same
  sampled items. A second failure excludes the task.

## Other reviewer observations (kept, disclosed)

- **ImplicatureX.** Several utterances keep the source's markers for implied meaning and
  cancellation, and two markers are empty. This is a property of the published stimuli,
  and the level agreement within one level is 0.88.
- **HalluTruthQA Choice.** In some items the correct option is worded differently from the
  distractors, even after length-rank balancing. One long item's options differ only in a
  place name.
- **WB reviews.** Some reviews state their own rating in the text.
- **innoduel.** Single votes between near-equivalent ideas (duplicate-pair agreement 55%)
  make this the noisiest task.
- **Legal.** A few target "sentences" are citation fragments from source sentence splits.

## Converter tests

A test subagent added 41 converter tests (DeliChess, legal, narrative-gold,
tutormoments). They record two minor boundary bugs as expected failures, and neither
affects the built items. `legal_case_law._nearest` counts a separator before the first
context sentence, so the window is at most 1,999 rather than 2,000 characters.
`narrative_gold._level` does not reject ±infinity, which never occurs in the data.
