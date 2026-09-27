# ConTRoL native-projection independent review: 4B source admission remains HOLD

This is a CPU-only, single-AI-reviewer quality screen of a **possible**
human-labeled data source. It is neither a model score nor a substitute for a
second independent adjudicator. No rows were added to student TRAIN, no GPU
experiment started, no publisher dev/test labels were opened, and no sealed
JevArena answers were read. Restricted passages, sample identities and
adjudication details stay in the private experiment record.

The source is [ConTRoL publisher TRAIN](https://github.com/csitfun/ConTRoL-dataset)
at commit `d7acc335bef6c716f2830e1413d0d90c133ad6e9`; its TRAIN file SHA-256
is `e51b63fa1da381a27fb5244e6f3c8f317eed51921e20fc05334023db3e2e834f`.
The protected prompt inventory SHA-256 is
`cc1988b9c7589b8011b2bb8ac02257bdd4423712a9c34b65bfb30980f4f0079b`.
The review used `prepare_control_blind_review.py` against the exact same
quarantine rule as the source screen: five protected-neighbor premise groups,
26 repeated pairs, and 13 conflicting-label groups excluded, leaving 6,618
source pairs in 1,509 independent premise groups. The reviewer saw passages,
claims and native Choice/Noul questions **without** publisher labels. Blind
judgments were sealed before opening a separate label file.

## Reproducible sample and readout

The fixed seed was `control-native-blind-review-v1-20260928`. Thirty distinct
premise groups were sampled: 10 each from publisher contradiction, neutral and
entailment strata, and 9 short / 12 medium / 9 long input strata. Task
assignments were balanced at 10 native Choice, 10 support Noul and 10
contradiction Noul. In the sample, Choice gold positions were 3 / 5 / 2;
support Noul positions 6 / 4 and contradiction Noul 5 / 5. Full-candidate
position balance is documented in the source screen; these sample counts do
not replace it.

| Single-reviewer measure | Count |
| --- | ---: |
| Blinded three-way relation matches source label | 21 / 30 |
| After reveal: clearly justified source-to-native relation | 18 / 30 |
| After reveal: ambiguous or rubric-dependent relation | 9 / 30 |
| After reveal: likely wrong source-to-native relation | 3 / 30 |

The blinded matches by assigned native task were Choice 6/10, support Noul
8/10 and contradiction Noul 7/10. Post-reveal review found one explicit
passage clue missed by the reviewer, illustrating why the nine initial
disagreements are not nine publisher mistakes. The three likely label errors
concern confusion between non-entailment and contradiction, a stated right
versus an obligation, and a below-threshold policy consequence. Ambiguities
involve attribution, modality, topic/theme judgment, and stronger wording than
the passage warrants. A second reviewer should adjudicate the three likely
errors and ambiguous cases before any training admission. These counts are a
small, source-stratified risk screen, not a corpus-wide defect-rate estimate.

The publisher TRAIN also embeds an explicit downstream non-sharing notice in
11 rows from two premise groups; six rows contain an all-rights-reserved
marker. These groups require private provenance and source-rights review and
should be quarantined meanwhile. The dataset-level license statement alone
does not settle permissions for embedded third-party passages. This note
deliberately does not reproduce or identify those passages.

## Protected long-leaf gap

The original near-duplicate scanner skipped protected leaves longer than 800
characters. We separately inspected the 10 long leaves with the highest
rare-word retrieval scores subject to at most three per role. They spanned
three sealed human-transfer leaves, three JevBench leaves, three Decision Bench
leaves, and one rights-clean TRAIN leaf. No selected source/leaf pair shared a
five-word sequence. Six appeared unrelated and four shared only a broad
subject, with no apparent same scenario or answer-bearing detail. This
**targeted 10-leaf sample does not clear** the 9,444 unscanned long leaves or
semantic paraphrases; a full long-leaf retrieval and adjudication gate remains.

## Private evidence hashes and next decision

| Private packet (not committed or uploaded) | SHA-256 |
| --- | --- |
| Blinded 30-item packet | `b265d55eb211231bf6971e2875d46700697a9442a376c9c9cb5886d884b77cd2` |
| Separate source-label reveal | `2e910919e9197cd197ea87578588402c52a76c018f9e6397b188a0a861fb9242` |
| Frozen pre-reveal judgments | `026c412187abc866a8d2a35cd4374f85d4748cd91c5c0882c21e323b599165c9` |
| Post-reveal adjudication | `3a39928f4492559dc3117df67ba93dbe439d04cfb051eb8598f2ce91bbe2dc31` |
| Ten long-leaf neighbor packet | `fabd3ab05c9c81d773ce6443124532e42cd8600c9d03defe555278470870dc6a` |
| Long-leaf judgments | `f701fe402c95ce94065194bfad46565be411e409f60c7a3973c73083c45321bf` |

The generation packets were byte-identical under Python hash seeds 0 and 42.
Raw judgments and passages remain private. **Training admission is HOLD**
pending independent adjudication, the embedded-passage rights screen, full
protected long-leaf overlap checks, a source-group-disjoint fresh diagnostic,
and a matched data-only arm with explicit Score retention. ConTRoL provides
Choice/Noul candidates but no ordinal Score supervision, so it cannot by
itself address the 4B Score regression.
