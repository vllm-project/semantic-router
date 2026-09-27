# Score v8.3 CPU pilot: independent blind review

## Scope and seal

This review used only the two gold-free `train-blind-packet.jsonl` and `select-blind-packet.jsonl` files. I did not inspect generator/source code, candidate/key files, training labels, model outputs, or formal evaluation labels. The train packet SHA-256 is `18ff1815d1c7911677c893dca80264b428efa6c175b925b40503c15b8646135b`; the select packet SHA-256 is `97e9394b61acfdda365749c11c19802ba145800f715ea4b4b408e62c48f4b5e3`. At 2026-09-27 11:59 UTC, I sealed all 54 independently derived item labels in a separate local JSONL file with SHA-256 `a7058ef1da37c37999c8f731eecf9caa56301336a5ce87bbc1b2ce99a29c7a11`. No oracle comparison was performed before or after sealing.

The packet contains 36 TRAIN and 18 SELECT items, arranged as 12 and 6 three-item ordinal groups. Every group has one each of levels 0, 1, and 2 under its printed rules. The three actual mechanics are transfer-window feasibility, transitive handoff dependencies, and stock with tentative inbound units. Each contributes four TRAIN and two SELECT groups. State text is only 55–62 words per item (median 58); all items are English. The instruction text and option set each have only three distinct forms across all 54 items.

## Sealed independent answers

Each triplet lists my labels for suffixes `-1/-2/-3` in order. These are blind review decisions, not a comparison against the generator's key.

| Split | Review group | Blind labels |
|---|---|---|
| TRAIN | `fbbf63fbe1560c20` | 2, 1, 0 |
| TRAIN | `4008168230ce7916` | 1, 2, 0 |
| TRAIN | `9d46bd5d509f3780` | 0, 1, 2 |
| TRAIN | `22dfaa796ba4a9bf` | 0, 2, 1 |
| TRAIN | `c2d436f52f3d5276` | 2, 0, 1 |
| TRAIN | `4811ec23c82f87d5` | 0, 1, 2 |
| TRAIN | `9e88ef2ffa8b2a95` | 1, 2, 0 |
| TRAIN | `60eb0c5e34577655` | 1, 2, 0 |
| TRAIN | `455da850fd18dfd9` | 2, 0, 1 |
| TRAIN | `5ceb836c5c309848` | 2, 0, 1 |
| TRAIN | `94cf38c18cb08cae` | 1, 0, 2 |
| TRAIN | `cb34dbaba4a10787` | 0, 2, 1 |
| SELECT | `37303d60ed4c1316` | 1, 2, 0 |
| SELECT | `d9b0f1a2206db63d` | 0, 1, 2 |
| SELECT | `dbd64ee6d6970168` | 0, 2, 1 |
| SELECT | `1536fed3c465e5cc` | 1, 2, 0 |
| SELECT | `04377841a508817f` | 1, 0, 2 |
| SELECT | `701c44671fa5a2df` | 2, 0, 1 |

## Findings

- **Answerability:** 54/54 have the fields required by their own instructions. The displayed evidence, rather than the options alone, determines the answer; all 18 triplets change the decisive status, time window, or stock arithmetic. I found no missing source or mathematical tie ambiguity. This establishes only rule-level answerability, not realistic transfer quality.
- **Realism defects:** Four handoff items mark a task complete while its stated prerequisite is still queued or failed: `9e88ef2ffa8b2a95-3` (battery certification complete, telemetry rehearsal failed), `60eb0c5e34577655-1` (rig inspection complete, operator signoff queued), `dbd64ee6d6970168-3` (water blank complete, custody review queued), and `1536fed3c465e5cc-3` (device pairing complete, gallery approval failed). The explicit rule still yields a label, but a real workflow would require a stale-status explanation or reconciliation rule. Quarantine these four items and their whole triads before training/selection unless repaired and re-blinded.
- **Copy quality:** `94cf38c18cb08cae-2`, `94cf38c18cb08cae-3`, `701c44671fa5a2df-1`, and `701c44671fa5a2df-2` say “1 tentative units.” Repair the grammar and re-hash the packet.
- **Limited diversity and shortcuts:** Eighteen groups reduce to three fixed instruction/option templates. Short synthetic notes repeat an explicit “other item is unrelated” cue, so entity filtering is weak; random place names and IDs are mostly cosmetic. There is no long-context evidence search, native non-English setting, multiple competing source records, or boundary-equality case. Numeric ordinal keys and option order never vary. The group-level counterfactual design is useful for a diagnostic smoke test but cannot establish broad Score transfer.

## Decision

**HOLD for scale-up or model-release claims from this batch.** It may be kept as a small deterministic logic smoke set. Before a training ablation, repair or quarantine the four inconsistent triads, correct singular-unit copy, add genuinely different documents and boundary cases, and obtain a new blind review of the revised packets. Preserve TRAIN/SELECT separation by whole group. Any resulting model must still be assessed on independent real transfer and Score distributions; this packet alone cannot justify a 27B release claim.
