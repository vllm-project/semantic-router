# Decision 1.0 training corpora: inventory for data arm A7 (2026-09-28)

Count-only record; no rows, text or panel items. Evidence: the six 1.0 model
cards at their runtime-bearing revisions (Kai `7185f514`, Lex `ee8e74d9`, Eos
`3c2d6326`, Sol `0665a411`, Nox `0bb83350`, Lux `bd45a30a`), the
[Decision 1.0 paper](https://vllm-sr.ai/decision-paper) (§4), the 1.0 project
directories on the two authorized nodes (read-only), and the census receipts
below. Native tokens are the Decision 2.0 decoder prompt (`segments`) under
`Qwen/Qwen3.5-0.8B-Base@dc7cdfe2`; they reproduce the 1.0 team's own totals
exactly (Stage3 52,506,425; Stage4 v2 125,553,787; natural 24k 22.01M versus
replay 24k 30.21M).

Receipts (node-private, count-only): decoder census
`/data/dev2/private/a7/runs/a7-dec10-v1/38d623633b22/inventory/census.json`
(node B, commit `38d623633`), encoder census
`/data/dev2/private/a7/runs/inventory-nodeA/840a56c596b9/census.json` (node A,
commit `840a56c59`).

## Where the corpora are

- **Hugging Face:** no Decision 1.0 training corpus was ever uploaded. The
  `llm-semantic-router` org has 30 datasets (Vela 1.0 data repositories,
  hallucination-span sets, cache triplets, and the private Decision 2.0
  dataset); `agentic-in` has none. The 1.0 model repositories carry only
  attribution/method documents (`TRAINING_ATTRIBUTION.md`,
  `TRAINING_PROVENANCE.json`, `ATTRIBUTIONS.md`, `METHODS.md`).
- **Node B, decoder project** `/root/jev-research-20260921/training-v1/data/`:
  the Sol/Nox curricula (Stage1–4), the natural-reading 24k study pool used by
  Eos, Lux, Kai and the Sol/Nox natural adaptation, its variants, and the 1.0
  builders and audits. Pinned copies of the A7 inputs (hash-identical) are in
  `/data/dev2/private/a7/sources/dec10/`.
- **Node A, encoder project** `/data/vela-jina-20260916/decision-1.0/data/`:
  134 Kai/Lex datasets; Kai's own history receipt covers 79 training/dev
  rosters with 1,782,397 rows. Eos's training file there
  (`eos-role-preserving-v1/materialized-v2/train.jsonl`) has the byte size of
  the node-B natural 24k pool.
- **Repository:** only the Decision 2.0 builders (`training/data/build_pilot.py`,
  `build_rights_clean_v1.py`, `build_rights_clean_v2.py`); the 1.0 builders are
  with the data on the nodes. Lux's final-stage run manifest was not found on
  either node; its card and the 2.0 research note identify its pool.

## Decoder-family corpora (node B)

| Corpus (file SHA-256 prefix) | Used by | Rows (Choice / Noul / Score) | Score levels | Languages | Native tokens (C / N / S) |
| --- | --- | --- | --- | --- | --- |
| Stage1 `train.jsonl` (`77d4ac13`) | Sol, Nox first round | 47,842 (36,469 / 6,486 / 4,887) | L4 4,887 (opaque keys) | en 33,210 · zh 14,632 | 10,772,063 (8,738,574 / 1,008,710 / 1,024,779) |
| Stage2 `pointer-train.jsonl` (`6360df30`) | Sol, Nox stage 2 | 50,000 (42,298 / 6,076 / 1,626), incl. 16,000 Stage1 replay | L4 1,626 (opaque keys) | en 29,135 · zh 20,865 | 21,105,335 (19,627,995 / 1,136,693 / 340,647) |
| Stage3 `train.jsonl` (`3d5ed286`) | Sol, Nox stage 3 | 69,000 (57,905 / 8,477 / 2,618): 47,000 new + 22,000 replay | L3 2,618 | en 39,444 · zh 29,556 | 52,506,425 (49,521,295 / 2,194,397 / 790,733) |
| Stage4 v2 `combined-v2/train.jsonl` (`063ac88e`) | Sol, Nox fourth round | 100,000 (65,230 / 24,248 / 10,522): 64,000 generated + 24,000 Stage3 replay (10,000 public intents) + 12,000 MultiNLI | L3 2,537 · L4 1,746 · L5 1,655 · L6 1,684 · L7 1,581 · L8 1,319 | en 61,009 · zh 38,991 | 125,553,787 (85,528,386 / 32,488,394 / 7,537,007) |
| Stage4 v2 SELECT / CAL (`f28a6c49` / `5ba3bd75`) | Sol, Nox selection / temperature | 2,000 / 1,000 | L3–L8 | en/zh | 2,745,580 / 1,441,025 |
| Natural 8k `natural-train.jsonl` (`b841d9e2`) | all natural adaptations | 8,000 (6,000 / 2,000 / 0): Cosmos QA 4,000, SNLI 2,000, SQuAD 2.0 answerability 2,000 | — | en | 1,825,771 |
| Natural 24k `natural-treatment.jsonl` (`d56175ab`) | **Eos** (all 24,000), **Lux** (24k mixture), **Kai** (16,529 admitted at 1,024 tokens), Sol/Nox natural adaptation | 24,000 (16,391 / 5,911 / 1,698) = natural 8k + 16,000 Stage4 v2 rows | L3 427 · L4 273 · L5 256 · L6 294 · L7 249 · L8 199 | en 17,710 · zh 6,290 | 22,008,009 (15,030,000 / 5,779,382 / 1,198,627) |
| Replay 24k `replay-control.jsonl` (`8d5c92cc`) | Sol/Nox natural A/B control | 24,000 (15,585 / 5,869 / 2,546), all Stage4 v2 rows | L3–L8 (2,546) | en 14,565 · zh 9,435 | 30,212,589 |
| Semantic 24k `nox-semantic-format-v1/treatment.jsonl` (`8f40cf03`) | Lux final stage (semantic option descriptions, per its card) | 24,000; 21,329 identical to natural 24k, 2,671 re-described options | as natural 24k | en/zh | 22,069,350 |
| Natural SELECT / CAL (`ec1c42b2` / `cc75c902`) | natural-study selection / temperature | 1,200 / 600 | — | en | 270,853 / 134,590 |

Other decoder files (quality-reading, order-pilot, balanced-exposure, distillation
teacher inputs, latent-robustness prototypes) are experimental branches that
did not produce released weights; Lux and Eos SELECT/CAL (3,419/1,814 and
3,180/1,590) are 1.0 holdouts, not training data.

**Labels:** program oracles of the 1.0 generators (Stage4 v2: all 66,400 new
labels recomputed by a separate evaluator, candidate-pool-first numeric
construction), and publisher-shipped human labels (Cosmos QA, SNLI, SQuAD 2.0
answerability, MultiNLI non-fiction, BANKING77, CLINC150). No LLM-written
states or labels, no own-model teacher targets in the released curricula, and
no Jev outputs ("No official API output is used as a training label").

**Licences:** project-generated; Cosmos QA and BANKING77 CC BY 4.0; CLINC150
CC BY 3.0; SNLI and SQuAD 2.0 CC BY-SA 4.0; MultiNLI government/slate/
telephone/travel under the OANC permissive terms (pinned card
`nyu-mll/multi_nli@da70db2a`). Training and redistribution with attribution
(share-alike where stated) are permitted.

**Known construction defects (1.0 records plus this audit):** (1) numeric
distractors symmetric around the gold in the legacy arithmetic/scalar/register
families (midrange rule 45.8–49.2% versus 13–25% chance; the 1.0 team excluded
them from Stage4 v2 replay); (2) Stage1/Stage2 Noul and Score rows use opaque
keys (`result_0`/`result_1`, `clover`, …); (3) **new here:** Stage1–3 Choice rows
keyed `result_<n>` number keys in construction order and then shuffle, so the
key reveals the gold (A7 amendment 1). This leak reached every decoder-family
1.0 model through replay rows, and 131 rows of the current rights-clean TRAIN.

## Encoder-family corpora (node A, Kai and Lex)

Kai-native rows (`state_text`, `question`, soft `target`, `provenance`).

| Roster | Rows (Choice / Noul / Score) | Score levels | Languages | Main sources |
| --- | --- | --- | --- | --- |
| Kai shared typed TRAIN / SELECT / CAL | 16,529 (10,923 / 4,157 / 1,449) / 1,894 / 951 | L3–L8 | en 13,480 · zh 3,049 | the natural 24k pool (current Kai Choice/Score paths; plus 4,838 historical Noul replay rows in 18 languages per Kai's card) |
| expanded-stage2 natural | 55,198 (7,268 / 14,292 / 33,638) | L3 6,875 · L5 5,114 · L6 21,649 | 12 (en 25,860, ja 11,916, ko 11,131, …) | SNLI, JSTS, KLUE-STS, MASSIVE, HelpSteer2, BoolQ, ARC |
| source-balanced stage2 | 172,800 exposures over 26,790 groups | L3/L5/L6 | 12 | oversampled expanded-stage2 sources |
| schema-transfer stage3 | 37,568 (14,784 / 14,784 / 8,000) | L3/L5/L6 | 12 | SNLI, BoolQ, MASSIVE, Natural Instructions, ARC, HelpSteer2 |
| natural-choice-transfer | 56,146 Choice | — | 12 | SocialIQA 33,156, JCommonsenseQA 8,938, MASSIVE, BoolQ, NI |
| natural-noul-transfer / BoolQ Noul / Noul category | 26,715 / 29,355 / 8,667 Noul | — | 16 | SNLI, TyDi QA, MASSIVE, BoolQ, NI |
| OASST score transfer / OASST1 score / OASST es-zh | 108,453 (14,052 / 19,275 / 75,126) / 59,146 S / 21,051 S | mostly L5 | up to 25 | OASST quality/helpfulness votes (soft), SNLI, MASSIVE, HelpSteer2, STS |
| HelpSteer2 score / score multirubric / score world diversity | 24,487 / 21,799 / 30,247 Score | L3/L5/L6 | up to 22 | HelpSteer2 five attributes, OASST, SentiMix, AfriSenti, generated long-rule |
| SentiMix Hinglish / AfriSenti Swahili / Hinglish mixed | 11,248 / 1,447 / 34,983 Score | L3 | hi-en, sw | coarse ordered sentiment |
| Cosmos QA choice / choice confidence / QASC choice | 20,369 / 36,195 / 1,812 Choice | — | 12 / 12 / en | Cosmos QA, QASC, MASSIVE, BoolQ, NI |
| NI multifamily / long-rule coverage | 13,568 (6,784 / 6,784 / 0) / 5,760 | L5 | en / en-zh | seven Natural Instructions families; project-generated rules |
| Kai Noul product mix (research) | 129,919 Noul | — | 17 | 100,374 rows without a recognized lineage field (PAWS/TabFact-type inputs), BoolQ, TyDi, SNLI, HANS, MASSIVE |
| **typed-decisions (Lex)** | 4,800 internal TRAIN + 1,200 dev = 6,000 decisions (1,800 / 1,800 / 2,400) | L4/L5 | en | LocalLLaMA/typed-decisions TRAIN@`ea930645` |

**Labels:** mostly publisher human annotations with documented
transformations ("released human annotation; transformation documented":
one-hot hard labels, vote distributions, adjacent-level interpolation of human
means); project-generated rule rows; **agent-authored instruction and rubric
text** (OASST Spanish/Chinese rubrics and schema paraphrases, native question
frames for multilingual MASSIVE/STS; the authoring model is not recorded; the
labels stay human); and **teacher-derived targets** in typed-decisions (states
written by an unnamed model, gold = mean of three samples of an unidentified
"roughly 4B-class" teacher endpoint). Laya-teacher experiments exist in the
encoder project; no evidence ties their outputs to released Kai weights.

**Licences (per Kai's `TRAINING_ATTRIBUTION.md`):** SNLI, ARC, KLUE, JGLUE
CC BY-SA 4.0; BoolQ CC BY-SA 3.0; MASSIVE, HelpSteer2, SocialIQA, QASC,
Cosmos QA, AfriSenti, SentiMix CC BY 4.0; TyDi QA and OpenAssistant Apache-2.0;
Natural Instructions instance terms per family (CC BY-SA 4.0, CC BY 4.0, MIT,
Apache-2.0); typed-decisions Apache-2.0 as declared by its publisher.

## Why the previous 2.0 session left these corpora out

From the 2.0 builders and records (`training/data/README.md`, data-rights v2):

1. **Scale choice, not a rights verdict.** The 2.0 TRAIN was built by sampling
   6,000-row LoRA pilot arms from one 24,000-row "legacy" pool
   (`build_pilot.py`), then removing sources: an 8,522-row Nox replay mix minus
   267 MultiNLI rows (8,255), minus 3,600 TweetEval rows (4,655), plus 2,800
   GoEmotions rows (7,455). The full Stage1–4 curricula (about 267,000 rows)
   and the encoder corpora were never considered as a whole.
2. **Rights reviews left open:** MultiNLI "derivative rights review pending"
   and "Stage3 CLINC/Banking source rights need explicit review". Both are
   resolved above (OANC permissive non-fiction; CC BY 4.0 / CC BY 3.0).
3. **Lineage of selection data:** legacy-derived SELECT/CAL shared lineage with
   the 1.0 weights and were replaced by independent SELECT/CAL; this concerned
   evaluation independence, not the TRAIN rows.
4. **Quality:** the 1.0 team's own numeric-shortcut finding; A7 applies their
   exclusion and adds the key-order fix.
5. **Publication lineage:** concerns about public-weight rights of 1.0-derived
   packages (e.g. the Lux ledger) are about weights, not about training on
   these corpora in the authorized environment.

## Admission summary (details in `a7-arms-v1-2026-09-28.md`)

- **Admitted to A7 (decoder family):** Stage4 v2 (generated, replay, MultiNLI),
  natural 8k, and Stage1–3 rows after the construction-defect rules.
- **Excluded by rule:** legacy numeric-choice rows 21,576 (Stage1 4,869; Stage2
  1,681; Stage3 15,026, exactly the 1.0 team's count); opaque-key Noul/Score
  rows held 19,075 (Stage1 6,486 N + 4,887 S; Stage2 6,076 N + 1,626 S).
- **Not admitted:** typed-decisions (Lex) 6,000 decisions (Choice 1,800, Noul
  1,800, Score 2,400): states and labels come from unnamed models whose terms
  cannot be verified (same principle as the Jev decision).
- **Closed third-party API labels:** none found in the decoder corpora. On the
  encoder side, agent-authored rubric/instruction text (provider unrecorded)
  covers at least the 21,051-row OASST es/zh roster and the native question
  frames of the multilingual rosters; flagged for the coordinator.
- **Deferred to a later A7 version:** the encoder-family rosters (per-source
  review of teacher and agent-authored parts; isolation from multilingual
  panels: PI-v2 already holds a MASSIVE-derived zh decision bench and
  multilingual dev panels, and the eval track's planned multilingual panel has
  no fixed sources yet).
