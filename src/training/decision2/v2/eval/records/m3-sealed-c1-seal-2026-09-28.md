# JevArena-C1 v1 — sealed confirmation set: seal record (2026-09-28)

Eval & peers track, Milestone 3.2. Design: [§5b](sealed-confirmation-set-design-2026-09-28.md)
(no paid annotation). Rules: [amendment 2](m3-prereg-amendment-2-sealed-c1-2026-09-28.md),
[3](m3-prereg-amendment-3-sealed-c1-2026-09-28.md), [4](m3-prereg-amendment-4-leak-audit-fixed-options-2026-09-28.md).
Reviews: [round 1](m3-sealed-c1-review-round1-2026-09-28.md), [round 2](m3-sealed-c1-review-round2-2026-09-28.md).
**Sealed 2026-09-28 12:08 UTC. No model has seen any C1 prompt.**

## Composition

2,874 items in 14 tasks from 8 post-cutoff public sources, covering 2,254 source groups.
All gold labels are the sources' own human labels. There is no AI-adjudicated tier.

- **Types.** Choice 1,434, Noul 695, Score 745.
- **Input length.** 539 items are ≥ 4,000 characters (18.8%).
- **Languages.** English 1,945, Arabic 555, Russian 300, Finnish 68, Swedish 4, Ukrainian 2.
  Non-English items total 929 (32%).
- **Licences.** 940 items come from NC-family sources and are for internal evaluation
  only: HalluTruthQA (NC-ND), WB reviews (NC-SA) and innoduel (NC).

| Task | Type | Items | Groups | Labels | Balance | Long | Language |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| delichess/communicative_function | Choice | 270 | 100 | 9 | class | 0 | en |
| delichess/epistemic_stance | Choice | 240 | 93 | 3 | class | 0 | en |
| hallutruthqa/find_truth | Choice | 300 | 300 | 6 | gold length rank | 2 | ar |
| hallutruthqa/hallucination | Noul | 255 | 255 | 2 | length | 0 | ar |
| implicaturex/likelihood | Score | 160 | 160 | 7 | length | 0 | en |
| innoduel/preferred_idea | Choice | 85 | 85 | 2 | class | 0 | fi/en/sv/uk |
| legal_case_law/argument_function | Choice | 210 | 41 | 5 | length | 190 | en |
| narrative_gold/event_causality | Choice | 129 | 129 | 3 | class | 0 | en |
| narrative_gold/setting_concreteness | Score | 150 | 150 | 4 | length | 0 | en |
| narrative_gold/setting_temporal_grounding | Score | 135 | 135 | 5 | length | 0 | en |
| narrative_gold/span_is_event | Noul | 240 | 240 | 2 | class | 0 | en |
| tutormoments/is_rapport | Noul | 200 | 134 | 2 | class | 175 | en |
| tutormoments/moment_type | Choice | 200 | 132 | 2 | length | 172 | en |
| wb_reviews/star_rating | Score | 300 | 300 | 5 | class | 0 | ru |

## How it was built

- **Sources.** 19 registry sources were checked and 11 admitted after re-verifying first
  release dates (amendment 2 §1). Review removed stance-it and MCJudgeBench. The build gate
  dropped GAPA, whose stimuli are unscreenable (too short to shingle).
- **Overlap screen** (15 corpora, 956M scanner tokens). The corpora were training data at
  5c0255ed, 39a120ca and ed87a03a (including the data-v2 `m2/` arms), the eight protected
  eval panels and the eight peer JEV aggregators. Of 40,707 candidates, 40,203 were
  CLEAN, 73 OVERLAP and 431 REVIEW. OVERLAP and REVIEW ≥ 0.2 containment were excluded
  before selection. Almost all hits were against peer aggregators: ImplicatureX 43,
  tutormoments 13, MCJudgeBench 14, legal 1. HalluTruthQA had 2 OVERLAP against the
  data-v2 arms.
- **Selection.** Salted order and class balance with group caps. Tasks with length cues
  were length-balanced. The selected set passes the leak audit (option and state surface
  CLEAN) and the per-task length gate.
- **Blind review.** Nine independent reviewer subagents took part. All 14 tasks passed:
  13 in round 1 and tutormoments after one template fix. Reviewer–gold majority agreement
  runs from 0.25 to 1.00 against chance levels of 0.11–0.50. It is disclosed and never
  used for selection.
- **Compute.** CPU only on node A. GPU-hours 0.

## Commitments (SHA-256)

| Object | SHA-256 |
| --- | --- |
| prompts.jsonl (gold-free, 2,874 rows) | `0b29686f60c980f3fbc8a03b88537fc0bf90ee967afa67d4fe0c958b1bfde16a` |
| gold.jsonl | `c02777713c0e58b40cf602947433420744b2765465252ce22d692eb676ca4fe1` |
| build-5 manifest (count-only) | `71297ecb5458a182c68e88f88b52eb37f8f45770d02e82aa93f39d04dcbedd86` |
| selection salt | `43e6a3bc8d736037e1d4fd3907b17d1470e65b39fea79334c644dd5e900ceaa5` |
| c1-config.json v3 | `42e8c9407ba55190c0f7bb0c9127395171ffde900a8fdc7329ab648e2926195f` |
| overlap receipt / hits | `e4e7d652b489c9b25f6a4fecf89e1e5b3cf00bce1a5c9f3f1a28cc7e357c7079` / `4e33adfe83cf9b0dd84b4f5c73760084fdce55d573c7fafce5f85b039f66c154` |
| overlap corpus manifest | `f9ad7ca907b0c42210bcde9c8044d6a4646b0cb14827e05f79c4711b016dc4e9` |
| review round 1 / 2 receipts | `eee825265cedebfbd7397c1078c45ab1080263ecdcf278808a4a5eda8b85b06f` / `ae213452bf4ad04619fdafefad45b7457bab5a4a94d8f7e6152881babfdfecf9` |
| encrypted bundle `c1-v1-bundle.tar.enc` (38 files) | `d924389a7ffc3d852d9204c3dfca1c12c32f82685a7e5c65277ba9980110534f` |
| bundle file manifest | `d1119ac316202b7786e664f1837935f6db8066f80de2b505dd06811582bb4834` |
| encryption key (verification only) | `9a3bab26b73492c66fc53df06ab92f433cb752f5bfc35c90a757ae162748e68d` |

Candidate files per source (build input):

| Source | SHA-256 |
| --- | --- |
| delichess | `21dc8c56…` |
| hallutruthqa | `c6b73c33…` |
| implicaturex | `0f63935e…` |
| innoduel | `72084dce…` |
| legal_case_law | `c7cc20fa…` |
| narrative_gold | `19dbdb6c…` |
| tutormoments | `c643dfb6…` (template of `51fd482a8`) |
| wb_reviews | `a503a245…` |

Code: converters and selection at `d35948d60`; the other candidate files came from
`b4a699a3b`, with converters unchanged since. The scorer is at `3dfeb9b4e`.

## Custody and access

- **Encrypted at rest.** The bundle holds the salt, every build output, the review packets
  and the reviewer answers. It is encrypted with AES-256-CBC + PBKDF2 (200,000 iterations)
  and stored in node A `/data/dev2/private/sealed/c1/` (mode 700). A copy sits in the
  private eval-artifacts dataset under `m3/sealed-c1/` (commit `356faac8`).
- **Plaintext deleted.** The plaintext salt, builds and review files are gone.
  Candidate pools and source snapshots stay, because they are public data; without the
  salt they do not identify the selected items.
- **Key.** The eval track, as custodian, holds the key only on the workstation, in its
  private config (`sealed-c1.key`, mode 600). It is never stored on a node, in git, the
  gist or HF. It is streamed over SSH only for encryption and decryption.
- **Logging.** Every decryption is appended to node A `ACCESS.log` and to the eval gist.
  Any read of prompts, gold, salt or key outside a logged event retires the affected items.

## One-shot scoring procedure (frozen release candidates only)

1. **Eligibility.** A frozen, hash-pinned release candidate (the coordinator's release
   decision). With it go the tier's Decision 1.0 model and at most two open same-tier
   peers, each scored once. Budget: one candidate per size and three scoring events in
   total. After that the set is declared post-key.
2. **Rescan.** Before the event, rescan every training file that landed after `ed87a03a`
   (and the candidate's own training data), using `v2.eval.sealed.overlap scan` over the
   selected items after decryption into a private temp dir. Any new overlap retires those
   items for this and later events. The retirement is disclosed.
3. **Collect.** Decrypt only the prompts to
   `/data/dev2/private/panels/goldfree/sealed-c1.prompts.jsonl` (the hash is registered
   as `sealed-c1` in `v2/eval/panels.py`). Run the frozen runner on the candidate's
   same-node runtime and packaging limit (the largest input limit its runtime supports,
   at least 8,192 tokens): `run_same_panel.sh ... -- --panels sealed-c1 --adapter-spec …`.
   Delete the prompts file right after collection and log it.
4. **Seal predictions** before any gold is decrypted:
   `python3 -m v2.eval.sealed.score seal --prompts <prompts> --predictions <run>/output/sealed-c1.predictions.jsonl --output <run>/SEAL-C1.json`.
5. **Score.** Decrypt the gold to a private temp dir, then run
   `python3 -m v2.eval.sealed.score score --gold <gold> --predictions … --seal … --label … --output <run>/REPORT-C1.json`.
   Then run `compare` against each comparator. Delete the gold and log the event.
6. **Report.** Label it "independent confirmation (JevArena-C1)", separately from post-key
   v3. Include C1 (mean task macro-F1), per type, per task, Score QWK, long and non-English
   slices, and the NC and permissive split. Use paired intervals by source group. Report
   whatever the direction. No checkpoint selection, calibration or threshold may use C1.

## Known limits (disclosed)

- **Old text, new labels.** Several sources have text that was public before the cutoff:
  ImplicatureX contexts, legal opinions (1935–1987), narrative-gold Dolma passages, and
  Instagram-era comments. They test label novelty, not text novelty.
- **Label noise.** innoduel rests on single votes (duplicate-pair agreement 55%).
  ImplicatureX and the narrative Score tasks are noisy at adjacent levels.
- **Thin coverage.**
  - HalluTruthQA supplies the only Arabic items and one of three Noul sources.
  - There is no Chinese, Japanese or German source.
  - After MCJudgeBench's exclusion, no instruction-following judgement task remains.
- **Content.** Tutoring transcripts contain minors' pseudonymised personal remarks.
- **Converter bugs.** Two minor boundary bugs are documented as expected-failure tests.
  Neither affects the built items.
- **What it can claim.** This is Option B evidence: independent of our data and of the
  peer aggregators by construction, but not human-re-verified. It supports "independent
  of project data" claims, not "human-verified held-out" claims.
