# DEV2.0-0.8B release support: E8F seed soup (identities, recipe, teacher use, disclosures)

> **Erratum (2026-09-29, per coordinator 05:25 note).** The staging revisions cited below (`16c0929a`, `5421582d`) are no
> longer in the `dev2-dec-staging` history, because LFS cleanups rewrote it. Their weight files return 403.
>
> - E8F soup identity = its 9 lines in hash list node B `m2/hf-staging/batch2.sha256` (`ab0aa4bc…`); released as
>   DEV2.0-0.8B `0b631a85c19fb573aee34fc68bb413271ebe89f4` (same weights and head).
> - Seeds: `batch1.sha256` (`74c04b5d…`) and `batch2.sha256`.
> - Node copies and file tables: [dec-staging-citation-errata-2026-09-29.md](dec-staging-citation-errata-2026-09-29.md).
>
> The per-file SHA-256s in the tables below remain valid.

Written 2026-09-28 ≈21:40 UTC+8 (decoder Milestone 3, item 1) for release
engineering and the coordinator. No new training. Every hash below was read
back from the node files or the staging upload lists; nothing here changes the
candidate. Parents: [release-candidate record](dec-m2-0p8b-release-candidate-2026-09-28.md)
(`23df7f5c6`), [batch-2 lock](dec-m2-batch2-formal-lock-2026-09-28.md) (`0a4de2b17`),
[amendment 3](dec-m2-amendment-3-2026-09-28.md) (`fc6d1e508`),
[M2 results](dec-m2-results-2026-09-28.md) (`3926ca445`).

## Release artifact

| Field | Value |
| --- | --- |
| Candidate | E8F seed soup: uniform FP32 average of the SELECT-chosen checkpoints of E8F-s1/s2/s3 (`v2.dec.soup`) |
| Private staging repo | `llm-semantic-router/dev2-dec-staging`, commit `16c0929ac0df649d8223483e1adf419d78a647ed`, folder `m2/E8F-soup/` [corrected 2026-09-29: revision no longer exists in the repo history; identity by hash list `batch2.sha256` `ab0aa4bc…`, see errata record] |
| Checkpoint | `m2/E8F-soup/checkpoint/` (full checkpoint: `backbone/` Transformers Qwen3.5 text model + `decision_head.safetensors` + tokenizer) |
| `model_sha256` (inference identity) | `60356482ceeb669c4a97eb14dcfae5144b1b181f6c8b7a628ea5d02c86a6dd8b` |
| `backbone/model.safetensors` | `9db82b841878b2f259701e11fdccdfd4b54021b857934304de181e28379f5f1e` |
| `decision_head.safetensors` / `decision_config.json` | `cf2a2ebdd0b007eca460d90889f129f80a75b2371be54ca1889f4bf15613af6a` / `3503b7cb43a1e4994c27916bf35653612a517c249f2ed9458000904a3f8da700` |
| `members.txt` (soup members) | `a631119c5076f03a3759dd2b4b46c4863acf29d662587efc9c7bd6c96595ccb0` |
| Loaded parameters | 753,446,208 (Qwen3.5 text backbone 752,393,024 + decision head 1,053,184); FP32 |
| **Release calibration** | `m2/E8F-soup/cal698-16k/calibration.json` `9f76867d5be618da315bd860aae689ed0cf2a2a20ea34d5bae485b011877e05c`; CAL698 `19cc1a8c…`; `selection_policy = frozen_checkpoint`; max length 16,384; temperatures Choice 1.12262 / Noul 1.07053 / Score 0.39532 |
| Development calibration (not for release) | `m2/E8F-soup/cal700/calibration.json` `4728f1e8f2c882e20ba8c06c02df529d92f31e0fa921c8a322d36d85a0bafad3` (CAL700; 1.12113 / 1.06975 / 0.39469) |
| Packaging | profile `qwen-full`, `max_input_tokens` 16,384; adapter `v2/dec/adapter-spec-infer-dec.json` (`v2.dec.infer_dec`; the full checkpoint carries every weight, `--source-path` is recorded only). The pipeline must accept `frozen_checkpoint` calibration reports (shared loader change `0a399c1d9`). |
| Scored run (node A, frozen runner, image `f83b1d10…`, GPU5) | `/data/dev2/runs/dec/formal/m2/m2-E8F-soup-nodeA` — `REPORT.json` `e3e08cd17fe38becdfa655c61d329de7d39c75ffb7db70861af4f88d87fe2682`, `SEAL.json` `7b886cdcda830f0ec982f644235b4a512b7fca28de832533f9167f996a8940ef`, `PAIRED-vs-adopted-1.0.json` `4489c0ea734a3827edf662419c11faba9fff90deb33325b04460a3f7fbdcd557`, `PAIRED-vs-same-limit-1.0.json` `15140c116b17512563ae37e3a6b53d7b11d077972a2978f90c283f7a688230eb`; persisted autotune cache `m2-E8F-soup-nodeA-triton` |
| `mlx-diag` run | `/data/dev2/runs/dec/formal/m2/m2-E8F-soup-nodeA-mlx` — `mlx-diag.score.json` `4e2529fe243b05f2b984c483804d7d81f6a6ba54993cc0018278f300bc47d1ae` |
| Comparators | Eos 1.0 adopted run `/data/dev2/runs/eval/m1/r4-eos1` (42.547); same-limit 16K control `/data/dev2/runs/dec/formal/m2/eos1-16k` (42.361) |

## The three seed checkpoints (seed evidence; not release artifacts)

| Seed | Staging commit / folder [corrected 2026-09-29: these revisions no longer exist in the repo history; identity by hash lists `batch1.sha256` / `batch2.sha256`, see errata record] | SELECT700 (BEST) | `model_sha256` | `backbone/model.safetensors` | `decision_head.safetensors` | CAL700 calibration (C / N / S) | CAL698 16K calibration (C / N / S) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E8F-s1 (20260926) | `5421582dfed51dd3b19b5686128ed75e0f8525e3` `m2/E8F-s1/checkpoint-0001776` | 592/700, .819722 | `5359b701c4a351f0f5130b34aa757494c4648f571092eaccf0912900a1a2281a` | `8b332483dc9a86faa4ac51dfb6f31c7c7f3de874bbe14abdaebea70b8bfe147c` | `bab6b40ce900b0f39d830ed86e76b81beabce4fcc71d3c329b73b7346a2efc2f` | `94edfa04…` (batch 1: `m2/E8F-s1/cal/`); 1.00520 / 1.01111 / 1.05724 | `97728822711cc891b3530211c4a16f021005e3291afb0e5b175e08d925b621ff` (batch 2: `m2/E8F-s1/cal698-16k/`); 1.00696 / 1.01286 / 1.05621 |
| E8F-s2 (20260927) | `16c0929ac0df649d8223483e1adf419d78a647ed` `m2/E8F-s2/checkpoint-0001520` | 603/700, .854907 | `b248e25dd4fcb123127d27a3f937ebccf715e501873ae558bf4b3de96f1c82c2` | `4501784c9c010779722668cbdb69f5f0b0b54d24ce46336c7d3e3952affcfa11` | `50f3931967794203a84413fab216507efbe36f312adfde110b28c83d3739e15c` | `175064b2b492dbcf05695433a457edf1c03164fc5ffe2aaf58b7b78b4717977c`; 1.37477 / 1.02621 / 0.28753 | `f4a628af5169b9d4d75082530cebcc62bc3ca23c5a1f41d9d1415ce31c2f5417`; 1.37791 / 1.02770 / 0.28775 |
| E8F-s3 (20260928) | `16c0929ac0df649d8223483e1adf419d78a647ed` `m2/E8F-s3/checkpoint-0002041` | 618/700, .872963 | `7805fd7c220f1abade079795e1423d96bb90d7d3d08aa0f4fa5d6935338b2ed3` | `74f8cda9ea2003434b5c800c13b6c6c4a030247c02816a8c3f80ee7b5292d3da` | `fba71d97fbfec204188df5382717d628846b44d789c271f639cc3190d1ae7817` | `7a50147417d0779e83a2d5309b53069eb9ed3f36d72fd9e6fffaeea1784d95e6`; 1.20529 / 1.21404 / 0.05 (Score at the fit's lower bound) | `aeaaeca33221e2eaed00b082204a963f88dcbc5548dfb9bab26195cecbffc486`; 1.20537 / 1.21892 / 0.05 |

Per-seed post-key runs (8,192 tokens, CAL700), node A: `m2-E8F-s1-nodeA` 49.777
(REPORT `83ec5a75…`), `m2-E8F-s2-nodeA` 47.360 (`ffffacb3…`), `m2-E8F-s3-nodeA`
41.084 (`45bd78ca…`), all under `/data/dev2/runs/dec/formal/m2/`. Batch 1 has 37
files verified on node A, batch 2 has 37; the per-file SHA-256 lists are
`/data/dev2/runs/dec/m2/hf-staging/batch{1,2}.sha256` on node B and
`/data/dev2/runs/dec/m2/batch{1,2}.nodeB.sha256` on node A.

## Recipe E8F (as trained; read back from every seed's `provenance.json`)

- **Start:** own `llm-semantic-router/Decision-1.0-Eos-0.8B@363c4a5e`
  (Apache-2.0; itself from `Qwen/Qwen3.5-0.8B@2fc06364`, Apache-2.0). Decision
  head kept (1.0 typed candidate head), `head_dim` 256.
- **Training:** full fine-tuning (backbone + head, FP32 weights and AdamW, BF16
  autocast backbone, FP32 head/loss), backbone lr 1e-5, head lr 1e-4, weight
  decay 0.01, 5% warmup then cosine to 10%, one epoch, objective CE + 0.5 Brier,
  no type balancing, no residual readouts, gradient checkpointing, token-budget
  micro-batches (≤ 32,768 tokens and ≤ 64 rows; ≥ 64 rows per update; 2,030 /
  2,026 / 2,041 updates), max length 8,192 (no truncation), SELECT700
  (`32a4352d…`) matrix-v1 selection over 8 evenly spaced checkpoints (earliest
  tie-break). Trainer code identical for the three seeds (`train_dec.py`
  `357e43ca…`, shared `loss.py` `c6fc39ed…`, `decision_model.py` `1ab1e49b…`);
  mirrors of `163d40dab` (s1) and `f316bad37` (s2, s3); image `dbe5f32b…`,
  node B GPU3 / GPU0 / GPU2, FLA + causal-conv1d kernels and a persisted
  autotune cache enforced.
- **Teacher targets: NONE.** Every seed's training contract records
  `teacher_kl_weight = 0.0`, `teacher_rows = 0`, no `--teacher` file; none of the
  162,777 mixture rows carries a `teacher_probs` field (checked on the node file).
  No own-1.0, Lux, AutoJev, Jev or other model outputs were used as targets.
- **Labels:** hard gold labels only — publisher-shipped human labels and
  program oracles of deterministic generators. A7 (own 1.0 corpora):
  program-oracle labels of the 1.0 generators plus human labels (Cosmos QA /
  SNLI / SQuAD 2.0 / MultiNLI non-fiction / BANKING77 / CLINC150); the A7
  inventory found no LLM-written states or labels, no own-model teacher targets
  and no Jev outputs. v1 arms: human labels (A1 ABCD + SGD, A3 MuSiQue twins,
  A5 KLUE YNAT + JGLUE, A6h ordinal Score sources) and programmatic verifiable
  generators (A2, A4v2h, A6g).
- **Mixture** `d1dc33fcb7a49fa9df9519545b1b0debeac38f86cbd760b481eb37334c9bd6f6`
  (spec `m2-full-a7-v1.json` `ba4eb52b…`; 162,777 rows / 138,393,350 native
  tokens; Choice / Noul / Score 105,657 / 35,400 / 21,720), built from private
  `llm-semantic-router/decision-2.0-training-data` revisions `39a120ca…` (A7
  v2, `v2/a7/registry.json` `9cb928be…`) and `5c0255ed…` (v1 arms,
  `v2/registry.json` `53e364f5…`):

  | Component | Rows | Tokens | Notes |
  | --- | ---: | ---: | --- |
  | A0s | 6,547 | 3,977,352 | the 752 rows of `natural_cosmos_qa` / `natural_squad2_answerability` **excluded**; rule 7d renumbered 117 rows (= pk1 A0s `d8eae3e4…` minus those families, row for row) |
  | A1 / A2 / A3 / A4v2h / A5 / A6g / A6h | 4,434 / 5,453 / 1,586 / 1,792 / 2,298 / 4,905 / 5,605 | 12,958,310 | A6h: 17 duplicate inputs dropped |
  | A7 (all six sub-arms, v2) | 130,157 | 121,457,688 | 5,339 excluded-family rows dropped; 2,817 duplicates of A0s rows dropped |

- **Licences:** training-data terms per the data registries
  (`v2/data/records/license-registry-v1.json`,
  `v2/data/a7/license-registry-a7-v2.json`): project-generated rows plus
  permissive human-labelled sources whose terms allow training and private
  redistribution (attribution / share-alike where stated). Weights: own Eos 1.0
  (Apache-2.0) from `Qwen/Qwen3.5-0.8B` (Apache-2.0).

## Disclosures for the card (post-key same-panel unless stated)

1. **Typed Score −13 items** (typed FINAL Score 107/400 vs Eos 1.0 120; resource
   ledger); Score is weak for both.
2. **Human transfer H −.021** (CSS15 median macro-F1 .4401 vs .4612): the gain
   is typed reasoning (Choice 529 vs 315, Noul 611 vs 410).
3. **Multilingual (`mlx-diag`, development diagnostic):** type macro 65.2 vs
   66.5; English 67.1 vs 67.9; non-English Noul 54.2 vs 59.2 (PAWS-X Noul
   −5.0); non-English Choice 68.7 vs 68.5, Score 71.9 vs 71.0; weakest language
   ko 55.0 vs 56.0.
4. **Seed dependence:** single seeds score 49.78 / 47.36 / 41.08 (seed 3's human
   transfer collapsed, H .3635); the soup (chosen on development data by the
   preregistered rule) is above every seed.
5. **A7 revision:** E8F trained on A7 v2 (`39a120ca`). A7 v3 (`3a4efc77`, the
   A7 release-candidate revision) later removed 64 groups after scanning
   against the stricter PI-v3 additions. 19 of the removed TRAIN rows are in
   E8F's mixture (A7m `stage4_natural_nli` 9, A7i `clinc_train` 6 and
   `stage4_replay_clinc_train` 4; 19 groups). They were flagged against the
   development held-out slices of the v1 arms (A3 / A1 AHO). One A7m group was
   flagged lexically against `mlx-diag`, and the public audit does not say
   whether it is one of the 9 TRAIN rows or one of the 3 A7m AHO rows. JevArena v3,
   public 231 and the C1 panels are unaffected: 0 groups ≥ 0.93 against the 38
   PI-v2 evaluation roles, and the C1 v1.1 independence check covered the
   training data. The id list is `/data/dev2/runs/dec/m3/e8f-a7v3-removed-in-mixture.ids.json`
   on node B, so the eval track can confirm the one possible `mlx-diag` item.
6. The two A7h shortcut families (and the matching 752 A0s rows) were **not**
   in E8F's training data.
