# GLiNER2.5-Decide native 0.6B continuation: preregistration

Status: **preregistered, no optimizer update yet**. This is a development pilot and does not authorize a Decision 2.0 model release or a JevArena rank. The existing source baseline is reused; no protected FINAL, CSS15, or authored release labels are opened.

## Source and rationale

- English [`fastino/GLiNER2.5-Decide`](https://huggingface.co/fastino/GLiNER2.5-Decide) weights revision `7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6`, model-safetensors SHA-256 `40a5a23ff860dc3dff426cecd1048cacdd29c648c96db209dad818e9686dc997`, measured **486,444,053** parameters. The model card calls it 340M, so all our parameter plots use the measured value. Its pinned `span` encoder has 512 positions. The model card marks Apache-2.0 and derives from GLiNER2 large.
- Training library [`fastino-ai/GLiNER2`](https://github.com/fastino-ai/GLiNER2) commit `55656fbfa01d3d4a77485e1a1eeeaf682990ccdf`, Apache-2.0. The published `GLiNER2Trainer` accepts classification records. The version's default training sampler drops/synthesizes classification labels, which is inappropriate for a one-of-K Decision target; this pilot disables those augmentations through an exact-schema processor. The trained checkpoint must be loaded through its original `Classifier` and scored by the existing `gliner25-native-exclusive-v3` Decision adapter.
- Source baseline already measured on the same development panels: typed DEV **652/1,600**, CSS pilot **561/1,430** with median task macro-F1 **.31035**, public JevBench **116/231** (easy48/48, standard46/72, hard22/111). These are development/public diagnostics, not an official JevBench ranking or sealed JevArena score. Source transfer is weak versus larger models and the native 512-position context causes invalid long items.

## Frozen data and construction

Use the existing rights-clean v2 TRAIN **7,455** and disjoint SELECT **700** only. Original file SHA-256 values are `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755` and `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`. The source dataset has already passed the prior rights/lineage audit; this pilot checks exact ID, group and canonical-input isolation again. It does not ingest calibration or benchmark gold.

Each flattened row is converted into the same native one-task `ClassificationSchema().single("decision", ...)` used by the inference adapter: state as input, question instruction as schema prompt, Choice/Score option descriptions as native label descriptions, Noul `true`/`false` mapped to native `yes`/`no`, and Score ordered by numeric level. Gold is inserted only as the native classification `true_label` in the private TRAIN file. Native processor token counts are computed before training; any row above 512 is excluded, with **no truncation**. This admitted 2,900 Choice, 2,345 Noul and 434 Score rows; 1,008/686/82 rows respectively exceeded the window. From admitted TRAIN, a fixed ID-hash draw with seed **20260927** takes **512 Choice + 360 Noul + 152 Score = 1,024** rows. Private converted TRAIN SHA-256: `e86598148a17bb263787371c818fe9b52bb253396c97a68f5b07fa921b25e642`.

Local implementation commit `0b533dae378843803cc65f0bca796b2270c93b81` on a separate vllm-sr worktree passed focused tests and `make check`. A mirrored source copy was verified before the read-only native-token preflight; adapter SHA-256 `5b52c74978206a5f0036b844933f71815da35fef729a175ed38c8783570e21e0`. Keep converted rows, model output, and individual predictions private.

## Fixed optimizer and decision rules

One full-backbone continuation arm, **64** optimizer updates, one epoch over 1,024 records, microbatch **2**, gradient accumulation **8** (effective batch 16), encoder LR `1e-5`, task LR `5e-5`, BF16 autocast, seed **20260927**, maximum native input **512** positions, strict error propagation, no invalid-sample skipping, no schema-label drop/synthesis or prompt omission, no W&B. Use the fixed **final step 64**, with no checkpoint selection on DEV/CSS/public panels and no CAL fitting. If BF16 or checkpoint save fails, record the failure; changing precision or recipe requires a new preregistration.

Before interpreting quality, verify final checkpoint completion, byte hashes and native reload; same weights and adapter must give identical categorical SELECT answers in a 32-row deterministic recheck. Evaluate pinned source and fixed final checkpoint on **the same full SELECT700**, counting all native overflows/invalids as wrong. The source is evaluated once for this isolated SELECT comparison; do not repeat existing larger panels.

Advance this checkpoint to broader development comparison only if all hold: SELECT total correct improves by **at least 21/700**, SELECT family-macro accuracy improves by **at least .03**, no Choice/Noul/Score type accuracy declines by more than **.05**, and invalid count does not rise. If the gate passes, score typed DEV1,600, CSS pilot1,430 and public JevBench231 with the exact existing native adapter and disclose every type/task/tier and invalid cause. For a credible 0.6B candidate, require typed DEV above **652/1,600**, CSS median task macro-F1 above **.31035**, and public231 above **116/231**; then audit multilingual transfer, calibration, paired robustness and sealed JevArena before any family release. A public subset win alone is not sufficient. If the SELECT gate fails, preserve the checkpoint and analyze the failure without treating it as 2.0.

The source has a strict short context. This pilot cannot solve long-input coverage by weight changes. A longer-context encoder architecture remains a separate, explicitly measured route if no broad gain emerges.

## Completed fixed-budget result

The source was evaluated once on SELECT700 under the same native Choice/Noul/Score mapping: **372/700**, family-macro **.38719**, all 700 valid. The preregistered full-backbone continuation completed exactly **64/64** optimizer updates in about **39.05 seconds** on the authorized GPU environment, and its full checkpoint saved with measured **486,444,053** parameters. Final weight SHA-256 is `cb909858779fe9c1ff61eab6d9ad7004a18e59ee060600a619e107e7810e33c4`. A separate native reload of the first deterministic 32 SELECT rows reproduced **32/32 categorical answers**. No precision or recipe change was made.

| Isolated SELECT700 | Source | Fixed step64 | Delta |
| --- | ---: | ---: | ---: |
| Total correct | 372 | 491 | +119 |
| Family-macro accuracy | .38719 | .56793 | +.18074 |
| Choice (320) | 163 | 207 | +44 |
| Noul (290) | 202 | 246 | +44 |
| Score (90) | 7 | 38 | +31 |
| Invalid | 0 | 0 | 0 |

The frozen SELECT gate passed, so the final step64 checkpoint was scored once on the predeclared broader development panels. Both source and candidate used the identical native inference adapter and the same original prompt files; no prompts were truncated and missing/overflow items count as wrong.

| Development panel | Source | Fixed step64 | Outcome |
| --- | ---: | ---: | --- |
| Typed DEV1,600 | **652** | 584 | −68; 1,600 valid in both |
| CSS three-task pilot1,430 | 561, median task macro-F1 **.31035** | 577, median macro-F1 .32581 | +16; 1,363 valid in both; these tasks' source families occur in TRAIN and this is **not** unseen-task transfer |
| JevBench public subset231 | 116 | 122 | +6; 175 valid in both, 56 native context overflows; this is **not** the official closed rank |

Typed per-type source→step64 is Choice **227→168/800**, Noul **208→208/400**, Score **217→208/400**. By independent 400-item families: attribute gate **206→164**, rule precedence **208→208**, set reconciliation **217→208**, transition table **21→4**. The existing 400-group, family-stratified paired bootstrap measured a typed family-macro delta (step64 minus source) of **−.0425**, 95% interval **[−.0631, −.0225]**. CSS pilot task correct counts rose only 149→157 discourse, 166→167 implicit hate, and 246→253 stance. The public subset paired changes were 9 wins and 3 losses, mostly within the 175 valid items; no new long-input coverage appeared.

The selected 1,024-row short-context TRAIN mix explains the direction: **463** human GoEmotions rows, **177** natural QA/NLI rows, **93** ordinal rows, but only **4** `stage4_replay_stage3_transition_set` rows. Longer structured `stage4_automaton`, dense-table and register examples were excluded by the 512-position admission check. The training objective can improve source-matched SELECT and short operational classification while forgetting or failing to learn the independent attribute/transition mechanisms. This is an evidence-backed data/window limitation, not a claim that all encoder continuation is ineffective.

**Decision: BLOCK_FOR_RELEASE.** The preregistered typed DEV gate (`>652/1,600`) failed materially, despite passing SELECT and slight CSS/public gains. Preserve this checkpoint only as a negative pilot; do not publish it as `dev-2.0-0.6b`, mix its public +6 with an official rank, or select it by CSS/public performance. The next defensible sub-0.6B experiment needs a longer-context encoder that actually admits the structured TRAIN families, with a new preregistered source-versus-trained protocol and a fully independent held-out transfer panel. Repeating more 512-position updates on this short sample is not supported by these results.

### Byte and source receipts

- Implementation signed commits: native TRAIN/SELECT `0b533dae378843803cc65f0bca796b2270c93b81`, candidate collector `9c2b55194`; the latter's SHA-256 is `07278a565f0504ebbb0ea88d0cb1db64b48447c97a839995ba246425d3a9ba85`. Full `make check` passed after each change.
- Private converted TRAIN manifest SHA-256 `9f6aea37b0d7c8cad11ad522b4f1e337c36846d2695183601254c483fa22bc8b`; source/candidate SELECT prediction SHA-256 `82c5c8c478789a5b7d958f5505b5b37fdd53366bbfc6e50c617b339d4c69badf` / `9dcc576eff40a90de09a4f4f708cefd2cd57bdbcfb7125eee6330d68ecde6eb4`; source/candidate SELECT score SHA-256 `dcf8badb9dcbc65f3c9fc9a5ff58416565adb83536ffb8a93415803bc1795b05` / `9efa9df2a75d81def7b12cf7eaa34b80e93b56ad33f26ce8a6f1f630441f72a9`.
- Candidate typed prediction/receipt/score SHA-256 `e8d4f5756b02c686cbe3357b132452c5a8bd7248d22f0f66742e4cbff7a37d24` / `91808c8d53dee62a30ebe2b5833245d34bfa772c57661325816fb11b1ff1e23f` / `91c25ba4dc00c43a6a981d9f2b68bebf266212b9a4c53d66286e154cdcff6d1f`.
- Candidate CSS prediction/receipt/score SHA-256 `6eab0af129faa908bcffcc2e095b5c49adc9437bf8dc78085326a7109440ed49` / `3c66061099b23adca8143f2bf331e8ede0f6fd27040e94605420e4e114fbfd9e` / `1de31f27d802421b93ed5ff88c1f4197fd2968cf9544d4a4248f8a5cd3266fdf`.
- Candidate public231 prediction/receipt/score SHA-256 `fd54f5cb3a9c0973a68a83bc4af6008b5cb09cbfa26d9498623241c3d97c4e40` / `a846325aa02eced87b840b6cb7a1ccc7cb802a8267aa4a216bc837f18d1ef58e` / `3e3eb0bfa9eb50738a34c9101941c819e53c3e868059c7010e7b3356a5f55437`. Group-paired typed report is retained privately.
