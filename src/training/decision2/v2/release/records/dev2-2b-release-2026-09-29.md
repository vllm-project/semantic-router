# DEV2.0-2B: private release (2026-09-29)

## Released, 2026-09-29 ≈04:45 UTC+8 — revision `5ad3e9a3`, in the private "Decision 2.0" collection

The coordinator finalized the release after the eval track's overlap-effect check (`v2/eval/records/m5-overlap-effects-2026-09-29.md`,
integration `cae64f4e8`). Removing the 84 flagged items moves v3 by at most 0.05 for every model, and there is no
contamination signature.

- **Revision `5ad3e9a3cc4865ce0360f4ecce2b345020bfdb38`** of private `llm-semantic-router/DEV2.0-2B` (now `main`).
  Manifest `6c4e1885…`. Against the verified `b2c5d7ea`, only `README.md` and `MODEL_MANIFEST.json` differ; all three
  safetensors files are identical ([readback](dev2-2b-release-2026-09-29/final/extra/final-readback.json)).
- **Final decision** [`DEV2.0-2B.decision.json`](dev2-2b-release-2026-09-29/DEV2.0-2B.decision.json)
  `de59a6c71e35b7cbdf32be18d72436a6b76aebc3e1adee84e068645bdd7cc85c`.
  - It is `status: final`, decided by the coordinator under the full-autonomy mandate.
  - It names report `08745acf…` and paired file `bca44e5f…`, and supersedes the drafts `31ae8a98…` and `fde8c87e…`.
  - Gate evidence: +7.66 [+3.26, +10.81] vs the Sol 1.0 16K control; human transfer not below Decider 2B or This-That
    1.2; no type collapsed; T = 1; Apache-2.0; the overlap result.
  - `gate.json` `8dbf4c87…` binds it to `5ad3e9a3…`.
- **Card.** The evaluation-familiarity item now ends: "Those rows touch 19 of the panel's items (18 media_ideology, 1
  tropes); rescored without them, or counting all 19 as errors, post-key v3 stays 53.44 and none of this card's
  comparisons change." The five eval-gate disclosures were already on the card, and the C1 placeholder stays until event 3.
- **Verification.** `release.sh --upload --collect` passed all 17 steps from mirror `33de83cea` on node A GPU5 (shared
  lease), with a fresh copy of the frozen scored cache (digest equal). [Receipts](dev2-2b-release-2026-09-29/final/)
  - Examples `1425b445…`, bit-identical across three processes and as before; card example reproduced.
  - Parity on 600 prompts before and after upload: 0 category changes.
  - Re-hash passed; readback private.
  - HTTP 12/12 with anonymous 401; links 15/15.
  - Collection private, holding exactly DEV2.0-0.6B, DEV2.0-0.8B and DEV2.0-2B.
- **GPU:** 0.063 GPU-hours. The DEV2.0-0.8B card-only revision followed in the same lease (0.047; see its record).
- **Storage cleanup** ([receipt](dev2-2b-release-2026-09-29/storage/free-duplicates-20260928T204746Z.receipt.json),
  [script](dev2-2b-release-2026-09-29/ops/free-released-duplicates.py.txt)), run after both publications. LFS objects
  were deleted with `permanently_delete_lfs_files`, so commit history is unchanged. An object was deleted only if its
  Hub OID equalled a fresh re-hash of the node copy, a file at the released repository's `main` had the same SHA-256,
  and no path outside the folder used it.

  | Staging copy | Released duplicate | Objects deleted | Bytes |
  | --- | --- | ---: | ---: |
  | `dev2-dec-staging` `m2/E8F-soup/` | DEV2.0-0.8B | 2 | 3,013,820,512 |
  | `dev2-dec-staging` `m3/S2T-soup/` | DEV2.0-2B | 3 | 7,535,759,320 |
  | `dev2-release-staging-06bm4` | DEV2.0-0.6B | 3 | 2,399,869,346 |
  | `dev2-release-staging` (Kai1 dry run) | Decision-1.0-Kai-0.6B | 5 (kept `assets/DEV2.0-0.6B-owl-banner.png`, not in Kai 1.0) | 2,322,051,120 |

  - **Total:** 15,271,500,298 bytes.
  - **Org private storage:** 99.17 GB before the cleanup (other tracks' uploads since 04:10; `dev2-9b-staging` alone is
    31.78 GB) and 83.90 GB after, under the 100 GB cap.
  - **Still served:** every released repository serves all its weight files (DEV2.0-2B 3/3, 0.8B 2/2, 0.6B 2/2,
    Kai 1.0 4/4).
  - **Not touched:** HOLD staging (4B N4LKr, 9B DW, 27B, 0.6B Z), the 0.8B seeds, datasets and eval artifacts.

**Status: verified private revision, stopped before `--collect`.** Private
`llm-semantic-router/DEV2.0-2B@b2c5d7eac4ef24648cbf9c23c26421f0960ca542` (manifest
**`63c61883ba52acc61b3782a9eed3fadde2d69fa4d1dd7078bcb46b8dd26f019d`**, 33 files, 7,556,619,709 bytes, 1,883,930,944
loaded parameters, temperature 1). Every `release.sh` step passed. The repository is not in the "Decision 2.0"
collection. Verified draft decision [`DEV2.0-2B.decision.draft.json`](dev2-2b-release-2026-09-29/DEV2.0-2B.decision.draft.json)
**`31ae8a9890bc3daf4d0dbd3cb8844b4a0164449d709fde69a441791d9701b536`**; the coordinator writes the final decision and runs
the collection add (§8).

- **Calibration:** CAL698 fails the 23:15 development-panel rule, so the package ships T = 1 (§2).
- **New disclosure:** the M3b XL r2 rescreen flags 45 S2T training rows that share long word sequences with CSS15
  items (§6). It is on the card.
- **Storage:** 59.78 GB of the org's 100 GB private tier were in use before the upload and 67.37 GB after; nothing was
  deleted (§7).
- **Shared-module change:** `card.py` now passes a peer's spec `repo_id` to the chart generator's fill-only model-id
  override (`0ca01698f`, with a test), because Decider 2B's adopted report names no model id (§5).

## 1. Candidate and scored identity

| Item | Value |
| --- | --- |
| Candidate | decoder S2T: full fine-tuning of Decision 1.0 Sol with own-Sol soft targets as a trust region, uniform soup of three seeds ([decoder record](../../dec/records/dec-m3-2b-candidate-2026-09-28.md), results `204074333`, candidate `48965e26d`) |
| Staged checkpoint | private `llm-semantic-router/dev2-dec-staging@545a6784175ce7a62319abeaec9502d775e586e2`, `m3/S2T-soup/`; node A copy `/data/dev2/runs/dec/m3/formal-candidates/staging-545a6784/`, re-verified 11/11 against node B's hash list before the T = 1 derivation |
| Identity | `model_sha256` `073bd1f2fe62e39fe993f57006bab17ece50a7e6fc7c5ee72107fefddeb81da4`; decision head `33b6541b…` (equals the soup head in the decoder record) |
| Loaded parameters | 1,883,930,944 (backbone 1,881,825,088 + head 2,105,856), from the safetensors headers: tier **2B** |
| Direct weight origin | `llm-semantic-router/Decision-1.0-Sol-2B@ce0c018a28de16d6639b1cd203b761bf643b89e6` (Apache-2.0; its `decision_config.json` names `base_revision` `15852e8c…`) |
| Upstream | `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc` (post-trained; card `license: apache-2.0`, `base_model: Qwen/Qwen3.5-2B-Base`); its LICENSE `bbedc3fd…` is byte-identical to Sol 1.0's `LICENSE` and `QWEN-LICENSE` |
| Profile / limit | `qwen-full`, 16,384 tokens (training inputs ≤ 8,192) |
| Scored run (CAL698) | `/data/dev2/runs/dec/formal/m3/m3-S2T-soup-nodeA` (REPORT `c04ca52a…`, SEAL `71e64188…`), image `decision20-train-fast:host2` `sha256:f83b1d10…`, runtime env `HIP_FORCE_DEV_KERNARG=1`, autotune cache `m3-S2T-soup-nodeA-triton` (1,051 files, digest `abdfd687…`); mlx-diag run `m3-S2T-soup-mlx` |
| T = 1 bindings | [derived run](dev2-2b-release-2026-09-29/derived-run/): REPORT `08745acf…`, SEAL `91a3df05…`; predictions typed-final `177e62c1…`, css15 `2cdcf415…`, public231 `f3af4853…`, mlx-diag `736a984a…`; paired vs the Sol 1.0 16K control `bca44e5f…`; mlx-diag score `63560f76…` |

Same-panel results at T = 1 (answers identical to the CAL698 run):

| Model | v3 | T | H | Choice / Noul / Score | Public 231 (easy / standard / hard) | Typed Brier / ECE |
| --- | ---: | ---: | ---: | --- | --- | --- |
| DEV2.0-2B | **53.437** | .5434 | .5255 | 445 / 567 / 175 | 171 (48 / 66 / 57) | .271 / .103 |
| Decider 2B | 49.499 | .5831 | .4202 | 545 / 497 / 133 | 175 (48 / 64 / 63) | .259 / .063 |
| This-That 1.2 | 46.112 | .5250 | .4050 | 529 / 431 / 79 | 147 (48 / 64 / 35) | .344 / .260 |
| Decision 1.0 Sol, 16K control | 45.781 | .4253 | .4928 | 374 / 438 / 155 | 160 (48 / 66 / 46) | .345 / .249 |
| Bosun v3.1 1.7B | 42.117 | .4669 | .3799 | 415 / 493 / 99 | 151 (46 / 62 / 43) | .312 / .127 |

Paired (joint bootstrap, 5,000 replicates; the T = 1 derivation reproduces the decoder and eval pairs exactly): vs the
Sol 1.0 16K control +7.66 [+3.26, +10.81] (T +.118 [+.094, +.143], H +.033 [−.044, +.092]); vs adopted Sol1 +7.86
[+3.36, +10.80]; vs Decider 2B +3.94 [−2.48, +5.67] (T −.040 [−.070, −.009], H +.105 [−.008, +.132]); vs This-That 1.2
+7.33 [−2.00, +13.60].

## 2. Calibration (coordinator rule 23:15)

The decoder track's CAL698 fit for this identity (16K file `d73e9ce4…`, identical temperatures and logits hash to the
8K development fit `f87a162b…`: Choice 0.74244, Noul 0.81608, Score 0.15391) was checked on the development panels
with `v2/release/dev_calibration.py` on node B (CPU; mirror `58f907b60`). It undid the 8K fit on the stored soup
development predictions (typed-DEV `c1c11d24…`, 1,600 slots; CSS pilot `15a9eef6…`, 1,430 slots), applied CAL698 and
scored both ([receipt](dev2-2b-release-2026-09-29/devcal/dev2-2b.json) `56de813b…`, [command](dev2-2b-release-2026-09-29/devcal/cmd.sh.txt)):

| Development metric | Raw (T = 1) | CAL698 |
| --- | ---: | ---: |
| Typed DEV Brier / ECE-10 | 0.267 / 0.095 | **0.289 / 0.135** (worse) |
| CSS pilot median task Brier-sum / ECE-15 | 0.748 / 0.080 | **0.757** (worse) / 0.079 |
| Typed DEV by type, Brier / ECE-10 | Choice .263 / .101, Noul .314 / .214, Score .229 / .131 | Choice .256 / .043, Noul .328 / .249, Score .315 / .290 |

Three of the four criteria worsen and no answer changes: **CAL698 rejected; the package ships T = 1** with the card
sentence used for 0.6B and 0.8B. For the record only (formal panels are not a decision input): T = 1 typed Brier / ECE
0.271 / 0.103 vs 0.300 / 0.174 with CAL698; CSS15 median Brier-sum / ECE 0.574 / 0.119 vs 0.608 / 0.137; public 231
0.178 / 0.076 vs 0.181 / 0.081.

The sealed CAL698 formal run was returned to T = 1 on node A (CPU; [script](dev2-2b-release-2026-09-29/ops/derive-t1.sh.txt)):
`retemper_predictions --undo` on typed-final 1,600, css15 6,547, public231 231 and mlx-diag 2,275 rows (0 answer changes
each), after checking the sealed prediction hashes and the staging copy; then `same_panel adopt` (reason in `ADOPT.json`),
`seal`, `report` (tier 2B, parameters from the safetensors headers), `compare` against the same four comparator runs, and
`multilingual_panel score`.

## 3. Package

Built by `v2/release/build.py` from mirror `336ac2a5a` (spec [`dev2-2b-release.json`](../specs/dev2-2b-release.json),
`spec_sha256` `cd32e3ee…`). 33 files; licence `apache-2.0` (Sol 1.0 and Qwen3.5-2B are Apache-2.0; training-data
licences credited in `ATTRIBUTIONS.md`); no `calibration.json`. All 9 staged checkpoint files are byte-identical in the
package ([manifest](dev2-2b-release-2026-09-29/release/package-text/MODEL_MANIFEST.json)).

| File | SHA-256 |
| --- | --- |
| `backbone/model-00001-of-00002.safetensors` | `1f083a66d8cdcd02887452b2801efae6af41a6e9529bdc37c962b994721dca08` |
| `backbone/model-00002-of-00002.safetensors` | `3a3f291be8d6ac079f1f737ec89092ed3947abd2cd5fd342823c91ce1084af8b` |
| `backbone/model.safetensors.index.json` | `12b12d9183d20062b4b5ab51d0d0a666ada0a2c53b8bd1b078f3456d5c3db781` |
| `backbone/config.json` | `071e97d8291168ba712237744c9a60e733acce554e6164322d22550dc96de19b` |
| `decision_head.safetensors` | `33b6541bb6636677eb91a4d8e06acd4db81152b11097088840796f11d49c707a` |
| `decision_config.json` | `0b6c3429ee06032739d7386659c332ed7bb62d0b96ccddfcad4fad999e22cd4f` |
| `tokenizer.json` | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| `tokenizer_config.json` | `bee8eba30f0eb4af73c0fe2cd06d0f89b657d7819941c438157ec42f7c80ea87` |
| `chat_template.jinja` | `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80` |
| `config.json` (root map) | `3d08526e2defacc65e64bfa6fe575d77bed7b320513edac99000ca5507a3e20a` |
| `README.md` | `d399add935bf587dfef2e8ca4c931f1b7e0da4eb51295b11f40cf9db026da3d9` |
| `LICENSE` / `LICENSES/Qwen3.5-2B-LICENSE.txt` | `bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a` |

## 4. Verification (node A GPU5, shared lease `owner.release`, 19:36:04–19:47:24 UTC)

Command: [launch-command.txt](dev2-2b-release-2026-09-29/release/extra/launch-command.txt) (`--gpu 5 --track release
--shared-lease release --site /opt/decision-fla --require-kernels --env HIP_FORCE_DEV_KERNARG=1`, a fresh copy of the
scored run's frozen autotune cache (digest equal to the frozen cache before the run), full-panel `--parity` on four panels,
`--upload`, no `--collect`). [Receipts](dev2-2b-release-2026-09-29/release/receipts/) · [log](dev2-2b-release-2026-09-29/release/extra/release.log.txt).

| Step | Evidence |
| --- | --- |
| build | manifest `63c61883…`; identity `073bd1f2…` recomputed; scored adapter sources equal the vendored runtime; build-time draft `fde8c87e…` |
| pre-a / pre-b / repeat-pre | six native System One requests (support routing, policy check, structured state, multilingual, null description, over-budget) with Choice, Noul and Score questions, in two containers: answers `1425b445…`, bit-identical; FLA gated-delta and causal-conv1d kernels bound |
| card-pre | the README's Python example, executed as written: 3 slots, drift 0 vs pre-a |
| parity-pre | typed-final 1,600 prompts / 2,000 slots, css15 6,547, public231 231, mlx-diag 2,275 against the T = 1 bindings: 0 category changes, 0 missing, max drift 8.9e-16 / 3.3e-16 / 2.2e-16 / 6.7e-16 |
| ensure / upload | private repository created; fixed uploader (each revision is exactly the package); 33 files → `b2c5d7ea…` |
| download / tree | real `hf download` at the revision into a fresh `HF_HUB_CACHE` (7,556,619,709 bytes); re-hash 33/33, nothing missing or extra |
| post / repeat-post / card-post | on the download: answers `1425b445…` (three processes identical), card example reproduced |
| parity-post | same 10,653 prompts on the download: 0 category changes, same drift |
| readback | private; every remote hash matches; card metadata `apache-2.0`, base model Sol 1.0; no card problems; collection "Decision 2.0" private with 2 items, this repo not in it |
| HTTP / links | `hub_card_http_check`: 12/12 card targets, anonymous model API / README / page refused (401); `hub_links`: 15/15 ([card-http](dev2-2b-release-2026-09-29/release/extra/card-http.json), [links](dev2-2b-release-2026-09-29/release/extra/hub-links.json)) |

The run wrote 33 new autotune entries into its cache copy (1,051 → 1,084 files); the frozen cache is unchanged.

## 5. Card

The 1.0 product design ([README](dev2-2b-release-2026-09-29/release/package-text/README.md)): DEV2.0-2B owl banner (Sol
mosaic owl, `94008faf…`), uses table, same-panel table (v3 with T / H, Choice / Noul / Score, public 231 easy / standard /
hard, mlx-diag non-English Choice / Noul, typed Brier / ECE), v3 rank, model×task and public-231 rank charts (no Pareto),
the automatic tradeoffs table versus Decision 1.0 Sol, the System One example, model details, training and limits.

- **Rows:** DEV2.0-2B, Decider 2B, This-That 1.2, Decision 1.0 Sol (the stricter same-renderer 16K control; the note
  gives the adopted 45.58) and Bosun v3.1 1.7B, all in the 2B roster and card-eligible. Decider 2B (Apache-2.0) and
  This-That 1.2 (MIT) declare licences in card metadata only, and the card says so. The mlx-diag Score (XNLI) part is
  not shown.
- **Disclosures:**
  - Weight origin (own Sol 1.0 → full fine-tune; Qwen3.5-2B lineage, Apache-2.0).
  - Teacher = own Sol 1.0 only.
  - Recipe, three seeds and uniform soup; training data by family with licences, from the program licence registries
    for all 44 sources of the 56,198-row mixture.
  - Not used: Jev, third-party decision models, the Cosmos QA / SQuAD 2.0 answerability families (both in Sol 1.0's own
    history, credited through its attributions), and the mlx-diag sources.
  - Regressions: CSS15 mrf −.094, wiki_corpus −.045, flute −.032 plus the full tradeoffs table; mlx-diag level
    overall, Korean Noul .59 vs .63, non-English Noul −2.2.
  - The eval gate disclosures (typed accuracy below Decider 2B −0.040 [−0.070, −0.009]; Choice weakest vs peers; margin
    over Decider not significant; gain over Sol typed only; Score level 0 recall .03).
  - Evaluation familiarity (§6), seed dependence, CPU not verified and tested hardware.
- **C1:** a marked placeholder line: "Independent sealed confirmation (JevArena-C1): PLACEHOLDER — to be added after the
  final sealed scoring event."
- **Korean note:** the decoder's Korean .59 vs .63 is the per-language mean, and Korean exists only in the
  PAWS-X Noul part, so it is card-eligible as a Noul figure.
- **Build fix:** the first build check failed with "Decider 2B: no licence class for None". The adopted Decider report
  names no model id, and `card.py` did not pass the spec's `repo_id` to the chart generator. `0ca01698f` fills a
  missing id from the spec entry through the generator's existing fill-only override. New test:
  `v2/release/tests/test_card_peer_model_id.py`; 47/47 release tests pass.

## 6. Evaluation-familiarity check against the M3b XL r2 rescreen

The data track's rescreen (`v2/data/records/m3b-xl-r2-2026-09-28.md`, private receipt `c01fe566…` on node A) landed
after S2T was trained on the r1-era `m3-v2m-ret` mixture. Its ids (id, group, source, type only; `efaf89d8…`) were
intersected with the flagged groups ([script](dev2-2b-release-2026-09-29/ops/s2t-overlap.py.txt),
[summary](dev2-2b-release-2026-09-29/overlap/s2t-overlap.summary.json)).

- **Coverage:** 55,086 of 56,198 rows are in the rescreen index. The remaining 1,112 are 1,111 program-generated A7
  `dec10:generated_stage4_v2` rows and one MTOP row.
- **Evaluation roles** (excluded in any new recipe; the methods are N n-gram and L long-window, no exact copies):
  - 57 rows in 24 groups (H3 15, H6 4, E11 3, H1 2).
  - CSS15: 45 rows in 18 groups (MuSiQue 30, SQuAD 2.0 8, QuAC 4, HotpotQA 2, CommonsenseQA 1).
  - Decision Bench v4: 15 rows in 7 groups.
  - None on the typed panels, the CSS pilot, JevBench public 231 or mlx-diag.
- **Disclosed roles:**
  - SELECT700: 252 rows in 198 groups. Checkpoint selection is familiar.
  - CAL700: 276 rows in 220 groups. This does not matter for the package, which ships T = 1.
  - v1 / A7 held-out development slices.
- **Card:** the limits state the CSS15 familiarity. **Coordinator / eval:** the H axis of every S2T comparison includes
  these 18 CSS15 groups. This check was not run for DEV2.0-0.8B or DEV2.0-0.6B; their v1-arm data (V1:A3) also has CSS15
  hits in the rescreen.
- **Draft wording:** the coordinator's 03:10 decision (after this check) has the eval track quantify the effect from
  stored predictions before the card wording is decided. The rescreen's excluded groups touch 82 CSS15 items, 73 of
  them `media_ideology`, through shared topical phrases. The card's familiarity sentence is therefore a draft; any
  change is a card-only edit to `card.text.limitations` before the collection add.

## 7. Hugging Face storage (step 0)

The org has no paid plan (`canPay: false`), so the free-organization private tier is 100 GB (Hub docs). The Hub's own
per-repository `usedStorage` was summed over all 125 org repositories ([probe](dev2-2b-release-2026-09-29/ops/hf-storage-probe.py.txt)):

| Moment | Private total | Largest Decision 2.0 repositories |
| --- | ---: | --- |
| before the upload | **59.78 GB** | dev2-dec-staging 36.71, decision-2.0-training-data 3.18, DEV2.0-0.8B 3.03, DEV2.0-0.6B 2.40, dev2-staging-06bm5-z 2.40, dev2-release-staging-06bm4 2.40, dev2-release-staging 2.32 |
| after the upload | **67.37 GB** | DEV2.0-2B 7.56 (charged in full, although its LFS objects equal the staging copy) |

- **Action:** the 7.6 GB package fit, so nothing was deleted. Headroom is now about 32.6 GB.
- **Next eligible deletion:** once DEV2.0-2B is released, the storage policy allows deleting its staging copy,
  `dev2-dec-staging` `m3/S2T-soup` (about 7.5 GB). It has a node copy and a hash list on both nodes. This is the decoder
  track's or coordinator's call.

## 8. Draft decision and finalization

- **Build-time binding draft:** [`DEV2.0-2B.decision.build-draft.json`](dev2-2b-release-2026-09-29/DEV2.0-2B.decision.build-draft.json)
  `fde8c87e…`, the spec's current `gate_receipt`.
- **Verified draft:** [`DEV2.0-2B.decision.draft.json`](dev2-2b-release-2026-09-29/DEV2.0-2B.decision.draft.json)
  `31ae8a98…`, also at `/data/dev2/runs/release/decisions/`. `gate.check` accepts it as a draft; a final seal refuses
  it.

To finalize (coordinator):

1. Write `/data/dev2/runs/release/decisions/DEV2.0-2B.decision.json` with `status: final`, `decided_by`, and the
   draft's identity `073bd1f2…`, `report_sha256` `08745acf…` and `paired_sha256` `bca44e5f…`.
2. Point `specs/dev2-2b-release.json` `gate_receipt` at that file. Edit `card.text.confirmation` only if the C1 wording
   should change before event 3.
3. Commit, push and mirror.
4. Rerun `extra/launch-command.txt` with the new mirror and `--upload --collect`, not `--already-collected`. A new
   revision results, in which only `builder.source_commit` in `MODEL_MANIFEST.json` changes (plus the README if the C1
   text changes).
5. After C1 event 3: a card-only revision with `--upload --collect --already-collected`.

## 9. GPU-hours and commits

- **GPU:** 0.189 GPU-hours (680 s wall on node A GPU5; `RELEASE-RECEIPT.json`). The shared-lease entry was removed at
  the end; the decoder's owner entry was never touched.
- **CPU only, no GPU:**
  - development calibration (node B, 0.3 s);
  - T = 1 derivation (node A, 45 s);
  - build check;
  - overlap intersection;
  - storage probes and Hub checks.
- **Commits:**
  - `0ca01698f`: shared `card.py` fix and test.
  - `336ac2a5a`: spec, calibration decision and derived bindings (the release mirror).
  - This record's commit, then a merge into `xunzhuo/decision-2-training`.
