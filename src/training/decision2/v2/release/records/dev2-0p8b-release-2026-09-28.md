# DEV2.0-0.8B: private release build, upload and verification (2026-09-28)

## Card-only revision (overlap and A7 v3 quarantine disclosures), 2026-09-29 ≈04:45 UTC+8 — current revision `f458c34c`

The coordinator ordered this card-only revision after the eval track's overlap-effect check
(`v2/eval/records/m5-overlap-effects-2026-09-29.md`, integration `cae64f4e8`). It was run by the DEV2.0-2B release
worker. Weights, calibration (T = 1) and every answer and score are unchanged.

- **Revision `f458c34ccfb4a5d4d32babeda1570919adb1a3c8`** (now `main`). Manifest `15e67dda…`. Against `7d08d0e1`, only
  `README.md` and `MODEL_MANIFEST.json` differ; both safetensors files are identical.
- **Final decision** [`DEV2.0-0.8B.decision.card2.json`](dev2-0p8b-release-2026-09-28/DEV2.0-0.8B.decision.card2.json)
  `dd397e041e0d0063782a1d8289b6b81297bd6a5199063a9568b863b4563e5c0c`. It is final, decided by the coordinator under the
  full-autonomy mandate, and supersedes `fedb18fa…`. `gate.json` `ebbad1d5…` binds it to `f458c34c…`.
- **Card.** Two new limits, inserted before the CPU limit (spec [`dev2-0p8b-card2.json`](../specs/dev2-0p8b-card2.json)):
  - Evaluation familiarity: 33 training groups, 11 evaluation items (9 media_ideology, 1 wiki_corpus, 1 multilingual
    diagnostic); rescored without them or counting them all as errors, v3 moves by at most 0.06 and no comparison changes.
  - Training-data quarantine: 19 training rows were later quarantined in A7 v3 against development held-out slices; at
    most one could match a multilingual-diagnostic item.
- **Verification.** `release.sh --upload --collect --already-collected` passed all 17 steps from mirror `33de83cea` on
  node A GPU5 (shared lease), with a fresh copy of the frozen E8F autotune cache.
  - Examples bit-identical across three processes (`f0a72416…`, as before); card example reproduced.
  - Parity on 600 prompts before and after upload: 0 category changes.
  - Re-hash passed; readback private.
  - HTTP 12/12 with anonymous 401; links 14/14.
  - Collection "Decision 2.0" (private): DEV2.0-0.6B, DEV2.0-0.8B, DEV2.0-2B.
  - Receipts: [card2/](dev2-0p8b-release-2026-09-28/card2/).
- **GPU:** 0.047 GPU-hours.

## Calibration-only revision (T = 1), 2026-09-29 ≈00:12 UTC+8 — current revision `7d08d0e1`

The coordinator approved a calibration-only revision (23:55 UTC+8, full-autonomy mandate). Under the
23:15 calibration rule, CAL698 temperatures are adopted only if they do not worsen calibration on the
development panels. The released temperatures (Choice 1.123, Noul 1.071, Score 0.395) worsen typed-DEV
Brier / ECE-10 from 0.269 / 0.128 (raw) to 0.300 / 0.180; the CSS pilot improves only slightly
(0.834 / 0.127 to 0.821 / 0.109) ([receipt](dev2-0p6b-release-2026-09-28/devcal/dev2-0p8b.json)).
So DEV2.0-0.8B now ships T = 1. The weights and every answer are unchanged: v3 50.236, +7.69
[+3.65, +13.32] vs Eos 1.0, public 231 156, and JevArena-C1 event 1 stay as they were.

- **Revision `7d08d0e12082ee6810a4221063342cefe45b6ac3`** of private `llm-semantic-router/DEV2.0-0.8B`
  (now `main`). Manifest **`2a19a65a0141a8d71939bbf82025e299266db404cadd8b2bc712d1a992e98e7e`**, 30 files,
  no `calibration.json`, 753,446,208 loaded parameters.
- **Final decision** [`DEV2.0-0.8B.decision.t1.json`](dev2-0p8b-release-2026-09-28/DEV2.0-0.8B.decision.t1.json),
  SHA-256 **`fedb18fa3634b1874b6508c75c4607fc59f3dd63961c4b43e0698b6faf378684`**. It is `status: final`,
  decided by the coordinator under the full-autonomy mandate. It names report `1f5cf33d…` and paired
  file `fa62c29a…`, and supersedes `b2338c45…` (revision `0b631a85`). The run's `gate.json`
  (`13398d42…`) binds it to `7d08d0e1…` and manifest `2a19a65a…`; all six gate items pass.
- **Card.** Two README lines differ from `0b631a85`:
  - typed Brier / ECE goes from 0.277 / 0.150 to **0.254 / 0.103** (report `1f5cf33d…`);
  - the calibration line now reads: raw model probabilities (temperature 1); the CAL698 temperatures were
    evaluated and not adopted because they worsened calibration on out-of-distribution development data.

  Scores, the C1 line, tradeoffs and disclosures are unchanged. The evaluation page shows the same
  0.254 / 0.103. It also drops the note that the scored report counted only the backbone, because the
  adopted T = 1 report counts all 753,446,208 parameters.
- **Package = the verified T = 1 candidate.** It was rebuilt from the mirror of `6d7e7a148`. Every file
  equals the candidate verified at `a1f5c332…` except `MODEL_MANIFEST.json`, and there the only changed
  field is `builder.source_commit` (`89294e89f` → `6d7e7a148`)
  ([compare](dev2-0p8b-release-2026-09-28/t1/extra/candidate-compare.json)). The decision allows exactly
  that difference; a CPU build-only preflight showed it before any GPU or Hub step.
- **Verification.** One `release.sh ... --upload --collect --already-collected` run passed all 17 steps
  in 180 s ([command](dev2-0p8b-release-2026-09-28/t1/extra/launch-command.txt)). It ran on node A GPU5 via
  the shared-lease entry `owner.release` (16:09:19–16:12:54Z, then removed; the owner entry was not
  touched), with image `f83b1d10…`, `--site /opt/decision-fla --require-kernels` and a copy of the
  candidate run's autotune cache.
  - Native examples are bit-identical in three processes (two before upload, one from the download).
    The answers hash `f0a72416…` equals the candidate's. The card example reproduced before and after upload.
  - Parity against the derived T = 1 predictions on 600 prompts (typed 150, transfer 200, public 100,
    mlx-diag 150; same prompt, id and prediction hashes as the candidate run): 0 changes, max drift
    4.4e-16, before and after the real download.
  - A real `hf download` of `7d08d0e1…` with a fresh cache re-hashes 30/30 files equal to the manifest
    and to the pre-upload package.
  - **Weights identical to `0b631a85`.** The six model files (backbone safetensors and config, head,
    decision config, both tokenizer files) have equal SHA-256 in the download and in the old manifest,
    and equal remote LFS / blob ids at both revisions. Only `calibration.json` (removed), `config.json`,
    `README.md`, `evaluation/*` and `MODEL_MANIFEST.json` differ
    ([compare](dev2-0p8b-release-2026-09-28/t1/extra/revision-compare.json)).
  - Readback: repository private at `7d08d0e1…` (= `main`), 30/30 remote hashes, Hub card `apache-2.0`
    with base model Eos 1.0, no card problems. HTTP check: 12/12 card images and files at the revision;
    anonymous model API, README and page refused (401). `hub_links`: 14/14.
  - Collection "Decision 2.0": private, **exactly one item** (`llm-semantic-router/DEV2.0-0.8B`, no note).
    Collection items name a repository, not a revision, so the item now shows `7d08d0e1…` and needed no change.
- **Pipeline fix `bf9717167`.** The pre-collect readback required the repository to be absent from the
  collection, which fails for any later revision of a collected release. `release.sh --already-collected`
  (only with `--collect`) makes that readback require the item instead (tests
  `v2.release.tests.test_hub_readback`). Default behaviour is unchanged.
- **GPU-hours 0.050** (node A GPU5, 180 s). The CPU preflight build and the extra readbacks used no GPU.
- `0b631a85` (CAL698) is superseded and stays in the repository history.

Receipts: [`t1/`](dev2-0p8b-release-2026-09-28/t1/) (pipeline receipts, extra checks, package text,
launch command and log).

## Final release (2026-09-28 ≈22:30 UTC+8) — in the private "Decision 2.0" collection (superseded by the T = 1 revision above)

The coordinator decided to release (full-autonomy mandate). The spec's card got the
JevArena-C1 event 1 line (DEV2.0-0.8B 40.24 vs Eos 1.0 37.94, +2.31 [+0.29, +4.30]; Kev 39.06;
Intern-Decision 34.58; JPT not shown), the Kev licence note, and two added limits (C1 weak spots;
Score level 0 never predicted on typed FINAL). Final decision
[`DEV2.0-0.8B.decision.json`](dev2-0p8b-release-2026-09-28/DEV2.0-0.8B.decision.json)
(SHA-256 `b2338c459d56a7b22b09365363532658060a0a67f53d19ec2784e47463b1a75e`, status final).
`release.sh --upload --collect` from the mirror of `df779eb5d` on node A GPU5 (borrowed
14:21–14:30Z, lease restored; 551 s, 0.153 GPU-hours) passed all 17 steps:

- **Final revision `0b631a85c19fb573aee34fc68bb413271ebe89f4`**, manifest `0af27b1c…`
  (only `README.md` differs from the draft package; every weight, config, tokenizer and
  calibration hash is unchanged), 753,446,208 loaded parameters.
- Real download + 31/31 re-hash; examples bit-identical across processes and after download
  (`a061d0a3…`, as before); card example reproduced; full-panel parity 8,378 prompts, 0 changes,
  max drift 0.0, before and after download.
- `gate.json` (`fb989b53…`) binds the final decision to `0b631a85…` and manifest `0af27b1c…`,
  six items pass; `collect` added the repository; collected readback: repository private,
  collection "Decision 2.0" **private with exactly one item** (`llm-semantic-router/DEV2.0-0.8B`),
  31/31 remote hashes, no card problems. HTTP readback: 12/12 card images and links, anonymous
  access refused; the card shows the C1 line, Kev note and new limits, and no JPT number.

Receipts: [`final/`](dev2-0p8b-release-2026-09-28/final/). The sections below describe the
draft verification at `2667d883…`.

**Result: every pipeline step and all six mechanical gate items pass; stopped before
`--collect`.** The first Decision 2.0 release candidate (decoder recipe E8F, three-seed
soup) is in the **private** repository `llm-semantic-router/DEV2.0-0.8B` at revision
`2667d883c5941356a436ba38824b0453dd2fa7b6`. Nothing was added to the private "Decision 2.0"
collection (still 0 items). The coordinator finalizes the decision after the JevArena-C1
event; the draft is [`DEV2.0-0.8B.decision.draft.json`](dev2-0p8b-release-2026-09-28/DEV2.0-0.8B.decision.draft.json).
Receipts: [`dev2-0p8b-release-2026-09-28/`](dev2-0p8b-release-2026-09-28/).

## Candidate and scored identity

| Item | Value |
| --- | --- |
| Recipe | E8F: own Decision 1.0 Eos `363c4a5e` → full fine-tuning (no teacher targets) on the M2 full mixture (162,777 rows, 138.4M tokens, `d1dc33fc…`), seeds 20260926/27/28, uniform FP32 soup ([release-candidate record](../../dec/records/dec-m2-0p8b-release-candidate-2026-09-28.md)) |
| Checkpoint | private `llm-semantic-router/dev2-dec-staging@16c0929ac0df649d8223483e1adf419d78a647ed`, `m2/E8F-soup/checkpoint`; all 9 staging files under `m2/E8F-soup/` equal the node-A copy the scored run loaded (LFS SHA-256) |
| Identity | `model_sha256` `60356482ceeb669c4a97eb14dcfae5144b1b181f6c8b7a628ea5d02c86a6dd8b` |
| Calibration | CAL698 `frozen_checkpoint`, 16,384 tokens, `9f76867d5be618da315bd860aae689ed0cf2a2a20ea34d5bae485b011877e05c` (temperatures 1.12262 / 1.07053 / 0.39532); accepted by the shared loader change `0a399c1d9` |
| Scored run | `m2-E8F-soup-nodeA` (frozen runner, image `f83b1d10…`, node A GPU5): REPORT `e3e08cd1…`, SEAL `7b886cdc…`, predictions typed `119256bd…` / css15 `5ab525e8…` / public `870abd5f…` |
| Paired | vs adopted Eos 1.0: 50.236 − 42.547 = **+7.689 [+3.646, +13.316]** (`4489c0ea…`); vs same-limit 16K control: +7.875 [+3.688, +13.362] (`15140c11…`) |

## Package (`qwen-full`, `dev2-package/1`)

Built by `v2.release.build` from the exact mirror of `4d94d6fd6` (build receipt `4cfb79bc…`):
31 files, 3,034,575,568 bytes, `MODEL_MANIFEST.json` **`d10bd95c40a1417b8416d73fe7af9a9213d3c113804a6f2bc600f57cbf379594`**.
Parameters from safetensors headers: backbone 752,393,024 + decision head 1,053,184 =
**753,446,208 loaded** (the runtime asserts this count; name `DEV2.0-0.8B` follows the tier
rule). The scored REPORT counted the backbone only (752,393,024); the card shows the
asserted count and says so in `evaluation/EVALUATION.md`.

| Model file (exact scored bytes) | SHA-256 |
| --- | --- |
| `backbone/model.safetensors` (FP32) | `9db82b841878b2f259701e11fdccdfd4b54021b857934304de181e28379f5f1e` |
| `backbone/config.json` | `2b5d7e14da681afa0f2717f4e450c8a4cd05d41a7aff99231578d5aa3a6bad73` |
| `decision_head.safetensors` | `cf2a2ebdd0b007eca460d90889f129f80a75b2371be54ca1889f4bf15613af6a` |
| `decision_config.json` | `3503b7cb43a1e4994c27916bf35653612a517c249f2ed9458000904a3f8da700` |
| `tokenizer.json` | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| `tokenizer_config.json` | `bee8eba30f0eb4af73c0fe2cd06d0f89b657d7819941c438157ec42f7c80ea87` |
| `calibration.json` | `9f76867d5be618da315bd860aae689ed0cf2a2a20ea34d5bae485b011877e05c` |

These equal the scored predictions' `model_files_sha256`, so the fingerprint recomputed from
the downloaded bytes is the scored identity. The vendored inference sources equal the scored
adapter's `adapter_files_sha256` (build-time check). Every other file (runtime, card, charts,
licences) is listed in [`package-text/MODEL_MANIFEST.json`](dev2-0p8b-release-2026-09-28/package-text/MODEL_MANIFEST.json).
Licence metadata `apache-2.0`: every lineage component is Apache-2.0 (own weights, Decision 1.0
Eos `363c4a5e`, Qwen3.5-0.8B `2fc06364` text backbone and tokenizer); `LICENSE`,
`LICENSES/Qwen3.5-0.8B-LICENSE.txt`, and the unchanged Eos `NOTICE` are shipped.

## Verification (node A GPU5, image `f83b1d10…`, 530 s, 0.147 GPU-hours)

One `release.sh --gpu 5 --track release --upload` run from the mirror of `4d94d6fd6` with the
scored kernel runtime: `--site /opt/decision-fla --require-kernels`, a copy of the scored run's
persisted Triton autotune cache (1,051 files, tree `5e37a143…`; 1,084 after the run: 33 entries
for the example prompts' shapes), and full-panel `--parity`.

| Step | Evidence |
| --- | --- |
| Build | exact bytes, identity and calibration checked; 30 text files screened (now including the manifest); licence filter kept Eos 1.0, Intern-Decision-0.8B, Kev-0.8B and excluded JPT-0.8B (CC BY-NC) |
| Native examples, two processes | 5 System One requests (Choice with criteria, Noul with and without criteria, Score with 3 and 5 levels, structured state, Chinese state, null descriptions) valid; over-budget input answered with explicit `max_length_exceeded`; FLA gated-delta and causal-conv1d kernels bound, persisted cache set; 753,446,208 loaded; answers `a061d0a3…` bit-identical across processes |
| Card example | the README's Python block, run unchanged in a fresh interpreter (only the image kernel directory on its path), printed the same answers (0 drift) |
| Scored parity (pre-upload) | **every scored prompt**: typed-final 1,600 (2,000 slots), css15 6,547, public231 231 — 0 category changes, 0 input-digest mismatches, **max drift 0.0** |
| Upload | repository created **private** (`77cd6b9c` initial), upload commit **`2667d883…`**, private afterwards |
| Real download | `hf download --revision 2667d883… --local-dir <fresh dir>` with a fresh cache: 31 files + `.gitattributes` |
| Re-hash | 31/31 equal the manifest and the pre-upload package |
| From the download | examples bit-identical to pre-upload (`a061d0a3…`), card example identical, full-panel parity identical (max drift 0.0) |
| Readback | private, exact revision, 31/31 remote LFS/blob hashes, Hub-parsed card `apache-2.0`, `base_model` Eos 1.0 (`finetune`), no card problems, collection private with 0 items |
| HTTP readback | every card image and file downloaded at the revision with manifest-equal bytes; Eos 1.0, Qwen3.5 and collection (API) links 200; anonymous requests to the model API, README and page refused (401) |
| Gate evaluate | six items pass ([`gate-evaluate-a1c087bd6.json`](dev2-0p8b-release-2026-09-28/receipts/gate-evaluate-a1c087bd6.json)); `gate seal` refuses the draft decision (no `gate.json`) |

## Card (1.0 product design)

Owl banner `DEV2.0-0.8B-owl-banner.png` (Eos mosaic owl), what it is for, the three decision
types, a runnable System One example, a same-panel table labeled post-key (v3, T, H,
Choice / Noul / Score, public 231 easy / standard / hard, mlx-diag non-English Choice / Noul,
typed Brier / ECE), the v3 rank, model × task and public-231 charts from REPORT.json files
only, the automatic tradeoffs table versus Eos 1.0 (typed Score, H, mlx-diag non-English Noul
and eight transfer tasks), model details, training data and licences, limits (seed dependence,
where it improves, multilingual Noul, CPU), and one clearly marked **placeholder line for the
JevArena-C1 result**. It says public 231 is not the official sealed JevBench rank. No Pareto
chart, no gate ledger. Text: [`package-text/README.md`](dev2-0p8b-release-2026-09-28/package-text/README.md).

Licence notes for the coordinator:

- **Kev-0.8B** declares Apache-2.0 in its card metadata at the scored revision `9a45d25e` (= current
  head) but ships no LICENSE file; kept on the card (known permissive licence, no extra terms in
  its README). **Intern-Decision-0.8B** is Apache-2.0 with `LICENSE` and `LICENSE-QWEN` (`85a0cc5a` =
  head). JPT-0.8B (CC BY-NC 4.0) is excluded.
- **mlx-diag Score** is built from XNLI (CC BY-NC 4.0, internal use only), so the card shows only
  the MASSIVE (Choice) and PAWS-X (Noul) parts.
- **Training data** include CC BY-SA 3.0/4.0 sets (SNLI, SGD, KLUE, JGLUE, ArgQ); share-alike
  applies to the data, and the weights follow the 15:30 lineage rule (Apache-2.0), as Decision 1.0
  did. Datasets are credited in `ATTRIBUTIONS.md`.
- The later A7 embedding / lexical rescreen (`a7-dec10-v3`) found no training group at the
  threshold against any evaluation or development panel; it removed rows near internal held-out
  slices and one A7m (MultiNLI) group lexically near an mlx-diag item (XNLI, not shown on the card).
  E8F trained on the earlier `39a120ca` files, so those rows were in its data.
- Intern-Decision answers more public-231 items (164 vs 156) and has lower typed ECE; both are
  visible in the table and charts.

## Pipeline changes (release track; shared modules called out)

| Commit | Change |
| --- | --- |
| `7d08d2edf` | release verification reproduces the scored Qwen3.5 kernel runtime: the image exposes FLA only via `PYTHONPATH`, which the isolated interpreter drops, so `release.sh` gains `--site`, `--require-kernels` (reuses `v2/dec/runtime_check.py`), non-secret `--env`, `--mount-rw` and a launcher receipt |
| `e110b9ab4` | **card renderer (shared)**: T/H, public tiers, mlx-diag Choice/Noul, calibration columns; asserted parameter count; H and mlx tradeoffs; post-key label; sealed-rank wording; confirmation / training / detail text |
| `7479b673c` | **gate**: draft and final decisions (`status`); drafts drive private build and verification, `seal`/`--collect` need `final` |
| `2e4340788` | **package builder (shared)**: screens `MODEL_MANIFEST.json` too |
| `1c0fc97d2` | **card renderer (shared)**: spec-controlled runtime sentence; behavioural device comment |
| `a1c087bd6` | **gate / hub**: download receipts carry `passed`; `gate evaluate` item 3 counts them |
| `52c7f23a0`, `44386648c` | HTTP readback check of a private card (collections checked via the API) |
| `08d597778`, `f457fac1c`, `4d94d6fd6` | DEV2.0-0.8B spec, build-time draft, Qwen licence file name, GPU-verified runtime text |

Tests: `python3 -m unittest v2.release.tests.test_release` (20 tests, stdlib only).

## Issues found and handled

1. **Isolated interpreter hid FLA** (found before any GPU use): the release examples would have
   run Transformers' reference gated-delta path instead of the scored kernels. Fixed by the
   explicit site path plus a fail-closed kernel check; the parity then came out exact.
2. **CPU rehearsal** (0 GPU): the first build stopped at the package screen (`Qwen3.5-0.8B-LICENSE`
   has the suffix `.8B-LICENSE`; renamed `.txt`); the second built the package but CPU examples
   failed because Transformers routes the convolution to the installed GPU-only causal-conv1d
   kernel (`Expected x.is_cuda()`). The card therefore names one CUDA/ROCm GPU and lists the CPU
   limit. Logs: [`rehearsal/`](dev2-0p8b-release-2026-09-28/rehearsal/).
3. **Shared GPU**: the eval track's one-shot C1 event 1 was running on GPU5 when verification was
   ready; release verification waited until the event finished and the lease was restored
   (13:45Z), took the lease for 13:46–13:57Z, then restored the decoder's idle lease record.
   No C1 file or result was read.
4. **`gate evaluate`** marked item 3 failed because download receipts had no `passed` field;
   fixed (`a1c087bd6`) and re-evaluated on the same receipts: all six items pass.
5. **HTTP check**: Hub web pages refuse token auth; the private collection link is checked through
   the collections API, and the rendered page status is only recorded.

## Finalization (coordinator)

After the C1 event: replace `card.text.confirmation` in
[`specs/dev2-0p8b-release.json`](../specs/dev2-0p8b-release.json) with the C1 result, write the
final decision (the draft with `status: final`, `decided_by`, updated rationale; identity,
report and paired hashes unchanged), point `gate_receipt` at it, commit, push, mirror, and rerun
`release.sh` with this run's launcher flags plus `--upload --collect` into a new work directory.
The README change yields a new revision; `gate seal` binds the final decision to it before the
collection add and the collected readback.

Suggested card addition for that final build: the eval track's gate check
([m4 record](../../eval/records/m4-dev2-08b-gates-and-c1-event1-2026-09-28.md)) found that on typed
FINAL Score the candidate never predicts level 0 of five (35 gold items); the Limits bullet on
Score could say so next to "107 vs 120 of 400".

## GPU-hours

0.147 (node A GPU5, the release run's wall clock). CPU rehearsals, HTTP readback and gate
evaluation: 0.
