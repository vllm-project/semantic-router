# DEV2.0-4B: private release build, upload and verification (2026-09-29)

**Status: verified private revision, stopped before `--collect`.** Private
`llm-semantic-router/DEV2.0-4B@8052eb6c99c0b3b6a9980dcd77fb868c27da353a` (card-only revision of the verified `c73123f3`; manifest
**`eab4e8ac26d3975e9fe83ddd35cf879b431b596f903bf06ae043819eeae151c9`**, 36 files, 16,854,371,330 bytes, 4,208,383,488 loaded parameters, temperature 1). Every
`release.sh` step passed in all three runs (§4). The repository is not in the "Decision 2.0" collection. Verified draft
decision [`DEV2.0-4B.decision.draft.json`](dev2-4b-release-2026-09-29/DEV2.0-4B.decision.draft.json)
**`ec50b04bdbea6440e0c51682bcfe693168e552b435f9f50855159ea155e1a2d7`**; the coordinator writes the final decision and runs the collection add (§8).

- **Calibration:** CAL698 improves three of the four development criteria but worsens the CSS-pilot ECE, so the 23:15
  rule (adopt only if none worsens) rejects it and the package ships T = 1 (§2). This is the first size where the
  outcome is this close; the rule has no tolerance band and was applied as written.
- **Gates:** the eval track's 4B gate checks pass and find no training-row exposure (`v2/eval/records/m4-dev2-4b-gates-2026-09-29.md`,
  integration `e1ea4f5c7`; coordinator approval 08:05). Their card disclosures are on the card (§5).
- **Parity with the right cache:** the scored mlx-diag run used its own autotune cache, which picked different kernel
  tiles than the formal run for 2 of 6 autotune keys. Each panel was therefore checked with a copy of the cache of the
  run that scored it (§4); every drift is ≤ 1.9e-14.
- **Storage:** 77.00 GB of the org's 100 GB private tier were in use before the upload and 93.85 GB after (the new
  repository is charged its full 16.85 GB although its LFS objects equal the staging copy). Nothing was deleted (§7).
- **Errata** to the 0.8B, 0.6B and 2B release records: commit `ce2eb1789` (§9).
- No shared module changed.

## 1. Candidate and scored identity

| Item | Value |
| --- | --- |
| Candidate | decoder N4XF soup: full fine-tuning of Decision 1.0 Nox on a token-matched 14.42% whole-group subsample of XL r2 full plus the M3 A0s rows, own-Lux 1.0 soft targets (KL 1.0; the 3,759 H7 / H8 gap rows gold-only), uniform soup of three seeds ([decoder record](../../dec/records/dec-m4-4b-candidate-2026-09-29.md), integration `9b8137e56`) |
| Staged checkpoint | private `llm-semantic-router/dev2-dec-staging@e86562218928fbb77f8071b45646962f63049fad` (still `main`; its 7 LFS objects equal node B's hash list), `m4/N4XF-soup/`; node A copy `/data/dev2/runs/dec/m4/formal-candidates/staging-e8656221/m4/N4XF-soup`, re-verified 14/14 against node B's hash list before the T = 1 derivation |
| Identity | `model_sha256` `11b5ca1c22f8cd38d67d718cbfa59a3fb4dc8f4e7bfd75b3e988f34e35fad0ad`; decision head `bd324aa6…`; soup members s1 `checkpoint-0000690`, s2 `-0000676`, s3 `-0000492` (`670ba59b…`, `993a5ba1…`, `d2399ce3…`) |
| Loaded parameters | 4,208,383,488 (backbone 4,205,751,296 + head 2,632,192), from the safetensors headers: tier **4B** (ratio 1.05 to 4e9) |
| Direct weight origin | `llm-semantic-router/Decision-1.0-Nox-4B@cde2a68dbaa557ea65dc458104d410a0802ee259` (Apache-2.0, `LICENSE` `bbedc3fd…`; its `decision_config.json` names `base_revision` `851bf6e8…`). The adopted Nox 1.0 comparator run used revision `0bb83350`, whose four weight files have the same Hub SHA-256 as `cde2a68d`'s (README, header image and `decision_config.json` differ) |
| Upstream | `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (post-trained; card `license: apache-2.0`, `base_model: Qwen/Qwen3.5-4B-Base`); its LICENSE is byte-identical to Nox 1.0's `LICENSE` and `QWEN-LICENSE` |
| Architecture | 32-layer Qwen3.5 text backbone (24 gated-delta, 8 attention; hidden 2,560), head `qwen3.5-text-endpoints-global-query-shared-bilinear-mlp`; FP32 safetensors (five shards) |
| Profile / limit | `qwen-full`, 16,384 tokens (training inputs ≤ 8,192) |
| Scored run (CAL698) | `/data/dev2/runs/dec/formal/m4/m4-N4XF-soup-nodeA` (REPORT `87ec711c…`, SEAL `12b44de9…`), image `decision20-train-fast:host2` `sha256:f83b1d10…`, runtime env `HIP_FORCE_DEV_KERNARG=1`, autotune cache `m4-N4XF-soup-nodeA-triton` (1,654 files, digest `438618a6…`); mlx-diag run `m4-N4XF-soup-mlx` with its own cache `m4-N4XF-soup-triton` (415 files, digest `ba8f2132…`) |
| T = 1 bindings | [derived run](dev2-4b-release-2026-09-29/derived-run/): REPORT `57fa23ef…`, SEAL `328cd927…`; predictions typed-final `003c5cd5…`, css15 `c966cfc7…`, public231 `ec085d45…`, mlx-diag `19a095e5…`; paired vs the adopted Nox 1.0 run `27f1419f…`; mlx-diag score `77d681c9…` |

Same-panel results at T = 1 (answers identical to the CAL698 run; the card shows the first four rows):

| Model | v3 | T | H | Choice / Noul / Score | Public 231 (easy / standard / hard) | Typed Brier / ECE |
| --- | ---: | ---: | ---: | --- | --- | --- |
| DEV2.0-4B | **63.151** | .6881 | .5796 | 582 / 734 / 185 | 171 (48 / 67 / 56) | .175 / .116 |
| Decider 4B | 61.882 | .6894 | .5555 | 750 / 623 / 114 | 192 (48 / 71 / 73) | .154 / .056 |
| Jet v6.2 | 60.375 | .6781 | .5375 | 720 / 640 / 125 | 174 (48 / 69 / 57) | .171 / .077 |
| Decision 1.0 Nox, adopted run (the bar) | 56.470 | .6144 | .5190 | 552 / 653 / 178 | 173 (48 / 66 / 59) | .205 / .092 |
| Decision 1.0 Nox, 16K same-renderer control | 55.689 | .5975 | .5190 | 549 / 627 / 180 | 173 (48 / 66 / 59) | .212 / .097 |

Paired (joint bootstrap, 5,000 replicates; the T = 1 derivation reproduces the decoder's five pairs exactly): vs the
adopted Nox 1.0 run +6.68 [+0.99, +9.64] (T +.074 [+.051, +.096], H +.061 [−.037, +.113]); vs the 16K control +7.46
[+1.80, +10.45]; vs Decider 4B +1.27 [−5.71, +4.31] (T −.001 [−.030, +.029], H +.024 [−.095, +.073]); vs Jet v6.2 +2.78
[−3.33, +5.94] (T +.010 [−.020, +.041], H +.042 [−.061, +.093]); vs M3 N4LKr +3.61 [+1.22, +12.13]. Hopper (G) 1.2
(research-and-demo licence) and JPT-4B (CC BY-NC) stay internal: not on the card, not in this table.

## 2. Calibration (coordinator rule 23:15)

The decoder's CAL698 fits for this identity are identical at 8K (development, `c822fafa…`) and 16K (formal,
`176a2f0e…`): Choice 0.66731, Noul 0.49631, Score 0.21176, logits `f134ec94…`. `v2/release/dev_calibration.py` on node B
(CPU; mirror `4fd7e3388`) undid the 8K fit on the stored soup development predictions (typed-DEV `f7572bb5…`, 1,600
slots; CSS pilot `49ec21a8…`, 1,430 slots; node-B image `dbe5f32b`; no input truncated, the longest 5,135 tokens),
applied CAL698 and scored both ([receipt](dev2-4b-release-2026-09-29/devcal/dev2-4b.json) `5b1d2e37…`,
[command](dev2-4b-release-2026-09-29/devcal/cmd.sh.txt)):

| Development metric | Raw (T = 1) | CAL698 |
| --- | ---: | ---: |
| Typed DEV Brier / ECE-10 | 0.216 / 0.119 | 0.206 / 0.090 (better) |
| CSS pilot median task Brier-sum / ECE-15 | 0.609 / 0.0632 | 0.602 (better) / **0.0679 (worse)** |
| Typed DEV by type, Brier / ECE-10 | Choice .269 / .167, Noul .202 / .118, Score .125 / .247 | Choice .256 / .113, Noul .243 / .202, Score .068 / .032 |

One of the four criteria worsens and no answer changes: **CAL698 rejected; the package ships T = 1.** The card states
the evaluated temperatures and why they were not adopted. For the record only (formal panels are not a decision
input): T = 1 typed Brier / ECE 0.175 / 0.116 vs 0.177 / 0.075 with CAL698; CSS15 median Brier-sum / ECE 0.561 / 0.081
vs 0.569 / 0.090; public 231 0.160 / 0.073 vs 0.174 / 0.107.

The sealed CAL698 formal run was returned to T = 1 on node A (CPU; [script](dev2-4b-release-2026-09-29/ops/derive-t1.sh.txt),
[log](dev2-4b-release-2026-09-29/derived-run/derive-t1.log.txt)): `retemper_predictions --undo` on typed-final 1,600,
css15 6,547, public231 231 and mlx-diag 2,275 rows (0 answer changes each), after checking the calibration file, the
staging copy and the sealed prediction hashes; then `same_panel adopt`, `seal`, `report` (tier 4B, parameters from
the safetensors headers), `compare` against the same five comparator runs, and `multilingual_panel score`.

## 3. Package

Built by `v2/release/build.py` (spec [`dev2-4b-release.json`](../specs/dev2-4b-release.json)). 36 files; licence
`apache-2.0` (Nox 1.0 and Qwen3.5-4B are Apache-2.0; training-data licences credited in `ATTRIBUTIONS.md`); no
`calibration.json`. All 11 staged checkpoint files are byte-identical in the package
([manifest](dev2-4b-release-2026-09-29/card2/package-text/MODEL_MANIFEST.json)).

| File | SHA-256 |
| --- | --- |
| `backbone/model-00001-of-00005.safetensors` | `06faf67f333d1136e4cbc583220e98602354b454d3b0fb917bf433a8bf4dbb3f` |
| `backbone/model-00002-of-00005.safetensors` | `abac23bcaad1d5fb2aa73d9d051db970514a4f427b0cff22336b50c5e7aa7302` |
| `backbone/model-00003-of-00005.safetensors` | `1f492a389f49f382b4bbf84686176b51f9621da50d143e436ed61f764a9fa988` |
| `backbone/model-00004-of-00005.safetensors` | `18024da979fb12651442e3a168deb92b4b8e41d41ea8076151df62879ecfa823` |
| `backbone/model-00005-of-00005.safetensors` | `0ad4368c4f04781d30c237c269a9911b9ffb037104a4ee469bb7f56d57e67d15` |
| `backbone/model.safetensors.index.json` | `1921033b850d1a219406de25ac8a8c13c45048572cd71b4a738246c1f690986c` |
| `backbone/config.json` | `a5ed4156fda05f0f9149c66964d6165916754e7355488a8a07d9b0398acdbdb9` |
| `decision_head.safetensors` | `bd324aa69788fa936293ff4959ea1caa8845a4e47589879b0748aed3c3df7c5a` |
| `decision_config.json` | `6658fbd01bf92030078ad1bbe735a0de77045431055d06c32bc6de8cf7d275d9` |
| `tokenizer.json` | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| `tokenizer_config.json` | `bee8eba30f0eb4af73c0fe2cd06d0f89b657d7819941c438157ec42f7c80ea87` |
| `chat_template.jinja` | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |
| `LICENSE` / `LICENSES/Qwen3.5-4B-LICENSE.txt` | `bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a` |

A CPU build check from mirror `4fd7e3388` (before any GPU time) rendered the card and charts; it needed no code change.

## 4. Verification (node A GPU5, shared lease `owner.release`)

GPU5 belongs to the decoder track (its owner entry was idle and never touched); the eval track ran short shared-lease
development collections on it at the same time. Every run used `--site /opt/decision-fla --require-kernels --env
HIP_FORCE_DEV_KERNARG=1`, the image of the scored runs, a fresh `cp -a` copy of a frozen autotune cache (digest equal to
the frozen cache before the run, frozen cache unchanged after), and no `--collect`.

**Why two caches.** The formal run and the mlx-diag run each started an empty cache. For mlx-diag's shapes, 2 of 6
autotune keys hold different best configurations in the two caches (`l2norm_fwd_kernel` BT 8 vs 16, `chunk_fwd_kernel_o`
BK / BV 64 vs 128), and tile choice alone moves this model's example probabilities by up to 1.5e-3 (0 category
changes) between the two runs below. Parity therefore used the cache of the run that scored each panel.

| Run | Work dir (node A `/data/dev2/runs/release/`) | Cache copy | Parity | Result |
| --- | --- | --- | --- | --- |
| no-upload ([command](dev2-4b-release-2026-09-29/release/extra/launch-mlx-parity.txt), [receipts](dev2-4b-release-2026-09-29/mlx-parity/receipts/)) | `dev2-4b-mlxparity-20260929T001839Z`, mirror `7855c46dd` | mlx-diag run's (415 files) | mlx-diag 2,275 prompts | 6/6 steps; 0 category changes, 0 missing, max drift 1.9e-14; manifest `18338b09…` |
| upload ([command](dev2-4b-release-2026-09-29/release/extra/launch-command.txt), [receipts](dev2-4b-release-2026-09-29/release/receipts/)) | `dev2-4b-release-20260929T002455Z`, mirror `7855c46dd` | formal run's (1,654 files) | typed-final 1,600 prompts / 2,000 slots, css15 6,547, public231 231, before and after upload | 15/15 steps; revision `c73123f3…`, manifest `18338b09…` |
| card-only ([command](dev2-4b-release-2026-09-29/card2/extra/launch-command.txt), [receipts](dev2-4b-release-2026-09-29/card2/receipts/)) | `dev2-4b-card2-20260929T004630Z`, mirror `b798099c5` | formal run's | subsets: typed-final 200, css15 300, public231 100 | 15/15 steps; 0 category changes; revision `8052eb6c…`, manifest `eab4e8ac…` |

Upload run, step by step:

| Step | Evidence |
| --- | --- |
| build | manifest `18338b09…` (equal to the no-upload run's build); identity `11b5ca1c…` recomputed; scored adapter sources equal the vendored runtime; build-time draft `00c25bb6…` |
| pre-a / pre-b / repeat-pre | six native System One requests (support routing, policy check, structured state, multilingual, null description, over-budget) with Choice, Noul and Score questions, in two containers: answers `3d909907…`, bit-identical; FLA gated-delta and causal-conv1d kernels bound |
| card-pre | the README's Python example, executed as written, reproduced |
| parity-pre | typed-final 1,600 prompts / 2,000 slots, css15 6,547, public231 231 against the T = 1 bindings: 0 category changes, 0 missing, max drift 1.2e-15 / 2.2e-16 / 1.8e-14 |
| ensure / upload | private repository created; fixed uploader (each revision is exactly the package); 36 files, 16,854,370,318 bytes → `c73123f36b6d9046ba7d85722adfd6031c7d9e23` (4.8 s: the LFS objects already existed in the org) |
| download / tree | real `hf download` at the revision into a fresh `HF_HUB_CACHE`; re-hash 36/36, nothing missing or extra |
| post / repeat-post / card-post | on the download: answers `3d909907…` (three processes identical), card example reproduced |
| parity-post | the same 8,378 prompts on the download: 0 category changes, same drift |
| readback | private; every remote hash matches (36 files); card metadata `apache-2.0`, base model Nox 1.0 (`finetune`); no card problems; collection "Decision 2.0" private with 3 items, this repository not in it |
| gate | `gate evaluate`: the six items pass ([output](dev2-4b-release-2026-09-29/release/extra/gate-evaluate.json)) |
| HTTP / links | `hub_card_http_check`: 13/13 card targets, anonymous model API / README / page refused (401); `hub_links`: 17/17 |

The card-only run repeated the same steps on the same weight files (369 s: parity on the 600-prompt subsets before and after upload with 0 category changes and the same drift, re-hash 36/36, readback private with no card problems, HTTP 13/13 with anonymous 401, links 17/17, gate items 6/6; on the Hub only `README.md` and `MODEL_MANIFEST.json` differ from `c73123f3` and all six safetensors are identical ([compare](dev2-4b-release-2026-09-29/card2/extra/revision-compare.json))).

## 5. Card

The 1.0 product design ([README](dev2-4b-release-2026-09-29/card2/package-text/README.md)): DEV2.0-4B owl banner (the
Nox 1.0 mosaic owl, `20d60396…`), uses table, same-panel table (v3 with T / H, Choice / Noul / Score, public 231 easy /
standard / hard, mlx-diag non-English Choice / Noul, typed Brier / ECE), v3 rank, model×task and public-231 rank charts
(no Pareto), the automatic tradeoffs table versus Decision 1.0 Nox, the System One example, model details, training and
limits.

- **Rows:** DEV2.0-4B, Decider 4B, Jet v6.2 and Decision 1.0 Nox (the adopted run, the stricter comparator; the note
  gives the 16K control's 55.69). The chart generator's licence filter admits exactly these four; Hopper (G) 1.2 and
  JPT-4B are not listed. Decider 4B declares Apache-2.0 in card metadata only and the card says so; Jet v6.2 ships an
  Apache-2.0 LICENSE and was measured on ROCm, which its card does not list. The mlx-diag Score (XNLI) part is not shown,
  so the diagnostic's overall .770 vs .795 (which includes it) stays off the card; its Choice / Noul parts carry the
  regression.
- **Disclosures** (decoder record, the eval track's 4B gate record and the coordinator's 08:05 list):
  - Weight origin (own Nox 1.0 → full fine-tune; Qwen3.5-4B lineage, Apache-2.0); teacher = own Lux 1.0 only, with its
    targets computed on a runtime without causal-conv1d; recipe, three seeds and uniform soup; training data by family
    with licences for all 58 sources (17,663 rows of own Decision 1.0 corpora, 41,079 rows of Decision 2.0 data; per-source
    licences from the program registries, none non-commercial or research-only); not used: Jev, third-party decision
    models, the Cosmos QA / SQuAD 2.0 answerability families, the mlx-diag sources.
  - The gain over Nox 1.0 is typed reasoning; human transfer higher but not significant; CSS15 losses wiki_corpus
    −.065, mrf −.045, media_ideology −.016, talklife −.001; public 231 level (171 vs 173; hard 56 vs 59); long inputs
    level with Nox 1.0.
  - Against the peers: v3 margins not significant; T and H level with Decider 4B; Choice far below both peers (.728 vs
    .938 / .900); public 231 significantly below Decider 4B (−21 [−32, −10]) and level with Jet v6.2; long inputs
    weaker than both.
  - Score rarely predicts level 0 (4 of 400; recall .11, Nox .14); level-2 recall .19 vs .30.
  - mlx-diag: non-English Choice level (75.1% vs 75.9%), non-English Noul 72.7% vs 80.0% (lowest of the four), Korean
    and Japanese Noul 63% vs 69% and 67% vs 77%.
  - Typed ECE at T = 1 higher than Nox 1.0's (0.116 vs 0.092); the calibration decision.
  - The eval track's evaluation-familiarity sentence (no exposure; v3 stays 63.15 without the 84 flagged items).
  - Seed dependence (development proxy 62.6 / 60.2 / 63.3; soup 62.9), CPU not verified, tested hardware, 4 invalid
    CSS15 answers (automatic limits).
- **C1:** a marked placeholder line: "Independent sealed confirmation (JevArena-C1): PLACEHOLDER — to be added after the
  final sealed scoring event." The eval track's event-3 independence recheck (XL r2, the H7 / H8 gap arms, Lux-XL
  waves 3–5) precedes it.
- **Card revisions:** `c73123f3` carried the pre-relay limits; `8052eb6c` adds the 08:05 disclosures (public 231
  vs Decider 4B, long inputs, Japanese Noul, all four CSS15 losses, the familiarity sentence, Jet's platform, the teacher
  runtime). Only `README.md` and `MODEL_MANIFEST.json` differ between them.

## 6. Evaluation familiarity

The eval track's 4B check (record above) found **no own exposure**: `m4-xl-full-29m` (`c7d51219…`) contains 0 of the r2
rescreen's 305 excluded groups (group id, row id and input hash), the H7 / H8 gap rows were quarantined at build time on
a stricter union screen, and removing the 84 flagged items leaves v3 at 63.151 and every comparison in place. The card
carries the eval track's proposed sentence. The selection and calibration splits (SELECT700, CAL700) share templates
with A0s-strict by construction (the 03:10 decision tolerates and discloses this); this package ships no calibration.

## 7. Hugging Face storage

`src/training/decision2/v2/common/hf_headroom.sh --node node-a` before each upload (and inside each launcher):

| Moment | Private total | Headroom | Largest repositories |
| --- | ---: | ---: | --- |
| first probe (00:05 UTC) | 74.60 GB | 25.40 GB | dev2-9b-staging 31.78, dev2-dec-staging 16.85 (the N4XF staging copy), DEV2.0-2B 7.56 |
| before the upload (00:23 UTC; threshold 18 GB) | 77.00 GB | 23.00 GB | a new 2.40 GB private repository of another track since the first probe |
| after the upload (00:39 UTC) | **93.85 GB** | **6.15 GB** | DEV2.0-4B 16.85 (charged in full, although its LFS objects equal the staging copy) |
| after the card-only revision (00:53 UTC) | 93.85 GB | 6.15 GB | unchanged (README and MODEL_MANIFEST only) |

- **Action:** the 16.85 GB package fit, so nothing was deleted.
- **Next deletion (after the collection add):** the staging copy `dev2-dec-staging` `m4/N4XF-soup/` (16.85 GB) with
  `rewrite_history=False`, once its Hub OIDs are re-checked against the released files and the node copies (both nodes
  hold hash-verified copies). That restores about 23 GB. Until then, other tracks have about 6 GB of headroom.

## 8. Draft decision and finalization

- **Build-time binding draft:** [`DEV2.0-4B.decision.build-draft.json`](dev2-4b-release-2026-09-29/DEV2.0-4B.decision.build-draft.json)
  `00c25bb6…`, the spec's current `gate_receipt` (also `/data/dev2/runs/release/decisions/` on node A, read-only).
- **Verified draft:** [`DEV2.0-4B.decision.draft.json`](dev2-4b-release-2026-09-29/DEV2.0-4B.decision.draft.json)
  `ec50b04bdbea6440e0c51682bcfe693168e552b435f9f50855159ea155e1a2d7`, also at `/data/dev2/runs/release/decisions/`.
  `gate.check` accepts it as a draft; a final seal refuses it. It names the card-only revision `8052eb6c` and
  supersedes the first verified draft `d5baa5bf…` (named `c73123f3`, before the 08:05 disclosures), kept as
  [`DEV2.0-4B.decision.draft-c73123f3.json`](dev2-4b-release-2026-09-29/DEV2.0-4B.decision.draft-c73123f3.json).

To finalize (coordinator):

1. Write `/data/dev2/runs/release/decisions/DEV2.0-4B.decision.json` with `status: final`, `decided_by`, and the
   draft's identity `11b5ca1c…`, `report_sha256` `57fa23ef…` and `paired_sha256` `27f1419f…`.
2. Point `specs/dev2-4b-release.json` `gate_receipt` at that file. Edit `card.text.confirmation` only if the C1 wording
   should change before event 3.
3. Commit, push and mirror.
4. Rerun [`card2/extra/launch-command.txt`](dev2-4b-release-2026-09-29/card2/extra/launch-command.txt) with the new
   mirror and `--upload --collect`, not `--already-collected` (the repository is not in the collection). Only
   `builder.source_commit` in `MODEL_MANIFEST.json` changes. Keep the formal run's cache and the three formal panels
   for parity (mlx-diag parity needs the mlx run's cache; it passed on these bytes).
5. After the collection add: delete the `m4/N4XF-soup/` staging copy with `rewrite_history=False` (§7).
6. After C1 event 3: a card-only revision with `--upload --collect --already-collected`.

## 9. Errata to earlier release records (the coordinator's 05:25 erratum)

Commit `ce2eb1789` adds a dated erratum under the title of each record, records only; no released repository was
touched. Each states that the storage cleanups of 2026-09-29 (02:15, 02:32, 04:47, 04:51 UTC+8) rewrote history because
`permanently_delete_lfs_files` defaults to `rewrite_history=True` in huggingface_hub 1.33, so the staging revisions
`545a6784`, `784a894f`, `16c0929a`, `62c61c10` and `5afd8fc8` no longer exist, and that identity rests on the per-file
SHA-256 in each manifest plus the verified node copies.

- [DEV2.0-0.8B](dev2-0p8b-release-2026-09-28.md): its checkpoint citation `dev2-dec-staging@16c0929a`.
- [DEV2.0-0.6B](dev2-0p6b-release-2026-09-28.md): its citations of `dev2-release-staging-06bm4@62c61c10`.
- [DEV2.0-2B](dev2-2b-release-2026-09-29.md): its checkpoint citation `dev2-dec-staging@545a6784`, and its statement
  "commit history is unchanged" is wrong (the 04:47 cleanup did not pass `rewrite_history=False`); do not reuse its
  cleanup script as is.
- A read-only Hub check (≈08:20 UTC+8) confirmed all five staging revisions absent and every cited released revision
  present and private (0.8B `0b631a85`, `2667d883`, `7d08d0e1`, `f458c34c`; 0.6B `7b5d3ff2`, `a87eeb72`, `e61b2b44`,
  `99c4e799`; 2B `b2c5d7ea`, `5ad3e9a3`).

## 10. GPU-hours and commits

- **GPU (node A GPU5, shared lease):** no-upload run 0.066, upload run 0.244, card-only run 0.103: **0.413
  GPU-hours**. The first upload attempt stopped before any GPU use (the mirrored `hf_headroom.sh` has no execute bit;
  the launcher now calls it with `bash`); an earlier launch wrote an empty launcher file (a backgrounded `cat` reads
  `/dev/null`) and ran nothing. The shared-lease entry was removed at the end of every run; the decoder's owner entry was
  never touched.
- **CPU only:** development calibration (node B), T = 1 derivation and type gates (node A), two build checks, Hub and
  storage probes.
- **Commits:** `7855c46dd` (spec, T = 1 bindings, calibration decision, build draft), `ce2eb1789` (errata),
  `b798099c5` (card disclosures relayed at 08:05), this record's commit, then a merge into `xunzhuo/decision-2-training`.
