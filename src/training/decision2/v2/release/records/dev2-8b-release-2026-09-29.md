# DEV2.0-8B (9B tier, candidate K-a13): private release build, upload and verification (2026-09-29)

## Renamed DEV2.0-9B, 2026-09-29 16:44 UTC+8 — card-only revision `ae683196` (current)

- **Why:** the user directive of 2026-09-29 16:05 UTC+8: names follow the base model's size, and this model's base is
  Qwen3.5-9B.
- **Move:** the repository was moved to **`llm-semantic-router/DEV2.0-9B`** at 08:44:34Z. History, revisions and
  privacy are unchanged, and the old ID redirects.
- **Card-only revision `ae6831960dd1114296cb15a59248b79832c42959`** (`main`):
  - manifest `996b28192928d9238657dce562721e10b850cae183926ff88396383192bef628`;
  - final decision `bad976354662ea807cfc444c48f39f723624b14bd8446f74195308b19c1b0b4a`, which supersedes `7666fd7c…`
    below;
  - gate.json `9de02171…`.
- **Weights:** every weight file is byte-identical to `53bac735`.
- **Card:** it shows the 9B owl and says "Named after its base model (Qwen3.5-9B); it loads 7,940,895,744
  parameters". The pending C1 line is unchanged.
- **Collection:** position 5 of 0.6B, 0.8B, 2B, 4B, 9B, 27B.
- Spec: `specs/dev2-9b-release.json`. Record: [`dev2-rename-9b-27b-2026-09-29.md`](dev2-rename-9b-27b-2026-09-29.md).
- The sections below describe the release as DEV2.0-8B.

## Released, 2026-09-29 ≈16:00 UTC+8 — revision `53bac735`, in the private "Decision 2.0" collection

The coordinator decided the release at 15:49 UTC+8 (full-autonomy mandate) on the verified draft `1db7683f…`.

- **Revision `53bac735be58def53673d0d290b9baa3f2af1cf9`** of private `llm-semantic-router/DEV2.0-8B` (now `main`), manifest
  `d5007cdb83cd9e29c353efb5dff92a72035c53a4312468c660cb73af2a3accfa`. Against the verified `0dee8017` only
  `MODEL_MANIFEST.json` differs (builder commit, decision binding); README and all weights are byte-identical.
- **Final decision** [`DEV2.0-8B.decision.json`](dev2-8b-release-2026-09-29/DEV2.0-8B.decision.json)
  `7666fd7c8676fac9052b7e10812c556dfac5e1bc040ddb86a6687639155f905a` (`status: final`, decided by the coordinator;
  supersedes `1db7683f…` and the build draft `5db122b0…`); `receipts/gate.json` `de1436a6…` seals it to `53bac735` and
  the manifest. The spec's `gate_receipt` names it.
- **C1:** the card keeps the family's pending line, verbatim as on the released DEV2.0-2B / DEV2.0-4B cards
  ("Independent sealed confirmation (JevArena-C1): PLACEHOLDER — to be added after the final sealed scoring event.");
  that event is C1 event 3, and the sealed result comes in a separate card-only revision. No card text changed.
- **Publishing run** ([launcher](dev2-8b-release-2026-09-29/ops/final-collect.sh), [receipts](dev2-8b-release-2026-09-29/final/receipts/),
  mirror `6d21c8980`, node A GPU6): build, examples in two processes (bit-identical), card, subset parity (typed-final 200,
  css15 300, public231 100, mlx-diag 100; 0 answer changes) before and after upload, real download and re-hash, readback,
  gate seal — all pass. `hub collect` then refused: the collection is now titled "🎲 Decision 2.0" (renamed outside
  the pipeline; `hub.py` expects "Decision 2.0"; both repo and collection verified private). DEV2.0-8B was already a
  collection item (added outside the pipeline between 07:41Z and 07:49Z; Vela-2.0-Encoder-307M-Unified, present at
  07:49Z, was gone by 08:00Z), so nothing needed adding. [`final-collect-finish.sh`](dev2-8b-release-2026-09-29/ops/final-collect-finish.sh)
  (mirror `2ab92d25c`) ran the pipeline's post-collect readback (`--expect-collected`: private, 40 files, 0 hash
  mismatches, 0 card problems, in the collection — pass), card HTTP 12/12 (anonymous 401), links 15/15. The title and
  the shared guard were left unchanged; the ~27B collection add will hit the same guard.
- **Collection** ([order receipt](dev2-8b-release-2026-09-29/final/extra/collection-order.json)): private, in size order
  via `update_collection_item`: DEV2.0-0.6B, DEV2.0-0.8B, DEV2.0-2B, DEV2.0-4B, DEV2.0-8B, DEV2.0-26B (0.6B and 0.8B were
  swapped). DEV2.0-26B was already an item before this run (not added by this worker, not removed).
- Storage 62.57 / 100 GB (37.43 GB free; no change). Final run ≈0.109 GPU-h; release total **0.737 GPU-hours**. GPU6
  lease back to `idle`.

## Verified package (before the collection add)

- **Private `llm-semantic-router/DEV2.0-8B@0dee801731a69a89c4219a2bf26381ce4d59af48`** (2026-09-29 07:32Z): manifest
  `80770483735c962454406e9723239af79aec8cc3ba7adb2d8546663ff7ca4ecc`, 40 files, 17,947,416,716 bytes; identity
  `b1ed5a71…` (the BF16-storage copy of the scored FP32 identity `b9d973b3…`, answers identical); T = 1; 7,940,895,744
  loaded parameters; licence `apache-2.0`. **Not in the collection** (readback: the private "Decision 2.0" collection has
  6 items and DEV2.0-8B is not among them).
- **Verification** (`release.sh` from mirror `05914b8e8`, node A GPU6, scored image and kernels required, fresh copy of the
  scored run's autotune cache): 15/15 steps pass — two isolated example processes and the card's Python block reproduce,
  answers bit-identical across processes and before / after the real `hf download`; re-hash 40/40; scored-panel parity
  before and after upload with **0 answer changes** on typed-final (2,000 slots), css15 (6,547), public231 (231) and
  mlx-diag (2,275) at tolerance 1e-4 (largest drift 8.9e-16); Hub readback private, card metadata clean; card HTTP 12/12
  with anonymous 401; links 15/15; `gate evaluate` 6/6.
- **Verified draft decision** [`DEV2.0-8B.decision.draft.json`](dev2-8b-release-2026-09-29/DEV2.0-8B.decision.draft.json)
  **`1db7683ff0657485ffdbdf84b3599c7780c5ea275b2f47f36989b38e40495725`** (node A `/data/dev2/runs/release/decisions/`,
  passes `gate.check` against the spec); build-time draft `5db122b0…`. **Stopped before `--collect`** (task scope).
- Pending (coordinator): final decision → JevArena-C1 line (card-only revision; the card's only placeholder) →
  `--upload --collect`.

## 1. Candidate, scored identity and name

- **Candidate** (coordinator 2026-09-29 13:45, from 9B Milestone 4; record
  [`lux9b-m4-formal-result-2026-09-29.md`](../../9b/records/lux9b-m4-formal-result-2026-09-29.md)): K-a13, the α = ⅓ point
  of the K line = ⅓ × K soup + ⅔ × Decision 1.0 Lux. The K soup (`20dbb8999f21…`) is the uniform average of three seeds
  (20260926 / 1 / 2, checkpoints 1,624 / 1,420 / 1,424 chosen on SELECT700) of own Lux 1.0 (`bd45a30a`) fully fine-tuned on
  x60 = 122,651 rows / 60,183,732 tokens of `mx-xl-full-r2` (human arms A7q, H1, H7, H8 included) with CE + 0.5 Brier +
  1.0 KL(own Lux 1.0) on every row. Node A `runs/9b/m4/K-a13-build/soup` (16 files, every tensor F32, 31.78 GB), model
  identity `b9d973b3ef555457…`, manifest `m4/K-a13-build/SHA256SUMS` sha256 `6913eb61836a…` (23 files incl. `m4/K-a13-cal`;
  re-hashed 23/23 before the derivation and before the BF16 copy).
- **Scored run:** node A `runs/9b/formal-m4/K-a13-16k` (+ `K-a13-16k-mlx`), post-key same-panel, image
  `decision20-train-fast:host2` = `f83b1d10…` with FLA 0.5.2 and causal-conv1d 1.7.0, 16,384 tokens (over-length invalid),
  BF16 backbone / FP32 head, `HIP_FORCE_DEV_KERNARG=1`, `TRITON_CACHE_AUTOTUNING=1`, Triton cache `formal-m4/triton-cache`
  (copy of the frozen `formal-m3` cache `af623300…`; final tree `5604ffdc5f19…`, shared by the panels and mlx-diag).
  **It was scored with the CAL698 temperatures** `m4/K-a13-cal/calibration.json` (`65297c6d…`; Choice 1.4203, Noul
  1.0368, Score 0.5626; fitted at 8,192 tokens): every output manifest binds model `b9d973b3` and that calibration file.
  SEAL `11abc1cc7818…`, REPORT `b0b805ccc3ef…`: v3 67.737 vs adopted native Lux1 16K 65.808, +1.929 [+0.607, +4.144];
  vs the same-renderer Lux1 65.231 +2.507 [+1.037, +4.599]; H +0.008 [−0.011, +0.042]; types OK ×3; public 231 178 vs 183;
  mlx-diag type macro .822 vs .832 (same-renderer; .828 for the eval track's native Lux1 run).
- **Name: DEV2.0-8B.** The brief (§二, §五) sets final names by the actually loaded parameter count. Counted from the
  safetensors headers of the scored files: text backbone 7,936,684,544 (embedding table 248,320 × 4,096 = 1,017,118,720;
  32 layers and the final norm 6,919,565,824) + decision head 4,211,200 = **7,940,895,744**; the checkpoint holds no
  lm_head, vision-tower or MTP tensors (the Qwen3.5-9B vision path is not part of the model, although `config.json` keeps
  the upstream `mtp_num_hidden_layers` / M-RoPE fields), and the runtime loads exactly these tensors (the release receipts
  report `loaded 7,940,895,744 = backbone 7,936,684,544 + head 4,211,200`). 7.94B rounds to 8B under the shared
  `count_name` rule (whole billions from 1B; `name_basis: loaded-parameters`, commit `2417f4033` of the 27B release
  worker), as the coordinator anticipated at 13:45; the tier stays 9B (successor of Decision 1.0 Lux-9B; reports keep
  `tier 9B`). The 9B-tier Lux owl banner is relabelled `DEV2.0-8B` (`banner.py --label 9B=8B`, `0347ccc2e`; the tool first
  reproduced the committed DEV2.0-9B banner bit for bit, `d318952d…`; new PNG `7263ca30…` differs only inside the size
  label box).

## 2. Calibration (coordinator rule 2026-09-28 23:15): T = 1

- Development panels only ([receipt](dev2-8b-release-2026-09-29/devcal/dev2-8b.json) `17fa4211…`,
  [command](dev2-8b-release-2026-09-29/devcal/cmd.sh.txt)): `v2.release.dev_calibration` undid the CAL698 temperatures of the
  stored typed-DEV (1,600 slots) and CSS-pilot (1,430) development predictions of the same weights and compared T = 1 with
  CAL698. **adopt = false**: typed-DEV Brier 0.05370 → 0.05809 and CSS-pilot Brier-sum 0.55352 → 0.55531 worsen; typed-DEV
  ECE-10 0.0157 → 0.0107 and CSS-pilot ECE-15 0.1326 → 0.0577 improve; 0 answer changes. The rule has no tolerance band,
  so the package ships **T = 1** (no calibration file) and the card names the rejected temperatures.
- (CAL698 would also have needed a 16K refit: the builder refuses a calibration file whose `inference.max_length` (8,192)
  differs from the package limit. Not needed at T = 1.)
- **T = 1 scored bindings** ([script](dev2-8b-release-2026-09-29/ops/derive-t1.sh),
  [log](dev2-8b-release-2026-09-29/derived-run/derive-t1.log.txt)): node copy re-hashed 23/23 and sealed prediction hashes
  checked, then `retemper_predictions --undo` on typed-final, css15, public231 and mlx-diag (0 answer changes each), adopt
  (prior receipts: COLLECT, SEAL, calibration, retemper), seal, report (`--count-safetensors` 7,940,895,744), compare.
  Derived run node A `runs/release/dev2-8b-t1-derived`: REPORT `66106ef9…` (v3 67.737, T .8106, H .5660, identical
  answers), SEAL `d00e453b…`, `PAIRED-vs-adopted-1.0.json` `ae559e9a…` (+1.929 [+0.607, +4.144]); same-renderer +2.507
  [+1.037, +4.599]; Nimble v2 +5.681 [+3.190, +10.090]; JPT-9B +6.743 [+1.114, +9.952] (internal only); mlx
  `runs/release/dev2-8b-t1-derived-mlx/mlx-diag.score.json` `0ab43a76…`. At T = 1 the typed panel has Brier 0.0997 / ECE
  0.0165 (CAL698 run: 0.1131 / 0.0482) and CSS15 median Brier 0.5548 / ECE 0.1137 (CAL698 run: 0.5272 / 0.0793).

## 3. Storage precision: BF16 copy tested and adopted

- **Rule** (coordinator 13:45 and the task brief): adopt a BF16 copy only with EXACT answer parity (0 changes) on every
  scored prompt and on mlx-diag; otherwise ship the scored bytes.
- **Design** (`v2.release.bf16_copy`, commit `7e20cd36b`, unit tests incl. a torch round trip passed in the scored image):
  the release runtime loads every parameter as FP32 (`from_pretrained(dtype=float32)` + `.float()`) and runs the backbone
  under BF16 autocast, so every Linear weight is rounded to BF16 (round-to-nearest-even) before each matmul. The copy stores
  exactly those 248 projection matrices (`q/k/v/o_proj`, `in_proj_qkv/z/a/b`, `out_proj`, `gate/up/down_proj`; 6,918,504,448
  parameters) in BF16 with the same cast and keeps the other 178 backbone tensors (embedding table, RMS norms, `A_log`,
  `dt_bias`, conv filters; 1,018,180,096 parameters) and the decision head bit for bit in FP32. A full BF16 cast would change
  weights that the runtime uses in FP32 and was not the tested design.
- **Copy** ([launcher](dev2-8b-release-2026-09-29/ops/bf16-copy.sh), [receipt](dev2-8b-release-2026-09-29/bf16/bf16-copy.json)
  `ccdcf1c3…`): node A `runs/release/inputs/dev2-8b-bf16/checkpoint`, identity **`b1ed5a71038b474902d2a6dfebeee0c6f7ce901097553b4621f6f0a23fb6da98`**
  (source re-fingerprinted to `b9d973b3`), 17,946,656,577 bytes (vs 31.78 GB); every non-shard file byte-identical; each
  stored tensor checked equal to its planned cast.
- **Parity test** ([launcher](dev2-8b-release-2026-09-29/ops/bf16-parity.sh),
  [receipts](dev2-8b-release-2026-09-29/bf16/receipts/)): no-upload staging run of the BF16 package through `release.sh`
  (spec `dev2-8b-bf16-parity-staging.json`, mirror `aa957cec7`) on node A GPU6 with a fresh copy of the scored run's cache
  (digest `5604ffdc…` = the scored tree), kernels required, category-only tolerance to count every change:

  | Panel | Prompts | Slots | Answer changes | Missing | Input mismatch | Max probability drift |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | typed-final | 1,600 | 2,000 | 0 | 0 | 0 | 8.9e-16 |
  | css15 | 6,547 | 6,547 | 0 | 0 | 0 | 3.3e-16 |
  | public231 | 231 | 231 | 0 | 0 | 0 | 4.4e-16 |
  | mlx-diag | 2,275 | 2,275 | 0 | 0 | 0 | 4.4e-16 |

  Build, both example processes, repeatability and the card example also passed (0.159 GPU-h). **Decision: BF16 adopted**;
  the release spec ships identity `b1ed5a71` and cites the scored identity `b9d973b3` in `runtime_equivalence`. The cache
  copy changed only in 14 Triton `__grp__*.json` path-group files (absolute paths; compiled kernels and the 56 autotune
  files unchanged), as in the scored run.

## 4. Package and verification

- `qwen-full` package from spec [`dev2-8b-release.json`](../specs/dev2-8b-release.json) (build receipt
  [`build.json`](dev2-8b-release-2026-09-29/release/receipts/build.json)): 40 files — nine backbone shards + index +
  `config.json`, `decision_head.safetensors`, `decision_config.json`, tokenizer, chat template, root `config.json` pointer,
  `MODEL_MANIFEST.json`, the `decision2/` runtime (training/model sources vendored from the scored mirror `3277dec9d`),
  `LICENSE` + `LICENSES/Qwen3.5-9B-LICENSE.txt`, `NOTICE`, `ATTRIBUTIONS.md`, `README.md`, banner + three charts,
  `evaluation/`. Loaded 7,940,895,744 (backbone 7,936,684,544 + head 4,211,200); 39 text files screened; licence
  apache-2.0; build-time draft `5db122b0…`. Card and evaluation text: [package-text](dev2-8b-release-2026-09-29/release/package-text/).
- Launch: [`ops/release-upload.sh`](dev2-8b-release-2026-09-29/ops/release-upload.sh) ([log](dev2-8b-release-2026-09-29/release/extra/launch.log.txt));
  receipts: [`release/receipts/`](dev2-8b-release-2026-09-29/release/receipts/), [`RELEASE-RECEIPT.json`](dev2-8b-release-2026-09-29/release/RELEASE-RECEIPT.json)
  `d729f471…`.

  | Step | Evidence |
  | --- | --- |
  | build | manifest `80770483…`; identity `b1ed5a71` recomputed; vendored sources equal the scored adapter hashes (8 files) |
  | pre-a / pre-b / repeat-pre | six System One Choice / Noul / Score examples in two isolated containers (no network, read-only package, `fla.*` and `causal_conv1d.*` kernels bound); answers `c5dbc71e…`, bit-identical |
  | card-pre | the README's Python block reproduces pre-a |
  | parity-pre | 0 answer changes, 0 missing, 0 input mismatches on typed-final 1,600 prompts / 2,000 slots, css15 6,547, public231 231, mlx-diag 2,275 (tolerance 1e-4; max drift 8.9e-16, 3.3e-16, 4.4e-16, 4.4e-16) |
  | ensure / upload | private repo; revision `0dee8017…`, 40 files, 17,947,416,716 bytes |
  | download / tree | real `hf download` of the revision into a fresh cache; 40/40 re-hashed, equal to the pre-upload package |
  | post / repeat-post / card-post | from the download: answers `c5dbc71e…` (= pre-upload), bit-identical; card block reproduces |
  | parity-post | identical to parity-pre on all four panels |
  | readback | private; `apache-2.0`, `base_model` Lux 1.0 (`finetune`); 0 card problems; 40 remote files, 0 hash mismatches; not in the collection |
  | card HTTP / links | 12/12 targets, anonymous 401 on the model API, README and page; 15/15 links |
  | gate evaluate | the six items pass ([output](dev2-8b-release-2026-09-29/release/extra/gate-evaluate.json)) |

- Runtime: `decision20-train-fast:host2` = `f83b1d10…`, torch 2.12.0+git6bbd260, Transformers 5.17.0, FLA 0.5.2,
  causal-conv1d 1.7.0, Triton 3.7.1, HIP 7.2, `--require-kernels`, `HIP_FORCE_DEV_KERNARG=1`, `TRITON_CACHE_AUTOTUNING=1`;
  cache copy digest `5604ffdc…` before (= the scored tree) and `16f05716…` after, again only 14 `__grp__` path files; the
  source cache is unchanged.

## 5. Card

- 1.0 product design, rendered by `v2/release/card.py` from spec `dev2-8b-release.json`: DEV2.0-8B owl banner, uses table,
  System One call example, same-panel score table, licence-filtered JevArena v3 rank, model × task and public-231 rank
  charts, tradeoffs table; no Pareto chart, no internal gate ledger. Rows: DEV2.0-8B (T = 1 derived run), Decision 1.0 Lux
  as the own 1.0 (native 16K run, the gate comparator; eval-track native mlx `x-lux1`), Nimble v2 (Apache-2.0 with LICENSE,
  card-eligible). JPT-9B (CC BY-NC 4.0) is excluded; Winnow-E4B has no run. mlx-diag shows the Choice and Noul parts only.
- Licence `apache-2.0`: the whole weight / tokenizer / code lineage is Apache-2.0 (Decision 1.0 Lux LICENSE and
  Qwen3.5-9B LICENSE, both `bbedc3fd…`); training-data licences (CC BY-SA, CC BY, OANC terms, MIT, CC0) are credited. The
  Lux `LICENSE` comes from the complete node copy of `bd45a30a` (`/data/decision20-20260926/models/Decision-1.0-Lux-9B`;
  the node HF-cache snapshot of that revision holds only `decision_config.json`; the file equals the `cdf4d3ef` blob).
- Disclosures: weight origin (own Lux 1.0 → full fine-tune with own-Lux KL → three-seed soup → interpolation ⅓ / ⅔ back
  to Lux 1.0; Qwen3.5-9B lineage and licence); teacher own Lux 1.0 only (no Jev, no third-party decision outputs); data
  families and licences of the x60 subsample incl. the human arms (cross-domain short, long evidence, multilingual,
  OASST1), all distilled; the gain is typed reasoning with human transfer level; CSS15 regressions (talklife −0.022,
  wiki_corpus −0.021, tropes −0.011, conv_go_awry −0.001); public 231 178 vs 183 (−5 [−11, +1]; hard 64 vs 68) and long
  public inputs 19 vs 22 of 37; vs Nimble v2 (+5.68 v3, typed Choice lower, public level); seed dependence (each seed and
  the soup were below Lux 1.0 on the development proxy; the interpolation carries the gain); multilingual (mlx-diag
  non-English Noul 80.8% vs 83.3%, Korean 70% vs 77%, Spanish 84% vs 88%, English Noul 84% vs 88%; the type macro .822 vs
  .832 includes the XNLI-based Score part and stays off the card per the 22:20 policy); Score level-0 use (42 vs 63);
  calibration at T = 1 and the rejected temperatures; 4 over-limit CSS15 inputs; CPU not verified; storage precision.
- **Eval-track 9B gate record** ([`m4-dev2-9b-gates-2026-09-29.md`](../../eval/records/m4-dev2-9b-gates-2026-09-29.md),
  `fbee86265`, all gates pass, no overlap exposure) is on the card before the upload: its evaluation-familiarity line
  verbatim (84 flagged items; margin +1.93 → +2.06 without them), Score level 4 under-predicted (104 for 128), the
  non-English Noul interval (−2.5 [−4.7, −0.3] points), the long-input intervals, and the Nimble v2 notes (LoRA over
  Qwen3.5-9B, bundled scorer on ROCm, native 8,192-token limit with 21 invalid human-transfer answers, parameter count with
  vision tower and unmerged LoRA); the evaluation page adds the paired Nimble v2 interval (`card.paired_peers`). Every
  number on the card agrees with that record.
- **Placeholder (marked):** the JevArena-C1 line only (a card-only revision after the coordinator's C1 event).
- Two upload runs were stopped by the worker during pre-upload parity, before any Hub write
  ([logs](dev2-8b-release-2026-09-29/release/extra/)): 07:07:46Z, 33 s into parity, to correct the card's calibration
  sentence (typed ECE 0.016 vs 0.017 as in the score table; the 27B worker's four-decimals lesson) and three wording
  details; 07:12:14Z, 3 min into parity, when the eval track's 9B gate record appeared on the integration branch, so that
  its disclosures ship with the first revision (the DEV2.0-4B lesson). Merging the integration branch brought a
  `training.model.infer` change (0.6B `8730d9413`, `--score-bias`), so the spec now vendors the inference sources from the
  scored run's own mirror `3277dec9d` (`vendor_source`, the 27B worker's `a60409303`); all eight vendored files equal the
  scored adapter hashes, and the runtime, example and launcher files are unchanged since the BF16 parity run.

## 6. Draft decision (stop before `--collect`)

- Build-time draft [`DEV2.0-8B.decision.build-draft.json`](dev2-8b-release-2026-09-29/DEV2.0-8B.decision.build-draft.json)
  `5db122b0…` (the spec's `gate_receipt`; bound in `build.json`).
- **Verified draft** [`DEV2.0-8B.decision.draft.json`](dev2-8b-release-2026-09-29/DEV2.0-8B.decision.draft.json)
  **`1db7683f…`**: `status: draft`, `decided_by: null`; identity `b1ed5a71…`, report `66106ef9…`, paired `ae559e9a…`;
  `verified_package` (revision, manifest, bytes, loaded count, scored identity, builder and vendor commits, receipt hashes),
  eleven disclosures, pending steps; `gate.check` passes against the spec. Both drafts are on node A in
  `/data/dev2/runs/release/decisions/`.
- Not done here (coordinator): the final decision (`DEV2.0-8B.decision.json`), the C1 line and `--collect`.

## 7. Hugging Face storage

- `hf_headroom.sh` before the upload: 44.63 GB used, 55.37 GB free (37 private repos; [receipt](dev2-8b-release-2026-09-29/storage/headroom-before-0637Z.txt));
  the launcher re-ran it with `--min-free-gb 19` right before the run. After: **62.57 GB used, 37.43 GB free** (38 repos;
  DEV2.0-8B 17.95 GB; [receipt](dev2-8b-release-2026-09-29/storage/headroom-after-0741Z.txt)).
- BF16 storage saved 13.83 GB against the scored FP32 bytes (31.78 GB). No staging upload (the BF16 test was a no-upload
  staging run; `dev2-release-staging-8bbf16` was never created), no deletions (no `rewrite_history` call was needed).

## 8. GPU-hours, commits and notes

- Node A GPU6 (taken explicitly; lease now `status=idle`): BF16 parity run 0.159 GPU-h; two upload runs stopped before
  upload 0.051 + 0.095; release run 0.324 → **0.628 GPU-hours**. CPU only (0 GPU-h): the development-panel calibration
  check, the T = 1 derivation and the BF16 copy.

- Commits (branch `xunzhuo/decision-2-training-release-9b`): merge of the 27B release worker's shared-module prefix
  `0347ccc2e` (`2417f4033` name_basis, `aeff0718d` no-1.0 profile, `9a1df2602` board-sibling licence, `0347ccc2e` banner
  --label; not re-implemented) → `5dcb8fc6e`; state `0b95ad29b`; `7e20cd36b` **new shared module** `v2.release.bf16_copy`
  with tests; ops `08382e424`; banner `efd36ac20`; specs `aa957cec7`; BF16 adoption, build draft and launcher
  `11486b412`; card fix `86e220861`; integration merge `5b7bc7a34`; gate disclosures and `vendor_source` `05914b8e8`;
  records, receipts and the merge into `xunzhuo/decision-2-training` (see `git log`).
- Notes: the scored run's `COLLECT.json` / `REPORT.json` `batch_policy` text ("LoRA over the pinned Decision 1.0
  source…") is stale for this full checkpoint and is not used on the card; `decision_config.json` keeps the soup block's
  container-relative member paths (as DEV2.0-4B); 2,224 of the 122,651 own-Lux targets (A0s-strict) have no recorded
  runtime (the other 120,427 came from the image without causal-conv1d), hence "mostly" on the card; node A GPU6 was taken
  explicitly (the 9B chain's `reserved` entry is kept as `gpu6.lock/owner.prev-9b-chain`); GPU7 was not used.
