# DEV2.0-0.6B: private build, upload and verification (2026-09-28)

## Erratum, 2026-09-29 ≈08:25 UTC+8: cited staging revisions no longer exist

Four storage cleanups on 2026-09-29 (decoder M3 at 02:15 and 02:32, release at 04:47, decoder M4 at 04:51 UTC+8)
called `permanently_delete_lfs_files` with its huggingface_hub 1.33 default `rewrite_history=True`, which rewrote every
commit of the staging repositories they touched. The staging revisions `545a6784`, `784a894f` and `16c0929a`
(`dev2-dec-staging`), `62c61c10` (`dev2-release-staging-06bm4`) and `5afd8fc8` (`dev2-release-staging`) no longer
exist. This record's citations of `dev2-release-staging-06bm4@62c61c10…` ("Update 23:50" and "2. Package") are among
them. Artifact identity rests on the per-file SHA-256 in each package's `MODEL_MANIFEST.json` plus the verified node
copies, not on those staging revision IDs. The released repository was not rewritten: `7b5d3ff2`, `a87eeb72`,
`e61b2b44` and `99c4e799` still resolve (Hub check, 2026-09-29 ≈08:20 UTC+8). Details:
[storage steward record](hf-storage-steward-2026-09-29.md) §4.

## Final release, 2026-09-29 ≈02:11 UTC+8 — revision `99c4e799`, in the private "Decision 2.0" collection

The coordinator approved the release at 02:05 UTC+8 (full-autonomy mandate) after JevArena-C1 event 2
([eval record](../../eval/records/m4-dev2-06b-c1-event2-2026-09-29.md)). The release is a card-only
revision of the frozen T = 1 package `e61b2b44`.

- **Final revision `99c4e799392afa73241915aeb16fefa5ef3518d7`** of private `llm-semantic-router/DEV2.0-0.6B`
  (now `main`). Manifest `0affd1b13fd41446011f97cb723ce0d48ef1ed01e72d2837123518b34ccc2282`, 31 files, no
  `calibration.json`, 597,103,104 loaded parameters.
- **Final decision** [`DEV2.0-0.6B.decision.json`](dev2-0p6b-release-2026-09-28/DEV2.0-0.6B.decision.json),
  SHA-256 **`e22fe181f6c245a15a1c341d136141ef529768c0b8fecb92d65ee909722b15b0`**. It is `status: final`,
  decided by the coordinator, and names identity `5b30b7e2…`, report `72cf01de…` and paired file `bd4ad6c7…`.
  It supersedes the drafts `bc59b2af…` and `4946f666…`. The run's `gate.json` (`787c31c9…`) binds it to
  `99c4e799…` and manifest `0affd1b1…`; all six gate items pass.
- **Card changes** (spec `specs/dev2-0p6b-release.json`). The C1 placeholder became the event 2 line:
  33.21 vs Kai 1.0 17.77, +15.44 [+13.51, +17.21]; Bosun 35.03, −1.82 [−4.05, +0.35]; GLiNER2.5-Decide 22.82;
  Lex 20.82; plus the input-limit caveat. Three limits were added:
  - typed accuracy below Bosun (−0.038 [−0.068, −0.010]);
  - typed Noul only narrowly above chance, and typed Score leaning on the lowest and highest levels;
  - the C1 weak spots versus Kai (Arabic HalluTruthQA, narrative event causality, Arabic overall; Noul the
    weakest type) and versus Bosun (weaker on Choice and Noul, stronger on Score).

  The existing disclosures are unchanged. Every named comparator passes the 22:20 licence policy: Bosun is
  Apache-2.0 with a LICENSE file; GLiNER2.5-Decide declares Apache-2.0 without a LICENSE file (noted on the card);
  Kai and Lex are our own.
- **Package = `e61b2b44` plus the card.** Only `README.md` and `MODEL_MANIFEST.json` differ from `e61b2b44`,
  both in a CPU build-only preflight and on the Hub. In the manifest, the changed fields are the README hash,
  the card hash and `builder.source_commit`. The evaluation page is unchanged.
- **Verification.** One `release.sh --upload --collect` run from the mirror of `61d22cd30` passed all 17
  steps in about 150 s ([command](dev2-0p6b-release-2026-09-28/final/extra/launch-command.txt)). It ran on
  node A GPU5 via the shared-lease entry `owner.release` (18:08:24–18:11:20Z, then removed; the owner entry
  was not touched).
  - Native examples are bit-identical in three processes, answers `a7da72d4…`, the same as `e61b2b44`.
    The card example reproduced before and after upload.
  - **Exact parity (tolerance 0)** against the raw scored predictions on 600 prompts (typed 150, transfer
    200, public 100, mlx-diag 150): 0 changes and drift 0.0, before and after the real download.
  - A real `hf download` of `99c4e799…` with a fresh cache re-hashes 31/31 files.
  - **Weights identical to `e61b2b44`.** All seven model files have equal SHA-256 in the download and in
    the old manifest, and equal remote LFS / blob ids at both revisions; the identity is still `5b30b7e2…`
    ([compare](dev2-0p6b-release-2026-09-28/final/extra/revision-compare.json)).
  - Readback: repository private at `99c4e799…` (= `main`), 31/31 remote hashes, Hub card `apache-2.0`,
    no card problems. HTTP check: 12/12 card images and files; anonymous model API, README and page refused
    (401). `hub_links`: 14/14.
  - Collection "Decision 2.0": private, **exactly two items**, `llm-semantic-router/DEV2.0-0.8B` and
    `llm-semantic-router/DEV2.0-0.6B`.
- **GPU-hours 0.042** (node A GPU5). The preflight build and the extra readbacks used no GPU.
- Superseded revisions stay in the history: `e61b2b44` (frozen T = 1 package, placeholder card) and the
  earlier `7b5d3ff2`, `cb2bfd76` and `a87eeb72` (never use `a87eeb72`).

Receipts: [`final/`](dev2-0p6b-release-2026-09-28/final/).

## Update 23:50 UTC+8: CAL698 rejected on development panels; T = 1 revision `e61b2b44` is current

Coordinator rule 23:15 (supersedes 22:30): CAL698 temperatures are adopted only if they do not worsen
aggregate calibration on the development panels. `v2/release/dev_calibration.py` re-tempered the stored
development predictions of the same weights (`m4-t-a7-soup` readout, identity `5b30b7e2…`, T = 1, 8,192
tokens; 0 answer changes) and scored them with the development scorers
([receipt](dev2-0p6b-release-2026-09-28/devcal/dev2-0p6b.json)):

| Development metric | Raw | CAL698 |
| --- | ---: | ---: |
| Typed DEV Brier / ECE-10 | 0.336 / 0.169 | **0.396 / 0.300** (worse) |
| CSS pilot median task Brier-sum / ECE-15 | 0.790 / 0.072 | 0.790 / 0.058 |

Typed-DEV Brier and ECE worsen, so **CAL698 is rejected and the package ships T = 1**.

- **Current private revision `e61b2b4419383672cb6a92d63699f7974e5f81ac`, manifest
  `a5cdabedf93835f028b871d0dc64ad8595e5369128f994d07bdee44dc1e999a9`** (31 files, no `calibration.json`).
  Model files and identity equal `7b5d3ff2`; only `README.md` and `evaluation/EVALUATION.md` differ (the
  card's calibration line now says the CAL698 temperatures were evaluated and not adopted because they
  worsened out-of-distribution calibration; the GLiNER2.5-Decide LICENSE note; score table back to the raw
  0.309 / 0.118). All 15 release steps pass ([receipts](dev2-0p6b-release-2026-09-28/t1/release/)):
  examples bit-identical in three processes (answers `a7da72d4…`, as for `7b5d3ff2`), card example,
  **exact parity (tolerance 0) against the raw scored predictions** on 600 prompts before and after the
  real download (full-panel exact parity was shown for the same model and runtime bytes at `7b5d3ff2`),
  re-hash 31/31, readback, `hub_links` 14/14, HTTP check 12/12 with anonymous 401, six gate items.
- **Do not use `a87eeb725b4edd5e7612342be662c4d11d470487`.** The first T = 1 upload kept the previous
  revision's `calibration.json` because `upload_folder` never deletes; the re-hash step failed and the
  chain stopped ([receipts](dev2-0p6b-release-2026-09-28/t1/failed-a87eeb72/)). Fixed in `89294e89f`
  (`hub upload` passes `delete_patterns=['*']`; the Hub keeps `.gitattributes`), then re-uploaded.
- **Draft decision:** build-time [`…build-draft-t1.json`](dev2-0p6b-release-2026-09-28/DEV2.0-0.6B.decision.build-draft-t1.json)
  (`4946f666…`); verified draft for the coordinator [`DEV2.0-0.6B.decision.draft.json`](dev2-0p6b-release-2026-09-28/DEV2.0-0.6B.decision.draft.json)
  (`bc59b2af…`, names `e61b2b44`, supersedes `f52e7873…`, `f752dad2…` and `72bc767e…`). Report `72cf01de…`,
  paired `bd4ad6c7…` (v3 43.541, +7.60 [+4.70, +10.76]).
- **Scored identity for C1 event 2:** `llm-semantic-router/DEV2.0-0.6B` at
  `e61b2b4419383672cb6a92d63699f7974e5f81ac` (manifest `a5cdabed…`), weights `model_sha256` `5b30b7e2…`,
  **no calibration (T = 1)**, 8,192-token limit, over-budget inputs invalid. Run the package runtime, or
  `training.model.infer` with the eval adapter spec `v2/06b/records/adapters/dev2-06b-causal-8k.json`
  (`--model-path <package or staging 62c61c10 checkpoint> --extra model_id=…`) on node A; this is exactly the
  configuration of the scored run `m4-t-a7-soup`.
- The CAL698 revision `cb2bfd76` (section below) is superseded.
- **GPU-hours this round:** 0.193 on shared node A GPU0 (0.6B failed upload 195 s, 0.6B re-upload 358 s,
  0.8B local verification 142 s); lease entry `owner.release` set back to idle.

## Update 23:15 UTC+8: CAL698-calibrated package, revision `cb2bfd76`, supersedes `7b5d3ff2`

Coordinator decision 22:30: the frozen release package includes its CAL698 calibration before any
C1 event or collection add. **New private revision `cb2bfd76e9d8a5e1b9a968dcafb7e18d8c01b3b9`,
`MODEL_MANIFEST.json` `83d6ec2b58a7a454b90711e05fc543cfadc7b13f0fb532e60a0802d6afb9db75`** (32
files). Stopped again before `--collect` (the collection holds only DEV2.0-0.8B). Receipts:
[`dev2-0p6b-release-2026-09-28/cal698/`](dev2-0p6b-release-2026-09-28/cal698/).

- **Fit.** `v2/release/calibrate_frozen.py` on CAL698 (`19cc1a8c…`, 698 rows: 319 Choice / 289 Noul /
  90 Score; never panel, SELECT or sealed items) for the frozen weights (identity `5b30b7e2…`), node A
  GPU0, scored image and execution (BF16 backbone, FP32 head, 8,192 tokens, one question per forward).
  [`calibration.json`](dev2-0p6b-release-2026-09-28/cal698/calibration.json) `e1f7c909…`,
  `frozen_checkpoint` policy: **T Choice 0.5917, Noul 0.5880, Score 0.3452** (all below 1: the soup is
  under-confident in distribution). CAL698 before → after: NLL 0.447 → 0.396, Brier 0.118 → 0.111,
  ECE 0.079 → 0.031 (Choice ECE 0.092 → 0.034, Noul 0.056 → 0.011, Score 0.110 → 0.161).
- **Calibrated predictions.** One re-score with the eval track's frozen runner (registry adapter
  `decision2-typed` = `training.model.infer` + `--calibration`, 8,192 tokens) on typed-final, css15,
  public231 and mlx-diag. `v2/release/temperature_parity.py` against the raw scored run: **0 answer
  changes on all 11,053 slots** (typed 2,000, transfer 6,547, public 231, mlx-diag 2,275), and the
  calibrated numbers re-derived offline as softmax(log p / T) from the raw scored probabilities equal the
  re-score within **1.3e-15** (so the logits were bit-identical and the calibration is exact).
  The calibrated report ([REPORT](dev2-0p6b-release-2026-09-28/cal698/rescore/REPORT.json) `ffbb32a4…`)
  has v3 43.541 (T 0.3953, H 0.4796), public 231 142 (48 / 54 / 40), identical per-task macro-F1, and the
  same paired interval vs Kai 1.0 (+7.60 [+4.70, +10.76], `53df7087…`); mlx-diag accuracies identical.
- **Calibration on the panels (raw → CAL698), disclosed on the card:**

  | Metric | Raw | CAL698 |
  | --- | ---: | ---: |
  | Typed Brier / ECE-10 | 0.309 / 0.118 | 0.332 / 0.191 |
  | Typed Choice Brier / ECE | 0.352 / 0.156 | 0.380 / 0.227 |
  | Typed Noul Brier / ECE | 0.231 / 0.137 | 0.240 / 0.177 |
  | Typed Score Brier / ECE | 0.378 / 0.040 | 0.423 / 0.226 |
  | Transfer median task Brier-sum / ECE-15 | 0.565 / 0.070 | 0.598 / 0.120 |
  | Public 231 Brier / ECE-15 | 0.233 / 0.116 | 0.253 / 0.144 |

  In-distribution sharpening makes the already slightly over-confident panel probabilities worse; still
  better than Kai 1.0's raw typed 0.390 / 0.209.
- **Package and verification** (`release.sh --gpu 0 --shared-lease release`, all 15 steps pass,
  [RELEASE-RECEIPT](dev2-0p6b-release-2026-09-28/cal698/release/RELEASE-RECEIPT.json)): every model file
  byte-identical to `7b5d3ff2` (identity unchanged); changed files only `calibration.json` (added),
  `config.json` (pointer names it), `README.md`, `evaluation/*`. Native examples bit-identical in three
  processes; card example reproduced; parity against the calibrated re-score on 600 prompts (typed 150,
  transfer 200, public 100, mlx-diag 150) **0 changes, drift 0.0** before and after the real
  `hf download`; full-panel equality follows from the re-score equivalence above and the earlier
  bit-exact raw parity on every prompt. Re-hash 32/32; readback private, 32 remote hashes, no card
  problems; `hub_links` 14/14 and HTTP check 12/12, anonymous 401; all six `gate evaluate` items pass.
- **Card.** Score table and evaluation page show the calibrated Brier / ECE; calibration line states the
  temperatures; a limit states the panel calibration tradeoff; the GLiNER2.5-Decide LICENSE note is
  now rendered; the C1 line stays a placeholder.
- **Draft decision.** Build-time draft
  [`…build-draft-cal698.json`](dev2-0p6b-release-2026-09-28/DEV2.0-0.6B.decision.build-draft-cal698.json)
  (`f752dad2…`, supersedes `72bc767e…`); verified draft for the coordinator
  [`DEV2.0-0.6B.decision.draft.json`](dev2-0p6b-release-2026-09-28/DEV2.0-0.6B.decision.draft.json)
  (`f52e7873…`, names revision `cb2bfd76` and manifest `83d6ec2b…`). The finalizing rerun must point the
  spec's `gate_receipt` at the final decision.
- **Scored identity for C1 event 2:** DEV2.0-0.6B = private `llm-semantic-router/DEV2.0-0.6B` at
  `cb2bfd76e9d8a5e1b9a968dcafb7e18d8c01b3b9` (manifest `83d6ec2b…`): weights `model_sha256`
  `5b30b7e2…` + `calibration.json` `e1f7c909…` (CAL698 `19cc1a8c…`), 8,192-token limit, over-budget
  inputs invalid. Run the package runtime, or `training.model.infer` through adapter `decision2-typed`
  with `--extra max_length=8192 --extra calibration=<the package's calibration.json>` on node A. Answers
  (so accuracy) are the same with or without calibration; probability metrics must use the calibrated
  package.
- **Launcher change.** `run_same_panel.sh` and `release.sh` gained `--shared-lease NAME` (writes only
  `owner.NAME`, never reads or rewrites the owner's entry, skips the idle-VRAM gate), because GPU0 is shared
  with the running 0.6B jobs. Default behaviour unchanged.
- **GPU-hours this round:** 0.497 on shared node A GPU0 (fit 93 s, re-score 1,087 s + 338 s, release run
  272 s; about 5× slower than idle because of the concurrent 0.6B training).

**Result: every verification step passed; the package is in the private repository
`llm-semantic-router/DEV2.0-0.6B` at revision `7b5d3ff2bf338194b0b485bda4d08b30b8fea88e`.
Stopped before `--collect`: nothing was added to the "Decision 2.0" collection (still 0
items).** The build-time decision is a `status: draft` binding
([`DEV2.0-0.6B.decision.build-draft.json`](dev2-0p6b-release-2026-09-28/DEV2.0-0.6B.decision.build-draft.json),
SHA-256 `72bc767e708364bc517ec6c331a05859924a651ddb0a72af59eaf34e23e1d1d0`); the coordinator
finalizes it after JevArena-C1 scoring event 2. Receipts:
[`dev2-0p6b-release-2026-09-28/`](dev2-0p6b-release-2026-09-28/).

The candidate is the 0.6B track's Milestone 4 `m4-t-a7-soup` (coordinator note 21:30;
[results](../../06b/records/m4-results-2026-09-28.md)): official Qwen3-0.6B-Base `da87bfb6`,
(a2) recipe on mixture T (A0s-r + A6 + the A7 v3 Eos/Lux natural pool, own-Lux teacher on
6,430 A0s-r rows), uniform soup of three same-init seeds. The previous, different package
with this name was deleted by the user and is neither restored nor referenced: the new
repository has exactly two commits (the Hub's initial commit and this upload), and the old
package's internal control run is not a card input.

## 1. Renumbered data (coordinator decision 21:30 (2)): PASS

`v2/release/pk1_rows.py` compared mixture T (`98e4e859…c5d549a`, the file all three seeds
trained on) with the published positional-key A0s (private dataset revision
`d8eae3e4fb5b91871c5aa7c13f0d94ea96e86ea7`, `m3/pk1/A0s/train.jsonl` `35c1f98f…ddc20b`, fresh
`hf download`) minus the two excluded families ([receipt](dev2-0p6b-release-2026-09-28/pk1-rows.json),
`5e79927c…`):

- 6,547 / 6,547 kept rows present, **byte-identical canonical lines** (ids, order, every
  field), forming the first 6,547 rows of mixture T; 0 differing fields.
- The 752 excluded rows (`natural_cosmos_qa` 448, `natural_squad2_answerability` 304) are
  absent from the mixture.
- Against the pre-renumbering A0s (`df90cb7f…`), 117 kept rows were renumbered; all 117
  appear only in renumbered form (0 still in the original form).
- No other mixture row repeats a kept row's input.

So the candidate's builder-renumbered A0s counts as the published renumbered data. The
own-Lux targets it used are the decoder file `752b7c8f…`; the canonical pk1 Lux file is
`56627939…` (agreement 99.5%, coordinator decision (3)). The 117 renumbered rows got no
teacher target.

## 2. Package (`dev2-package/1`, profile `qwen-full`)

| Item | Value |
| --- | --- |
| Checkpoint | fresh `hf download` of private `llm-semantic-router/dev2-release-staging-06bm4` at `62c61c10b969658e855399a8b357701be96783ff`; all 7 model files byte-identical to the scored export on node A and to the scored `model_files_sha256` |
| Scored identity | `model_sha256` `5b30b7e2e612cbf4a45dcab1cbe2bd05b2b6d25cfe91c376031cb22d5ce303a8` (recomputed by the builder from the checkpoint and by the runtime from the downloaded bytes) |
| Scored run | `/data/dev2/runs/06b/m4/formal/m4-t-a7-soup` (+ `-mlx`): report `72cf01de…`, seal `4e02a0b0…`, predictions typed `16f01c21…`, transfer `7b1b07f2…`, public `7c1e9c61…`, mlx-diag `3cf5b5e0…`; paired vs adopted Kai 1.0 `bd4ad6c7…` |
| Inference sources | vendored `infer.py`, `decision_model.py`, `data.py`, `lora.py`, `source.py` equal the scored adapter sources (build-time check against the scored predictions manifest `c0057be0…`) |
| Calibration | none: the scored run used raw probabilities (temperature 1.0), so the package ships no `calibration.json` and reproduces the scored probabilities exactly |
| Input limit | 8,192 tokens, as scored (Qwen3 position limit 32,768; nothing was evaluated above 8,192); longer inputs are rejected, never truncated |
| Parameters | **597,103,104 loaded** (backbone 596,049,920 + head 1,053,184) from safetensors headers = the count the runtime asserts on load; tier 0.6B (ratio 1.005), so the name `DEV2.0-0.6B` |
| Package | 31 files, 2,400,645,139 bytes; **`MODEL_MANIFEST.json` `ce81f9b809d21b6cde22d8d4ce7835154926fd5aacb9b0411575d199fcefdb66`** ([copy](dev2-0p6b-release-2026-09-28/release/MODEL_MANIFEST.json)); root `config.json` `c9d6b9cc…` |
| Builder | exact mirror of `b655e2d5d744bd9d03d5cbafd815dbf955b25979` (subtree `src/training/decision2`) |
| Licence | `apache-2.0`: own weights and runtime + Qwen3-0.6B-Base (Apache-2.0); root `LICENSE` (Apache-2.0 text `cfc7749b…`), `LICENSES/Qwen3-0.6B-Base-LICENSE.txt` (`832dd9e0…`, Hub file at `da87bfb6`) |

Model-file SHA-256 (every other file is in the manifest copy):

| File | SHA-256 |
| --- | --- |
| `backbone/model.safetensors` | `7e2d1f9d7010cb59fe0e38d3e0fb848222ba12657663dc96776a22221bd82b8e` |
| `backbone/config.json` | `fcf62f425b626755c6df66630215aed57c70b38bf779e6ac9aeff376224310ac` |
| `decision_head.safetensors` | `70dc7451862bec4c88f2f8abcdaae66b7f1f118baa8582e3fa863c1794810046` |
| `decision_config.json` | `5a4a56cae499e02042546424fcdc829425ffe71803a87e888ba754b5070f2246` |
| `tokenizer.json` | `be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506` |
| `tokenizer_config.json` | `cbea7bca9904d56693d2226b00fd564e3be053609d1e50f8f1885510f0f6e790` |
| `chat_template.jinja` | `87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5` |

## 3. Verification (node A GPU0, image `decision20-train-fast:host2` `f83b1d10…`, the scored run's GPU and image)

`release.sh --gpu 0 --track release-06b --env HIP_FORCE_DEV_KERNARG=1 --parity (4 panels) --upload`
(no `--collect`); all 15 steps passed ([RELEASE-RECEIPT](dev2-0p6b-release-2026-09-28/release/RELEASE-RECEIPT.json),
`9dedb0c1…`).

| Step | Evidence |
| --- | --- |
| Native System One examples, pre-upload, two processes | 5 requests (Choice with criteria and with null descriptions, Noul with and without criteria, Score with 3 and 5 levels, structured state, Chinese state) valid; the over-budget request answered with explicit `max_length_exceeded` errors. Answers `a7da72d4…` in both processes, bit-identical (11/11 slots) |
| Card example | the README's Python block, run unchanged in a fresh isolated interpreter, printed the same 3 answers bit for bit (before and after download) |
| Scored-panel parity, pre-upload and post-download | **every scored prompt** vs the sealed predictions: typed-final 1,600 prompts / 2,000 slots, css15 6,547, public231 231, mlx-diag 2,275: 0 category changes, 0 missing, 0 input-digest mismatches, **max drift 0.0** (bit-exact) |
| Upload | repository created **private** by `ensure` (it did not exist), revision `7b5d3ff2…`, still private afterwards |
| Real download | `hf download … --revision 7b5d3ff2… --local-dir <fresh dir>` with a fresh cache: 31 files + the Hub's `.gitattributes`, 2,400,645,139 bytes |
| Re-hash | every downloaded file equals `MODEL_MANIFEST.json` and the pre-upload package (0 missing, 0 extra, 0 mismatched) |
| Load and examples from the download | 597,103,104 parameters loaded and asserted; answers `a7da72d4…`, bit-identical to both pre-upload processes (third process) |
| CPU | rehearsal build + two CPU processes bit-identical (`7db090d4…`) + card example; the downloaded package on CPU gives the same `7db090d4…`; CPU vs GPU: same answer on all 11 slots, probabilities within 0.0071 (FP32 vs BF16 backbone) |
| Gate items (`gate evaluate`) | all six pass with the draft decision ([receipt](dev2-0p6b-release-2026-09-28/release/extra/gate-evaluate-fixed.json)); the first evaluation flagged item 3 only because download receipts carried no `passed` field, fixed by the release track in `a1c087bd6` (merged here) |

## 4. Model card

Rendered by `card.py` ([README at the revision](https://huggingface.co/llm-semantic-router/DEV2.0-0.6B/blob/7b5d3ff2bf338194b0b485bda4d08b30b8fea88e/README.md),
private): the DEV2.0 0.6B owl banner, uses and the three decision types, "Post-key same-panel
results", a clearly marked **C1 placeholder line**, the score table (v3 with T / H, typed
Choice / Noul / Score, public 231 with easy / standard / hard, mlx-diag non-English Choice /
Noul, typed Brier / ECE), the v3 rank, model × task and public-231 rank charts (same-panel
REPORT.json only), the automatic tradeoffs table versus Decision 1.0 Kai (typed Choice 255 vs
277; transfer task `mrf` 46.0% vs 56.6%), the System One example, model details (direct
weight origin Qwen3-0.6B-Base, not Kai; three-seed soup), training (recipe, own Lux 1.0
teacher, data families with licences, what was not used) and limits. No Pareto chart, no
internal gate ledger. Comparators shown after the fail-closed licence filter: Decision 1.0
Kai and Lex (own, Apache-2.0), GLiNER2.5-Decide and Bosun v3.1 0.6B (Apache-2.0); none
excluded. Card metadata: `license: apache-2.0`, `base_model: Qwen/Qwen3-0.6B-Base`,
`base_model_relation: finetune`.

Choices made here:

- **Kai comparator.** The table uses the adopted Kai 1.0 run (35.938; the published 1,024-token
  profile and the eval track's 0.6B comparator); a limits line gives the stricter same-limit
  control (Kai 1.0 at 8,192 tokens: 35.97, public 127, paired +7.57 [+4.43, +10.59]). Its report
  carries a track-local model id, so it cannot be the card's own-1.0 row without relabeling.
- **Paired interval rounding.** The card prints +7.60 [+4.70, +10.76]; the lower bound is
  4.7049 (the M4 record's "+4.71" is a rounding slip).
- **Tier leader.** Ranked 1 of 5 on v3, but the limits say the margin over GLiNER2.5-Decide
  (+1.02 [−1.77, +7.80]) includes zero and its typed accuracy is higher.
- **Tokenizer warning.** Transformers 5.17 warns about a Mistral tokenizer regex whenever a local
  tokenizer directory contains a `config.json` without `transformers_version` (our root pointer);
  it only sets `fix_mistral_regex=False` and leaves tokenization unchanged (bit-exact parity above).
  The card tells users it can be ignored; a pipeline-level fix is optional.
- **Collection link.** The card links the private "Decision 2.0" collection it will join after the
  coordinator's final decision; the repository is not in it now.
- **Peer without a LICENSE file (coordinator card policy, 22:20, after this upload).** GLiNER2.5-Decide
  declares Apache-2.0 in its model card metadata but ships no LICENSE file (pinned roster). The
  uploaded revision `7b5d3ff2…` does not say so yet; the spec now carries `comparator_note` (a small,
  tested card-renderer addition), so the finalization rerun prints it after the rank scope and on the
  evaluation page. The rest of that policy already holds here: the mlx-diag Score part (XNLI) is not
  shown, the label is `apache-2.0` with training-data licences credited, and CPU support is claimed
  only because it was verified. A CPU build of the staged spec on node A (mirror of integration
  `f41783939`, no upload) succeeds: only `README.md` and `evaluation/EVALUATION.md` differ from the
  uploaded revision (the note appears once in each); identity, parameters and every model file are
  unchanged; card checks report no problems.

## 5. Hub readback: PASS

- `hub readback`: private, exact revision, 31 remote files with matching LFS SHA-256 or git blob
  ids, Hub-parsed card metadata equals the front matter, no card problems, collection
  "Decision 2.0" private with **0 items** and without this repository.
- `hub_links` ([receipt](dev2-0p6b-release-2026-09-28/release/extra/hub-links.json)): 14 links and
  images of README.md and ATTRIBUTIONS.md fetched from the Hub at `7b5d3ff2…`; every relative file
  re-hashed equal to the manifest (banner, three charts, EVALUATION.md, LICENSE, NOTICE,
  ATTRIBUTIONS.md), Qwen3-0.6B-Base and Lux 1.0 pages exist (public), the collection exists
  (private), anchors resolve.
- The release track's `hub_card_http_check` ([receipt](dev2-0p6b-release-2026-09-28/release/extra/hub-card-http-check.json)):
  12 targets pass; the model API at the revision reports private, `apache-2.0`, base model and
  32 files; **anonymous requests for the model API, README and page are refused (401)**.
- Rendered page: Hub web pages refuse token auth (401), so rendering is evidenced by the
  Hub-parsed card metadata and the per-file checks.
- Repository history ([receipt](dev2-0p6b-release-2026-09-28/release/extra/repo-history.json)):
  created 2026-09-28 14:06 UTC, two commits, branch `main` only, no tags.

## 6. Draft decision and what finalizing needs

[`DEV2.0-0.6B.decision.build-draft.json`](dev2-0p6b-release-2026-09-28/DEV2.0-0.6B.decision.build-draft.json)
(`72bc767e…`, also at `/data/dev2/runs/release/decisions/` on node A) binds the identity
`5b30b7e2…`, the report `72cf01de…` and the paired comparison `bd4ad6c7…`, cites the pk1 result,
and has `decided_by: null`. `gate seal` (and so `--collect`) refuses it. To finalize, the
coordinator (1) replaces the card's `confirmation` line in `specs/dev2-0p6b-release.json` with
the C1 event 2 result, (2) writes a `status: final` decision with `decided_by` and points the
spec's `gate_receipt` at it, and (3) reruns the same command with `--upload --collect` into a new
work directory (a card-only change: the model files and identity stay the same, the README and
manifest change, so the collection gets the new revision):

```bash
SRC=<mirror of the finalizing commit>; S=/data/dev2/src/$SRC/src/training/decision2
G=/data/dev2/private/panels/goldfree; R=/data/dev2/runs/06b/m4/formal/m4-t-a7-soup
$S/v2/release/release.sh --spec $S/v2/release/specs/dev2-0p6b-release.json --src $SRC \
  --work /data/dev2/runs/release/dev2-0p6b-final-<utc> --gpu 0 --track release-06b --threads 4 \
  --env HIP_FORCE_DEV_KERNARG=1 --mount $G --mount $R/output --mount $R-mlx/output \
  --parity typed-final:$G/typed-final.prompts.jsonl:$R/output/typed-final.predictions.jsonl:1600 \
  --parity css15:$G/css15.prompts.jsonl:$R/output/css15.predictions.jsonl:6547 \
  --parity public231:$G/public231.prompts.jsonl:$R/output/public231.predictions.jsonl:231 \
  --parity mlx-diag:$G/mlx-diag.prompts.jsonl:$R-mlx/output/mlx-diag.predictions.jsonl:2275 \
  --upload --collect
```

## 7. Open questions for the coordinator

- **Calibration.** Research & data's 18:45 note says release candidates fit their final
  calibration on CAL698; the 0.6B track scored raw probabilities, so this package ships none
  (typed Brier 0.309 / ECE 0.118, both better than Kai 1.0's 0.390 / 0.209). A CAL698 fit would
  change probabilities (not the argmax) relative to the scored run and would need a new scored run
  to keep the card honest; a card-and-calibration revision can follow if wanted.
- **Licence label (settled by the 22:20 card licence policy).** `apache-2.0`: every shipped
  upstream part (Qwen3-0.6B-Base weights and tokenizer) is Apache-2.0. Training data under
  share-alike terms (SNLI, KLUE STS, JGLUE JSTS: CC BY-SA 4.0; IBM ArgQ-30k: CC BY-SA 3.0) and
  MultiNLI (OANC terms) are credited on the card and in ATTRIBUTIONS.md, as for the 0.8B package.
- **Tool overlap.** `v2/release/hub_links.py` (this worker) and `v2/release/tests/hub_card_http_check.py`
  (0.8B worker) were written in parallel and check nearly the same things; both ran here.
  Consolidating them into one module is a small follow-up for the release track.

## 8. GPU-hours and commits

GPU jobs on node A GPU0: release run 507.5 s wall (**0.141**) + the CPU-versus-GPU examples check
14 s (0.004) = **0.145 GPU-hours**; the lease was held 14:00–14:12 UTC and returned to the 0.6B
track (idle). The CPU rehearsal, the pk1 check and the CPU runs used no GPU.

Commits on `xunzhuo/decision-2-training-release-06b`: `f92e0e66c` (spec, `pk1_rows.py`),
`aafd9c4c5` (pk1 receipt, draft decision), `b655e2d5d` (card wording: measured CPU difference,
tokenizer warning; the builder source of the uploaded package), `ed91f045c` (`hub_links.py`),
merges of the release and integration branches, and this record.
