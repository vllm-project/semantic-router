# 0.8B / 2B Index path — state (continuation of 1afc17e8; polled ≤ 30 min; newest first)

Assignment: COORDINATION 2026-10-02 02:05 (Index path), 04:25, 04:40 and the 09:40 watchdog note: apply the Index path
to M16's 0.8B / 2B finalists `08b-RA-a75`, `08b-RASD-a75` and `2b-RASD-a25` and release the best qualifier per tier
to `Decision-2.0-Eos-0.8B` / `Decision-2.0-Sol-2B`. **Superseded at 09:55 by the user's Index-first rule** (below).
Branch `xunzhuo/decision-2-training-dec-08bfast`. Index values stay private (node C
`/data/dev2/private/eval/index021/ix1/runs/<name>/` and `decision2-program/private/m16-indexpath/`); this file holds
verdicts, counts and hashes only.

## 2026-10-02 06:55Z — 2B release inputs final; upload waits for another worker's `release.sh` on node A

- **Index run `M16-2b-RASD-bf16` (node C GPU1–3):** shards 05:47–06:27Z, 120,224 ok + 2 unsupported, scorers
  pass; co-tenant entries removed. FP32 and BF16 runs agree on 119,520 of 120,226 choices. Full-panel and
  transfer-only bootstraps (2,000 replicates, seed 20261002) done 06:41Z: **IF1 passes** (95% lower bound > 0;
  values private).
- **`stage_private` (mirror `41fb29c6c`):** bootstraps, both IX1 receipts and the `out-2bRA` audit copied to node
  A (hash-checked); `card_index` input `7ee4035e…` (decision 1.0 entrants and footnote equal to the 9B release's).
  Assets rendered on node A (card-render venv; receipt `5c7a370a…`, 5 files).
- **Spec / decision (`a1382cee2`):** `dev2-2b-ixf.json` `c7f34871…`, `Decision-2.0-Sol-2B.decision.ixf.json`
  `091db6d7…` (`decided_utc` 05:05Z = the coordinator's 13:05 UTC+8 choice); `make_ixf --check` on node A is
  byte-equal for 2B and 0.8B. Sol main is still `8ed41433`; Transformers 5.18 site `93df9002…`.
- **Held:** a 27B prerelease `release.sh` (lease `release-27b-27bif`, started 05:57Z) runs on node A. The 2B
  upload starts only after it ends (never concurrent), on node A GPU0 or GPU1, whichever is free.

## 2026-10-02 06:20Z — poll: `M16-2b-RASD-bf16` shards at about 80%

- Node C GPU1–3 (co-tenant `release-2b-ixf`; parity 86 / 86 ok, max |Δp| 0.0) started 05:47Z; each shard is at
  about 32k of 40k rows at about 20 rows/s. `make_ixf` regenerates the committed 0.8B spec / decision byte for byte
  after the card-round change (`e039ed550`). Next: score, bootstraps, `stage_private`, assets, the 2B release.

## 2026-10-02 05:50Z — coordinator 13:05 (option 1): 2B is `2b-RASD`; formal done, types OK; BF16 Index running

- **Formal path on node B GPU1 / GPU5** (reserved `track=dec-2b-formal`). The GPU3 / GPU4 pin was a script default:
  a static render map in `m6-formal-lib.sh` plus `m16-formal.sh`'s guard; the runner itself passes only the GPU
  index on node B. Added `M6_B_GPUS` (render nodes looked up by PCI address, as on nodes E / F), `M6_FORCE_T1` (T = 1
  staging without the CAL698 fit; calibration decision none) and `M16_FORMAL_{GPUS,SELECT,TRACK}` (`b9d56b999`).
  Select `formal/m16/select-ixf/2b-finalists.json` (node B): RAUP = M14 soup `37c85fff…` (9 files, list
  `c1c5601f…`), RASD = node B's M16 copy of the M13 soup `341c2bd2…` (list `8639209f…`); each identity matches its
  M16 lineage receipt. Smoke (8 items) then the finalist collection, both in parallel 05:16–05:22Z: `m16-2b-RAUP` seal
  `5a010225…`, `m16-2b-RASD` seal `478e1b15…` (package revision `847fcf65…`). Relayed to node A; `m16-fscore.sh
  2b run` for both exited 0. **RASD types: choice / noul / score all OK** (R3 passes). The GPU1 / GPU5 owner files
  are released (05:45Z).
- **Choice:** RASD has the largest FP32 Index lower bound (RASD > RAUP > `2b-RA-a75`; values private). The gap to
  RAUP is about 7× the FP32 → BF16 shift `2b-RA-a75` showed, so (as the coordinator allowed) only RASD gets a BF16
  Index run. `2b-RA-a75` is not released.
- **Contamination audit:** both RASD (M13) and RAUP (M14) train on M12's 2B RA TRAIN `08140409…` (hash-checked on
  nodes F / B), so `audit/out-2bRA` covers them.
- **BF16 copy (node C, `ix.sh ckpt` + `bf16`, `2a0010c3a`):** the sweep package's 9 checkpoint files equal node B's
  formal list; `341c2bd2…` → **`40061c70…`** (receipt `dd1c0b9f…`).
- **Index run:** node A GPU6 (the only free GPU) ran 05:33–05:43Z (one shard at a time, about 2.3 h). It was stopped
  and set aside when node C GPU1–3 turned out idle: the 4B run that holds them has finished, merged and
  bootstrapped. Restaged on node C (manifest `bc0a4ffa…`); parity, then all 3 shards in parallel as co-tenant
  `release-2b-ixf`. Node A GPU6 was released.
- **Release inputs (node A, `9b5e2bd32`):** card4 merged in (Sol main `8ed41433`'s spec, gate receipt and decision).
  `make_ixf` / `prep` / `release_ixf` / `render_assets` use the tier's current card (2B round 4) and skip mlx-diag
  / exposure for a point without them. `release_ixf --stage`: current gate `445ca889…`, decision `ea4b72cf…`.
  `prep 2b bf16 / adopt / paired` passed: sealed predictions match, adopted run reproduces the formal report, types
  OK, every paired file equal to the same_panel compare. The formal autotune cache was relayed to node A (manifest
  `f56b0a82…`). The unreleased RA-a75 inputs were moved aside (`*.ra-a75-unreleased-*`).
- **Next:** shards → score → node C bootstraps → `stage_private` (card_index) → assets on node A (its card-render venv
  matches Python 3.12.13 / matplotlib 3.11.2 / Pillow 12.3.0, with the same logo and fonts) → `make_ixf 2b` →
  `release_ixf 2b --release` on top of `8ed41433`, then the purge.

## 2026-10-02 05:05Z — Eos 0.8B released (`3de61185`); 2B choice needs the coordinator

- **Decision-2.0-Eos-0.8B** main = **`3de61185`** (M16 `08b-RA-a75`, BF16 `d9712799…`). Every post-upload check passed;
  gate evaluate PASS. The superseded weights of `9c7f3ea0` were purged (2.02 GB, `rewrite_history=False`, every
  check true; headroom 47.45 GB). The 9B purge script was pinned to Lux-9B, so `ops/purge_superseded.py` takes
  Eos / Sol (`74cc2d9de`). The purge was run with `release_ixf.sh`'s own arguments.
- **2B `2b-RA-a75` BF16 run done** (node A GPU0 / GPU6, 120,224 `ok` + 2 `unsupported`, scorer gate PASS, 1.20
  GPU-h). GPU6 was released and the GPU0 co-tenant entry removed. The run was pulled to node C, and its bootstraps
  are running.
- **2B candidates by lower bound (values private):** `IS-2b-RAUP` > `2b-RA-a75` > `IS-2b-RA`. `IS-2b-RASD` is merged
  and its bootstrap is pending; its point delta is the highest. Neither RAUP nor RASD has a formal run
  (typed-FINAL for R3, and the sealed predictions the release parity needs). The formal library runs on node B
  GPU3 / GPU4, both held by 9B M10. Sol main is `8ed41433` (card-only over `b42b6ff3`).

## 2026-10-02 04:42Z — 0.8B release running; 2B pick moves to `2b-RAUP`, whose formal collection is blocked

- **0.8B:** BF16-run bootstraps (full + transfer-only, 2,000 replicates) done; lower bound > 0 (private). The BF16
  rows match the FP32 run's choices on 120,173 / 120,226 rows. Private inputs staged; Index input
  (`card_index`, default generator) `42055aca…`; assets rendered (banner A, receipt `980fe5c6…`); spec / decision
  committed (`38478152`, `--check` byte-equal). Two release attempts stopped before any upload and were fixed:
  - `gate.first_items` assumed a tier gate (0.8B has none). Fixed in `5e2c3fa92` with a test.
  - The build screen rejected "node A" in `runtime_equivalence`; the text now names only the parity this release
    runs.
  - `--release` has run on node A GPU1 since 04:35:41Z; pre-upload native parity passed. Eos main is `9c7f3ea0`.
- **2B:** the sweep's `IS-2b-RAUP` (M14 arm, `37c85fff…`) now has the largest lower bound, above `2b-RA-a75`;
  `IS-2b-RASD` is still running. RAUP never went formal (no typed-FINAL / CSS15 / public 231 predictions). The M6
  formal library collects only on node B GPU3 / GPU4, which the 9B M10 worker (`7e1c9ce8`) now holds with GPU2 / GPU7.
  This needs the coordinator. The `2b-RA-a75` BF16 Index run (formal run complete, types OK) continues as the
  ready alternative. Sol main is `8ed41433` (card-only: `MODEL_MANIFEST.json`, `assets/banner.png`).

## 2026-10-02 04:05Z — coordinator interrupt (11:40 UTC+8): one release per tier; 2B point is now `2b-RA-a75`

- **Directive:** release only the best candidate per tier, directly. 0.8B: `08b-RA-a75` on its BF16 Index run. 2B:
  not `2b-RASD-a25`; choose among the Index sweep's 2B runs by the largest Index lower bound vs Sol-2B. No
  reference-only evals (v3, C1, human transfer, mlx-diag).
- **Stopped:** `M16-2b-RASD-a25-bf16` (node A GPU0, shard 1 of 3) at 03:39Z; marked `STOPPED.txt`, co-tenant entry
  removed. Integration (`cd565a588`) was already merged.
- **2B choice:** the sweep's `IS-2b-RA-a75` (= M16 `2b-RA-a75`, `bbba9fad…`) has the largest lower bound, above the
  M12 soup `IS-2b-RA` and `2b-RASD-a25` (values private). `IS-2b-RAUP` is running and `IS-2b-RASD` is queued; both
  are re-checked before the upload. `2b-RA-a75`'s M16 formal run passed item 3 (no collapsed type), so no new
  typed-FINAL run is needed. The contamination audit is the RA TRAIN audit (`08140409`).
- **2B BF16 release weights:** the FP32 checkpoint was assembled on node C from the sweep's package and checked
  file by file against the formal `PACKAGE.sha256` (9 files equal). `bf16_copy`: `bbba9fad…` → **`d58577a2…`**
  (receipt `5e44bafc…`). Restaged onto `a53cf66a` (manifest `ec0bc4f1…`, loaded 1,883,930,944). Parity 86 / 86,
  max |Δp| 0.0. Two shards (`panel-2`, same run-ID set `6455d7be…`) have run since 04:01Z on node A GPU0
  (co-tenant `owner.release-2b-ixf`) and GPU6. `ops/ix.sh` now takes `IX_GPUS` / `IX_FP32` (`4ea5e7d29`).
- **0.8B BF16 Index run done** (03:55Z): 120,224 `ok` + 2 `unsupported`, scorer gate PASS, 1.15 GPU-h. Pulled to
  node C (13 files, SHA-256 lists equal); bootstraps running.

## 2026-10-02 03:25Z — release tooling committed; BF16 Index runs at shard 1 / shard 0 of 3

- Decision record drafted ([`dec-08bfast-indexpath-2026-10-02.md`](dec-08bfast-indexpath-2026-10-02.md); release
  section pending). Release tooling: `ops/make_ixf.py` (spec + final decision, `index_first` profile),
  `ops/release_ixf.sh` (`--stage` / `--mlx` / `--release`), `ops/stage_private.sh` (private inputs and the default
  Index generator), `ops/index_runs.py`, `ops/render_assets.sh` (workstation render, Python 3.12.13 / matplotlib
  3.11.2 / Pillow 12.3.0, Inter and logo hashes equal to round 4's receipt).
- `--stage` done for both tiers on node A (current gate / decision byte-equal to card round 3: Eos `221c331e…` /
  `83e15d20…`, Sol `c5d5a137…` / `bf8d2200…`). Hub mains unchanged (Eos `9c7f3ea0`, Sol `b42b6ff3`); HF headroom
  47.45 GB; the Transformers 5.18 site on node A is `93df9002…` (as the 4B / 9B releases).
- Next: 0.8B BF16 run ends ≈ 03:55Z → pull to node C, bootstraps, private inputs, Index input, assets, spec /
  decision, `--mlx`, `--release` (node A GPU1, co-tenant `owner.release-0p8b-ixf`); then the same for 2B.

## 2026-10-02 03:05Z — winners 08b-RA-a75 / 2b-RASD-a25; Index runs on the BF16 release weights running

- **0.8B choice (09:55 rule, largest Index-gain 95% lower bound):** `08b-RA-a75` > M12 `08b-RA` (`DEV2.0-0.8B-08bRA`,
  full-panel bootstrap `fbd0ff1e…`, transfer-only `8f397e3f…`; lower bound > 0 too) > `08b-RASD-a75`. **2B:**
  `2b-RASD-a25` (the only measured 2B candidate). Both pass every reference item (v3 not below, H, types, mlx-diag,
  tier gates, exposure 0, public 231) and the integrity checks so far (types OK; contamination audits above).
- **BF16 copies** (node C, `v2.release.bf16_copy` in the host2 image, the FP32 checkpoint checked file by file
  against the formal run's `PACKAGE.sha256`): 0.8B `fee82408…` → **`d9712799…`** (receipt `d1f51ee0…`), 2B
  `7afafef9…` → **`d6cbd9f0…`** (receipt `bec0e4ef…`); restaged onto `bede7938` / `a53cf66a` (manifests `3062aba3…`
  / `fc3036b0…`).
- **GPU contention:** after the 01:42Z release of node C GPU1–7, the 27B M6 Index run took node C GPU1–4 and node D
  GPU4–7 and the Index sweep node C GPU5–7 (02:12–02:27Z). Two parity starts (node C GPU4, node D GPU7) were refused
  by the launcher's busy-GPU check before any GPU work and were voided (`ix1/void/`). Node A GPU7 went to an unrelated
  study at 02:38Z.
- **Running on node A** (IX1 suite on node A, panel-3 = the same 120,226 rows): 0.8B `M16-08b-RA-a75-bf16` on GPU6
  (released by 9B M9 for any track; parity 86 / 86, max |Δp| 0.0; shards 0–2 one after another since 02:45Z, ETA
  ≈ 04:05Z); 2B `M16-2b-RASD-a25-bf16` on GPU0 as the recorded co-tenant `owner.release-2b-ixf` (the 0.6B allocation
  allows release co-tenants; launcher option `IX1_SHARED_LEASE`, `9cc92103d`; parity 86 / 86, max |Δp| 0.0; shards
  since 03:03Z, ETA ≈ 04:30Z).
- **Release inputs on node A (CPU, `ops/prep.sh`, mirror `3bd92fe82`):** BF16 copies staged; the M16 formal runs
  adopted unchanged at T = 1 as `dev2-0p8b-ixf-t1` (v3 53.913, seal `ea2f2dcb…`) and `dev2-2b-ixf-t1` (v3 53.488, seal
  `fde2a8af…`), each reproducing its formal report; mlx-diag scored; reference gates and named paired files written.
  Formal and mlx-diag autotune caches relayed node B → node E → node A, each equal to its run's after-manifest.
- **Gate code:** `index_first` profile in `v2/release/gate.py` (`3bd92fe82`, tests `test_gate_index_first.py`):
  I1 full-panel Index bootstrap of the release weights minus the current revision (both IX1 receipts bound),
  lower bound > 0, ≥ 2,000 replicates; I2 no type collapsed; I3 contamination audit. References bound, not gating.

## 2026-10-02 02:10Z — Index runs found complete; bootstraps done; rule now Index-first

- **Found finished (predecessor, node C GPU1–7, mirror `00401c9b5`, 20:22–21:04Z):** for each candidate the
  restaged package (onto DEV2.0-0.8B `bede7938` / DEV2.0-2B `a53cf66a`, FP32 checkpoint of the M16 formal package,
  files lists `bca5ec1d…` / `a57e236b…` / `b5248c80…`), the 86-request parity gate (86 / 86 ok, max |Δp| 0.0), the
  full panel-7 run (120,226 rows: 120,224 ok, 2 unsupported `max_length_exceeded`, 0 errors), dual scoring (port vs
  kit `87d4650b`) PASS:

  | Candidate | Results | IX1 identity | GPU-h |
  | --- | --- | --- | --- |
  | `M16-08b-RA-a75` | `23b90341…` | `fee82408…` | 1.28 |
  | `M16-08b-RASD-a75` | `3697e0f8…` | `690e5b97…` | 1.22 |
  | `M16-2b-RASD-a25` | `aabe4111…` | `7afafef9…` | 1.35 |

  Only `08b-RASD-a75`'s transfer-only bootstrap had run. The node C GPU1–7 owner files still named the finished 2B
  run; they were rewritten to `status=released` at 01:42Z (GPUs idle, no IX1 container).
- **Bootstraps (node C CPU, `paired_boot.py` of mirror `42960045f`, 2,000 paired replicates, seed `20261002`):** for
  each candidate vs its tier's DEV2.0 IX1 run, the full panel (no exclusion; the item-1′ evidence) and the
  transfer-only variant (HoVer, When2Call, iSarcasmEval, GSM8K, BPoMP excluded); family deltas. **All three have a
  95% Index lower bound > 0.** Files (private): `paired-boot-full-vs-ref.json` `e50577de…` / `dc2634d3…` /
  `982430b8…`, `paired-boot-vs-ref.json` (transfer) `544c3fb9…` / `0aa5acd0…` / `9611e6bd…`.
- **Item 1′ and 6(b)′ (from M16's two-bar successor files on node A):** v3 upper bound > 0 against both bars on the
  full panel (item 1′, v3 part) and with the flagged items removed (6(b)′) for all three; the rule-5 parts of 6(b)
  pass; 6 / 6 stored pairs reproduce. **All three qualify on the Index path.**
- **Rule change (COORDINATION 09:55, user decision):** the release gate is now the Index delta alone (95% lower
  bound > 0) plus integrity checks (exact package parity, Hub / `trust_remote_code`, the row-level contamination
  audit, no collapsed type); v3, human transfer, mlx-diag, public 231 and C1 are references; C1 item 8 is not run
  unless in progress (it was not); selection = the largest Index-gain lower bound. The coordinator compares M12's
  frozen `08b-RA` (IX1 run `DEV2.0-0.8B-08bRA`, results `9c28a430…`) with the M16 a75 points: its results were
  copied node D → node C through node A (hash equal) and its two bootstraps are running (node C CPU, 02:04Z).
- **Contamination audits (IX1 method, planted 200 / 200) cover every candidate:** the RASD arms' TRAIN files are hard
  links of M12's RA files (M13 data lock). 0.8B RA TRAIN `12bd63d8…`: 18 duplicate-class rows, 1 item row (the same
  public-split BANKING77 item as the DEV2.0-0.8B TRAIN); 2B RA TRAIN `08140409…`: 76 duplicate-class rows, 0 item
  rows (DEV2.0-2B TRAIN: 80 / 0).
- **Next:** pick the 0.8B winner; BF16 copies of the winners, an Index run of each on exactly the released weights
  (card Index and gate evidence), then the releases on the then-current `main` with the default card generator
  (banner A, round-3 Index conventions).
