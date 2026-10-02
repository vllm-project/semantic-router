# 9B M9: K-a13IB on the Index path — items 1′–8 and the successor release (2026-10-02)

> **Result: K-a13IB passes items 1′ and 2–8.** It is released as the 9B successor on `Decision-2.0-Lux-9B`,
> `main` `259a4550`, verified (section 4). Index values stay in private files; this record carries verdicts, counts
> and hashes only.

## 0. The decision

**Coordinator decision 2026-10-02 02:05 UTC+8 (the Index path for frontier-targeted finalists).** The user did not
answer, so the recommended rule was adopted; the user may override it.

- **Item 1′** replaces item 1 for this path. Both parts must hold:
  - v3 is not significantly below the current revision (paired 95% CI upper bound > 0);
  - the private Index delta vs the reference is significantly positive (paired bootstrap 95% CI lower bound > 0,
    the same bootstrap as the follow-up A receipt).
- **Items 2–8** are unchanged.
- Internal records also report the transfer-only Index delta (the five benchmarks with in-domain training rows
  excluded). It is reported privately; it is not a gate.

The gate code carries the rule as the successor profile's `index_path` key (`v2/release/gate.py`, `391db11fb`):
R1 becomes `1_successor_R1p_v3_not_below_index_gain`, bound by `evidence_sha256.index_path`. A bootstrap that
excludes any benchmark is refused as R1′ evidence.

## 1. Items 1′ and 2–7 (formal run `formal-m9/K-a13IB-16k`, T = 1 derivation)

The T = 1 derivation (`retemper_predictions --undo`) changes 0 answers on every scored panel and on mlx-diag
(seal `bedbb51c…`).

| Item | Verdict | Evidence |
|---|---|---|
| 1′ v3 | PASS | 68.024 vs 67.737, +0.29 [−1.57, +1.20]: not significantly below |
| 1′ Index | PASS | private paired bootstrap vs DEV2.0-9B: 95% lower bound > 0 (file `92fd2fb0…`, node A private tree) |
| 2 human transfer | PASS | −0.021 to +0.020 (not below) |
| 3 types | PASS | every type floor holds |
| 4 card-eligible mlx-diag | PASS | −0.0037 [−0.0110, +0.0034] |
| 5 tier gates | PASS | vs adopted Lux 1.0 +2.22 [+0.11, +3.95]; v3 ≥ 0.9 × Nimble v2; human transfer vs Nimble v2, upper +0.097 |
| 6 exposure | PASS | 0 exposed training groups in both receipts (`m6/exposure/x60.json`, `m9/exposure/ib1-ib2-train.json`) |
| 7 public 231 | PASS | 182 vs 178 (McNemar p 0.125) |

Prediction files: typed-final `a049ce2d…`, public231 `1f4c2321…`, formal `COLLECT.json` `f6487676…`. Gate
paired files: vs adopted-1.0 `156a2ad1…`, vs DEV2.0-9B `bb7e76ab…`, vs Nimble v2 `6e73a204…`.

## 2. The private Index run on the release weights (receipt; values private)

The Index evidence was scored on exactly the weights that ship: the BF16-storage copy (`v2.release.bf16_copy`,
identity `b9a65d60…`, receipt `773d9b0c…`), not the FP32 soup of follow-up A.

- **Where:** node C GPU1–7 (IX1 harness `v2/9b/lux9b/m9/ix.sh`, name `K-a13IB-bf16`), restaged into the DEV2.0-9B
  runtime `e51f9881` (package manifest `4471a7a2…`).
- **Parity gate:** 86 / 86 ok, identical choices, max |Δp| 0.0.
- **Run:** 120,226 rows: 120,224 ok, 2 unsupported (`max_length_exceeded`), 0 errors. Panel run IDs
  `6455d7be…`; results `207b867f…`; kit `index.json` `ac4f2da3…`; receipt `aee534c3…`. 2.73 GPU-h.
- **Scorers:** pass. Leases on node C GPU1–7 released.
- **Bootstrap:** 2,000 paired replicates vs DEV2.0-9B (`paired_boot.py`). The transfer-only variant
  (`--exclude`, weighted contributions renormalised by the kept weight, `1bfd1fb73`) is reported privately.

Values: `decision2-program/private/9b-ka13ib/` only.

## 3. Item 8: JevArena-C1 v1.2, post-key (not an independent validation)

- **Content recheck:** `v2/eval/records/c1-recheck-r1-2026-10-01.md`, PASS (IB1-r3, IB2, PN1-r2: 0 scored items
  exposed). `c1-recheck-registry.json` `arms_exposure_0` lists K-a13IB under 9B M9, so its C1 exposure is 0.
- **Frozen package:** the pre-release build of `v2/release/specs/dev2-9b-ka13ib.draft.json` on node A, GPU6
  (`release_ka13ib.sh --prerelease`, 19:17–19:22Z, mirror `d531076d3`). Manifest `055c86f1…`, 39 files, identity
  `b9a65d60…`. Pre-upload examples, card example and Transformers remote code passed; `verify_bundle` ok.
- **Spec:** `v2/eval/sealed/c1-postkey/dev2-9b-ka13ib.json` (`24228e13b`). Formal runtime: image host2 `f83b1d10`,
  adapter `dev2-dec-package-t1`, 16,384 tokens, `HIP_FORCE_DEV_KERNARG=1`, a fresh copy of the formal run's autotune
  cache (`48a2611a…`); parity against the T = 1 derivation.
- **Runs (node A GPU6, shared lease `owner.release-9b-ka13ib`):** verify-only 19:24Z, then preflight 19:24–19:25Z
  (smoke identical to the stored formal run), then **one** keyed run 19:25–19:32Z. The key arrived on stdin only;
  prompts and gold were decrypted, used and removed on node A.
- **Result:** C1 **53.49** (valid 2,840) vs the registered 9B baseline 53.77 (event-3 run `cand9b`):
  **−0.28 [−1.31, +0.65], p 0.526: PASS** (not a regression). Seal `e5b7e2db…` (sealed before any gold), summary
  `16fd38fb…`, gate `8042a25d…`, predictions `31c6bd94…`. 0.064 GPU-h. The custodial ledger has one line.

The final decision binds this summary as R8 (`evidence_sha256.c1_postkey`).

## 4. The release

### 4.1 Inputs

- **Spec:** `v2/release/specs/dev2-9b-ka13ib.json`, built by `ops/make_ka13ib.py final` on node A. It derives from
  the round-2 product spec of the current revision (`586af779`) and carries over the roster, peers, runtime
  (`99432d1a7`: BF16-resident, Transformers remote code, forward token budget), remote code, licence and product
  card. The weights, scored run, gate evidence, Index input and assets are K-a13IB's.
- **Decision:** `release/records/dev2-9b-ka13ib-2026-10-02/Decision-2.0-Lux-9B.decision.ka13ib.json`, gate profile
  `successor` with `index_path`. All of R1′ and R2–R8 pass.
- **Card Index:**
  - **Conventions:** the user's card fix of 2026-10-02 03:04 UTC+8 (card round 3) came after the coordinator
    decision. It sets one audited footnote for every card (`card_index.FOOTNOTE`; `card_index.load` refuses any
    other) and the board's served-parameter count on the size axis. It leaves Lux 9B to this release.
  - **Footnote:** the card therefore carries the round-3 footnote, not the 02:05 decision's draft wording.
    Both say that the Decision 2.0 points are an independent reproduction with the official 0.2.1 kit on the
    released weights, and that the training data was checked against the Index test items. The round-3 sentence
    is "Training data audited at row level against all Index test items."
  - **Index input:** the round-3 input with only the 9B scores replaced by the BF16 IX1 run (section 2): balanced
    skill, area skills × 100 and the kit `index.json` hash. The 9B point keeps its board-served and loaded counts.
    It lives at node A `/data/dev2/private/release/ka13ib/` (mode 700), `0899dca4…`, built by
    `ops/make_index_ka13ib.py`.
  - **Assets:** rendered by the round-3 `card_assets.py` in the round-2 render environment (Python 3.12.13,
    matplotlib 3.11.2, Pillow 12.3.0, Inter 4.1, the same logo). Receipt `b911708f…`. The footnote sits clear of the
    logo on both Index charts.
- **No training details on the card:** the package README names no data source, recipe or soup.
- **Speed:** the card keeps the 400-request bench receipt of this runtime (same architecture, runtime and shapes).

The C1-scored package and the released one differ only in the card (README and Index assets): the weights,
identity and runtime are equal. `release_ka13ib.sh --release` checks this on the released manifest
(`c1-package-diff`).

### 4.2 Publication

**`main` = `259a45502bfca2f585a59ef99079533106e0b136`** (private), published at 20:37:55Z by
`release_ka13ib.sh --release --gpu 6` (mirror `9e84f05f4`, node A GPU6, 20:19–20:50Z). The full chain, receipts and
verification are in the release record
[`dev2-9b-ka13ib-2026-10-02.md`](../../release/records/dev2-9b-ka13ib-2026-10-02.md).

- **Package:** manifest `01d642a1…`, identity `b9a65d60…`, 7,940,895,744 loaded parameters. The 10 weight files
  equal the C1-scored package `055c86f1`.
- **Gate:** sealed in `gate.json` `f3d79e24…` (decision `99621f30…`, supersedes `586af779`); `gate evaluate`
  14 / 14.
- **Parity:** 0 answer changes on typed-final, css15, public231 and mlx-diag, before the upload and on the real
  download. AutoModel equals native. The Hub `trust_remote_code` smoke passes under Transformers 5.17.0 and 5.18.0.
- **Hub:** the old ID `DEV2.0-9B` redirects. The collection is unchanged.
- **Purge:** the 10 superseded weight objects of `586af779` were deleted with `rewrite_history=False`. Storage is
  back to 52.55 GB of 100.
- **Interruption:** a session interruption stopped this worker before its receipts were committed. The verification
  after it (worker 68fece59, 2026-10-02 01:35–01:50Z) found every required receipt present and passing. A post-purge
  real download re-hashes equal, and the card is the default generator's round-3 product card.
- **Issue:** the driver's stale collection-order step (former `DEV2.0-*` IDs) printed `post_checks=FAILED`, so the
  purge was run by hand right after. See the release record, section 5.
