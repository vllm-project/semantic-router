# Index sweep — state

Branch `xunzhuo/decision-2-training-index-sweep`, worktree `vllm-sr-dev2-index-sweep`. Prereg
[`dec-index-sweep-prereg-2026-10-02.md`](dec-index-sweep-prereg-2026-10-02.md) (`e6cb71541`, mirrored to nodes
A / C / D before any job). Index values are private (node runs, local private folder); this file has none.

## 2026-10-02 06:10Z

- **New verdicts** (values private):
  - `IS-2b-RASD` **PASS**, the largest 2B lower bound measured (above `IS-2b-RAUP`'s); node C `ix1/runs/IS-2b-RASD`.
    b49d1f36 has already chosen it and runs its BF16 release copy (`M16-2b-RASD-bf16`, node C GPU1–3), so this
    sweep does not. A paired RASD-vs-RAUP bootstrap (FP32 runs) is written to node C
    `ix1/index-sweep/cross/IS-2b-RASD-vs-IS-2b-RAUP-{full,transfer}.json` for that decision.
  - `IS-08b-RAUP` **FAIL** vs DEV2.0-0.8B (CI includes 0); node A `ix1/runs/IS-08b-RAUP`.
- **GPU sharing.** The 4B pool started `DEV2.0-4B-LHA10SD-a50-bf16` on node D GPU4–7 at 05:38Z while this sweep's
  node D chain was waiting; that chain now follows it (4B `LHA10SDML`, 08b `RASDML`, `RAAG`). `IS-08b-RA10SDML-bf16`
  moved to node A GPU6 (free, released), one GPU in three waves.

## 2026-10-02 05:10Z

- **New verdicts** (Index-first gate vs the tier's current release; values private):

  | Candidate | Gate | Run (node, `ix1/runs/…`) | Hand to |
  | --- | --- | --- | --- |
  | `IS-L9IB` | **FAIL** (the whole CI is below K-a13IB; confirmed) | D `IS-L9IB` | 7e1c9ce8 |
  | `IS-2b-RAUP` | PASS; the largest 2B lower bound measured so far (above `IS-2b-RA-a75`'s) | C `IS-2b-RAUP` | b49d1f36 |
  | `IS-08b-RASD` | PASS vs DEV2.0-0.8B; lower bound below M16 `08b-RA-a75`'s | D `IS-08b-RASD` | b49d1f36 / ce74f1e5 |

  `IS-2b-RASD` (C) and `IS-08b-RAUP` (A) are scored; their bootstraps run on those nodes.
- **No overlap with M18:** M18 (prereg 04:50Z) owns the RAUP interpolation points (`2b-RAUP-a75/-a50`,
  `08b-RAUP-a75/-a50`) and runs them on node E GPU0–3 / 6–7 and node A GPU1 / 2 / 7, so this sweep dropped them and
  does not use node E's GPUs. Node E only did CPU work (BF16 copies of `08b-RAAG`, `08b-RA10SDML`).
- **Widened queue, all on BF16 release copies** (mirror `789356e4c`; copies node to node, C–F through node A):
  - node A GPU4–5 (`panel-2`): `IS-K-a13IBX-bf16`, then `IS-L9IBX-bf16` (9B, to 7e1c9ce8);
  - node A GPU3 (`panel-3`, waves): `IS-08b-RA-a50-bf16`, `IS-08b-RASD-a50-bf16`;
  - node D GPU4–7 (`panel-4`): `IS-4b-LHA10SDML-bf16` vs `DEV2.0-4B-LH` (to 5e7b8132), `IS-08b-RASDML-bf16`,
    `IS-08b-RAAG-bf16`, `IS-08b-RA10SDML-bf16`;
  - node C GPU6–7 (`panel-2`): `IS-2b-RA-a50-bf16`, `IS-2b-RASD-a75-bf16`, `IS-2b-RASDML-bf16`,
    `IS-2b-RA10SDML-bf16`.
  - 0.8B / 2B results go to ce74f1e5 after b49d1f36's releases; each is bootstrapped vs the tier's current-release
    run (and re-based on the new release when it lands).
- GPU-h so far ≈ 16 (well within 45).

## 2026-10-02 04:45Z (COORDINATION 11:35 / 12:30 / 12:40: no 2B or 9B release from this sweep)

- **9B verdicts (for the Lux-9B publisher, 7e1c9ce8): no 9B candidate of this sweep qualifies.**
  - `IS-K-a12IB` **FAIL** vs K-a13IB (CI includes 0). Node A `ix1/runs/IS-K-a12IB` (merged, family delta, both
    bootstraps); formal `formal-m9/K-a12IB-16k`.
  - `IS-L9IB` **FAIL**: the full-panel point estimate is below K-a13IB, so the lower bound cannot be > 0; the
    confirming bootstraps are running on node D, `ix1/runs/IS-L9IB`. Formal `formal-m9/L9IB-16k`.
  - The release-weight runs `IS-K-a12IB-bf16` and `IS-L9IB-bf16` (node A) are stopped (`STOPPED.json`); node A
    GPU4–5 are released.
- **2B verdicts (for b49d1f36, the Sol-2B publisher):** `IS-2b-RA` PASS and `IS-2b-RA-a75` PASS (node C
  `ix1/runs/IS-2b-RA`, `IS-2b-RA-a75`); `IS-2b-RAUP` is scored (node C `ix1/runs/IS-2b-RAUP`, merged and family
  delta) and its point estimate is the highest 2B so far; its bootstraps are running there.
- **Still running (original queue):** `IS-08b-RAUP` node A GPU3 (`panel-3`, last wave); `IS-08b-RASD` node D GPU4–7
  (`panel-4`); `IS-2b-RASD` node C GPU6–7 (`panel-2`). `IS-08b-RASDML` moves to node E on its BF16 release copy
  (node C's chain is stopped after 2b-RASD is scored).
- **Widened queue (12:40), each measured once on its BF16 release copy, on node E GPU0–3 / 6–7 (never GPU4–5):**
  9B `IS-K-a13IBX-bf16`, `IS-L9IBX-bf16` (to 7e1c9ce8); 4B `IS-4b-LHA10SDML-bf16` vs `DEV2.0-4B-LH` (to 5e7b8132);
  0.8B `IS-08b-RASDML-bf16` (to ce74f1e5). Copies are node to node (C–F through node A); node E gets the bootstrap
  environment from node A, so every bootstrap runs on the node of its run. Mirror `c9a45e4e3`.

## 2026-10-02 03:55Z

- **Verdicts so far** (Index-first gate: full-panel paired bootstrap 95% lower bound > 0 vs the tier's current
  release; values private):

  | Candidate | Gate vs current release | Transfer-only delta (private record) |
  | --- | --- | --- |
  | `IS-2b-RA` | PASS | significantly negative |
  | `IS-2b-RA-a75` | PASS; the largest 2B lower bound so far, also larger than M16 `2b-RASD-a25`'s | not significant |
  | `IS-K-a12IB` | **FAIL** (CI includes 0 vs Lux-9B K-a13IB) | significantly negative |

- **K-a12IB is out.**
  - Its release-weight run (`IS-K-a12IB-bf16`) was stopped at about 6.5k rows (the gate cannot pass), and the
    partial run is kept as `runs/IS-K-a12IB-bf16` with `STOPPED.json`.
  - Its formal panel did run: types OK / OK / OK; v3, human transfer and public 231 are flat vs K-a13IB; mlx-diag
    card-eligible is significantly below K-a13IB (reference).
  - The prepared release ops (`release/records/dev2-9b-ka12ib-2026-10-02/ops/`) are not used.
- **L9IB** is the remaining 9B candidate. Its formal panel: types OK / OK / OK; v3 not significantly different from
  K-a13 (reference).
  - **Deviation from the prereg (recorded before its result):** its Index run uses the BF16 release copy
    (`IS-L9IB-bf16`, identity `78836b25…`) instead of the FP32 soup. The copy is answer-identical by construction
    (Linear weights rounded as the runtime's autocast rounds them), and if L9IB passes it is the release evidence,
    which saves a second 2.7 GPU-h run.
  - The FP32 run started at 03:41Z was stopped after about 900 rows per shard and moved to `void/`.
- **2B release weights:** the BF16 copy of `2b-RA-a75` (`IS-2b-RA-a75-bf16`, identity `d58577a2…`) is restaged on
  node C and queued after `IS-2b-RAUP`. If it stays the 2B selection, its run is the card Index and gate evidence.
- **Queues:**
  - node A GPU3: 08b-RAUP, then 08b-RASD (`panel-3` in waves);
  - node A GPU4–5: L9IB-bf16 (`panel-2`);
  - node C GPU6–7: 2b-RAUP, 2b-RA-a75-bf16, 2b-RASD, 08b-RASDML.
  - Node D was left to the 27B and 4B workers (its GPUs were taken one at a time); the L9IB / 08b packages staged
    there are unused.

## 2026-10-02 03:10Z

- **Shared-module change:** the Index-first successor gate profile (`gate_profile.index_first`, `f3590ef2a`,
  tests in `test_gate_successor`, 23 new cases) is merged into `xunzhuo/decision-2-training` at `d3000ffcb`, for
  every Index-first release worker. IF1 is the Index gain, R3 no collapsed type, IF3 the row-level Index audit; the
  other successor items are references only.
- **GPU sharing.**
  - The 0.8B / 2B release worker (b49d1f36) and the 4B worker's pool also need node C / A GPUs for their release
    weights' Index runs.
  - Node C GPU5 was released after `IS-2b-RA` (the 4B pool now holds it). The node C queue continues on GPU6–7 with
    a 2-shard panel (`panel-2`, same run-ID set).
  - Node A GPU6 / GPU7 are held by b49d1f36 and the open-jev-fast study; this sweep never touched them.
  - Node D GPU4–7 (free once the 27B M6-IB shards end) take L9IB, 08b-RAUP and 08b-RASD on `panel-4` (built on node
    D, same run-ID set; packages restaged there with identical manifests).
- **Done:** `IS-2b-RA`: parity PASS, 120,224 `ok` + 2 `unsupported`, scorer gate PASS, both bootstraps written;
  1.30 GPU-h.
- **Running:**
  - node A GPU3–5: `IS-K-a12IB`;
  - node C GPU6–7: `IS-2b-RA-a75`, then 2b-RAUP, 2b-RASD, 08b-RASDML;
  - node D: waiting for GPU4–7.
- **Release weights.** The BF16 storage copy of K-a12IB (`v2.release.bf16_copy`, identity `5cb68d17…`, receipt
  `c58c5e25…`) is restaged as `IS-K-a12IB-bf16`. Its Index run queues on node A GPU4–5 after K-a12IB; GPU3 then
  takes K-a12IB's formal panel on the 9B formal path (`M9_FORMAL_GPUS`, `165dff605`).

## 2026-10-02 02:25Z

- **GPUs.**
  - The 27B M6 worker started its M6-IB Index run on node D GPU4–7 and node C GPU1–4 at 02:11Z. Its state file
    leaves node C GPU5–7 free.
  - This sweep therefore uses node A GPU3–5 (`panel-3`) and node C GPU5–7 (`panel-3`, copied from node A, SHA-256
    lists equal). Lease owners are `track=eval-ix1`; node A's previous M16 owner files are kept as `owner.prev-*`.
  - Node D is not used, and node B is skipped (no suite there).
- **Copies** (02:13–02:14Z; every SHA-256 list equal):
  - to node A: the 9B / 0.8B IX1 packages (`e51f9881`, `bede7938`) and the reference runs `K-a13IB-bf16` and
    `DEV2.0-0.8B` (merged results, compare, receipt) under `ix1/index-sweep/refs/`;
  - relayed through node A: the M15 `08b-RASDML` soup from node E.
- **Restaged** (02:15–02:17Z; identity = the soup's `model_sha256`, loaded count = the package's, T = 1):

  | Name | Node | Package manifest |
  | --- | --- | --- |
  | `IS-K-a12IB` | A | `fe2dd99d…` |
  | `IS-L9IB` | A | `f5714e48…` |
  | `IS-08b-RAUP` | A | `7f205094…` |
  | `IS-08b-RASD` | A | `88069c0f…` |
  | `IS-2b-RA` | C | `48f08d61…` |
  | `IS-2b-RAUP` | C | `566ab9bf…` |
  | `IS-2b-RA-a75` | C | `9109f7cb…` |
  | `IS-2b-RASD` | C | `1ce3a901…` |
  | `IS-08b-RASDML` | C | `413d92f4…` |

- **Chains** (`chain.sh`, mirror `e6cb71541`) started at 02:19Z:
  - node A: K-a12IB → L9IB → 08b-RAUP → 08b-RASD;
  - node C: 2b-RA → 2b-RAUP → 2b-RA-a75 → 2b-RASD → 08b-RASDML.
  - Parity gates run first: `IS-2b-RA` and `IS-08b-RAUP` PASS so far (86 / 86 `ok`, max |Δp| 0.0).
