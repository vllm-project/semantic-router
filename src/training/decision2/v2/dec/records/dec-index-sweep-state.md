# Index sweep — state

Branch `xunzhuo/decision-2-training-index-sweep`, worktree `vllm-sr-dev2-index-sweep`. Prereg
[`dec-index-sweep-prereg-2026-10-02.md`](dec-index-sweep-prereg-2026-10-02.md) (`e6cb71541`, mirrored to nodes
A / C / D before any job). Index values are private (node runs, local private folder); this file has none.

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
