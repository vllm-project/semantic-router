# Index sweep — state

Branch `xunzhuo/decision-2-training-index-sweep`, worktree `vllm-sr-dev2-index-sweep`. Prereg
[`dec-index-sweep-prereg-2026-10-02.md`](dec-index-sweep-prereg-2026-10-02.md) (`e6cb71541`, mirrored to nodes
A / C / D before any job). Index values are private (node runs, local private folder); this file has none.

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
