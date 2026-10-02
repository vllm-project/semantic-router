# Index sweep — preregistration (2026-10-02)

COORDINATION 2026-10-02 10:00 (the Index sweep) under the user's **Index-first** release rule of 09:55. Written and
pushed before any sweep job. Index values never appear in this branch, the gist or a card: they stay in node-private
run directories and the local private folder.

## Candidates (frozen soups; read-only; nothing is trained)

| Name | Tier | Soup (`model_sha256`) | Lineage |
| --- | --- | --- | --- |
| `IS-08b-RASD` | 0.8B | M13 `08b-RASD` `02c17086…` | node A's M16 copy (content manifest equal to node E's soup) |
| `IS-08b-RAUP` | 0.8B | M14 `08b-RAUP` `0bf0401d…` | node A |
| `IS-08b-RASDML` | 0.8B | M15 `08b-RASDML` `db5cccad…` | node E |
| `IS-2b-RA` | 2B | M12 `2b-RA` `3d4fde06…` | node B's M16 copy (equal to node E's soup) |
| `IS-2b-RASD` | 2B | M13 `2b-RASD` `341c2bd2…` | node B's M16 copy (equal to node F's soup) |
| `IS-2b-RAUP` | 2B | M14 `2b-RAUP` `37c85fff…` | node B |
| `IS-2b-RA-a75` | 2B | M16 `2b-RA-a75` `bbba9fad…` | node B |
| `IS-L9IB` | 9B | M9 stage 2 `L9IB` `d7c48f9a…` | node A |
| `IS-K-a12IB` | 9B | M9 stage 4 `K-a12IB` `68fed4cb…` | node A |

Already measured, read from their stored runs: 0.8B `08b-RA` (`f25beedd…`, run `DEV2.0-0.8B-08bRA`, node D) and the
M16 points of the 0.8B / 2B Index-path continuation (b49d1f36).

## Harness (IX1, unchanged)

- Each FP32 soup is restaged (`v2.eval.ix1.restage`) into its tier's IX1-scored package: 0.8B `bede7938`, 2B
  `a53cf66a`, 9B `e51f9881`. T = 1, calibration none. The identity is the soup's `model_sha256`, and the loaded
  parameter count must equal the package's.
- Then the 86-request parity gate (pass = 86 / 86 `ok`, identical choices, max |Δp| = 0), the full 120,226-row panel
  with the candidate's own frozen autotune cache, and dual scoring (port vs kit `87d4650b`; the scorer gate must pass).
- Image `decision20-train-fast:host2` (`f83b1d10`), kit `87d4650b`, panels `panel-3` / `panel-7` / `panel-8` (one
  run-ID set, `6455d7be…`).
- Tooling: `v2/dec/ops/index-sweep/` (`ix.sh` workstation side, `chain.sh` node side), launcher entries `IS-*` in
  `v2/eval/ix1/launch.sh`, and `paired_boot.py --exclude` from the 9B track (`1bfd1fb73`, content-identical).

## Comparison and rule (Index-first, user decision 09:55)

- **Reference** = the tier's current release, through its IX1 run:
  - Eos-0.8B: run `DEV2.0-0.8B`;
  - Sol-2B: run `DEV2.0-2B`;
  - Lux-9B (K-a13IB): run `K-a13IB-bf16`, the shipped weights.
- If the Index-path continuation releases a new 0.8B / 2B model first, that release's IX1 run becomes the
  reference. This is re-checked on `main` and in its records before each tier decision.
- **Gate:** a paired bootstrap over scoring cases within benchmarks through the board's area weights; 2,000
  replicates, seed `20261002`. Gate = 95% CI lower bound > 0 on the full headline.
- **Selection** within a tier: the largest lower bound among the candidates that pass. `08b-RA`'s stored rows take
  part at 0.8B.
- **Recorded (private, not a gate):** the transfer-only delta (`--exclude HoVer When2Call iSarcasmEval GSM8K
  BPoMP`).
- **Integrity checks for the winner (all required, no quality comparison):**
  - exact package parity of the release build;
  - Hub `trust_remote_code` smoke under Transformers 5.17 / 5.18;
  - a row-level Index contamination audit of the training file (`v2.eval.ix1.contamination`, card footnote
    "audited");
  - no decision type collapsed: formal typed-FINAL item 3 (run once for a winner without a formal run).
- **References** (recorded, not blocking): JevArena v3, human transfer, mlx-diag, public 231, C1, whatever is
  already measured.
- **One Index run per candidate.** A failed parity gate, shard or scorer gate is recorded and that candidate stops.
  No rerun, except the launcher's own `resume` of transient row errors.

## Resources

- GPUs:
  - node A GPU3–5 (`panel-3`);
  - free node C GPU1–7 (`panel-7`; node C GPU0 never);
  - free node D GPU4–7 (`panel-8` in two waves).
- Lease-checked; each container sees only its own render node.
- Node B is skipped (no Index suite there).
- Cap: ≤ 30 GPU-h, including parity, runs and the integrity checks.
