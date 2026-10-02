# Index sweep — results (2026-10-02)

Prereg [`dec-index-sweep-prereg-2026-10-02.md`](dec-index-sweep-prereg-2026-10-02.md) (`e6cb71541`); running state
[`dec-index-sweep-state.md`](dec-index-sweep-state.md). Rule: the user's Index-first release rule (COORDINATION 09:55)
with the progressive directive (11:35). A candidate qualifies when the full-panel paired bootstrap vs the tier's
current-release Index run (private Jev Decision Index 0.2.1, IX1 harness, kit `87d4650b`, 2,000 replicates, seed
20261002) has a 95% lower bound > 0. The transfer-only delta (HoVer, When2Call, iSarcasmEval, GSM8K, BPoMP excluded)
is recorded privately. **Index values are private** (node runs `ix1/runs/<name>/`, local
`decision2-program/private/index-sweep/`); this file holds verdicts only.

Ownership changed during the sweep (COORDINATION 12:30 / 12:40): 2B and 0.8B releases belong to b49d1f36 (then the
2B / 0.8B worker ce74f1e5), 9B to the 9B frontier worker 7e1c9ce8, 4B to 5e7b8132. **This sweep released nothing**;
it hands each verdict to the tier's publisher.

## Verdicts

Reference runs: 0.8B `DEV2.0-0.8B`, 2B `DEV2.0-2B`, 4B `DEV2.0-4B-LH`, 9B `K-a13IB-bf16` (Lux-9B).

| Candidate | Weights measured | Gate vs current release | Run (node) | Hand to |
| --- | --- | --- | --- | --- |
| `IS-2b-RA` | FP32 soup, restaged | PASS | C | b49d1f36 |
| `IS-2b-RA-a75` | FP32 soup, restaged | PASS | C | b49d1f36 |
| `IS-2b-RAUP` | FP32 soup, restaged | PASS | C | b49d1f36 |
| `IS-2b-RASD` | FP32 soup, restaged | PASS (largest 2B lower bound) | C | b49d1f36 (chosen; its BF16 copy runs there) |
| `IS-08b-RAUP` | FP32 soup, restaged | FAIL | A | — |
| `IS-08b-RASD` | FP32 soup, restaged | PASS; lower bound below M16 `08b-RA-a75`'s | D | b49d1f36 / ce74f1e5 |
| `IS-K-a12IB` | FP32 soup, restaged | FAIL | A | 7e1c9ce8 |
| `IS-L9IB` | FP32 soup, restaged | FAIL (whole CI below) | D | 7e1c9ce8 |
| `IS-K-a13IBX-bf16` | BF16 release copy | FAIL (whole CI below) | A | 7e1c9ce8 |
| `IS-4b-LHA10SDML-bf16` | BF16 release copy | PASS vs LH; **not significant vs the new Nox release (`LHS17SD`)** | D | 5e7b8132 |
| `IS-2b-RASD-a75-bf16` | BF16 release copy | PASS; lower bound marginally above `M16-2b-RASD-bf16`'s; paired test vs it on node C | C | b49d1f36 |
| `IS-08b-RA10SDML-bf16` | BF16 release copy | PASS; below M16 `08b-RA-a75` | A | ce74f1e5 |
| `IS-08b-RASDML-bf16` | BF16 release copy | FAIL (significantly below) | D | — |
| `IS-08b-RAAG-bf16` | BF16 release copy | FAIL | D | — |
| `IS-08b-RA-a50-bf16` | BF16 release copy | PASS; below M16 `08b-RA-a75` | A | ce74f1e5 |
| `IS-2b-RA-a50-bf16` | BF16 release copy | PASS; below `IS-2b-RASD` | C | ce74f1e5 |

Pending (BF16 release copies, running unattended; see the state file): `IS-L9IBX-bf16`, `IS-08b-RASD-a50-bf16`,
`IS-2b-RASDML-bf16`, `IS-2b-RA10SDML-bf16`. M18 owns the RAUP interpolation points, so they are not in this sweep.

## Integrity

- Every run passed its 86-request parity gate (86 / 86 `ok`) and the scorer gate (120,224 `ok` + 2 `unsupported`).
- Packages are each tier's IX1-scored package restaged with the candidate's weights (identity = the soup's or BF16
  copy's `model_sha256`, loaded parameter count = the package's, T = 1, calibration none).
- The 9B formal panels (K-a12IB, L9IB): types OK / OK / OK.
- No candidate of this sweep was released, so the release-time checks (Hub smoke, row-level audit, card) were not run
  here; the tier publishers run them for the candidates they release.
