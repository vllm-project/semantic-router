# Decoder M17 stage 2, amendment 2: the released 4b-LHA10SDML is the base (2026-10-02)

Coordinator interrupt, 15:45 UTC+8 (user decisions):

- M15 `4b-LHA10SDML` is released to `Decision-2.0-Nox-4B`, by user override of the Index-first significance rule.
- The 4B goal is to overtake JPT-4B on the Index.
- Stage-2 arms use `4b-LHA10SDML` as the new base reference.

This amendment changes three things:

- **Reference:** from the release of `4b-LHA10SDML`, every stage-2 candidate is compared with the IX1 run
  `IS-4b-LHA10SDML-bf16` (node D), not with `DEV2.0-4B-LHS17SD-bf16`.
- **New arms:** wave 3, trained on node F GPU2/3/6/7.
- **New interpolation:** one CPU-only point.

The recipe, the seeds (20260926 / 20260927), the stop rules, the measurement on the BF16 release copy and the
release rule are unchanged from the prereg and amendment 1. A candidate is released only if its Index lower bound vs
the then-current Nox is > 0, and only after the integrity checks pass.

## Arms (wave 3; `ops/m17/m17-prep4.sh`, `data/4b-s4`, locked by `data/READY-m17s4.json`)

| Arm | TRAIN | Teacher | What it tests |
| --- | --- | --- | --- |
| `4b-SDMLIB4` | the `4b-LHA10SDML` TRAIN (M15 lock `fef6b036…`, byte for byte), then IB4 phase 1 (`6045b456…`, every row), then IB3-r2 (`9d92d92a…`, every row) | a hard link of `4b-LHA10SDML`'s (`b95c5e63…`; IB rows gold only) | stage 2's arm (a) on the new base |
| `4b-LHS17ML` | the `4b-LHS17SD` TRAIN (`14bce13c…`), then the 4,252 multilingual copies of `4b-LHA10SDML` (its `~m2` rows, file order) | `4b-LHS17SD`'s (`374f4fa6…`), then those copies' teacher rows from `4b-LHA10SDML`'s | the 17% swap and the multilingual copies combined |

## Interpolation points (CPU; `ops/m17/m17-soup.sh`)

- `4b-SDMLxS17-m50`: the uniform FP32 average of the `4b-LHA10SDML` soup (`1b515675…`) and the `4b-LHS17SD` soup.
- `4b-SDMLIB4-m50` and `4b-LHS17ML-m50`: each arm's soup averaged with the `4b-LHA10SDML` soup.

## Budget

- Wave 3 is 4 seeds of about 1.2 GPU-h each.
- The Index runs use node F GPU2–7 (lease-checked), node C GPU1–7, node D GPU4–7 and the eval fast lane (node E
  GPU0–3 / 6–7).
- The M17 cap stays 50 GPU-h; the chains' gate on node F stops new seeds above it.
