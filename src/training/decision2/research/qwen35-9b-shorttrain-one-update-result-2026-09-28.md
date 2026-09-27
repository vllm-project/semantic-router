# Official Qwen3.5-9B short-TRAIN one-update/reload result

**Technical PASS; model-quality and release HOLD.** This is the sole optimizer
smoke specified by the signed [prospective note](qwen35-9b-shorttrain-one-update-prereg-2026-09-28.md).
It used fresh official general Qwen3.5-9B weights, not the earlier failed run,
the old one-update checkpoint, or any third-party Decision weights. The
complete-group filtered TRAIN and unchanged SELECT/CAL hashes matched before
and after; source 12/12 and trainer 7/7 file hashes also matched. One isolated
accelerator was idle again after the two processes.

| Gate | Observation |
| --- | --- |
| TRAIN/SELECT/CAL | 7,324/700/700; fixed hashes in prospective note |
| Updates | Exactly 1, 16 TRAIN rows and 13,328 tokens |
| Numeric | Loss 1.590555; gradient norm 11.178479; finite |
| SELECT diagnostic | Baseline 231/700, family macro .301132; after update 251/700, family macro .303860; 700/700 valid both times |
| Checkpoint | Complete saved step 1; original SELECT predictions SHA-256 `86a425fff67126e314a753c52678e25a8d49953bd80a0d0b5441678c6956bcd7` |
| Independent reload | 32/32 rows, Choice13/Noul14/Score5, zero category changes, p99 and maximum probability drift both zero |
| Reload implementation | Signed `a5d3e4e9a`, SHA-256 `8695c27e8c598301009934f1a75dafbce63dea3a460a4d6261f8849207bd66dd` |
| Container GPU time | 86.558 seconds for train + 29.960 seconds for reload = **0.03236 GPU-hour** |

The SELECT figures diagnose one update only and are not generalization or
release scores. No CAL, typed DEV, CSS pilot, formal JevArena or public
JevBench labels were accessed. The one-update checkpoint remains a private
technical artifact and will not initialize the new candidate. This pass
admits the separately frozen [458-update development arm](qwen35-9b-shorttrain-full-prereg-2026-09-28.md).
Training on a 4,096-token filtered subset still leaves long-input capability
unproven; the final arm may fail quality or parity gates and remain HOLD.
