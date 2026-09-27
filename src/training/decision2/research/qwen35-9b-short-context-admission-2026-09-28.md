# Official Qwen3.5-9B: prospective short-TRAIN admission

**Status: CPU data audit only; optimizer arm remains HOLD.** This proposal
does not resume the faulted 9B training run, change its checkpoint, or claim a
new model score. It uses the official general `Qwen/Qwen3.5-9B` source at
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, not a third-party Decision
model. The existing own-Lux arm and official-source failed runs are retained as
separate historical evidence.

## Why this cell is worth testing

The native rights-clean v2 TRAIN has 7,455 rows, 4,194,465 Qwen3.5-9B tokens,
and a maximum 6,596-token row. A separate read-only, exact-source/tokenizer
CPU audit with maximum length 4,096 found **131 overlength TRAIN rows** and
zero overlength SELECT/CAL rows. The source TRAIN/SELECT/CAL SHA-256 values
were `61740be4…243755`, `32a4352d…0f2a6`, and `3e34f6cb…f60a`; the
tokenizer JSON was `5f9e4d49…b8ed`. The complete earlier 8,192-token audit
and two 9B backward faults are retained unchanged. Isolated reference
gated-delta and SDPA both passed 4,096 and 6,144 tokens, so this length
restriction is only a diagnostic hypothesis, not a proven fix.

## Ordered gates

1. Use the checked-in group-preserving `training.data.filter_native_length`
   tool on the pinned inputs and tokenizer. Remove a **complete original
   source group** if any member exceeds 4,096 tokens. Never clip text or
   recompute the answer. Write new TRAIN bytes and a private manifest with
   exact retained/removed group, row and token counts, type/language mix,
   source/output hashes and removed IDs. Require all retained native inputs
   at most 4,096, unchanged SELECT/CAL, valid rows and complete split
   isolation. If the removed groups materially erase any family, language or
   all genuine long evidence, stop and redesign instead of calling the subset
   representative.
2. Only after local implementation/tests and a CPU output review, run **one**
   bounded no-optimizer long-backward cell from the existing pinned 9B source
   and update-64 diagnostic checkpoint. Use the already checked-in
   `rocm_9b_no_checkpoint_probe.py` with seed `20260927`, 20 iterations and
   fixed cycle `[512, 1024, 2048, 4096]`, synthetic IDs/labels, BF16
   autocast, the same decision head/loss, and no real task or evaluation row.
   On one freshly verified idle GPU, cap at 600 seconds (0.167 GPU-hour),
   preserve source/checkpoint hashes before and after, and require 20/20
   finite gradients, no native fault, zero optimizer updates, and unchanged
   trainable tensors. An exit 139, OOM, timeout or hash mismatch stops this
   route. Do not search alternate lengths/checkpoints after a failure.
3. Even a passing synthetic cell only admits a separately versioned
   **one-update/reload smoke** using the filtered TRAIN and unchanged SELECT.
   Before an optimizer start, freeze the new file SHA, row/token budget,
   schedule, model initialization, source hashes and native parity thresholds.
   A passing smoke would admit a newly registered complete development arm;
   it does not revive the original 8,192-token arm or authorize formal v3.

Any resulting model must evaluate the **complete** same-panel v3 and public
JevBench inputs without truncation, counting overbudget outputs as failures.
Training at 4,096 cannot establish long-input competence. Report length-bin
results and compare own Lux1 plus an Index-selected near-size peer rerun under
the same native protocol. A later post-key formal comparison remains post-key;
independent corroboration is required before claiming broad improvement.
