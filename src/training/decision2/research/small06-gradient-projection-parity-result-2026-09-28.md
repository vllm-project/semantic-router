# 0.6B projected-gradient arm: one-update technical gate passed

This is a **technical parity result**, not a new trained checkpoint,
development score, or release claim. The preceding read-only diagnostic
measured gradient conflict in six of eight frozen TRAIN windows; the separate
one-update check now confirms that per-type bookkeeping with projection
disabled reproduces the ordinary update.

The opt-in parity implementation is signed commit
`bc90b9c90e7e5a5a0e889f4e9106f2ba95ffe7e9`. The exact runtime archive
SHA-256 was `9ebc4b81a65289aab3edc838b17db4516cd7ae1daa36217f1e53683eda77fead`.
Local `make check`, `make test-training-contracts`, five focused CPU tests,
Black and Ruff passed before the remote run. The runtime source was mirrored
from that commit; no trainer default changed.

The CPU plan verified the official `Qwen/Qwen3-0.6B-Base` revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, the frozen TRAIN and
SELECT hashes, all 7,455 TRAIN rows / 4,094,489 native tokens, the exact
epoch-zero order, and **466 planned updates**. The first-step warmup factor
was `1/23`, rather than the factor from a standalone one-step schedule.
The private plan receipt SHA-256 is
`20174e684605aadbc19b3b16b32cbbcc091a8f0f6bdeaaa28a7898157b1f5201`.

Two fresh, independent model loads received the same first 16-example
training window and one AdamW update each. The ordinary arm independently
reproduced the trainer's usual operations; the other grouped backbone
gradients by task type and then summed them **without projection**. The
ordinary head gradient, CE + 0.5 Brier loss, BF16/FP32 computation, global
clip and 466-step-horizon learning rate were held fixed. Both unclipped
gradient norms were `27.687129974365234` and finite. The arm receipt
SHA-256 values are `273cd0a7a5ab5d5aba43a88c125d2dd3f78a6dd86d533603a62bb78cf2bc517a`
and `4f14d7c4716c39c98fecdcc88f91d81f0e61278249b931c060b70218db7c1d40`.

On the same fixed 32 SELECT inputs, using the native two-item BF16 batch
shape and **no gold labels for scoring**, the independent paths had zero
categorical changes and `0.0` maximum absolute option-probability difference
at both zero step and after the update. This passed the prospectively fixed
gate of zero categorical changes and maximum drift `1e-5`. The aggregate
comparison receipt SHA-256 is
`0e71c153ae5ca9970ed44ec10f92fc85e9e5282bf0f06576d76c9b7daba8d726`.
Model-load-through-inference time summed to **29.617 seconds = 0.008227
GPU-hour**; the conservative sum of whole GPU-visible container lifetimes was
**55.819 seconds = 0.015505 GPU-hour**. The card was released after both arms.

The frozen first optimizer window contained 11 Choice and 5 Noul rows, with
no Score row. Thus this single update establishes control-path parity but
does not exercise a real Score gradient. CPU integration tests exercised all
three task types; the earlier read-only measurement tested eight actual
three-type windows. No full projection treatment, SELECT checkpoint selection,
formal evaluation, or publication decision occurred in this gate. Next is
review and the single already preregistered projected-gradient arm; failed
gates will retain HOLD without threshold changes.
