# Qwen3.5-9B Base exact-state replay: deterministic fault not reproduced

**Technical result: narrow negative; official Base arm remains HOLD.** This
reports the single [prospectively frozen optimizer-state probe](qwen35-9b-base-state-continuation-prereg-2026-09-28.md).
It is not a resumed model candidate, SELECT result or release evaluation. No
protected benchmark partition was mounted, no model package was saved, and no
second GPU launch was made.

## Inputs and trajectory

The CPU audit reverified the original official Base shard/config/tokenizer,
7,324-row TRAIN, trainer, provenance, console, and step-64 state hashes. It
reproduced the sealed update-107 16-row token sequence and canonical manifest
digest `d1554946856a3c9ee29a22376a61951d6934ad8a2b77d54c9d11900bb28eaa08`.
The saved step-64 state had two AdamW groups, 506 parameter states and the
Python, PyTorch CPU and accelerator RNG states. Its contract and trainer hash
matched the original run. The pinned image reported PyTorch
`2.12.0+git6bbd260`, HIP `7.2.53211` and PEFT `0.21.0` as required.

From that state, one fresh-process container replayed original TRAIN updates
**65–107** with the native model, LoRA/head, BF16 autocast, loss, AdamW and
learning-rate schedule. Every update 65–106 matched the original log's exact
token total, loss and pre-clip gradient norm: maximum observed relative loss
and gradient deviations were both **0.0**, within the prospectively frozen
1% / 5% limits. Update 107 completed **16/16 forward and backward calls**, the
optimizer step, and 1,536/1,536 paired module forward/backward markers. No
native fault or nonfinite value appeared.

| Bounded execution | Observation |
| --- | --- |
| Window | 43 optimizer updates, 65–107 only; original full target was 458 |
| Time | 2026-09-27 19:41:53–19:47:01 UTC; 308 s = 0.08556 one-GPU-hour |
| Exit / OOM | 0 / false; unchanged host kernel tail |
| Peak allocation | 116,673,514,496 bytes, about 108.66 GiB |
| TRAIN phase ledger | 43 completed updates; target 16/16 forward and backward completions |
| Original saved artifact integrity | Step-64 state and original failed-run console SHA-256 unchanged |
| Cleanup | Task-owned container and reservation removed; accelerator returned idle |

The private receipt SHA-256 is
`a38640d6bcf9782738a56b976d39a4164663bbf290452cbd08763cef36f9f096`,
phase ledger SHA-256 is
`7da4dcd8dca7c4845c1839e7de614fbdc5e6d9ccfeb4742f60361696d77d1bc6`,
console SHA-256 is
`10804fb5a094923b939f857ae3dbb0561ef698d2ebfc8918625ec98ab6273913`,
and local launcher SHA-256 is
`a714b5348a025586c641b27f4eb5110e23f44959c5c4111860675f46edcc4c1f`.
No private row content, paths, GPU address or original predictions are
published here.

## Interpretation and next decision

The original SIGSEGV is **not** a deterministic consequence of the durable
step-64 model/AdamW/RNG state and the exact subsequent TRAIN order in a fresh
process under the same pinned runtime. This is stronger than the earlier
no-update step-107 pass because the trajectory through update 106 was
numerically identical to the failed run. It does **not** prove the original
native operator was safe: unsaved long-lived process/allocator/stream state,
or a nondeterministic ROCm fault, remains possible. The original crash had no
operator-level trace, and no fault occurred here from which the module markers
could isolate an ATen or ROCm operator. Do not claim a root-cause repair.

**Keep the official 9B Base training arm HOLD.** Do not restart its 458-update
run in the unchanged runtime or search partial checkpoints. The next
discriminating step is a separately signed, bounded runtime/operator stress
protocol with native output-parity controls; only a reproduced fault and
demonstrated repair should admit the previously specified one fresh full arm.
