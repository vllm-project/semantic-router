# Official 9B Base runtime fault: independent read-only audit

**Disposition: HOLD.** This note audits the failed official-Base arm after the
[signed result](qwen35-9b-base-init-contrast-result-2026-09-28.md). It does not
resume training, change the frozen selector or interpret a partial checkpoint
as a model result. No GPU cell was run for this audit.

## What is established

- The frozen Base source, 7,324-row TRAIN, SELECT, CAL, code and image identities
  were verified before launch. Zero-step, one-update and 32-row fresh reload
  passed their technical gates. The full arm then logged 106 complete finite
  updates and stopped during the next backward; the full console log matches
  its recorded SHA-256 `59e2aecf8e2c6d684598d0cbd994c305cd38c9ed83834ba2673f5bd559f13ded`.
  The last observed allocation high-water mark was 108.434 GiB, well below
  device capacity. There is no completed 458-step run or eligible BEST.
- The bounded kernel record at the failure time shows a `pt_autograd_0` thread
  faulting at instruction pointer `0x100000001`, followed by a Python process
  fault at address `0x8` inside `libhsa-runtime64.so.1.18.0`. This identifies
  a native backward/runtime failure, **not the specific tensor operation or
  original corruption site**. No application-level traceback or OOM event was
  recorded. Because the trainer logs only at optimizer boundaries, the exact
  failing microbatch and its token length are unknown.
- This Base run used `--no-gradient-checkpointing`, LoRA plus the native
  candidate head, FP32 stored weights/trainables and BF16 backbone autocast.
  The trainer does not call `torch.compile`; the autograd thread name is not
  evidence of TorchInductor involvement. Transformers reported its PyTorch
  reference gated-delta fallback, not an optimized FLA path.
- The earlier official 9B full-model synthetic backward fault reproduced on
  both authorized nodes under the same image; disabling checkpointing also
  faulted. Isolated reference gated-delta and SDPA backward cells passed.
  Those probes used another source checkpoint and included 6,144-token cases,
  so they narrow but do not exactly reproduce this Base arm's 4,096-token
  maximum. The separately completed Posttrained 458-step arm shows the image
  is not deterministically broken for every 9B training sequence.

**Working hypothesis:** the pinned PyTorch/ROCm HSA runtime interacts badly
with a long-lived, combined Qwen3.5 hybrid-attention autograd graph or its
memory/stream state. The kernel record and cross-node failures make this more
plausible than a Python label error, one bad accelerator, gradient
checkpointing alone, simple capacity exhaustion or an Inductor compilation
fault. None of those alternatives is fully excluded, and the faulting native
operator remains unidentified.

## Smallest discriminating technical preflight

Freeze a new technical protocol and code/image hashes **before** any GPU run.
First, on CPU, reconstruct the original deterministic TRAIN ordering and
native token lengths around updates 105–107 from the pinned tokenizer, data,
seed, and saved step-64 cursor. Privately retain only row identity hashes,
type, token length, order and their manifest hash; inspect length/type/label
validity without opening evaluation labels. Verify the existing log and
checkpoint receipts again. This identifies which real TRAIN inputs a bounded
probe should exercise; it cannot localize the original fault by itself.

If that audit passes, run **one** no-optimizer, no-weight-update combined-model
cell from the exact saved Base step-64 adapter/head and immutable official
source. Replay the 16 TRAIN microbatches scheduled for update 107 twice in
their frozen order, interleaved with one short and one maximum-length TRAIN
control. Preserve the trainer's BF16 autocast, head/loss, LoRA targets,
reference gated-delta fallback and disabled checkpointing. Emit and flush a
private event with hashed row identity, token length and forward/backward
phase; synchronize the device at phase boundaries, check finite loss/gradient
and verify trainable tensors are bitwise unchanged. Capture container exit,
OOM flag, memory peak and bounded kernel events. Use one independently idle
accelerator, a pinned image, a single container, no network, at most 600
seconds (0.167 one-GPU-hour), and no automatic retry. Keep raw prompts,
model state and logs private. A source mismatch, timeout or OOM is
**INDETERMINATE**; exit 139 is **FAIL**; all finite backwards are only a PASS
for this narrow cell and do not certify a 458-step optimizer run. The step-64
state is the last durable state, not the unknown exact step-106 state.

## Prospective one-retry amendment, if a mitigation exists

Do not repeat the full arm in the same image merely because this cell passes:
earlier full-model cells and this optimizer arm already faulted under that
runtime. A retry becomes rational only after a **separately pinned runtime or
single operator-path mitigation** passes a full combined-graph stress cell at
the frozen 4,096-token cap, with the exact Base source and native output
parity on 32 fixed SELECT inputs (zero category changes, p99 absolute
probability drift <= .005, maximum <= .02). Record the changed runtime or
operator path as the sole experimental difference and its digest before
launch. A changed image makes the prior Posttrained arm a descriptive control,
not a clean initialization-only causal contrast.

Then allow at most **one fresh** Base-initialized 458-update arm, not an exact
resume from the failed run or a search among partial checkpoints. Retain the
same source/data/seed/LoRA/head/objective/length/batch/update budget, eight
SELECT checkpoints, frozen BEST selector, 4.0 one-GPU-hour cap and downstream
reload/development gates. Add only phase and row-hash instrumentation; never
log private text. A native fault, OOM, nonfinite value, identity drift or cap
breach returns HOLD without changing thresholds or trying another checkpoint.
If a validated mitigation is unavailable, keep this official-Base cell HOLD
and prioritize a separately preregistered 9B training/data arm instead.
