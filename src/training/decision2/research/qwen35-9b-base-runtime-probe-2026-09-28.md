# Official 9B Base backward fault: sealed technical replay

This is a **runtime diagnostic**, not a training candidate, benchmark result,
checkpoint selector or release gate. It follows the independent
[fault audit](qwen35-9b-base-runtime-audit-2026-09-28.md) and leaves the
458-update official Base arm **HOLD**. No evaluation partition is mounted.

## Pre-GPU freeze

The original official Base source is
`Qwen/Qwen3.5-9B-Base@68c46c4b3498877f3ef123c856ecfde50c39f404`.
The pinned original trainer's seven source hashes, frozen 7,324-row TRAIN
SHA-256 `fe9c419a3e751e4a4173b5c90c49e0a83bba7e1c35c683441bb036138acb971c`,
run provenance SHA-256 `21a544b82a3b98a5bcb5a123ae37d5802ecf5fa301e2ce0e03572ea8fdc6d94c`,
full console SHA-256 `59e2aecf8e2c6d684598d0cbd994c305cd38c9ed83834ba2673f5bd559f13ded`,
train metrics SHA-256 `10ff2bd1934304350e68e12751f9d70eef91709569a66599192dfd7435024e79`,
and step-64 trainer state SHA-256
`cbe3b4be2f9fd42db9fd7c11e631608eb10a3b4feef829ffaf8f8f509f5c6786`
were reverified. The saved cursor is epoch 0, next batch 1,024. The original
log contains exactly 106 completed optimizer updates. The original source and
frozen data, not any benchmark gold, were used for reconstruction.

The deterministic one-example/16-accumulation schedule was reconstructed from
the original seed and tokenizer. Updates 105 and 106 exactly match the logged
TRAIN token totals. The 16 update-107 token lengths, in order, are
`[138,436,128,128,4065,93,108,277,3196,240,298,225,463,97,77,261]`.
The shortest and longest TRAIN controls are 75 and 4,089 tokens. The private
row-hash/type/token manifest is mode 0600; its canonical content digest is
`d1554946856a3c9ee29a22376a61951d6934ad8a2b77d54c9d11900bb28eaa08`
and file SHA-256 is
`e72f04219ef4d267fb3c851c8ff4c1b4c5d9a3d44f40e8463cf3eea7679f1ed1`.
No raw TRAIN text or row identifiers are published. The script SHA-256 is
`4263bf8c9ca2b5605280144a847cd3f46ccf4c0dff80f07b131cbd20c7a41956`;
the remote mirror hash matched before this note. The original pinned image is
`sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`.

**Single allowed probe:** load the immutable official source plus step-64
adapter/head, preserve the BF16 autocast and FP32 head/loss, no gradient
checkpointing, and replay the exact update-107 16-microbatch order twice,
with the short and maximum-length TRAIN controls between rounds. Synchronize
and flush private row-hash/phase events around forward and backward. Require
finite loss/gradient, exact token identities, and bitwise unchanged trainable
tensors. Use one independently idle GPU, a single container, no network, at
most 600 seconds (0.167 one-GPU-hour), and **no automatic retry**. Original
source/checkpoint mismatch or GPU unavailability stops before launch. Exit 139
is fault reproduction; OOM/timeout/source mismatch is indeterminate; all
finite backwards is a narrow pass only. Record exit, time, memory and bounded
kernel evidence privately, then add the public-safe conclusion below.

The original model/data/checkpoint are present on an idle authorized node; the
other authorized node has a free accelerator but lacks these ~20 GiB pinned
artifacts. Reusing the node with the exact artifacts avoids copying large
private weights and data for a ten-minute bounded probe. This probe does not
recover the unknown exact step-106 parameter or RNG state, and does not
authorize resuming or repeating the failed optimizer arm.

## Outcome

Pending the one frozen GPU diagnostic.
