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

**INDETERMINATE; no backward occurred.** The single launcher ran for two
seconds, exited 1 before model loading and wrote no row-phase events. The
inside-container BF16/device admission check rejected the environment because
no accelerator was visible. The container was not OOM-killed, the host GPU
remained idle, no tensors or weights were loaded or changed, and the task-owned
container and reservation were removed. Private receipt SHA-256:
`e647d6bc4b572071ff345718143ea3ff8e1bdbb882ee33c07afada1f5e09acad`;
private launcher log SHA-256:
`0f4e1cdbb50ec4e3f2090947ddb4f720df3b0ce3324ac2abcf57e7a3ce881704`.
There were zero GPU-compute hours and two seconds of launcher wall time.

A read-only host bus-to-render-node inspection then showed that the selected
device node was a virtual partition rather than the physical idle accelerator.
The exact corrected mapping is retained only in the private execution record.
No same-protocol automatic retry was run. The faulting native backward operator
is still unidentified; this launcher failure neither reproduces nor clears
the original SIGSEGV. The 9B Base arm remains HOLD.

## Prospective device-only amendment, signed before corrected launch

The failed launcher performed no model or GPU work. A second, final technical
launch is admitted with **only** the corrected physical GPU device mapping.
The private mode-0600 bus-to-device receipt has SHA-256
`710a0f0b9d4fd7036d877cf1f6e87aed8a3be12f83d58df7208ed1b0e7f6f693`.
It matches the requested idle accelerator's physical PCI bus to its actual
render node, unlike the virtual node mistakenly used on the first launch.
Recheck the bus/node mapping and zero utilization immediately before launch;
stop if either differs. The original source, step-64 weights, exact TRAIN
schedule, controls, pinned image, probe script, no-network setting, 600-second
cap and finite/unchanged-weight stop rules above are **unchanged**. No other
device or automatic retry is allowed. Preserve the first failed receipt as a
separate negative event. A corrected launch still cannot justify an optimizer
run by itself.

## Corrected-launch outcome

**NARROW PASS, 9B Base full arm remains HOLD.** The corrected single-container
probe ran 54 seconds (0.015 one-GPU-hour), exited 0 and was not OOM-killed.
It completed all **34/34** real TRAIN backward calls: 16 in the frozen
update-107 order, the 75-token short and 4,089-token longest controls, and
the same 16 in order again. The private phase ledger contains 34 starts and
completions for both forward and backward and four complete rounds (106
events total). All losses and gradients were finite; trainable tensor bytes
were identical before and after. Peak allocated device memory was
116,537,272,832 bytes (approximately 108.5 GiB), comparable to the failed
optimizer arm's last observed high-water mark. The host kernel tail did not
change across this short run. The container and reservation were removed and
the accelerator returned idle.

Private receipt SHA-256:
`dc0a6617cbc16596255e29fb5e32227b94e301aa549b060d3d28ab3813c19524`;
private phase-ledger SHA-256:
`0fa7d3f54dc2a2c88f4ec091b1f37a61aa57636b52656ab49bd54f407f8b7dd0`;
private console SHA-256:
`9f14a8e216b7d4170b199e85fcd428618f9bf04524452e4f5918567f119eb191`.
No evaluation predictions or weights were created.

This rules out a deterministic failure of those encoded TRAIN inputs under
**this** short, no-update step-64-state cell. It does not recover the actual
step-106 weights/RNG, exercise optimizer steps 65–106, prove that the original
faulting operation is safe, or establish a viable 458-update training path.
The original fault could depend on accumulated optimizer/runtime state or a
different transient condition; the operator remains unidentified. **Do not
resume or repeat the Base optimizer arm in the unchanged image.** The next
discriminating experiment is a separately pinned operator/runtime mitigation
stress cell with native zero-step output parity, followed at most by the one
fresh full arm allowed by the parent fault audit if that mitigation passes.
