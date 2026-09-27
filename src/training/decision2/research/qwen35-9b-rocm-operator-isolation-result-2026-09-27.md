# Qwen3.5 9B standalone backward operators: PASS, training HOLD

The one GPU cell frozen in the [operator-isolation protocol](qwen35-9b-rocm-operator-isolation-prereg-2026-09-27.md)
ran after the signed [CPU setup amendment](qwen35-9b-rocm-operator-isolation-cpu-amendment-2026-09-27.md)
passed its no-device source/hash/shape check. A fresh GPU inventory and
per-device process/memory check showed an exclusively idle accelerator. No
other workload was stopped. The official-source 9B optimizer arm and release
status remain **HOLD**.

| Frozen field | Observed result |
| --- | --- |
| Pinned image | `sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54` |
| Runner / official config SHA-256, before and after | `c00c8ada89122acea19b035f25383e9bc3f9e2dc8504a80651c111918a5c7134` / `d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05` |
| Container interval, UTC | 2026-09-27 14:43:34–14:43:47 |
| Container exit / OOM / watchdog | 0 / false / completed before cap |
| Charged GPU time | 13.641 s, approximately 0.00379 GPU-hour |
| PyTorch reference gated-delta | 4,096 × 1 and 6,144 × 3: all finite forwards/backwards |
| PyTorch SDPA | 4,096 × 1 and 6,144 × 3: all finite forwards/backwards |
| Peak allocated accelerator memory | 2.656 GiB |
| Full private raw-log SHA-256 | `23c5f859209ec2cad7b0b3728e0579914d7d79c0256bb21331c6ec492a764464` |

The isolated operators **passed this exact synthetic test**. It rules out a
simple deterministic failure of one reference gated-delta call or one SDPA
call at these shapes and on this pinned runtime. It does **not** rule out
their combined autograd graph, 32 stacked transformer layers, LoRA/head
gradients, checkpoint recomputation, model-produced values, or allocator
pressure. The gated-delta cell called the pinned unwrapped PyTorch reference
directly; the full model reaches it through the Transformers fallback wrapper.
The SDPA cell used the same Q/K/V head dimensions but omitted Qwen's projected
values, positional transform and surrounding layers. The complete 9B model
previously crashed at 38.093 GiB with checkpointing and 107.987 GiB without;
the 2.656 GiB isolated peak is a materially different graph and memory state.

No model weights, update-64 checkpoint, optimizer, real task data, teacher
output, evaluation labels, or release candidate were read. The post-exit
device had zero allocated VRAM and no KFD process. No automatic retry or next
GPU cell was launched. A future combined-graph diagnosis would require its
own prospective protocol and cap; this result alone does not justify resuming
the frozen 9B optimizer arm.
