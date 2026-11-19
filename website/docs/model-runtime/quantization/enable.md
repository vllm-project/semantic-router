---
title: Serve a quantized package
sidebar_label: Serve a quantized package
sidebar_position: 2
description: Serve an FP8, int4 or NVFP4 package of a Decision 3.0 model, from its repository or a local copy, and check that it loaded.
---

import Tabs from '@theme/Tabs'
import TabItem from '@theme/TabItem'

# Serve a quantized package

A quantized package is served like any other model: name its repository with
the commit to load, or the absolute path of a local copy. There is no switch to
turn on; the runtime reads the format from the package's `config.json` and
checks that the device can run it.

## Pick the package for your GPU

<Tabs groupId="quantization">
<TabItem value="fp8" label="FP8" default>

For an AMD Instinct MI300 or MI350 series GPU, or an NVIDIA GPU of compute
capability 8.9 or later:

```bash
vllm-sr serve vllm-sr/d3-FP8 --revision 2fb271adbdaf6dc969ca1ca746b61a6a5127ed62 --engine --platform rocm
```

</TabItem>
<TabItem value="int4" label="int4">

For any GPU that runs Decision 3.0:

```bash
vllm-sr serve vllm-sr/d3-W4A16-GPTQ --revision 93aa927c5bbf184770ecd144972bed9a1450a4bd --engine --platform cuda
```

</TabItem>
<TabItem value="nvfp4" label="NVFP4">

For an NVIDIA Blackwell GPU (compute capability 10.x or 12.x):

```bash
vllm-sr serve vllm-sr/d3-NVFP4 --revision 07fa5a04859155eb99a034e2e424d423dd50ce1d --engine --platform cuda
```

</TabItem>
</Tabs>

`--revision` takes a full commit, a branch or a tag; the CLI resolves it once,
before startup, into the commit it stores in the active deployment.
[Support by hardware and model](./support.md) lists every published package
and the revision its record measured.

## Use it from the Router

Declare the package as a deployment, with the 40-character commit to load, and
name it where it answers:

```yaml title="config.yaml"
global:
  model_catalog:
    deployments:
      d3-fp8:
        provider: model_runtime
        artifact: vllm-sr/d3-FP8
        revision: 2fb271adbdaf6dc969ca1ca746b61a6a5127ed62
        device: rocm:0
      spans:
        provider: model_runtime
        artifact: vllm-sr/vela3-fast
    bindings:
      pii: spans
      hallucination: spans
    system:
      decision_model: {deployment: d3-fp8}
```

- A package that is not built in needs `revision`, a 40-character commit; the
  Router refuses a branch or a tag in YAML.
- Decision 3.0 answers no span questions, so `pii` and `hallucination` stay on
  a span-capable deployment.
- `device: auto` skips devices that lack the format's matrix units. Naming the
  GPU (`rocm:0`, `cuda:0`) makes a device that cannot run the package fail at
  once, with the reason.

To serve a local copy instead, give the absolute path of the package directory
as `artifact` and leave `revision` out. A runtime you run yourself takes the
same package:

```bash
vllm-srun serve vllm-sr/d3-FP8 --revision 2fb271adbdaf6dc969ca1ca746b61a6a5127ed62 --device rocm:0 --host 0.0.0.0 --port 8100
```

## Check that it loaded

- `GET /v1/models` on the worker shows the package's revision, its identity
  (`model_sha256`) and its self-check, which reports `unverified`: published
  packages have no recorded reference answers.
- `vllm_srun_model_memory_bytes` on the worker's `/metrics` shows the bytes its
  loaded weights take: about 31.9 GB for FP8 against 52.2 GB for BF16.
- Ask it the questions your rules ask, on a few dozen representative requests,
  and compare the answers with the BF16 package's before you switch traffic.

## If it is refused

| The reason says | Do this |
| --- | --- |
| FP8 weights need an NVIDIA GPU of compute capability 8.9 or later or an AMD Instinct MI300 or MI350 series GPU | Serve the FP8 package on such a GPU, or serve the int4 or BF16 package instead. The runtime does not retry. |
| NVFP4 packages need FP4 tensor cores; serve NVFP4 packages on a Blackwell GPU | The device is a CPU, a ROCm GPU or an NVIDIA GPU without FP4 tensor cores, or its PyTorch has no FP4 tensors or Triton is missing; the reason names which. Serve it on a Blackwell GPU with a recent PyTorch and Triton, or serve the FP8, int4 or BF16 package. |
| only compressed-tensors int4 (pack-quantized), FP8 (float-quantized) and NVFP4 (nvfp4-pack-quantized) weights are supported | The package's `quantization_config` is another method or format. Requantize it as one of these, with one group of weights. |
| needs about X GiB, Y GiB free | The GPU has less free memory than the package's files and an activation reserve. Free memory on that GPU, choose another, or use a smaller package. |

[Troubleshooting](../deploy/troubleshooting.md#the-runtime-reports-failed) lists
the other load errors.

## Bring your own package

The runtime loads a Decision 3.0 package whose `config.json` carries a
compressed-tensors `quantization_config` with one group of weights, in one of
these schemes:

- **int4:** `pack-quantized`, 4-bit integer weights in groups of input columns
  (a multiple of 8), symmetric (GPTQ) or asymmetric with zero points (AWQ), and
  BF16 activations.
- **FP8:** `float-quantized`, FP8 E4M3 weights with one scale per output
  channel, and activations quantized to FP8 per token as each call runs
  (`FP8_DYNAMIC`).
- **NVFP4:** `nvfp4-pack-quantized`, FP4 E2M1 weights and activations in groups
  of 16 with FP8 E4M3 block scales under a tensor global scale, weights static
  and activations dynamic per block.

The package also needs its `MODEL_MANIFEST.json`, as the published packages
have. Anything else is refused rather than loaded as BF16.
