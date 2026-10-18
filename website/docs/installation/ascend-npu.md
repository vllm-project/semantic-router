---
title: Ascend NPU
description: Run the Router's local signal models on Ascend NPUs through the npu accelerator.
---

# Deploy with Ascend NPU

The model runtime serves the Router's local signal and decision models on
Ascend NPUs (`device: npu:N`), the way it does on CUDA and ROCm. Ascend support
is functional and measured, but not yet validated: the accelerator stays
opt-in, `device: auto` never selects it, and the Vela models check their
answers against NPU reference answers recorded on an Ascend 910B1
([parity record](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/npu-parity-ascend910b1.md)).
The paths below were measured on Ascend 910B1 and 950DT hosts
(torch_npu 2.10.0.post4, CANN 8.5 / 9.1).

Large-model inference on Ascend belongs to vLLM behind an HTTP endpoint; the
`npu` accelerator covers the Router's own 307M signal encoders and
decision models inside the model runtime.

## Prerequisites

- an aarch64 Linux host with an Ascend NPU (910B series or 950) and the NPU
  driver installed; `npu-smi info` must list the device;
- the CANN toolkit on the host, under `/usr/local/Ascend`;
- `torch_npu` installed next to the runtime (`pip install torch_npu`,
  an aarch64-only wheel); and
- the CANN environment loaded before the runtime starts.

Always load the CANN environment first. Without it, importing `torch_npu`
fails with a backend-extension error because `ASCEND_OPP_PATH` is unset:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## Run the model runtime on NPU

Serve a model the same way as on a GPU, naming the device:

```bash
export ASCEND_RT_VISIBLE_DEVICES=3   # pick a free card; without it npu:0 is physical card 0
vllm-srun serve vllm-sr/Vela-2.0-0.3B --device npu:0 --port 8100
curl --fail http://127.0.0.1:8100/health   # {"status":"ready", ...}
```

Readiness runs the model's golden check: two reference forwards, each compared
against the released NPU reference answers, with the tolerance policy other
GPUs use. A model that fails the check reports it at startup instead of
serving unverified answers.

`device: auto` still selects the CPU on an Ascend host: accelerators that are
not yet validated are only used when named explicitly. In a full
configuration, place a deployment on the NPU with `device: npu:0` — the same
[placement rules](../model-runtime/deploy.md#place-and-scale-replicas) as for
GPUs, with the runtime installed next to torch_npu.

## Run the NPU runtime in a container

There is no dedicated NPU image (decided in
[#4827](https://github.com/vllm-project/semantic-router/pull/4827)). The
validated path runs the runtime inside the arm64 `vllm-sr` (CPU) image with
the host CANN mounted. Three things the CPU image does not ship are required:

- `--privileged`: the Ascend 950 driver rejects device enumeration from
  non-privileged containers (error 87) unless the host runs
  ascend-docker-runtime;
- the host's UB user-space libraries (`libummu.so.1` and the `liburma*`
  family) mounted **by file** — mounting all of `/usr/lib64` breaks the
  image's glibc; list the exact files with
  `ls /usr/lib64 | grep -E 'urma|ummu'`; and
- `torch_npu` installed in the container at start.

```bash
IMG=ghcr.io/vllm-project/semantic-router/vllm-sr:latest

docker run --rm -it --privileged --entrypoint bash --net=host \
  -v /usr/local/Ascend:/usr/local/Ascend \
  -v /opt/models:/opt/models \
  -v /usr/lib64/urma:/usr/lib64/urma:ro \
  -v /usr/lib64/liburma.so:/usr/lib64/liburma.so:ro \
  -v /usr/lib64/liburma.so.0:/usr/lib64/liburma.so.0:ro \
  -v /usr/lib64/liburma_common.so:/usr/lib64/liburma_common.so:ro \
  -v /usr/lib64/liburma_common.so.0:/usr/lib64/liburma_common.so.0:ro \
  -v /usr/lib64/libummu.so.1:/usr/lib64/libummu.so.1:ro \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /var/log/npu:/usr/slog \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  $IMG
```

The exact `liburma*` sonames vary with the host driver build; mount every file
the `ls` above shows. Then, inside the container:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/lib64:${LD_LIBRARY_PATH:-}
pip install torch_npu
python3 -c "import torch, torch_npu; print(torch.npu.is_available(), torch.npu.device_count())"
# → True 8
```

The driver library paths under `/usr/local/Ascend` differ between CANN
releases (9.1 adds an `aarch64-linux/` level); adjust them to what the
container sees. The torch_npu version must match the PyTorch the image ships.

`vllm-sr serve` orchestration provisions its own runtime container without
these mounts, so a fully orchestrated local stack keeps its models on the CPU.
Run the runtime in the prepared container and point the Router's deployments
at it ([External runtime](runtime/external.md)), or keep the whole stack on
the host.

## Troubleshooting

### Device enumeration fails with error 87

`drvGetDevNum` returns 87 ("Can't get ascend_hal device count") in a
non-privileged container: the 950 driver's manager ioctl rejects it. Capability
flags, device rules, or an unconfined seccomp profile do not change this. Start
the container with `--privileged`, or install ascend-docker-runtime on the
host for a non-privileged path.

### set_device fails with TsdOpen or `urma_attr is NULL`

The container lacks the host's UB user-space libraries, so the `liburma`
load chain fails. Mount `libummu.so.1` and the `liburma*` family from
`/usr/lib64` by file. Do not bind-mount the whole directory: the image's
glibc then fails with GLIBC_PRIVATE symbol errors.

### torch_npu fails to import

`Failed to load the backend extension: torch_npu` means the CANN environment
is not loaded. `source /usr/local/Ascend/ascend-toolkit/set_env.sh` before
starting the runtime, and check `ASCEND_OPP_PATH` is set in the container.

### The runtime serves on the wrong card

`ASCEND_RT_VISIBLE_DEVICES` selects the visible cards; without it, `npu:0` is
physical card 0. A card that opens is not necessarily free — the driver's
device-open call is the real gate. Check `npu-smi info` for HBM usage and pick
a card with enough free memory.
