---
title: 昇腾 NPU
description: 通过 npu 加速器在昇腾 NPU 上运行路由器的本地信号模型。
is_mtpe: false
---

# 使用昇腾 NPU 部署

模型运行时可以通过 `device: npu:N` 在昇腾 NPU 上运行路由器的本地信号模型和
decision 模型，方式与 CUDA、ROCm 相同。昇腾支持已经打通并有实测数据，但**尚未
转正（validated）**：该加速器只接受显式指定，`device: auto` 永远不会选中它；
Vela 系列模型加载时会与在 Ascend 910B1 上采集的 NPU 参考答案对照自检
（[parity 记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/npu-parity-ascend910b1.md)）。
以下路径在 Ascend 910B1 与 950DT 主机上实测（torch_npu 2.10.0.post4、CANN 8.5 / 9.1）。

大模型推理在昇腾上由 vLLM 经 HTTP endpoint 承担；`npu` 加速器覆盖的是模型运行时
内部的路由器自有模型（307M 信号编码器与 decision 模型）。

## 前提条件

- 一台 aarch64 Linux 主机，带昇腾 NPU（910B 系列或 950）并装好 NPU 驱动；
  `npu-smi info` 必须能列出设备；
- 宿主机已安装 CANN 工具包，位于 `/usr/local/Ascend`；
- 运行时旁已安装 `torch_npu`（`pip install torch_npu`，仅 aarch64 wheel）；以及
- 启动运行时之前加载 CANN 环境。

先加载 CANN 环境。否则导入 `torch_npu` 时会因 `ASCEND_OPP_PATH` 未设置而报
backend-extension 错误：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 在 NPU 上运行模型运行时

与 GPU 相同的起服方式，显式指定设备：

```bash
export ASCEND_RT_VISIBLE_DEVICES=3   # 选空闲卡；不设置时 npu:0 即物理 0 卡
vllm-srun serve vllm-sr/Vela-2.0-0.3B --device npu:0 --port 8100
curl --fail http://127.0.0.1:8100/health   # {"status":"ready", ...}
```

readiness 会执行模型的 golden 自检：两次参考前向，各自与发布版 NPU 参考答案
对照，采用与其他 GPU 相同的容差判据。自检不通过的模型会在启动时报告，而不是
带着未验证的答案继续服务。

在昇腾主机上 `device: auto` 仍会选择 CPU：尚未转正的加速器只在显式指定时使用。
完整配置里，用 `device: npu:0` 把 deployment 放到 NPU 上——放置规则与 GPU 相同
（见[放置和扩展副本](../model-runtime/deploy.md#place-and-scale-replicas)），
前提是运行时旁装好了 torch_npu。

## 在容器中运行 NPU 运行时

没有专门的 NPU 镜像（[#4827](https://github.com/vllm-project/semantic-router/pull/4827) 中的结论）。
经过验证的路径是：在 arm64 `vllm-sr`（CPU）镜像内运行运行时，并挂载宿主机的
CANN。需要补齐 CPU 镜像不带的三样东西：

- `--privileged`：昇腾 950 驱动会拒绝非 privileged 容器的设备枚举
  （错误码 87），除非宿主机装了 ascend-docker-runtime；
- 宿主机的 UB 用户态库（`libummu.so.1` 与 `liburma*` 家族）**按文件**挂载——
  整个 `/usr/lib64` 挂进去会破坏镜像自身的 glibc；用
  `ls /usr/lib64 | grep -E 'urma|ummu'` 列出确切文件；以及
- 容器启动后安装 `torch_npu`。

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

`liburma*` 的确切 soname 随宿主机驱动构建而不同；把上面 `ls` 列出的文件都挂进去。
然后在容器内：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/lib64:${LD_LIBRARY_PATH:-}
pip install torch_npu
python3 -c "import torch, torch_npu; print(torch.npu.is_available(), torch.npu.device_count())"
# → True 8
```

`/usr/local/Ascend` 下的驱动库路径随 CANN 版本不同（9.1 多一层
`aarch64-linux/`），按容器内实际情况调整。torch_npu 版本必须与镜像自带的
PyTorch 匹配。

`vllm-sr serve` 编排会自建不带这些挂载的运行时容器，因此全栈编排的本地栈
模型仍跑在 CPU 上。在准备好的容器里运行运行时，再把 Router 的 deployment
指向它（[外部运行时](runtime/external.md)），或者整栈留在宿主机上。

## 排障

### 设备枚举报错误码 87

非 privileged 容器里 `drvGetDevNum` 返回 87（"Can't get ascend_hal device
count"）：950 驱动的 manager ioctl 拒绝了它。capability 标志、设备规则或
unconfined seccomp 都改变不了这一点。用 `--privileged` 起容器，或在宿主机
安装 ascend-docker-runtime 走非 privileged 路径。

### set_device 报 TsdOpen 或 `urma_attr is NULL`

容器缺少宿主机的 UB 用户态库，`liburma` 加载链断掉。从 `/usr/lib64` 按文件
挂载 `libummu.so.1` 与 `liburma*` 家族。不要整个目录挂载：镜像自身的 glibc
会报 GLIBC_PRIVATE 符号错误。

### torch_npu 导入失败

`Failed to load the backend extension: torch_npu` 说明 CANN 环境没有加载。
启动运行时之前 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`，容器内
还要确认 `ASCEND_OPP_PATH` 已设置。

### 运行时跑在了错误的卡上

`ASCEND_RT_VISIBLE_DEVICES` 决定可见的卡；不设置时 `npu:0` 即物理 0 卡。
能打开的卡不一定是空闲卡——驱动层的设备打开调用才是真正的门禁。先看
`npu-smi info` 的 HBM 占用，选一张余量足够的卡。
