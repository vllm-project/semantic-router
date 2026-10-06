---
title: macOS 上的 Apple GPU
description: 通过 MPS 在宿主机上运行 Router 模型，同时保留 Docker 中的 Router 服务栈。
---

# macOS 上的 Apple GPU

Apple 宿主机桥接目前是实验性功能。模型运行时已经实现 MPS，但仍需要在
Apple silicon 上验证各模型的输出和性能。进程运行或请求成功不代表准确性
已经通过验证；缺少 MPS 黄金答案的模型会报告 `golden.status: unverified`。

## 启动与停止

需要 Apple silicon Mac、原生 arm64 CLI 和运行中的本地 Docker 引擎。
Linux、Intel Mac、Rosetta Python、Kubernetes、Podman 和远程 Docker
上下文不受支持。Docker 容器必须能通过 `host.docker.internal` 访问宿主机
回环接口上的服务。

```bash
vllm-sr serve --config config.yaml --platform apple
vllm-sr status model-runtime
vllm-sr logs model-runtime -f
vllm-sr stop
```

Router 和 Dashboard 保留在容器中，现有网关服务栈不变。CLI 管理宿主机上的
原生监督进程；Router 把活动 recipe 所需的模型进程组交给它启动，并通过
带认证的 TCP 代理调用现有模型 HTTP/JSON API。配置重载时，不变的模型组
继续共享，最后一个使用者释放它时停止模型进程。声明但未使用的模型不会
加载；显式外部运行时端点仍保持外部部署。

引擎模式直接在宿主机回环接口上提供模型 API，不启动 Router：

```bash
vllm-sr serve vllm-sr/Decision-1.0-Kai-0.6B --platform apple --port 8000
curl http://127.0.0.1:8000/health
vllm-sr stop
```

引擎模式在后台加载模型，调用前应检查 `/health`。支持 MODEL 名称、固定
`MODEL@REVISION` 和 `--models` 文件；使用 MPS 和回环 TCP，不支持 `--uds`。
引擎模式允许宿主机上的本地模型目录。Router 模式必须使用 Hub 模型标识，
镜像内的路径不能作为宿主机模型路径。独立的 Router 或引擎服务栈应使用
不同的 `VLLM_SR_STACK_NAME`。

## 安装与缓存

首次启动从选定 Router 镜像复制纯 Python 运行时和插件元数据，以不可变
镜像 ID 区分环境缓存。CLI 自动准备原生 Python，并安装与镜像中运行时
直接依赖版本匹配的 macOS wheel；不会复制 Linux torch 库或修改用户 Python。
无需另外安装模型运行时，但首次下载解释器、依赖及未缓存模型需要网络。

状态、原生环境、模型缓存和有大小限制的日志位于
`~/Library/Caches/vllm-sr/apple`。停止服务保留缓存和日志，镜像变化会选择
独立环境。环境缓存可复用，但镜像拉取策略和未缓存的模型仍可能需要网络。
宿主机下载使用宿主机代理和证书设置，可能与 Docker 设置不同。

开发时使用既有本地镜像工作流，构建包含桥接实现的 Router 镜像，并显式选择：

```bash
vllm-sr serve --config config.yaml --platform apple \
  --router-image YOUR_LOCAL_ROUTER_IMAGE --image-pull-policy never
```

镜像必须同时包含宿主机连接实现及匹配的模型运行时。仅升级 CLI 并传入
`--platform apple`，不会让旧版镜像自动获得该功能。

## 限制与排查

MPS 当前使用 FP32 参考内核，不支持 bf16 autocast、Triton 或 GPU graphs。
受管理的模型进程禁用 CPU 回退；不支持的算子或模型族会报错，而不是悄悄
转为 CPU 推理。仅存在于镜像中的预处理资源和任意加速器插件尚未完成验证。
该功能不会把配置中的回答生成后端 LLM 自动搬到 Mac 上。

GPU 与 macOS、应用程序及 Docker VM 共享系统内存。4B 模型仅 FP32 权重大约
需要 16 GB，9B 大约需要 36 GB，尚未计入激活和临时分配。宿主机桥接移除了
Docker VM 的模型内存限制，但不会移除 Mac 的总内存限制。

`status all` 会额外报告已记录的宿主机监督进程。监督进程运行不代表每个模型
都已就绪，模型状态仍由运行时及 Router 清单提供。即使 Docker 不可用，
`stop` 也会尝试清理宿主机进程，并只停止身份与记录匹配的进程。Router
崩溃后，未释放模型组的租约在 45 秒后过期；监督进程保留到执行 stop。
停止后仍可查看日志。

连接失败时检查 Docker 上下文、防火墙/VPN、`host.docker.internal` 和日志。
依赖安装失败时检查镜像依赖版本是否有 macOS wheel，以及宿主机网络。
模型加载失败时检查 MPS 算子支持、模型访问权限及可用内存。

## 验证

每个固定模型版本的 MPS 黄金答案都需要在真实 Apple silicon 上记录，并保留
依赖、硬件信息和测量得到的误差界限。
`src/model-runtime/tools/golden_answers.py --device mps --record FILE`
要求显式提供 `--tolerance`；就绪检查使用该模型的容差，而不是通用 GPU 容差。
缺少记录时保持 unverified。不要虚构记录、直接把加速器标记为已验证，
或修改 `exact` profile 来掩盖差异。普通 CI 运行生命周期/HTTP fixture，
GPU 验收需要 Apple silicon runner。
