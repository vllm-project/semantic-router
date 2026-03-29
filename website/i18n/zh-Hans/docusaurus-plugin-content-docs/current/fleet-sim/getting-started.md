---
title: 快速开始
translation:
  source_commit: "0c5b1d02"
  source_file: "docs/fleet-sim/getting-started.md"
  outdated: false
---

# 快速开始

本文介绍运行 `vllm-sr-sim` 的推荐方式。

## 推荐本地流程：与 `vllm-sr serve` 搭配的边车

若希望模拟器与仪表盘在本地连通，请使用仓库原生流程：

```bash
make vllm-sr-dev
cd src/vllm-sr
vllm-sr serve --image-pull-policy never
```

该流程会：

- 构建路由器镜像
- 构建 `vllm-sr-sim` 镜像
- 以可编辑模式安装两个 CLI
- 在共享运行时网络上自动启动模拟器边车

## 独立 CLI

仅需本地规划命令时使用独立 CLI：

```bash
cd src/fleet-sim
pip install -e .[dev]

vllm-sr-sim --version
vllm-sr-sim optimize --cdf data/azure_cdf.json --lam 200 --slo 500 --b-short 6144
```

## 独立服务

当仪表盘或其他调用方需通过 HTTP 访问模拟器时，使用服务模式：

```bash
cd src/fleet-sim
pip install -e .[dev]

vllm-sr-sim serve --host 0.0.0.0 --port 8000
```

模拟器在外部运行时，通过 `TARGET_FLEET_SIM_URL` 让仪表盘或 `vllm-sr serve` 指向该地址。

## 使用外部服务替代本地边车

若已设置 `TARGET_FLEET_SIM_URL`，`vllm-sr serve` 会使用该外部服务，而不会启动本地边车。

若要在不提供外部服务的情况下关闭默认边车行为，可设置：

```bash
export VLLM_SR_SIM_ENABLED=false
```
