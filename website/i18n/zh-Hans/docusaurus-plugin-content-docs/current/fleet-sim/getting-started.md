---
title: 快速开始
translation:
  source_commit: "33349fdab9ad294da19ebd11588f8adbe8771b4a"
  source_file: "docs/fleet-sim/getting-started.md"
  outdated: false
---

# 快速开始

Fleet Sim 是独立的规划工具。它不会由 `vllm-sr serve` 启动，也不会通过 Semantic Router 仪表盘对外暴露。

## 安装 CLI {#install-the-cli}

在源码检出目录中：

```bash
cd src/fleet-sim
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
vllm-sr-sim --version
```

Windows PowerShell 用户可用 `.venv\Scripts\Activate.ps1` 激活环境。

### 运行第一次研究 {#run-a-first-study}

下面的命令会搜索低成本的双池机队，并对前列候选做 DES 校验：

```bash
vllm-sr-sim optimize \
  --cdf data/azure_cdf.json \
  --lam 200 \
  --slo 500 \
  --b-short 6144 \
  --verify-top 3 \
  --n-sim-req 30000
```

先理解输入，再看输出：

- `--cdf` 描述累计 token 长度分布。
- `--lam` 是假定到达率，单位为每秒请求数。
- `--slo` 是 P99 TTFT 目标，单位为毫秒。
- `--b-short` 将不超过该阈值的请求发往短池。
- `--verify-top` 选出解析候选，供 DES 验证。

得到的 GPU 数量和成本是基于所选内置 profile 的估计值。在校准目标部署之前，请替换这些 profile，或把结果当作相对参考。

## 选择命令 {#choose-a-command}

| 命令 | 用途 |
| --- | --- |
| `optimize` | 搜索双池机队，并可选择对候选做 DES 校验 |
| `simulate` | 对固定的短池和长池数量运行 DES |
| `whatif` | 扫描到达率或内置 GPU profile |
| `pareto` | 根据工作负载 CDF 比较 token 阈值 |
| `compare-routers` | 在同一固定机队上比较 CLI 的长度、压缩后路由和随机策略 |
| `disagg` | 分别为 prefill 和 decode 池做规模估算 |
| `grid-flex` | 在降低建模并发和功耗时估计延迟 |
| `tok-per-watt` | 比较建模能效 |
| `simulate-fleet` | 模拟任意多池 JSON 拓扑 |
| `serve` | 启动 Fleet Sim HTTP 服务 |

运行 `vllm-sr-sim <command> --help` 查看当前选项。需要机器可读 JSON 时，对支持的命令加上 `--out FILE`。

## 启动独立服务 {#start-the-standalone-service}

安装 API 依赖并启动 FastAPI：

```bash
cd src/fleet-sim
python -m pip install -e '.[api]'
vllm-sr-sim serve --host 127.0.0.1 --port 8000
```

然后检查：

```bash
curl -sS http://127.0.0.1:8000/healthz
```

交互式 API 文档位于 `http://127.0.0.1:8000/api/docs`，OpenAPI 文档位于 `/api/openapi.json`。

仅当其他主机或容器必须连接时才使用 `--host 0.0.0.0`，并在服务前部署认证和网络控制。Fleet Sim 的 FastAPI 应用本身不增加认证层。

在把示例结果当作部署建议之前，请继续阅读[容量规划工作流](./use-cases)。
