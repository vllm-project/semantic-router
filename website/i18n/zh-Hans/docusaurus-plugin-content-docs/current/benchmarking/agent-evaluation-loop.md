---
title: Agent 评测循环
description: 不依赖控制面板，对 vLLM Semantic Router 进行校验、路由、探测、基准测试和优化。
translation:
  source_commit: "12c2aa4feb5c5d40d90104d8b09cade1facf0bf5"
  source_file: "docs/benchmarking/agent-evaluation-loop.md"
  outdated: false
---

# Agent 评测循环 {#agent-evaluation-loop}

Agent 直接使用两份运行时契约：Router 管理 API 用于配置和路由检查，Envoy 监听器用于真实模型请求。控制面板是可选查看器，从不属于执行路径。

```text
canonical YAML
    │
    ├── validate → plan → compare-and-swap apply       Router :8080
    │
    ├── route preview                                  Router :8080
    │       └── signals + decision; no backend call
    │
    ├── route probe                                    Envoy listener
    │       └── real response + route receipt
    │
    └── benchmark
            ├── routing workload → recipe behavior and system outcome
            └── Intelligence 1.0 → physical or virtual model quality
```

## 1. 发现并更改配置 {#1-discover-and-change-configuration}

只发现当前编辑所需的契约：

```bash
curl -sS 'http://localhost:8080/api/v1?audience=agent&visibility=primary'
curl -sS 'http://localhost:8080/openapi.json?capability=config&audience=agent'
vllm-sr config schema --endpoint http://localhost:8080 \
  --surface algorithm:multi_factor
```

YAML 是运维编写配置的唯一格式。CLI 和 HTTP API 是同一套 Router 校验器上的等价传输：

```bash
vllm-sr config validate --config candidate.yaml \
  --endpoint http://localhost:8080
vllm-sr config plan --config candidate.yaml --mode replace \
  --endpoint http://localhost:8080
vllm-sr config apply --config candidate.yaml --mode replace \
  --endpoint http://localhost:8080
```

`plan` 执行与变更相同的解析、规范化、语义校验和热重载可行性检查，但不写入。`apply` 会再次规划，并将返回的 ETag 作为比较并交换的前置条件。会更改 listeners 或 provider 后端拓扑的 plan 返回 `RESTART_REQUIRED`，因为这些字段会渲染进 Envoy；应通过部署工作流激活该候选，而不是 Router 变更 API。对于本地 Docker，替换运行中的栈前先询问，然后使用 `vllm-sr serve --config candidate.yaml --replace-active-config`。普通 `serve` 会保留控制面板编辑过的生效状态。

## 2. 分两阶段验证路由 {#2-verify-routing-in-two-stages}

Preview 检查路由策略，不消耗模型 token：

```bash
vllm-sr route preview \
  --endpoint http://localhost:8080 \
  --model vllm-sr/auto \
  --prompt 'Implement a lock-free queue' \
  --json
```

随后 Probe 通过 Envoy 发送真实请求，并断言结果路由：

```bash
vllm-sr route probe \
  --base-url http://localhost:8899/v1 \
  --model vllm-sr/auto \
  --prompt 'Implement a lock-free queue' \
  --expect-recipe balanced \
  --expect-decision coding \
  --expect-algorithm multi_factor \
  --expect-selected-model qwen \
  --expect-response-model Qwen/Qwen3.8-Flash-Next
```

Probe 会发出机器可读回执，包含 HTTP 状态、延迟、路由头、响应和断言。`--expect-selected-model` 检查 Router 回执；`--expect-response-model` 检查上游 OpenAI 响应体。仅当该后端暴露稳定的顶层 `model` 值时使用后者。断言失败以退出码 `2` 退出。
基础 URL 可以是 Envoy 监听器源，也可以是以 `/v1` 结尾的标准 OpenAI 根。
Preview 成功只证明决策行为。selected-model 头证明 Router 的选择，但不证明哪个后端作答；可用时，response-model 证据补上这一缺口。单次 Probe 仍不能替代基准测试。

## 3. 运行可比较的基准测试 {#3-run-comparable-benchmarks}

对不可变路由工作负载使用 `vllm-sr benchmark`。对独立或虚拟模型质量使用固定的 Intelligence 1.0 harness：

```bash
vllm-sr benchmark intelligence list
vllm-sr benchmark intelligence plan \
  --model vllm-sr/quality \
  --base-url http://localhost:8899 \
  --source-root .vllm-sr/benchmark-sources \
  --output .vllm-sr/benchmark-results/quality-1
```

六个 1.0 叶子是 MMLU-Pro、GPQA Diamond、HLE 1.0 纯文本、LiveCodeBench v6、SciCode 和 Terminal-Bench 2.1。HLE 始终是冻结的 2158 题纯文本子集。Harness 会验证干净的 runner 修订；在执行前证明 AIPerf 使用的 Hugging Face 修订；并使用 Inspect Evals 带校验和固定的 SciCode 题目与数值测试资产。它会在无私密信息的私有回执中记录这些身份和执行条件。`--sample-limit` 是冒烟证据，不能进入指数。

物理模型名和虚拟模型名使用同一个 `--model` 字段和同一套端点契约。虚拟分数是端到端测得的；绝不会由其成员模型的分数拼装而成。

## 4. 依据证据优化 {#4-optimize-from-evidence}

将基准测试结果与路由回执、回放决策、结果反馈、延迟、token 用量、失败和成本连接起来。每次只更改一项已评审的策略，对基线和候选重跑同一冻结工作负载，仅当其声明的质量、成本、可靠性和安全门槛都通过时才保留候选。

凭据应放在由 `--token-env` 或 `--api-key-env` 命名的环境变量中。不要把凭据值放进 YAML、URL、命令参数、日志或回执。
