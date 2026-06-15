---
title: 概览
translation:
  source_commit: "0c5b1d02"
  source_file: "docs/fleet-sim/overview.md"
  outdated: false
---

# Fleet Sim 概览

Fleet Sim 是 vLLM Semantic Router 维护的机队模拟器。`vllm-sr-sim` 包提供其 CLI 与服务入口，用于在部署前规划 GPU 机队、比较路由与拆分策略，并在仪表盘中暴露这些工作流，而无需恢复单独的模拟器前端。

## Fleet Sim 的用途

- 在延迟目标下为同构、异构或拆分的机队做规模估算
- 比较不同 GPU、路由策略与阈值下的年化成本
- 用仿真运行、轨迹回放与假设分析验证规划假设
- 通过维护中的后端代理在仪表盘中呈现上述工作流

## Fleet Sim 不做什么

- 不是路由器的在线请求路径
- 不是运行时自动扩缩或突发控制器
- 不是针对单个部署副本的逐算子分析器
- 不能替代路由器配置文档

## 部署形态

`vllm-sr-sim` 可以：

- 作为独立 Python CLI，用于本地规模估算与假设分析
- 以 HTTP 服务运行：`vllm-sr-sim serve`
- 作为 `vllm-sr serve` 默认在共享 `vllm-sr-network` 上启动的边车容器

## 建议阅读顺序

1. [快速开始](./getting-started.md)：本地边车、独立 CLI 与外部服务
2. [仪表盘集成](./dashboard-integration.md)：代理路径与 UI 入口
3. [容量规划场景](./use-cases.md)：示例驱动的决策流程
4. 需要底层机制时查阅 [仿真模型参考](./sim-algorithms.md) 与 [功耗模型参考](./power-model.md)
5. 需要可打印版本或源文件时使用 [指南 PDF](pathname:///files/fleet-sim/fleet-sim.pdf) 与 [指南资源](./guide.md)
