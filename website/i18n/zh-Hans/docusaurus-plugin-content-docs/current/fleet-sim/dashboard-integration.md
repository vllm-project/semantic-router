---
title: 仪表盘集成
translation:
  source_commit: "0c5b1d02"
  source_file: "docs/fleet-sim/dashboard-integration.md"
  outdated: false
---

# 仪表盘集成

Fleet Sim 通过后端代理层集成到仪表盘中；仪表盘不再连接已弃用的独立模拟器前端。

## 默认本地行为

运行：

```bash
cd src/vllm-sr
vllm-sr serve --image-pull-policy never
```

时，CLI 会在同一运行时网络上将 `vllm-sr-sim` 作为同级容器启动，并在路由器栈内将 `TARGET_FLEET_SIM_URL` 设为边车服务 URL。

## 代理路径

仪表盘后端在以下路径代理模拟器请求：

```text
/api/fleet-sim/*
```

若未配置模拟器，该代理会返回结构化的「服务不可用」响应，而不是静默失败。

## 外部服务模式

当模拟器位于其他容器、主机或环境中时，设置 `TARGET_FLEET_SIM_URL`：

```bash
export TARGET_FLEET_SIM_URL=http://your-simulator:8000
```

设置该变量后，`vllm-sr serve` 会使用外部模拟器并跳过默认的本地边车启动。

## 仪表盘入口

顶部栏 **Fleet Sim** 菜单提供：

- `Overview`：模拟器高层状态与最近产物
- `Workloads`：内置负载库与轨迹输入
- `Fleets`：已保存的机队定义与规划输出
- `Runs`：优化、仿真与假设任务
