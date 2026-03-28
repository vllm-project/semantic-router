---
title: 仿真模型参考
translation:
  source_commit: "0c5b1d02"
  source_file: "docs/fleet-sim/sim-algorithms.md"
  outdated: false
---

# 仿真模型参考

本文说明 `vllm-sr-sim` 背后的数学模型、各模型假设、假设失效之处，以及如何按工作负载调参。

面向进阶用户与维护者。操作路径请先读 [快速开始](./getting-started.md) 与 [容量规划场景](./use-cases.md)。

---

## 1. GPU 迭代延迟模型

### 公式

每个 GPU profile 暴露迭代延迟函数：

```
iter_t(n_active, mean_seq_len) = W + H_eff × n_active
```

| 符号 | 含义 | 单位 |
|---|---|---|
| `W` | 基础迭代成本：权重内存读、AllReduce、核启动 | 秒 |
| `H` | 在 `calibration_ctx` 上测得的每序列开销 | s/seq |
| `H_eff` | 按实际平均序列长度缩放后的有效每序列开销 | s/seq |
| `n_active` | 当前 batch 中活跃序列数 | — |
| `mean_seq_len` | 活跃序列的平均总 token（prompt + 输出） | tokens |
| `calibration_ctx` | 测量 `H` 时的上下文长度（默认 8192） | tokens |

**序列长度缩放：**

```
H_eff = H × (mean_seq_len / calibration_ctx)
```

注意力计算成本随序列长度近似线性：O(n_active × seq_len × d_head)。池服务 800 token 的 LMSYS 请求时，`H_eff` 比服务 8192 token 上限时小约 10×。

### 异构池为何重要

`iter_t` 与 `n_slots` 均随上下文长度反向缩放；合起来使短池相对同构池有显著吞吐优势（A100-80 GB 例：W=8 ms，H=0.65 ms，`calibration_ctx=8192`）：

| Pool | max_ctx | n_slots | mean_seq | H_eff | iter_t @full | GPU 吞吐 |
|---|---|---|---|---|---|---|
| Short pool | 2 048 | 512 | 800 | 6.5 × 10⁻⁵ | 40 ms | ~78 req/s |
| Homo pool  | 8 192 | 128 | 1 600 | 1.3 × 10⁻⁴ | 25 ms | ~49 req/s |
| Long pool  | 8 192 | 128 | 5 000 | 4.0 × 10⁻⁴ | 59 ms | ~2 req/s  |
| Homo (agent) | 65 536 | 16 | 15 000 | 1.2 × 10⁻³ | 27 ms | ~1 req/s |

若省略 `mean_seq_len`（兼容旧调用），`H_eff = H`，等价于假设每条活跃序列都是 `calibration_ctx` 长 —— 仅当池内平均请求长度与标定上下文一致时才正确。明显更短的请求应传入 `mean_seq_len`。

### 预置 H100 profile 常数

```
W = 0.004 s   （4 ms 基础迭代，权重内存读主导）
H = 0.00032 s/seq @ calibration_ctx = 8192（Llama-3-70B，TP8，BF16）
```

由微基准拟合；任意 GPU+模型可用 `ProfileBuilder` 从第一性原理计算。

### 假设与局限

| 假设 | 失效情形 |
|---|---|
| 注意力成本 ∝ seq_len（线性） | 无 FlashAttention 的二次核或极长序列上的次线性核 |
| 解析模型中活跃 batch 的 `mean_seq_len` = 单请求长度 | 同构池混合 batch：iter_t 由混合决定，非单条请求长度 |
| 跨 batch 常数 W | 极小 batch（n_active &lt; 4）时核启动开销占比变大 |
| H 对 n_active 线性 | n_active 超过算力或带宽饱和点后 H 非线性上升 |

---

## 2. KV-cache 槽模型

### 公式

池中每 GPU 须同时满足两类并发上限：

```
kv_limit     = total_kv_blks ÷ ⌈max_ctx / blk_size⌉
compute_cap  = max_slots × calibration_ctx ÷ max_ctx

n_slots      = min(kv_limit, compute_cap)
```

| 符号 | 含义 | A100_80GB 示例 |
|---|---|---|
| `total_kv_blks` | GPU 上 PagedAttention 块总数 | 65 536 |
| `blk_size` | 每 KV 块 token 数 | 16 |
| `max_ctx` | 池的最大上下文 | 按池配置 |
| `max_slots` | 在 `calibration_ctx` 测得的注意力带宽饱和上限 | 128 |
| `calibration_ctx` | 测量 `max_slots` 时的上下文 | 8 192 |

更短 `max_ctx` 下，相同带宽预算可支持更多并发序列，故 `compute_cap` 与 `max_ctx` 成反比。在 `max_ctx = calibration_ctx` 时两限制按构造相等。

**A100-80 GB 示例：**

| max_ctx | kv_limit | compute_cap | n_slots |
|---|---|---|---|
| 2 048 | 512 | 512 | **512** |
| 4 096 | 256 | 256 | **256** |
| 8 192 | 128 | 128 | **128** |
| 16 384 | 64 | 64 | **64** |
| 65 536 | 16 | 16 | **16** |

这是双池效率的主要来源：`max_ctx = 4096` 时可跑 256 条并发 —— 比 8192 时多 2× —— 而每迭代注意力成本按同因子下降。

### KV 块准入门限

新请求准入前 DES 检查块预算：

```
blocks_needed = ⌈(l_in + l_out) / blk_size⌉
if used_blocks + blocks_needed > total_kv_blks:
    → 抢占当前最长活跃请求（回到队列头重排）
```

刻画 PagedAttention 块级分配，避免未检查 KV 容量即准入导致 OOM。

### 假设

| 假设 | 失效情形 |
|---|---|
| 最坏块分配：准入时 `l_in + l_out` 块 | 推测解码或早停使实际输出 &lt; `l_out` |
| 固定 `total_kv_blks` | 解耦 serving 时 prefill/decode GPU 块预算不同 |
| 块粒度 16 token | vLLM 0.4+ 默认 block_size=16；用 `ManualProfile(blk_size=...)` 调整 |

---

## 3. 请求服务时间模型

### Prefill 与 decode 成本

| 阶段 | 瓶颈 | 成本模型 |
|---|---|---|
| **Decode** | 内存带宽（KV 读、权重流式） | `W + H_eff × n_active` |
| **Prefill** | 大块以算力为主；小块可能 memory-bound | `max(compute_bound, memory_bound)` |

大块 prefill 的 FlashAttention FLOPs 与 ridge point 比较见英文原文；大模型 prefill 在 H100 上常为 **算力受限**。模拟器对 `ComputedProfile` 用 `prefill_iter_latency()` 取 max；`ManualProfile` 无算力规格时退化为 memory-bound。

### 服务时间（物理）

```
S_raw = prefill_iters × prefill_iter_t(n_active)
      + l_out        × decode_iter_t(n_active)
```

`TTFT = slot_wait + prefill_iters × prefill_iter_t(n_active)`。  
`TPOT = (physical_end_time − first_token_time) / (l_out − 1)`（l_out &gt; 1）。

---

## 4. 抢占模型

KV 块预算不足时抢占最长活跃请求：失效完成事件、释放槽与块、受害者回到**队列头**重试。刻画 vLLM swap-out 策略；受害者按 `l_in + l_out` 近似 KV 压力。

**局限：** 重排队从头开始（无部分 KV 保存）；真实 vLLM 可将 KV 换到 CPU DRAM；多轮抢占迭代处理。

---

## 5. 解析机队规模（Erlang-C / Kimura）

优化器在运行较慢 DES 前用闭式近似求最小 GPU 数。

**流程：** CDF → `_calibrate` → `(μ_gpu, cv², n_slots, mean_prefill)` → `_min_gpus_analytical` → 满足 P99_TTFT ≤ SLO 的最小 `n_gpus`。

**槽级 Kimura M/G/c：** 每 GPU 有 `n_slots` 个并行「服务台」，有效池为 `c_slots = n_gpus × n_slots`，`ρ = λ/(n_gpus × μ_gpu)`，P99 等待用 Kimura 尾公式，P99_TTFT = P99_wait + mean_prefill。**稳定性：** ρ_max = 0.85。

**可靠性规模 `node_avail`：** GPU/NVLink 故障率与 MTTR 导出稳态可用性 A，`n_provisioned = ceil(n_for_slo / A)`。与 `ρ_max` 的 15% 余量**独立且可乘**。API 见英文原文与 [容量规划场景](./use-cases.md)。

**假设局限：** Poisson 到达、i.i.d. 服务时间、P99 尾近似 —— 突发流量、聚类长请求、重尾分布下可能偏差；此时用 DES。

---

## 6. 路由算法

- **LengthRouter：** `total ≤ threshold` → 短池，否则长池。生产默认，零额外开销。
- **SpilloverRouter：** 长度主路由 + 短池压力高时溢流到长池（`spill_threshold` 可调）。
- **CompressAndRouteRouter：** 边界区间压缩后走短池；**仅建议离线规模/γ 扫描**，实时仿真相对 LengthRouter 可显著恶化 P99。
- **LeastLoadedRouter：** 同质多池间按 `(active+queued)/total_slots` 最低路由。

---

## 7. 由工作负载导出的阈值选择（Pareto 扫描）

`threshold_pareto()`（CLI：`vllm-sr-sim pareto`）遍历 CDF 断点上的候选 `B_short`，记录成本与 P99，标出 Pareto 最优集。推荐阈值通常为满足 SLO 的最便宜 Pareto 点（未必全局最低成本点）。

---

## 8. 解耦机队规模

`DisaggFleetOptimizer`（`vllm-sr-sim disagg`）模型：prefill 池与 decode 池独立扩缩。

```
r_sys = min(r_pre, r_dec)
TTFT_eff = prefill_base_ms × β_TTFT
```

默认 `α_pre=0.90`、`α_dec=0.92`、`β_TTFT=1.80`（KV 传输开销），可覆盖。λ ≥ 100 req/s 时相对聚合可大幅降本，TTFT 升高；&lt; 50 req/s 时常不值得运维复杂度。

---

## 9. DES 事件流

Poisson 工作负载生成到达序列 → 按时间顺序路由、准入、堆合并推进各 `Instance` 事件 → 完成时释放槽并从队列启动下一请求。默认丢弃前 20% 时间为预热。建议 `n_sim_req ≥ 15000` 以得到稳定 P99。

---

## 10. Profile 调参

- 用两测点拟合 W、H（见英文公式）。
- `blk_size` 通常 16–32，与部署一致。
- `max_slots` 对应计划使用的 `--max-num-seqs` 附近饱和点。
- `DisaggFleetOptimizer` 常数可按实测覆盖。
- `ProfileBuilder`：`alpha_bw`、`layer_overhead_s`、`gpu_memory_utilization` 等见英文原文表与 `builder.py`。

---

## 11. 与 vLLM、Vidur 的对比

简要对照 vLLM v1 行为与 Vidur 经验表方法与本案屋顶线模型的异同；已知缺口包括：`ComputedProfile` 中 TP all_reduce 未显式建模（TP≥4 时 W 可能低估约 10–20%）、CPU 调度开销、GQA batching +10% 等 ——  workaround 见英文 §11.1–11.3。

---

## 12. 已知局限

| 局限 | 影响 | 缓解 |
|---|---|---|
| `ManualProfile.prefill_iter_latency` 退化为 memory-bound | 大模型 prompt 重时 TTFT 高估 | 用 `ComputedProfile` |
| `ComputedProfile` 未建 TP all_reduce | W 低估 | 手动加大 `W` |
| 无 CPU 调度开销 | iter_t 低估 1–5 ms | 校准进 `W` |
| 无 GQA +10% | H 略低估 | H × 1.10 |
| 静态 Poisson 流量 | 昼夜/突发不符 | 峰值 λ、`SageServe` 类策略 |
| 长度路由子流近似独立 Poisson | 解析队列近似偏差 | 用 DES 验证 |
| 抢占从头开始 | 抢占延迟高估 | 规划可接受 |

---

## 13. 功耗与能量分析

详细功耗估计、需求响应与每瓦 token 内容见 [功耗模型参考](./power-model.md)。本文聚焦延迟、排队、路由与机队规模；操作示例见 [容量规划场景](./use-cases.md) 中的 `grid-flex` 与 `tok-per-watt`。
