---
title: 功耗模型参考
translation:
  source_commit: "0c5b1d02"
  source_file: "docs/fleet-sim/power-model.md"
  outdated: false
---

# 功耗模型参考

本文档说明模拟器的功耗估算方法，面向进阶分析，**不是**主要入门路径。

模拟器中共存**两种功耗建模思路**：

1. **第一性原理模型**（`ComputedProfile.power_at_concurrency`）——与延迟相同的屋顶线分解推导功耗；除 TDP 外无拟合标量。在已知 `HardwareSpec + ModelSpec` 且希望对任意 (GPU, 模型) 组合得到可移植估计时使用。

2. **经验 logistic 模型**（`ManualProfile` 字段）——拟合真实硅片数据；对所覆盖的 (GPU, 模型) 组合更准确，但需要实测。在已有数据、用于生产机队决策时使用。

两种模型共享 ML.ENERGY Benchmark v3.0 的两处经验锚点（H100-SXM5，batch=1 → 300 W，batch=128 → 600 W），用于定义 idle/活跃 TDP 比例。

---

## 第一部分 — 第一性原理功耗模型（`ComputedProfile`）

### 推导链

各量均可追溯到硬件规格与模型结构参数；除下文两个 TDP 比例外无拟合参数：

```
W = model_bytes_per_gpu / (mem_bw × 0.80)     # 每次 decode 迭代权重流式时间
H = kv_bytes_per_token × ctx / mem_bw / tp    # 每条在飞序列的边际延迟

iter_latency(n)  = W + H_eff × n              # H_eff = H × (mean_ctx / calib_ctx)
decode_tps(n)    = n / iter_latency(n)         # 每 GPU 每秒输出 token 数

kv_frac(n)       = n × H_eff / iter_latency(n)
                   ← KV-cache 占 HBM 流量比例（仅由 W、H 推导）

compute_frac(n)  = 2 × active_params/tp × n / (iter_latency(n) × fp16_tc_flops)
                   ← 峰值张量核心吞吐利用率

activity(n)      = min(1.0,  kv_frac(n) + compute_frac(n))
power(n)         = hw.power × (P_IDLE_FRAC + (P_ACTIVE_FRAC − P_IDLE_FRAC) × activity(n))

tok/W            = decode_tps(n) / power(n)
```

### 物理含义

`kv_frac` 描述核心过渡：batch=1 时 GPU 仅流式加载模型权重（HBM 流量为权重字节）；batch 增大时 KV-cache 读逐渐主导 HBM 流量与功耗。`compute_frac` 描述每步矩阵乘增加带来的 SM 功耗。二者之和截断在 1.0。

| n_active | kv_frac | compute_frac | activity | P (W) | tok/W |
|---|---|---|---|---|---|
| 1   | 0.009 | 0.003 | 0.012 | 305 | 0.48 |
| 8   | 0.069 | 0.019 | 0.089 | 328 | 3.38 |
| 16  | 0.130 | 0.036 | 0.166 | 351 | 5.90 |
| 32  | 0.230 | 0.064 | 0.294 | 389 | 9.41 |
| 44  | 0.291 | 0.082 | 0.372 | 413 | 11.24 |

*H100-SXM5 + Llama-3.1-70B，TP=8，fp16，mean_ctx=4096。*

### 经验 TDP 比例

| 常量 | 取值 | 来源 |
|---|---|---|
| `_POWER_IDLE_FRAC` | 0.43 | ML.ENERGY v3.0：H100-SXM5 batch=1 → 300 W / 700 W TDP |
| `_POWER_ACTIVE_FRAC` | 0.86 | ML.ENERGY v3.0：H100-SXM5 batch=128 → 600 W / 700 W TDP |

上述比例可迁移到同类 HBM 带宽受限体制（Ampere、Hopper、Blackwell 稠密 decode）。若 GPU 不在该体制（如 PCIe、GDDR6），比例可能不同；请使用下文经验 logistic 校准。

### 校验

| n | 预测 (W) | ML.ENERGY (W) | 误差 |
|---|---|---|---|
| 1   | 305 | 300 | +1.5 % |
| 32  | 389 | ~480 | ~19 %  ← 见说明 |
| 128 | 465 | 600 | ~22 %（超出 n_slots） |

高 n 下约 20% 低估来自模型未包含的三点：(a) NVLink all-reduce 随 batch×hidden×TP 增长；(b) LayerNorm 与 softmax 核开销；(c) 高算力负载下 GPU 动态调压调频。对规划通常足够；若需 ±5% 的 demand-response 合同，请用 logistic 模型。

### 模型依赖性

因 W、H 均依赖模型结构，**tok/W 是 (GPU, 模型) 对的性质，而非仅 GPU**。同一 H100 上：

| Model | W (ms) | H (ms/seq) | tok/W @ n=32 |
|---|---|---|---|
| Llama-3.1-8B, TP=1 | ~5.0 | ~0.016 | ~35 |
| Llama-3.1-70B, TP=8 | ~6.7 | ~0.063 | ~9.4 |
| Llama-3.1-405B, TP=8 | ~38 | ~0.063 | ~1.8 |

勿跨不同模型 profile 比较 tok/W。`ComputedProfile.decode_efficiency()` 暴露全部中间量，推导可审计。

### 多池机队 tok/W

单一同构池内 N_gpus 相消：

```
tok/W (单池) = decode_tps(n) / power(n)   [与 N 无关]
```

多池异构机队（异构路由、semantic-router 小+大模型）**跨池不能相消**。`fleet_tpw_analysis()` 计算正确聚合：

```
fleet_tok/W = Σ_i(λ_i × mean_L_out_i) / Σ_i(N_i × power_i(n_i))
```

各池有各自的 (GPU, 模型) W、H 与功耗模型。算例见 [容量规划场景](./use-cases.md)。

---

## 第二部分 — 经验 logistic 模型（`ManualProfile`）

---

## Logistic 功耗曲线

G2G 论文（Hassan et al.，arXiv:2602.05116v1，式 2）建议将每 GPU 功耗建模为 batch 大小对数的 logistic：

```
P(b) = P_range / (1 + exp(-k * (log2(b) - x0))) + P_idle
```

其中：

- `b`        = 并发在飞请求数（≈ vLLM `max_num_seqs`）
- `P_idle`   = b → 0 时 GPU 功耗（渐近下限；实践中常取 b=1）
- `P_nominal`= b → ∞ 时 GPU 功耗（渐近上限；≈ TDP × 利用率因子）
- `P_range`  = `P_nominal − P_idle`
- `k`        = 从 idle 到 nominal 过渡的陡度
- `x0`       = 功率为 `P_idle + P_range / 2` 时的 log₂(batch)

### x0 与 k 的物理含义

- **x0** 由 GPU 带宽（HBM 或 GDDR6）约 50% 饱和时的 batch 决定。带宽越大 → 饱和 batch 越高 → x0 越大。
- **k** 刻画从算力受限（低 batch、低利用率、低功耗）到 memory-bandwidth 受限（高 batch、高利用率、高功耗）的过渡陡峭程度。HBM GPU（H100、A100）过渡更陡（k ≈ 1.0）；GDDR6 GPU（A10G）较缓（k ≈ 0.7），因带宽以较小突发交付。

### 带宽饱和缩放规则（用于投影）

同一模型与上下文长度下，饱和 batch n_sat 缩放为：

```
n_sat ∝ GPU_memory_bandwidth / KV_cache_demand_per_seq_per_iter
```

若每条序列的 KV 相对需求不变（同一模型），则：

```
x0(GPU_target) = x0(GPU_ref) + log2(BW(GPU_target) / BW(GPU_ref))
```

下文 A100、A10G 投影主要据此推导。

---

## H100-SXM5 — 实测数据（来源质量：高）

| 参数        | 取值  | 来源                                         |
|------------------|--------|------------------------------------------------|
| `power_idle_w`   | 300 W  | ML.ENERGY Benchmark v3.0，H100-SXM5 + vLLM    |
| `power_nominal_w`| 600 W  | ML.ENERGY Benchmark v3.0，H100-SXM5 + vLLM    |
| `power_logistic_k` | 1.0  | 拟合 G2G 论文图 2 数据点          |
| `power_logistic_x0`| 4.2  | 拟合 G2G 论文图 2 数据点          |
| 饱和 batch | ≈ 18   | 2^4.2 ≈ 18 并发请求                 |
| HBM3 带宽   | 3.35 TB/s | NVIDIA H100 规格书                       |
| TDP              | 700 W  | NVIDIA H100 规格书 (DS-10313-001_v1.6)      |

**拟合方法：** G2G 论文图 2 给出 H100-SXM5 上 Llama-3.1 类模型在 vLLM 中 batch ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256} 的功耗。由图读数近似：

| batch | ≈ W (图) | 模型 W (k=1.0, x0=4.2) |
|-------|-----------|--------------------------|
|     1 |     ~304  |                     304  |
|     4 |     ~330  |                     330  |
|    16 |     ~435  |                     435  |
|    32 |     ~507  |                     507  |
|    64 |     ~557  |                     557  |
|   128 |     ~583  |                     583  |

40–100% 负载范围内拟合误差 &lt; 3%。

ML.ENERGY v3.0（Chung et al.，arXiv:2601.22076）确认：H100-SXM5 上 batch=128 时每 GPU 功耗 ≈ 600 W（≈ 86% TDP）；batch=1 时 ≈ 300 W（≈ 43% TDP）。

---

## A100-SXM4 — 单锚点 + FLOPS 缩放推导（来源质量：中）

### 实测锚点

来自 Sada et al.（arXiv:2507.00418，Table 1，2025-10）：

> 8×A100-SXM4-80GB 通过 vLLM 服务 Nemotron-70B，32 并发请求：节点总功耗 **2,983 W** → **373 W / GPU**（nvidia-smi 实测）

张量并行 TP=8；每 GPU 处理相同 32 条并发序列。

### 为何带宽缩放在此失效

带宽缩放给出 `x0 = 4.2 + log2(2.0/3.35) = 3.5`，预测 P(32)=334 W，比实测 373 W 低约 10%，误差系统性。

70B 模型在 A100-SXM4、TP=8 下，每迭代权重内存读取 = 17.5 GB/GPU。在 A100 的 2.0 TB/s HBM2e 带宽下约需 8.75 ms，与性能 profile 中 W=8 ms 一致。**在 b=1 时仅加载权重即已饱和 memory bandwidth**。batch 增大时带宽无法继续增加，上升的是算力利用率（更大 matmul）。因此 x0 反映的是**算力饱和**，而非 KV-cache 带宽饱和。

### FLOPS 缩放方法

大模型、高 TP 部署中权重占满带宽时，x0 随 GPU TFLOPS 缩放（多少并发序列能饱和张量核心）：

```
x0(GPU) = x0(H100) + log2(TFLOPS_GPU / TFLOPS_H100)
```

FP16 峰值 TFLOPS（NVIDIA 规格书）：

- H100-SXM5: 989 TFLOPS (DS-10313-001_v1.6)
- A100-SXM4: 312 TFLOPS (DS-10031-001_v1.6)

```
x0(A100) = 4.2 + log2(312 / 989) = 4.2 − 1.66 = 2.54  →  2.5
```

饱和 batch：2^2.5 ≈ 5.7 并发请求。

### 参数推导

**k = 1.0** — A100 HBM2e 与 H100 HBM3 同属内存族，突发特性相似。

**x0 = 2.5** — 见上 FLOPS 缩放。

**P_nominal = 385 W（96.3% TDP）** — 由实测锚点反推（略）。

**P_idle = 175 W（43.75% TDP）** — 无直接 b=1 实测；由 H100 TDP 比例投影：400 W × 0.4375 = 175 W。

### 校准核对表

| batch | 模型 W (k=1.0, x0=2.5, P_nom=385) | 说明                              |
|-------|--------------------------------------|------------------------------------|
|     1 |                                 191  | 接近 P_idle=175 ✓                  |
|     4 |                                 254  | 上升；算力唤醒          |
|     6 |                                 272  | 中点（2^2.5 ≈ 5.7）            |
|    32 |                                 369  | **实测 ≈ 373；误差 1.1% ✓**  |
|   128 |                                 383  | 接近 P_nominal=385 ✓               |

---

## A10G — 仅投影（来源质量：低）

**未找到可靠的 A10G 在 LLM 推理中 batch 与功耗的公开实测数据。** ML.ENERGY v3.0 不含 A10G。截至 2026 年 3 月，无 arXiv 论文给出 A10G + vLLM 的 batch–功耗曲线。

**所有 A10G 参数仅由硬件规格投影。在用于 DR 合同前强烈建议进行经验校准。**

### 参数推导（摘要）

**P_nominal（120 W）与 P_idle（75 W）**

- A10G TDP = 150 W（NVIDIA A10G 规格书 DS-10012-001_v1.3）
- P_nominal = 120 W = 80% TDP（PCIe 卡满载通常 75–85% TDP）
- P_idle = 75 W = 50% TDP（GDDR6 刷新与 idle 特性与 HBM 不同）

**k = 0.7（投影）** — GDDR6 突发更小、更均匀，过渡不如 HBM 陡。

**x0 = 3.7（投影）** — 典型 7B 场景下 KV 与带宽比推导见英文原文；13B 与 7B 时 x0 可不同，默认 3.7（7B）偏保守。

---

## 如何校准自有 logistic 参数

在离线 batch 模式下运行 vLLM，用 `nvidia-smi --query-gpu=power.draw` 采样功耗（见英文原文中的完整 Python 拟合示例）。

---

## 来源质量汇总

| GPU     | P_idle  | P_nominal | k    | x0   | 可靠性 |
|---------|---------|-----------|------|------|-------------|
| H100    | 300 W — 实测（ML.ENERGY v3.0） | 600 W — 实测（ML.ENERGY v3.0） | 1.0 — 拟合（G2G 图2） | 4.2 — 拟合（G2G 图2） | **高** |
| A100    | 175 W — 投影（H100 TDP 比） | 385 W — 反推（Sada et al. 锚点，1.1% 误差） | 1.0 — 投影（HBM 族） | 2.5 — FLOPS 缩放 | **中** |
| A10G    | 75 W — 投影 | 120 W — 投影 | 0.7 — 投影（GDDR6） | 3.0 — 保守中值；无实测 | **低** |

生产 DR 承诺：**务必**对实际 (GPU, 模型) 做剖析并更新 profile 的 logistic 参数后再签约。

---

## 参考文献

- Hassan et al. (2025). "GPU-to-Grid: Voltage Regulation via GPU Utilization Control." arXiv:2602.05116v1.
- Chung et al. (2025). "The ML.ENERGY Benchmark." arXiv:2505.06371v2.
- Chung et al. (2026). "Where Do the Joules Go?" arXiv:2601.22076v2.
- Sada et al. (2025). "Serving LLMs in HPC Clusters: QAic vs A100." arXiv:2507.00418v3.
- NVIDIA A100 / H100 / A10G 规格书（TDP 与带宽）。
