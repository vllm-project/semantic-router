---
title: 容量规划场景
translation:
  source_commit: "0c5b1d02"
  source_file: "docs/fleet-sim/use-cases.md"
  outdated: false
---

# 容量规划场景

`vllm-sr-sim` 回答仅靠第一性原理无法解决的机队规划问题：拆分阈值设在哪、在真实排队动态下机队是否真能达标、某工作负载下哪种 GPU 最便宜、何时应预置下一档资源。

全文使用的 GPU 单价：

| GPU | $/hr | $/yr |
|---|---|---|
| A10G 24 GB | $1.01 | $8.85 K |
| A100 80 GB | $2.21 | $19.4 K |
| H100 80 GB | $4.02 | $35.2 K |

> **P99 TTFT** = P99(KV 槽排队等待) + 平均 prefill 时间。  
> 每个 KV-cache 槽在模型中相当于 M/G/c 排队的一个服务台。

---

## 何时拆分池 — 简短版

在打开模拟器前，先用下列筛选：

```
重尾服务时间（智能体 / 长上下文）？
  → 必须拆分。同构池无论加多少 GPU 都无法满足 SLO。

ctx 比例 R = long_max_ctx / B_short，长请求比例 f：

  R ≤ 2×  或  f > 30%   →  同构通常更便宜；拆分为隔离延迟
  R ≥ 4×  且  f &lt; 10%  →  高流量（λ > ~100 req/s）下拆分更便宜
  R ≥ 16× 且  f &lt; 5%   →  任意有意义流量下拆分更便宜
```

以下各「谜题」是单靠经验法则无法解决的。

---

## 谜题 1 — 究竟应在何处拆分？

**问题：**规则说「要拆分」——但 token 阈值应设在哪？

最优阈值完全取决于 CDF 形状。太低则长池承担过多流量；太高则短池的槽位优势消失。`pareto` 命令会扫过 CDF 的每个断点并给出成本–延迟前沿。

```bash
vllm-sr-sim pareto \
  --cdf data/lmsys_cdf.json --lam 100 --slo 500 --long-max-ctx 65536
```

**LMSYS 结果**（λ=100，A100，同构基线 = $271K / 14 GPU）：

```
 B_short   α-short   n_s  n_l   GPUs      $/yr   saving    P99-s    P99-l   SLO  Pareto
---------------------------------------------------------------------------------------
     512    63.8%     2   13     15  $   290K    -7.1%       9ms      25ms     ✓       ★
   1,024    83.1%     2   10     12  $   232K   +14.3%      10ms      36ms     ✓       ★
   2,048    94.8%     2    7      9  $   174K   +35.7%      12ms      63ms     ✓       ★
   4,096    98.4%     3    5      8  $   155K   +42.9%      13ms     108ms     ✓       ★  ← 最优
   8,192    99.7%     4    4      8  $   155K   +42.9%      14ms     212ms     ✓       ★
  12,288    99.9%     5    3      8  $   155K   +42.9%      14ms     319ms     ✓       ★
```

**洞见：**B_short=4096 最优 —— 约 98% LMSYS 流量低于该阈值，短池（max_ctx=4096 时 256 个 KV 槽）比同构池（max_ctx=65536 时 16 槽）槽效率高 16×。结果：14 GPU → 8 GPU，**成本 −43%**。若选 B_short=512，仅约 64% 流量走短池，多数仍进昂贵长池，成本比同构高 7%。

**Azure 结果**（λ=200，A100，同构基线 = $465K / 24 GPU）：

整段 Azure CDF 在 8192 token 内，ctx 比例仅约 2×。最佳 Pareto（B_short=3072）仅省约 4% —— 槽位增益不足以抵消 Erlang 碎片化。价值在于**延迟隔离**：短池 P99 从同构 26 ms 降至 19 ms，利于分层 SLA。

**智能体结果**（λ=200，A100，同构基线 = $9293K / 480 GPU）：

```
  B_short   saving    P99-s    P99-l   SLO
  16,384    +13.3%     69ms    339ms    ✓   ← 最优
  32,768    +12.9%     86ms    593ms    ✗   ← SLO 失败：长池 prefill 主导
```

B_short=16384（相对同构 64 槽 vs 16 槽）节省 64 GPU。超过 32768 后长池请求的 prefill 达 300–600 ms，直接突破 500 ms SLO —— **P99 失败来自 prefill 成本而非排队等待**，只有完整仿真才能看见。

---

## 谜题 2 — 为何智能体机队达不到 SLO？

**问题：**24 张 H100、λ=20 req/s，利用率仅约 30%。解析说机队健康，DES 却说不行。

```bash
# 同构基线 —— 解析上可行
vllm-sr-sim optimize \
  --cdf data/agent_heavy_cdf.json --lam 20 --slo 1000 \
  --gpu-short h100 --gpu-long h100 \
  --b-short 65536 --long-max-ctx 65536

# 双池 —— 修复
vllm-sr-sim optimize \
  --cdf data/agent_heavy_cdf.json --lam 20 --slo 1000 \
  --gpu-short h100 --gpu-long h100 \
  --b-short 4096 --long-max-ctx 65536 --verify-top 3
```

| 配置 | GPU | 成本/年 | P99 TTFT | SLO 1000ms |
|---|---|---|---|---|
| 同构 65K ctx | 24 | $845 K | **1 052 ms** | **✗ 失败** |
| 双池 4K/65K | 25 | $880 K | 17ms / 147ms | ✓ |

**解析为何失效：** M/G/c 假设服务时间 i.i.d. 且方差有限。智能体请求服务时间跨度 10–300 秒，变异系数 cv²≫1。单条长请求可占用 KV 槽数分钟，使其他请求排队，即使 GPU 利用率看起来不高。DES 重放真实到达序列并暴露尖峰；Erlang-C 不能。

**双池**将约 46% 长请求（&gt;4K token）路由到专用池，使其慢服务不会阻塞短请求。成本溢价约 +4% —— 相对 SLO 失败几乎是免费保险。

---

## 谜题 3 — 哪种 GPU 实际最便宜？

**问题：**A10G 单卡便宜但慢；H100 贵但快。给定工作负载谁最便宜？

```bash
for gpu in a10g a100 h100; do
  vllm-sr-sim optimize --cdf data/azure_cdf.json --lam 100 --slo 500 \
    --gpu-short $gpu --gpu-long $gpu --b-short 8192 --long-max-ctx 8192
  vllm-sr-sim optimize --cdf data/azure_cdf.json --lam 100 --slo 500 \
    --gpu-short $gpu --gpu-long $gpu --long-max-ctx 8192 --verify-top 3
done
```

**结果（Azure，λ=100，SLO=500ms）：**

| GPU | 布局 | GPU 数 | 成本/年 | P99 TTFT |
|---|---|---|---|---|
| **A10G** | **双池** | **19** | **$168 K ← 最低** | 155ms / 335ms |
| H100 | 同构 | 6 | $211 K | 26 ms |
| A100 | 双池 | 12 | $232 K | 52ms / 112ms |
| H100 | 双池 | 7 | $247 K | 13ms / 30ms |

**非显然结论：** A10G 双池（$168K）比 H100 同构（$211K）更便宜 —— 更慢的 GPU 用双池路由补偿。因 Azure ctx 比（8192/4096=2×）使 A10G 每 GPU KV 槽从 64 变 128，足以抵消较低吞吐。

**取决于你的约束：**

| 优先级 | 选择 |
|---|---|
| 最低成本 | A10G 双池（$168K） |
| 最少机架 / 功耗 | H100 同构（6 GPU） |
| 最低延迟 | H100 双池（短请求 P99 13ms） |
| 长上下文 / 智能体 | H100 或 A100 —— A10G 24GB VRAM 限制 KV |

---

## 谜题 4 — 何时需要加 GPU？

**问题：**流量增长 —— 在精确哪个 λ 下需要预置下一档 GPU，以避免被动违反 SLO？

```bash
vllm-sr-sim whatif \
  --cdf data/azure_cdf.json --slo 500 \
  --gpu-short h100 --gpu-long h100 --long-max-ctx 8192 \
  --lam-range 25 50 100 150 200 300 400
```

**GPU 阶梯阈值（双池，H100）：**

| λ (req/s) | GPU | 成本/年 | 在达到下列 λ 之前应加 GPU… |
|---|---|---|---|
| 25 | 4 | $141 K | λ = 65 |
| 50 | 5 | $176 K | λ = 90 |
| 100 | 7 | $247 K | λ = 130 |
| 150 | 10 | $352 K | λ = 185 |
| 200 | 12 | $423 K | λ = 270 |
| 300 | 18 | $634 K | λ = 370 |
| 400 | 23 | $810 K | — |

**洞见：** GPU 扩展次线性 —— 流量增 16×（25→400）而 GPU 仅约 5.75×（4→23）。用此表在**阶梯前**预置，而非事后。等到 SLO 已违反再扩容，至少会有一个流量档 P99 劣化。

---

## 谜题 5 — 哪种路由器会导致 SLO 违反？

**问题：**机队规模已算对。路由器是否重要？

```bash
vllm-sr-sim compare-routers \
  --cdf data/agent_heavy_cdf.json --lam 20 --slo 1000 \
  --gpu-short h100 --gpu-long h100 \
  --n-s 2 --n-l 23 --long-max-ctx 65536 --n-req 5000
```

**智能体机队（λ=20，n_s=2，n_l=23）：**

| Router | P99 TTFT | SLO 1000ms |
|---|---|---|
| LengthRouter | 495 ms | ✓ 99.98% |
| **CompressAndRoute** | **534 ms** | **✗ 99.94%** |
| RandomRouter | 292 ms | ✓ 100% |

**两点意外：**

1. **CompressAndRoute 违反 SLO**，尽管其设计为缩小机队规模。它将边界长度请求压缩并路由到短池；多请求同时到达时会压垮仅 2 张 GPU 的短池并抬高 P99。它是**规划**工具 —— 用于规模阶段发现更低 GPU 数，生产应部署 LengthRouter。

2. **RandomRouter 通过 SLO**，因负载均匀摊到 25 张 GPU，稀释重尾请求。其 P99 最低（292 ms），但脆弱：短请求与长请求共享槽，流量混合变化会导致不可预测的延迟劣化。

对聊天负载（Azure、低利用率）三者均可过 SLO —— 差异仅在智能体或近饱和机队才显著。

---

## 谜题 6 — 双池混合 GPU 类型能否省钱？

**问题：**短请求受内存带宽约束、便宜；长请求需要大 KV 与快 prefill。短池放便宜卡、长池只放高端卡能否降本？

```bash
vllm-sr-sim optimize \
  --cdf data/azure_cdf.json --lam 100 --slo 500 \
  --gpu-short a10g --gpu-long h100 --long-max-ctx 8192

vllm-sr-sim optimize \
  --cdf data/lmsys_cdf.json --lam 100 --slo 500 \
  --gpu-short a10g --gpu-long h100 --long-max-ctx 65536
vllm-sr-sim optimize \
  --cdf data/lmsys_cdf.json --lam 100 --slo 500 \
  --gpu-short a10g --gpu-long a100 --long-max-ctx 65536
```

**Azure λ=100：**

| 配置 | GPU | 成本/年 | P99 短 | P99 长 |
|---|---|---|---|---|
| 全 A100（基线） | 12 | $232 K | 52 ms | 112 ms |
| **A10G 短 + H100 长** | **12** | **$212 K** | 155 ms | 30 ms |
| A10G 短 + A100 长 | 15 | $206 K | 155 ms | 112 ms |

**LMSYS λ=100（max_ctx=65536）：**

| 配置 | GPU | 成本/年 | P99 短 | P99 长 |
|---|---|---|---|---|
| 全 A100（基线） | 8 | $155 K | 43 ms | **2 822 ms ✗** |
| **A10G 短 + H100 长** | **7** | **$141 K** | 129 ms | 181 ms ✓ |
| A10G 短 + A100 长 | 9 | $132 K | 129 ms | **2 822 ms ✗** |

**两点洞见：**

1. **Azure（短上下文）：** A10G+H100 相对全 A100 同 12 卡省约 9%。贵卡集中在长池，便宜 A10G 承担约 98% 短流量。

2. **LMSYS（长上下文）：** A10G+A100 纸面便宜 11%，但 A100 长池**无法满足 500 ms SLO**。对高达 65536 token 的请求，A100 prefill 约 700–2800 ms；H100 凭借更大 chunk（1024 vs 512）与更低 W 将其减半。此处 H100 不是「奢侈」而是长池**正确性**要求。A10G 短 + H100 长相对全 A100 省约 9% **且**修复全 A100 无法达到的 SLO。

---

## 谜题 7 — 何时应切换到解耦 prefill/decode？

**问题：**prefill 算力受限；decode 内存带宽受限。若独立扩缩两池，哪种配对成本最低？

```bash
vllm-sr-sim disagg \
  --cdf data/azure_cdf.json --lam 100 \
  --slo-ttft 500 --slo-tpot 100 \
  --gpu-prefill h100 --gpu-decode a100 --max-ctx 8192

vllm-sr-sim disagg \
  --cdf data/azure_cdf.json --lam 100 \
  --slo-ttft 500 --slo-tpot 100 \
  --gpu-prefill a100 --gpu-decode h100 --max-ctx 8192
```

**Azure λ=100**（平均 ISL≈1450 tok，OSL≈483 tok，TTFT 含 1.8× KV 传输开销）：

| 配置 | GPU | 成本/年 | TTFT | TPOT |
|---|---|---|---|---|
| 全 A100 聚合 | 12 | $232 K | 26 ms | — |
| 全 H100 聚合 | 6 | $211 K | 8 ms | — |
| H100P + A100D | 7 (1P+6D) | $151 K | 162 ms | 91 ms |
| H100P + H100D | 4 (1P+3D) | $141 K | 162 ms | 45 ms |
| **A100P + H100D** | **4 (1P+3D)** | **$125 K ← 最低** | **492 ms** | **45 ms** |

**洞见：**

1. **解耦相对聚合省 35–46%**，代价是更高 TTFT（KV 传输使 raw prefill 约 ×1.8）。

2. **A100P + H100D 优于 H100P + A100D**，尽管 H100 单价约为 A100 的 1.82×。因 H100 decode 工人每秒处理请求数约为 A100 的 2.5×，所需张数更少（3 vs 6）。一张便宜 A100 在 λ=100 时已能承担全部 prefill。反直觉地，**溢价 GPU 在 decode 池**摊销成本。

3. **解耦何时值得：** 高吞吐、成本效率优先于延迟。解耦使每 GPU 成本约降 46%，但 TTFT 升至约 500 ms。若 SLO ≤ 200 ms，用 H100P（TTFT 约 162 ms）约 $141 K —— 仍比全 H100 便宜约 33%。

4. **何时保持聚合：** 低流量或延迟极关键（&lt; 50 req/s，TTFT SLO ≤ 100 ms）。解耦增加运维复杂度（两套扩缩、KV 网络），在约 100 req/s 以下难以justify。

---

## 小结

### 相对同构一览

| 工作负载 | 配置 | 成本差 | SLO |
|---|---|---|---|
| Azure 聊天，λ=50 | 双池 H100 | **+25%** | ✓ |
| Azure 聊天，λ=200 | 双池 H100 | **持平** | ✓ |
| LMSYS 聊天，λ=100 | 双池 H100 | **−38%** | ✓ |
| 智能体重，λ=20 | 同构 H100 | — | **✗ 失败** |
| 智能体重，λ=20 | 双池 H100 | +4% | ✓ |
| Azure，λ=100 | A10G 双池 | **相对 H100 同构 −21%** | ✓ |

### 经验法则

| 问题 | 答案 |
|---|---|
| 要拆分吗？ | 重尾则必须；或 ctx 比 ≥4× 且 λ 高 |
| 在哪拆？ | 跑 `pareto` —— 不能仅从 CDF 形状单独推出 |
| 选哪种 GPU？ | 跑 `whatif --gpu-compare` —— A10G 双池可击败 H100 同构 |
| 混合 GPU？ | 可以 —— 短池便宜卡，长池按 max_ctx 需求匹配 |
| 长上下文 GPU？ | A100 在约 20K token P99 以上易失败；65K ctx 机队长池常需 H100 |
| 解耦 P/D？ | λ ≥ 100 req/s：A100P + H100D 相对全 A100 约省 46%（TTFT 约升 20×） |
| 何时预置？ | `whatif` 看阶梯；在阶梯前预置 |
| 生产用哪个路由器？ | 始终 LengthRouter；CompressAndRoute 仅用于规模阶段 |
| 需要 DES 吗？ | 智能体 / 重尾需要 —— 解析会说「可行」实际不可行 |
| 可靠性余量？ | 见下文 —— 在 SLO 规模上乘 `node_avail` |

---

## 可靠性规模 —— 将 SLO 计数转为生产计数

> **上文所有 GPU 数均假设 `node_avail=1.0`（节点全健康）。**  
> 生产部署应按稳态节点可用性的倒数放大。

GPU 与 NVLink 硬件存在可测故障率（Kokolis et al. 2024；Cui et al. 2025）：

| GPU | 故障率 r_f | MTTR | 稳态可用性 |
|---|---|---|---|
| A100（Meta RSC-1） | 0.0065 / 节点·天 | ~4h 软（驱动重置）→ **A ≈ 99.89%** | `A100_AVAIL_RSC1_FAST` |
| A100（Meta RSC-1） | 0.0065 / 节点·天 | ~48h 硬（换 GPU/NVLink）→ **A ≈ 98.71%** | `A100_AVAIL_RSC1_SLOW` |
| 规模 H100 | — | 5% 过度预置规则（Cui 2025）→ **A = 95%** | `H100_AVAIL_5PCT` |

**应用方式：** 见英文原文 Python 示例（`FleetOptimizer(..., node_avail=...)`）。

**对上文数字的影响：** SLO 规模是在无故障下的最小值。乘可靠性余量后，每档约额外 `ceil(n/A)−n` 张 GPU；多数场景 **+1 张 GPU** 可覆盖 A100 现实故障率。H100 5% 规则可能 +1～2 张。

> `ρ_max=0.85` 利用率上限提供**排队稳定性**余量（约 15% 过度预置），**不是**可靠性余量；二者独立且可乘。

---

## 谜题 8 — 不违反 SLO 的前提下最多可削减多少电网功率？

**场景：** 数据中心参与需求响应（DR）。电网要求在未来 15–60 分钟内将功耗降低 X%。你希望知道：在 P99 TTFT 不越界的前提下，**最大可承诺的削减**是多少？

**机制：** GPU-to-Grid（G2G）表明，限制服务引擎最大在飞 batch（`vLLM max_num_seqs`）是调节 GPU 功耗最有效的软件旋钮。更低上限 → 并发序列更少 → 内存带宽压力与功耗下降；代价是可用的 KV 槽更少，排队更长，P99 TTFT 上升。

**模拟器行为：** `grid_flex_analysis()` 扫功率目标，反解功耗模型得隐含 batch 上限，并在该并发下重算 P99 TTFT。两级验证：

- **解析**（快，约 1 s）：M/G/c 近似，在每档 batch 上限下重标定服务率。
- **DES 验证**（`--verify-des N`）：每档 flex 跑 N 条请求离散事件仿真，直接验证延迟断言；DR 合同签字前建议采用。

**功耗模型：** 各 profile 使用 **logistic 曲线**（Hassan et al.，式 2），拟合 ML.ENERGY v3.0。H100-SXM5（k=1.0，x0=4.2）：P(1)≈304W，P(128)≈583W（实测约 600W）。低 batch 深度削减时比旧线性模型更准确。

**数值示例（含 DES 验证）：** 40×H100，λ=200 req/s，SLO=500 ms：

```bash
vllm-sr-sim grid-flex \
    --cdf data/azure_cdf.json \
    --lam 200 --n-gpus 40 --gpu h100 --slo 500 \
    --verify-des 15000
```

```
======================================================================
  Grid Flexibility Analysis  [logistic power model]
  Fleet: 40 GPUs  λ=200 req/s  SLO=500 ms
  Baseline: 23.3 kW fleet power  (583 W/GPU)
======================================================================
    Flex  n_max   W/GPU  Fleet kW  P99 analyt   P99 DES   SLO
  ------ ------ ------- --------- ----------- --------- -----
      0%    128    583W     23.3kW        7.9ms     35.2ms   OK
      5%     84    570W     22.8kW        7.9ms     36.1ms   OK
     10%     48    540W     21.6kW        7.9ms     38.4ms   OK
     15%     33    510W     20.4kW        7.9ms     39.9ms   OK
     20%     24    479W     19.1kW        7.9ms     41.0ms   OK
     25%     17    442W     17.7kW        7.9ms     42.3ms   OK
     30%     13    413W     16.5kW        7.9ms     51.0ms   OK
     40%      6    350W     14.0kW        infms    189.8ms   OK
     50%      1    304W     12.2kW        infms 449791ms BREACH

  Max safe flex depth: 40%  (saves 9.3 kW fleet-wide, P99=189.8ms (DES))
```

短时 DR 窗口（约 15000 请求 ≈ 75 s @200 req/s）内可承诺约 **40% 功率削减** 且 P99 仍在 SLO 内；50%（n_max=1）时队列灾难性崩溃（P99 约 449 s）。

**解析 vs 事件：** 解析给出*持续*降功率的稳态极限；DES 给出*短时事件*（分钟级）可承受的深度。二者都有用。

**与过度预置的区别：** 可靠性规模是加 GPU；电网 flex 是在**固定 GPU 数**下节流 batch —— 可叠加。

**Python / CLI：**

```python
from fleet_sim import grid_flex_analysis, print_grid_flex_table, H100_80GB
import json

cdf = json.load(open("data/azure_cdf.json"))
cdf = [(int(t), float(f)) for t, f in (cdf["cdf"] if isinstance(cdf, dict) else cdf)]

results = grid_flex_analysis(
    cdf=cdf, lam=200, n_gpus=40, gpu=H100_80GB,
    t_slo_ms=500, max_ctx=8192,
    n_sim_requests=15000,
    verbose=True,
)
print_grid_flex_table(results, t_slo_ms=500, n_gpus=40, lam=200)
```

```bash
vllm-sr-sim grid-flex \
    --cdf data/azure_cdf.json \
    --lam 200 --n-gpus 40 --gpu h100 --slo 500
vllm-sr-sim grid-flex \
    --cdf data/azure_cdf.json \
    --lam 200 --n-gpus 40 --gpu h100 --slo 500 \
    --verify-des 15000
```

> batch ≥ 16 时 logistic 曲线误差约 ±3%；batch &lt; 8 时约束较弱，深度削减时请用自有 `vllm serve` 剖析重拟合 `power_logistic_k` / `power_logistic_x0`。

---

## 谜题 9 — 哪种 GPU 对我的工作负载能效最高？

**问题：** 推理电费占比上升。哪种 GPU 每瓦输出 token 最多？机队低利用率时能效下降多少？

**重要前提：** 预置 `ManualProfile` 针对不同模型校准 —— H100/A100 对 Llama-3-70B TP8，`A10G` 对 7B 级单卡。`tok-per-watt --gpus h100 a100 a10g` 将同一 CDF 套在三个**代表不同模型**的 profile 上 —— tok/W 差异部分来自模型规模而非仅 GPU。

**干净 GPU 对比（同模型、不同硬件）** 请用 `ComputedProfile`（见英文 Python 示例）。

**单池每 GPU 对比（假设同模型）：**

```bash
vllm-sr-sim tok-per-watt \
  --cdf data/azure_cdf.json --lam 100 --slo 500 \
  --gpus h100 a100 a10g --rho-sweep
```

**结果（Azure CDF，λ=100 req/s，SLO=500 ms）— SLO 最优机队：**

```
Pool / GPU               GPUs    ρ   n_act  Tok/W  P/GPU(W)  $/1M tok  P99(ms)  PwrQ
------------------------------------------------------------------------------------
H100-80GB                   6  0.82  104.4   9.38    577 W   $  0.21      7.9   HIGH
A100-80GB                  12  0.83  106.1   7.09    382 W   $  0.23     25.9   FAIR
A10G                       21  0.82   52.3  13.56    114 W   $  0.18     71.1   LOW *
```

*A10G 功耗模型质量低：仅投影；无公开 batch–功耗数据。且 A10G profile 代表 7B 级模型，与 H100/A100 的 70B 不同。*

**Tok/W 与利用率（仅 H100、A100 — HIGH/FAIR）：**

```
GPU           ρ=0.20  ρ=0.40  ρ=0.60  ρ=0.80   @ SLO-opt
----------------------------------------------------------
H100-80GB      2.69    4.63    6.43    8.12      9.38 ★
A100-80GB      1.79    3.45    5.03    6.55      7.09 ★
```

**要点：** H100 在已验证 tok/W 上领先（9.38 vs 7.09 @ ρ≈0.82）。低 ρ 使 tok/W 降 3–4×（idle 地板）。`--cdf data/lmsys_cdf.json` 时排序可能反转 —— H100 机队过度预置、ρ≈0.46 时 idle 功耗使 A100 反超。

**决策指南：** 紧 SLO + 长上下文能效优先选 H100；最低每 token 成本看 A10G/A100 并用 `--rho-sweep`；同模型硬件对比用 `ComputedProfile.decode_efficiency()`；生产 DR 合同前先校准自有 logistic 曲线。

> 勿过度解读 tok/W 绝对值；曲线形状与同模型下 H100 vs A100 相对排序更可靠。A10G 绝对值为投影。

---

## 谜题 10 — 短请求路由到小模型能否省电？

**问题：** 若语义路由器将短/简单请求送到小模型（A10G+7B），复杂请求到大模型（H100+70B），机队级能效相对全 H100 同构如何？

这是多池 tok/W 问题：**跨池 N 不能相消**：

```
fleet_tok/W = Σ_i(λ_i × mean_L_out_i) / Σ_i(N_i × power_i(n_i))
```

`fleet_tpw_analysis()` 用各池子 CDF 正确计算（见英文 `tok-per-watt` 与 Python 示例）。

**注意：** 不同池模型能力不同；tok/W 衡量能量而非质量；语义分类误差会改变实际分流比例。

---

## 复现全部结果

```bash
# Puzzle 6: 混合 GPU 类型池
vllm-sr-sim optimize --cdf data/azure_cdf.json  --lam 100 --slo 500 \
  --gpu-short a10g --gpu-long h100 --long-max-ctx  8192
vllm-sr-sim optimize --cdf data/lmsys_cdf.json --lam 100 --slo 500 \
  --gpu-short a10g --gpu-long h100 --long-max-ctx 65536
vllm-sr-sim optimize --cdf data/lmsys_cdf.json --lam 100 --slo 500 \
  --gpu-short a10g --gpu-long a100 --long-max-ctx 65536

# Puzzle 7: 解耦 prefill/decode
vllm-sr-sim disagg --cdf data/azure_cdf.json --lam 100 \
  --slo-ttft 500 --slo-tpot 100 --gpu-prefill h100 --gpu-decode a100 --max-ctx 8192
vllm-sr-sim disagg --cdf data/azure_cdf.json --lam 100 \
  --slo-ttft 500 --slo-tpot 100 --gpu-prefill a100 --gpu-decode h100 --max-ctx 8192

# Puzzle 1: Pareto 阈值
vllm-sr-sim pareto --cdf data/lmsys_cdf.json       --lam 100 --slo 500 --long-max-ctx 65536
vllm-sr-sim pareto --cdf data/azure_cdf.json       --lam 200 --slo 500 --long-max-ctx  8192
vllm-sr-sim pareto --cdf data/agent_heavy_cdf.json --lam 200 --slo 500 --long-max-ctx 65536

# Puzzle 2: 智能体 SLO 失败
vllm-sr-sim optimize --cdf data/agent_heavy_cdf.json --lam 20 --slo 1000 \
  --gpu-short h100 --gpu-long h100 --b-short 65536 --long-max-ctx 65536
vllm-sr-sim optimize --cdf data/agent_heavy_cdf.json --lam 20 --slo 1000 \
  --gpu-short h100 --gpu-long h100 --b-short 4096  --long-max-ctx 65536 --verify-top 3

# Puzzle 3: GPU 类型选择
for gpu in a10g a100 h100; do
  vllm-sr-sim optimize --cdf data/azure_cdf.json --lam 100 --slo 500 \
    --gpu-short $gpu --gpu-long $gpu --b-short 8192 --long-max-ctx 8192
  vllm-sr-sim optimize --cdf data/azure_cdf.json --lam 100 --slo 500 \
    --gpu-short $gpu --gpu-long $gpu --long-max-ctx 8192 --verify-top 3
done

# Puzzle 4: 流量阶梯
vllm-sr-sim whatif --cdf data/azure_cdf.json --slo 500 \
  --gpu-short h100 --gpu-long h100 --long-max-ctx 8192 \
  --lam-range 25 50 100 150 200 300 400

# Puzzle 5: 路由器对比
vllm-sr-sim compare-routers --cdf data/azure_cdf.json --lam 200 --slo 500 \
  --gpu-short h100 --gpu-long h100 --n-s 5 --n-l 7 --long-max-ctx 8192 --n-req 5000
vllm-sr-sim compare-routers --cdf data/agent_heavy_cdf.json --lam 20 --slo 1000 \
  --gpu-short h100 --gpu-long h100 --n-s 2 --n-l 23 --long-max-ctx 65536 --n-req 5000

# Puzzle 9: GPU 能效（单池、同模型假设）
vllm-sr-sim tok-per-watt --cdf data/azure_cdf.json --lam 100 --slo 500 \
  --gpus h100 a100 a10g --rho-sweep
vllm-sr-sim tok-per-watt --cdf data/lmsys_cdf.json --lam 100 --slo 500 \
  --gpus h100 a100

# Puzzle 10: 多池路由 tok/W
vllm-sr-sim tok-per-watt --cdf data/azure_cdf.json --lam 100 --slo 500 \
  --b-short 4096 --gpu-short a10g --gpu-long h100
vllm-sr-sim tok-per-watt --cdf data/lmsys_cdf.json --lam 100 --slo 500 \
  --b-short 2048 --gpu-short a10g --gpu-long h100

# Puzzle 8: 电网 flex（解析 + DES）
vllm-sr-sim grid-flex --cdf data/azure_cdf.json \
  --lam 200 --n-gpus 40 --gpu h100 --slo 500
vllm-sr-sim grid-flex --cdf data/azure_cdf.json \
  --lam 200 --n-gpus 40 --gpu h100 --slo 500 --verify-des 15000
vllm-sr-sim grid-flex --cdf data/azure_cdf.json \
  --lam 100 --n-gpus 15 --gpu a100 --slo 500 --verify-des 15000
```
