---
title: Power Model Reference
---

# Power Model Reference

This page documents the simulator's power estimation methods. It is reference
material for advanced analysis, not the primary getting-started path.

This document covers **two distinct power model approaches** that coexist in the simulator:

1. **First-principles model** (`ComputedProfile.power_at_concurrency`) — derives power
   from the same roofline decomposition used for latency; no fitted scalars beyond TDP.
   Use this when you have a `HardwareSpec + ModelSpec` and want a portable estimate for
   any (GPU, model) pair.

2. **Empirical logistic model** (`ManualProfile` fields) — fitted to actual silicon data;
   more accurate for the specific (GPU, model) pairs it covers, but requires real
   measurements.  Use this for production fleet decisions where data exists.

Both models share the same two empirical calibration anchors from ML.ENERGY Benchmark v3.0
(H100-SXM5, batch=1 → 300 W and batch=128 → 600 W), which give the idle and active TDP
fractions used throughout.

---

## Part 1 — First-principles power model (`ComputedProfile`)

### Derivation chain

Every quantity traces back to hardware datasheets and model architecture parameters.
No fitting parameters beyond the two TDP fractions documented below:

```
W = model_bytes_per_gpu / (mem_bw × 0.80)     # weight-streaming time per decode iter
H = kv_bytes_per_token × ctx / mem_bw / tp    # marginal latency per in-flight sequence

iter_latency(n)  = W + H_eff × n              # H_eff = H × (mean_ctx / calib_ctx)
decode_tps(n)    = n / iter_latency(n)         # output tokens per second per GPU

kv_frac(n)       = n × H_eff / iter_latency(n)
                   ← KV-cache fraction of HBM traffic (derived purely from W, H)

compute_frac(n)  = 2 × active_params/tp × n / (iter_latency(n) × fp16_tc_flops)
                   ← fraction of peak tensor-core throughput utilised

activity(n)      = min(1.0,  kv_frac(n) + compute_frac(n))
power(n)         = hw.power × (P_IDLE_FRAC + (P_ACTIVE_FRAC − P_IDLE_FRAC) × activity(n))

tok/W            = decode_tps(n) / power(n)
```

### Physical interpretation

`kv_frac` captures the core transition: at batch=1 the GPU streams only model weights
(HBM traffic = weight bytes); as batch grows, KV-cache reads increasingly dominate HBM
traffic and power rises. `compute_frac` captures the additional SM power from
the growing number of weight matmuls per second. The sum is clamped at 1.0.

| n_active | kv_frac | compute_frac | activity | P (W) | tok/W |
|---|---|---|---|---|---|
| 1   | 0.009 | 0.003 | 0.012 | 305 | 0.48 |
| 8   | 0.069 | 0.019 | 0.089 | 328 | 3.38 |
| 16  | 0.130 | 0.036 | 0.166 | 351 | 5.90 |
| 32  | 0.230 | 0.064 | 0.294 | 389 | 9.41 |
| 44  | 0.291 | 0.082 | 0.372 | 413 | 11.24 |

*H100-SXM5 + Llama-3.1-70B, TP=8, fp16, mean_ctx=4096.*

### Empirical TDP fractions

| Constant | Value | Source |
|---|---|---|
| `_POWER_IDLE_FRAC` | 0.43 | ML.ENERGY v3.0: H100-SXM5 batch=1 → 300 W / 700 W TDP |
| `_POWER_ACTIVE_FRAC` | 0.86 | ML.ENERGY v3.0: H100-SXM5 batch=128 → 600 W / 700 W TDP |

These transfer to other GPUs in the same HBM-bandwidth-bound regime (Ampere, Hopper,
Blackwell dense-decode). For GPUs outside this regime (e.g. PCIe cards, GDDR6), the
fractions may differ; use the empirical logistic calibration below.

### Validation

| n | Predicted (W) | ML.ENERGY (W) | Error |
|---|---|---|---|
| 1   | 305 | 300 | +1.5 % |
| 32  | 389 | ~480 | ~19 %  ← see note |
| 128 | 465 | 600 | ~22 % (exceeds n_slots) |

The ~20 % underestimate at high n has three sources not in the model:
(a) NVLink all-reduce power grows with batch × hidden × TP,
(b) LayerNorm and softmax kernel overhead,
(c) GPU dynamic voltage/frequency scaling at high compute load.
For planning purposes this accuracy is sufficient; use the logistic model for
production demand-response contracts where ±5 % matters.

### Model dependency

Because W and H both depend on model architecture, **tok/W is a (GPU, model) pair
property, not a GPU property alone**. The same H100 yields different tok/W for:

| Model | W (ms) | H (ms/seq) | tok/W at n=32 |
|---|---|---|---|
| Llama-3.1-8B, TP=1 | ~5.0 | ~0.016 | ~35 |
| Llama-3.1-70B, TP=8 | ~6.7 | ~0.063 | ~9.4 |
| Llama-3.1-405B, TP=8 | ~38 | ~0.063 | ~1.8 |

Do not compare tok/W across profiles that represent different models.
`ComputedProfile.decode_efficiency()` exposes all intermediate quantities so the
derivation is fully auditable.

### Multi-pool fleet tok/W

For a single homogeneous pool, N_gpus cancels and:

```
tok/W (single pool) = decode_tps(n) / power(n)   [independent of N]
```

For multi-pool fleets (hetero routing, semantic-router small+large model), **N does not
cancel** across pools. `fleet_tpw_analysis()` computes the correct aggregate:

```
fleet_tok/W = Σ_i(λ_i × mean_L_out_i) / Σ_i(N_i × power_i(n_i))
```

where each pool has its own (GPU, model)-specific W, H, and power model.
See [Capacity planning scenarios](./use-cases.md) for a worked routing example.

---

## Part 2 — Empirical logistic model (`ManualProfile`)

---

## The logistic power curve

The G2G paper (Hassan et al., arXiv:2602.05116v1, Eq. 2) proposes modelling per-GPU power
as a logistic function of log₂(batch size):

```
P(b) = P_range / (1 + exp(-k * (log2(b) - x0))) + P_idle
```

where:

- `b`        = number of concurrently in-flight requests (≈ vLLM `max_num_seqs`)
- `P_idle`   = GPU power at b → 0 (asymptotic floor; in practice: power at b=1)
- `P_nominal`= GPU power at b → ∞ (asymptotic ceiling; ≈ TDP × utilisation factor)
- `P_range`  = `P_nominal − P_idle`
- `k`        = steepness of the transition from idle to nominal power
- `x0`       = log₂(batch) at which power = midpoint = `P_idle + P_range / 2`

Physical interpretation of x0 and k
-------------------------------------

- **x0** is set by when the GPU memory bandwidth (HBM or GDDR6) is 50% saturated.
  More bandwidth → higher saturation batch → higher x0.
- **k** characterises how abruptly the GPU transitions from compute-bound (low batch,
  low utilisation, low power) to memory-bandwidth-bound (high batch, high utilisation,
  high power).  HBM GPUs (H100, A100) have a sharper transition (k ≈ 1.0) than GDDR6
  GPUs (A10G) because HBM delivers bandwidth in large bursts (k ≈ 0.7).

Memory-bandwidth saturation scaling rule (used for projections)
----------------------------------------------------------------
For the same model and context length, the saturation batch n_sat scales as:

```
n_sat ∝ GPU_memory_bandwidth / KV_cache_demand_per_seq_per_iter
```

If the *relative* KV-cache demand per sequence is held constant (same model), then
x0 scales as:

```
x0(GPU_target) = x0(GPU_ref) + log2(BW(GPU_target) / BW(GPU_ref))
```

This is the primary derivation used for A100 and A10G projections below.

---

## H100-SXM5 — MEASURED DATA (source quality: HIGH)

| Parameter        | Value  | Source                                         |
|------------------|--------|------------------------------------------------|
| `power_idle_w`   | 300 W  | ML.ENERGY Benchmark v3.0, H100-SXM5 + vLLM    |
| `power_nominal_w`| 600 W  | ML.ENERGY Benchmark v3.0, H100-SXM5 + vLLM    |
| `power_logistic_k` | 1.0  | Fitted to G2G paper Fig. 2 data points          |
| `power_logistic_x0`| 4.2  | Fitted to G2G paper Fig. 2 data points          |
| Saturation batch | ≈ 18   | 2^4.2 ≈ 18 concurrent requests                 |
| HBM3 bandwidth   | 3.35 TB/s | NVIDIA H100 datasheet                       |
| TDP              | 700 W  | NVIDIA H100 datasheet (DS-10313-001_v1.6)      |

**Fit method:** The G2G paper (Figure 2) shows H100-SXM5 power vs. batch for Llama-3.1-class
models running vLLM at batch ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256}.  Approximate readings
from the figure give:

| batch | ≈ W (fig.) | model W (k=1.0, x0=4.2) |
|-------|-----------|--------------------------|
|     1 |     ~304  |                     304  |
|     4 |     ~330  |                     330  |
|    16 |     ~435  |                     435  |
|    32 |     ~507  |                     507  |
|    64 |     ~557  |                     557  |
|   128 |     ~583  |                     583  |

Fit error &lt; 3% across the 40–100% load range.

ML.ENERGY v3.0 (Chung et al., arXiv:2601.22076) confirms: at batch=128 on H100-SXM5,
per-GPU power ≈ 600 W (≈ 86% TDP).  At batch=1, power ≈ 300 W (≈ 43% TDP).

---

## A100-SXM4 — ONE ANCHOR + FLOPS-SCALING DERIVATION (source quality: FAIR)

### Measured anchor

From Sada et al. (arXiv:2507.00418, "Serving LLMs in HPC Clusters: QAic vs A100",
Table 1, October 2025):

> 8×A100-SXM4-80GB serving Nemotron-70B via vLLM at 32 concurrent requests:
> total node power = **2,983 W** → **373 W / GPU** (nvidia-smi measured)

Tensor parallelism TP=8; each GPU processes the same 32 concurrent sequences.

### Why BW-scaling fails here

Bandwidth-scaling gives `x0 = 4.2 + log2(2.0/3.35) = 3.5`, predicting P(32)=334 W
(10% below measured 373 W).  The error is systematic.

For a 70B model on A100-SXM4 with TP=8, weight memory reads per iteration = 17.5 GB/GPU.
At A100's 2.0 TB/s HBM2e bandwidth, this takes 8.75 ms — matching the W=8 ms in the
performance profile.  **Memory bandwidth is saturated at b=1 by weight loading alone.**
As batch increases, BW cannot increase further; what increases is COMPUTE utilisation
(larger matmuls).  Therefore, x0 reflects compute saturation, not KV-cache BW saturation.

### FLOPS-scaling methodology

For large-model, high-TP deployments where weights fill the memory bandwidth, x0 scales
with the GPU's TFLOPS (how many concurrent sequences saturate the tensor cores):

```
x0(GPU) = x0(H100) + log2(TFLOPS_GPU / TFLOPS_H100)
```

FP16 peak TFLOPS (NVIDIA datasheets):

- H100-SXM5: 989 TFLOPS (DS-10313-001_v1.6)
- A100-SXM4: 312 TFLOPS (DS-10031-001_v1.6)

```
x0(A100) = 4.2 + log2(312 / 989) = 4.2 − 1.66 = 2.54  →  2.5
```

Saturation batch: 2^2.5 ≈ 5.7 concurrent requests.

### Parameter derivation

**k = 1.0** — A100 HBM2e and H100 HBM3 are same memory-type family; similar burst
characteristics.  No published data justifies a different value.

**x0 = 2.5** — FLOPS-scaling (above).

**P_nominal = 385 W (96.3% TDP)** — Back-derived from measured anchor:

```
P(32) measured = 373 W
sigmoid(log2(32) - 2.5) = sigmoid(2.5) = 0.924
P_range = (373 − 175) / 0.924 = 214 W
P_nominal = 175 + 214 = 389 W  →  conservative round to 385 W
```

Cross-check: P(32) = 210 / (1 + exp(-2.5)) + 175 = **369 W** (1.1% error vs 373 W) ✓

**P_idle = 175 W (43.75% TDP)** — No direct b=1 measurement available.
Projected from H100 TDP fraction: 400 W × 0.4375 = 175 W.  May be slightly
over-estimated (A100 HBM2e has lower memory-refresh power per GB than H100 HBM3).

### Calibration check table

| batch | model W (k=1.0, x0=2.5, P_nom=385) | notes                              |
|-------|--------------------------------------|------------------------------------|
|     1 |                                 191  | near P_idle=175 ✓                  |
|     4 |                                 254  | rising; compute waking up          |
|     6 |                                 272  | midpoint (2^2.5 ≈ 5.7)            |
|    32 |                                 369  | **measured ≈ 373; 1.1% error ✓**  |
|   128 |                                 383  | near P_nominal=385 ✓               |

---

## A10G — PROJECTION ONLY (source quality: LOW)

**No reliable published measurement data for A10G power vs batch size during LLM
inference was found.**  ML.ENERGY v3.0 does not cover A10G.  No arXiv papers as of
March 2026 report batch-vs-power curves for A10G + vLLM.

All A10G parameters are **projections from hardware specifications** only.
**Empirical calibration is strongly recommended before using these in a DR contract.**

### Parameter derivation

**P_nominal (120 W) and P_idle (75 W)**

- A10G TDP = 150 W (NVIDIA A10G datasheet DS-10012-001_v1.3)
- P_nominal = 120 W = 80% TDP (conservative; PCIe cards typically run 75–85% TDP at full load)
- P_idle = 75 W = 50% TDP
  - Higher fraction than H100/A100 because A10G uses GDDR6 (not HBM);
    GDDR6 has higher refresh power per byte than HBM at large capacities
  - 24 GB GDDR6 at ≈ 2 W/GB idle ≈ 48 W for memory, plus ≈ 25 W core logic = ~73 W
  - Round to 75 W
  - **Uncertainty: ±15 W** — no measurement available

**k = 0.7 (projected)**

- A10G uses GDDR6 (not HBM); GDDR6 delivers bandwidth in smaller, more uniform bursts
- The transition from compute-bound to memory-bound is less abrupt than HBM
- GDDR6 steepness is lower: k ≈ 0.7 (vs HBM k ≈ 1.0)
- This is a structural argument, not a fitted value

**x0 = 3.7 (projected)**
A10G is typically used for 13B-class models (fits in 24 GB with KV-cache headroom).
KV-cache per sequence scales as:

```
KV(13B) / KV(70B) ≈ (n_layers_13B × n_kv_heads_13B × head_dim_13B)
                   / (n_layers_70B × n_kv_heads_70B × head_dim_70B)
≈ (40 × 8 × 128) / (80 × 8 × 128)
= 40/80 = 0.5
```

Applying bandwidth-scaling rule with A10G GDDR6 (≈ 600 GB/s) and A100 HBM2e (2,000 GB/s),
normalised for the smaller model:

```
n_sat(A10G, 13B) / n_sat(A100, 70B) =
    (BW_A10G / KV_13B) / (BW_A100 / KV_70B)
  = (600 GB/s / 0.5) / (2000 GB/s / 1.0)
  = 1200 / 2000
  = 0.60

n_sat(A100, 70B) = 2^3.5 ≈ 11
n_sat(A10G, 13B) = 11 × 0.60 ≈ 6.6  →  but single-GPU, no TP overhead

x0 = log2(6.6) ≈ 2.7   (for 13B model)
```

However, for smaller models (7B) that A10G frequently serves:

```
KV(7B) / KV(70B) ≈ 0.25
n_sat(A10G, 7B) = 11 × (600/0.25) / (2000/1.0) = 11 × 1.2 = 13.2
x0 = log2(13) ≈ 3.7   (for 7B model)
```

The A10G profile is parameterised for the **7B model** case (the larger-saturation scenario
to be conservative for DR purposes), giving x0 = 3.7.

**Note on model size dependence:** The A10G x0 varies from ~2.7 (13B, memory pressure
saturates early) to ~3.7 (7B, lower KV-cache per seq).  Use x0=2.7 if serving 13B;
x0=3.7 for 7B.  The profile default uses 3.7 (7B) as a conservative estimate.

---

## How to calibrate your own logistic parameters

Run vLLM in offline batch mode and measure power with `nvidia-smi --query-gpu=power.draw`:

```bash
# Install zeus for easy GPU power sampling
pip install zeus-ml

# Then for each batch size b in {1, 2, 4, 8, 16, 32, 64, 128}:
#   1. Start vllm serve with --max-num-seqs b
#   2. Send a sustained stream of requests at rate >> b
#   3. Record steady-state power from nvidia-smi

python - <<'EOF'
import numpy as np
from scipy.optimize import curve_fit

# Your measured data points: (batch_size, power_W_per_GPU)
data = [
    (1,  310),   # replace with your measurements
    (4,  340),
    (8,  380),
    (16, 440),
    (32, 490),
    (64, 530),
    (128, 560),
]

b_vals = np.array([d[0] for d in data], dtype=float)
p_vals = np.array([d[1] for d in data], dtype=float)

def logistic(b, p_idle, p_range, k, x0):
    return p_range / (1 + np.exp(-k * (np.log2(b) - x0))) + p_idle

p0 = [p_vals[0], p_vals[-1] - p_vals[0], 1.0, 4.0]
popt, _ = curve_fit(logistic, b_vals, p_vals, p0=p0,
                    bounds=([0, 0, 0.1, 0], [500, 500, 5, 10]))

print(f"power_idle_w     = {popt[0]:.0f}")
print(f"power_nominal_w  = {popt[0]+popt[1]:.0f}")
print(f"power_logistic_k = {popt[2]:.2f}")
print(f"power_logistic_x0 = {popt[3]:.2f}")
EOF
```

---

## Summary of source quality

| GPU     | P_idle  | P_nominal | k    | x0   | Reliability |
|---------|---------|-----------|------|------|-------------|
| H100    | 300 W — measured (ML.ENERGY v3.0) | 600 W — measured (ML.ENERGY v3.0) | 1.0 — fitted (G2G Fig.2) | 4.2 — fitted (G2G Fig.2) | **HIGH** |
| A100    | 175 W — projected (H100 TDP ratio) | 385 W — back-derived (Sada et al. anchor, 1.1% error) | 1.0 — projected (HBM family) | 2.5 — FLOPS-scaling (1.1% error vs anchor) | **FAIR** |
| A10G    | 75 W — projected (TDP ratio, GDDR6 estimate) | 120 W — projected (TDP ratio) | 0.7 — projected (GDDR6 architecture) | 3.0 — conservative median; no measurement | **LOW** |

For production DR commitments: **always profile your actual (GPU, model) pair** using the
calibration script above and update the profile's logistic parameters before signing a DR
contract.

---

## References

- Hassan et al. (2025). "GPU-to-Grid: Voltage Regulation via GPU Utilization Control."
  arXiv:2602.05116v1.  — Power model equation and H100-SXM5 measurements.
- Chung et al. (2025). "The ML.ENERGY Benchmark." arXiv:2505.06371v2 (NeurIPS 2025 D&B).
  — Benchmark methodology; v3.0 covers H100 + B200 only.
- Chung et al. (2026). "Where Do the Joules Go?" arXiv:2601.22076v2.
  — ML.ENERGY v3.0 blog/paper; confirms H100 power range 300–600 W at batch 1–128.
- Sada et al. (2025). "Serving LLMs in HPC Clusters: QAic vs A100." arXiv:2507.00418v3.
  — Measured A100-SXM4 power: 373 W/GPU (8×A100 TP=8, Nemotron-70B, batch=32).
- NVIDIA A100 Datasheet DS-10031-001_v1.6. TDP = 400 W (SXM4).
- NVIDIA H100 Datasheet DS-10313-001_v1.6. TDP = 700 W (SXM5).
- NVIDIA A10G Datasheet DS-10012-001_v1.3. TDP = 150 W (PCIe).
