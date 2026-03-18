# Simulation Algorithms, Assumptions, and Tuning Guide

This document describes the mathematical models underlying `inference-fleet-sim`,
the assumptions each model makes, where those assumptions break down, and how
to tune key parameters for your workload.

---

## 1. GPU Iteration Latency Model

### Formula

Every GPU profile exposes an iteration latency function:

```
iter_t(n_active, mean_seq_len) = W + H_eff × n_active
```

where:

| Symbol | Meaning | Unit |
|---|---|---|
| `W` | Base iteration cost — model-weight memory read, AllReduce, kernel dispatch | seconds |
| `H` | Per-sequence overhead at `calibration_ctx`, measured on silicon | s/seq |
| `H_eff` | Effective per-sequence overhead scaled to actual mean sequence length | s/seq |
| `n_active` | Number of sequences currently in the batch | — |
| `mean_seq_len` | Mean total token count of active sequences (prompt + output) | tokens |
| `calibration_ctx` | Context length at which `H` was measured (default: 8 192 tokens) | tokens |

**Sequence-length scaling:**

```
H_eff = H × (mean_seq_len / calibration_ctx)
```

This reflects that attention computation cost scales linearly with sequence
length: O(n_active × seq_len × d_head).  When a pool serves 800-token LMSYS
requests, `H_eff` is 10× smaller than when it serves 8 192-token max-context
requests.

### Why this matters for heterogeneous pools

`iter_t` and `n_slots` each scale inversely with context length; together they
give the short pool a substantial throughput advantage (A100-80 GB, W=8 ms,
H=0.65 ms, `calibration_ctx=8192`):

| Pool | max_ctx | n_slots | mean_seq | H_eff | iter_t @full | GPU throughput |
|---|---|---|---|---|---|---|
| Short pool | 2 048 | 512 | 800 | 6.5 × 10⁻⁵ | 40 ms | ~78 req/s |
| Homo pool  | 8 192 | 128 | 1 600 | 1.3 × 10⁻⁴ | 25 ms | ~49 req/s |
| Long pool  | 8 192 | 128 | 5 000 | 4.0 × 10⁻⁴ | 59 ms | ~2 req/s  |
| Homo (agent) | 65 536 | 16 | 15 000 | 1.2 × 10⁻³ | 27 ms | ~1 req/s |

Note: even though `iter_t` is higher for the short pool (because n_slots is 4×
larger), throughput = `n_slots / service_time` is still 1.6× better because 4×
more sequences are processed concurrently.  For agent workloads where the homo
pool must support 65 K context, the n_slots advantage (16 → 64 at max_ctx=16 K)
drives a 13 % GPU saving.

If `mean_seq_len` is omitted (backward-compatible call), `H_eff = H`, which
assumes every active sequence is `calibration_ctx` tokens long.  This gives
correct results only when the pool's mean request length matches the
calibration context.  For pools serving requests significantly shorter than
`calibration_ctx`, always pass `mean_seq_len` so `iter_t` reflects actual
attention work.

### Pre-built H100 profile constants

```
W = 0.004 s   (4 ms base iteration — dominated by model weight memory read)
H = 0.00032 s/seq at calibration_ctx = 8 192 tokens (Llama-3-70B, TP8, BF16)
```

These were fitted to microbenchmark data; see `ProfileBuilder` for computing
them from first principles for any GPU + model combination.

### Assumptions and limitations

| Assumption | When it breaks down |
|---|---|
| Attention cost ∝ seq_len (linear) | Quadratic attention kernels (no FlashAttention) or very long sequences (>32K) with sub-linear kernels |
| `mean_seq_len` of active batch = request's own seq_len (analytical model) | Mixed batches in the homo pool: iter_t is determined by the mix, not by one request's length.  The analytical model is conservative. |
| Constant W across batch sizes | At very small batches (n_active < 4), kernel launch overhead becomes a larger fraction of W |
| Linear n_active scaling for H | When n_active exceeds the GPU's compute or memory bandwidth saturation point, H increases non-linearly |

---

## 2. KV-Cache Slot Model

### Formula

Each GPU in a pool has two concurrent-sequence limits that must both be
respected:

```
kv_limit     = total_kv_blks ÷ ⌈max_ctx / blk_size⌉   (KV-cache memory)
compute_cap  = max_slots × calibration_ctx ÷ max_ctx    (attention bandwidth)

n_slots      = min(kv_limit, compute_cap)
```

| Symbol | Meaning | A100_80GB value |
|---|---|---|
| `total_kv_blks` | Total PagedAttention blocks on the GPU | 65 536 |
| `blk_size` | Tokens per KV block | 16 |
| `max_ctx` | Pool's maximum context length | configured per pool |
| `max_slots` | Attention-bandwidth saturation limit at `calibration_ctx` | 128 |
| `calibration_ctx` | Context length at which `max_slots` was measured | 8 192 |

`max_slots` is the scheduler limit measured when the GPU first saturates its
memory-bandwidth budget at `calibration_ctx`.  At shorter `max_ctx` the same
bandwidth budget supports proportionally more concurrent sequences, so the
compute cap scales inversely with `max_ctx`.  Both limits are equal at
`max_ctx = calibration_ctx` by construction of the profile constants.

**A100-80 GB examples:**

| max_ctx | kv_limit | compute_cap | n_slots |
|---|---|---|---|
| 2 048 | 512 | 512 | **512** |
| 4 096 | 256 | 256 | **256** |
| 8 192 | 128 | 128 | **128** (calibration point) |
| 16 384 | 64 | 64 | **64** |
| 65 536 | 16 | 16 | **16** |

This scaling is the primary source of two-pool efficiency: a GPU configured
for `max_ctx = 4 096` can run 256 concurrent sequences — 2× more than at
`max_ctx = 8 192` — while the attention cost per iteration scales down by the
same factor, keeping total GPU utilisation unchanged.

### KV block admission gate

Before admitting a new request, the DES checks the block budget:

```
blocks_needed = ⌈(l_in + l_out) / blk_size⌉
if used_blocks + blocks_needed > total_kv_blks:
    → preempt longest active request (re-queue at head)
```

This models PagedAttention's block-level allocation and prevents OOM errors
that would occur if requests were admitted without checking KV capacity.

### Assumptions

| Assumption | When it breaks down |
|---|---|
| Worst-case block allocation: `l_in + l_out` blocks per request at admission | With speculative decoding or early stopping, actual output length < `l_out`. Use conservative `l_out` estimates. |
| Fixed `total_kv_blks` | Disaggregated serving (prefill and decode on separate GPUs) has different per-GPU block budgets |
| PagedAttention block granularity = 16 tokens | vLLM 0.4+ defaults to block_size=16; adjust via `ManualProfile(blk_size=...)` |

---

## 3. Request Service Time Model

### Prefill vs decode cost

Prefill and decode have fundamentally different performance characteristics:

| Phase | Bottleneck | Cost model |
|---|---|---|
| **Decode** | Memory bandwidth (KV cache reads, weight streaming) | `W + H_eff × n_active` |
| **Prefill** | Compute for large chunks; memory-bound for small | `max(compute_bound, memory_bound)` |

For a prefill chunk of `c` tokens attending to `q` tokens already in KV cache,
FlashAttention FLOPs ≈ `4 × n_heads × head_dim × c × (c/2 + q) × n_layers / tp`.
This can exceed the memory-bandwidth ceiling on high-FLOP/byte GPUs (H100, B200):

```
arithmetic_intensity = 4 × c × q × n_heads / ((2c + 2q) × n_kv_heads)
ridge_point = fp16_tflops / mem_bw_effective

if arithmetic_intensity > ridge_point → compute-bound
else                                  → memory-bound
```

**Example** — LLaMA-3-70B (n_heads=64, n_kv_heads=8, head_dim=128) on H100
(ridge ≈ 295 ops/byte), `chunk=512`, `q=2048`:
```
intensity ≈ 4 × 512 × 2048 × 64 / ((1024 + 4096) × 8) ≈ 16384 ops/byte
```
Far above the ridge → **compute-bound** for large-model prefill.  Our simulator
uses `ComputedProfile.prefill_iter_latency()` to take the maximum.  `ManualProfile`
falls back to the memory-bound cost since it has no compute-throughput spec.

### Service time (physical)

The DES computes one physical (wall-clock) service time per request:

```
S_raw = prefill_iters × prefill_iter_t(n_active)
      + l_out        × decode_iter_t(n_active)
```
where `prefill_iter_t = max(compute_bound, mem_bound)` and
`decode_iter_t = W + H_eff × n_active`.

`S_raw` is the time from when a KV-slot is granted until the last decode
token is produced.  It is also the quantity used to compute throughput in the
analytical model:

```
μ_gpu = n_slots / E[S_raw]     (requests per second per GPU at full concurrency)
cv²   = Var[S_raw] / E[S_raw]² (service-time coefficient of variation squared)
```

**TTFT:**
```
TTFT = slot_wait + prefill_iters × prefill_iter_t(n_active)
```

where `slot_wait` is the time a request queues before a KV-cache slot is
available.  This is the quantity whose P99 is measured against the TTFT SLO.

**TPOT** (time per output token):
```
TPOT = (physical_end_time − first_token_time) / (l_out − 1)    [for l_out > 1]
```

---

## 4. Preemption Model

When a new request's KV block requirement would exceed the GPU's budget, the
longest currently-active request is preempted:

1. Invalidate the victim's pending completion event (`req.preempted = True`).
2. Release victim's KV blocks and slot.
3. Re-queue the victim at the **head of the queue** (priority re-admission).
4. Retry admission for the new request.

This models vLLM's swap-out preemption strategy.  The victim is chosen by
total context length (l_in + l_out) as a proxy for KV memory pressure.

**Metrics exposed**: `request.preempted` flag; `instance.total_preempted` counter.

### Limitations

| Limitation | Impact |
|---|---|
| Re-queued victim restarts from scratch (no partial KV cache saved) | Overstates preemption cost; real vLLM can swap KV to CPU DRAM |
| Preemption victim = longest request (greedy) | Real vLLM uses a more sophisticated eviction policy |
| No multi-step preemption (preemption stops when budget is met) | Multiple concurrent preemptions in a single step are handled iteratively |

---

## 5. Analytical Fleet Sizing (Erlang-C / Kimura)

The optimizer uses a closed-form approximation to find the minimum GPU count
before running the slower DES.

### Flow

```
CDF  →  _calibrate(gpu, max_ctx, alpha)
         ↓
    (μ_gpu, cv², n_slots, mean_prefill)
         ↓
    _min_gpus_analytical(λ_eff, μ_gpu, cv², slo_budget, n_slots)
         ↓
    n_gpus  [minimum GPUs satisfying P99_TTFT = P99_wait + mean_prefill ≤ SLO]
```

### `_calibrate`

Samples `N` requests from the workload CDF (filtered to the pool's traffic
fraction α), computes `service_time(l_in, l_out, max_ctx)` for each, and
returns:

```
μ_gpu        = n_slots / E[S_raw]      (GPU-level throughput, req/s per GPU)
cv²          = Var[S_raw] / E[S_raw]²  (service-time variability)
n_slots      = gpu.n_slots(max_ctx)    (KV-cache concurrency per GPU)
mean_prefill = E[ceil(l_in/chunk) × iter_t(1)]   (mean prefill time)
```

### `_min_gpus_analytical`

Uses the **slot-level Kimura M/G/c approximation** for P99 queue wait.
Each GPU has `n_slots` KV-cache slots that act as parallel servers.
The effective server pool has `c_slots = n_gpus × n_slots` servers at
per-slot rate `μ_slot = μ_gpu / n_slots`:

```
ρ       = λ / (n_gpus × μ_gpu)          (GPU utilisation, same as slot utilisation)
a_slots = λ / μ_slot                    (Erlang load in slot-units)
C       = Erlang-C(n_gpus × n_slots, a_slots)  (slot-level blocking probability)
P99_wait = −ln(C / 0.01) / decay        (Kimura tail formula)
P99_TTFT = P99_wait + mean_prefill       (total TTFT)
```

**Why slot-level matters**: treating each GPU as a single server inflates the
P99 estimate for high-concurrency profiles (e.g., n_slots=128 for H100 at
max_ctx=8192).  The correct Erlang-C formula uses `c × n_slots` servers, which
gives near-zero Erlang-C when c is much larger than the Erlang load `a_slots/n_slots`.

**Stability constraint:** ρ_max = 0.85.  Above this, the Erlang-C approximation
becomes inaccurate and queue wait diverges non-linearly.

### Reliability-aware sizing (`node_avail`)

GPU and NVLink hardware fails at a measurable rate.  Kokolis et al. 2024
(arXiv:2410.21680) report for Meta's RSC clusters:

| Cluster | r_f (failures / node-day) | MTTF (1 node) | MTTF (1024-GPU job) |
|---|---|---|---|
| RSC-1 (16k A100, general ML) | 6.50 / 1000 = 0.0065 | ~154 days | ~7.9 h |
| RSC-2 (8k A100, vision) | 2.34 / 1000 = 0.00234 | ~427 days | ~22 h |

For an **inference fleet**, individual node failures reduce effective capacity.
At steady state, a fraction `(1 − A)` of nodes are under repair, where:

```
A = MTTF / (MTTF + MTTR)  =  1 / (1 + r_f × MTTR_days)

MTTR=4h  (driver reset / health-check quarantine): RSC-1 A100 → A ≈ 99.89%
MTTR=48h (GPU / NVLink hardware swap):             RSC-1 A100 → A ≈ 98.71%
H100 at scale (5% overprovisioning rule, Cui et al. 2025):   A = 0.95
```

**MTTR is failure-type-driven, not GPU-model-driven** (Cui et al. 2025, arXiv:2503.11901):

| Failure type | Repair path | MTTR | A100 vs H100 |
|---|---|---|---|
| Soft (driver/GSP/ECC recoverable) | Driver reset / node reboot | ~1–4 h | Same |
| Medium (health-check, reimaging) | Quarantine + OS reinstall | ~4–24 h | Same |
| Hard (HBM uncorrectable, NVLink die, PCIe) | Physical HGX/GPU swap by vendor | ~24–72 h | Same (both SXM, vendor-swapped) |

What **does** differ between A100 and H100 is the **failure rate r_f**:
- H100 has **3.2× more memory MTBE** (more soft ECC events, short MTTR) but **better critical-hardware resilience** (fewer hard failures, long MTTR).
- Net practical rule: **~5% overprovisioning for H100 at scale** → use `H100_AVAIL_5PCT = 0.95` constant.

The optimizer applies a **reliability margin** by inflating the raw SLO-sized
GPU count:

```
n_provisioned = ceil(n_for_slo / A)
```

**Important distinction** from the `ρ_max=0.85` utilisation cap:

| Parameter | Purpose | Typical magnitude |
|---|---|---|
| `ρ_max=0.85` | Keep Erlang-C approximation accurate; prevent runaway queue near saturation | 15% over-provision |
| `node_avail` | Cover nodes under active repair so SLO still holds | 0.1–1.3% over-provision |

They are independent and multiplicative.  For most inference workloads the
existing `ρ_max` already absorbs the availability margin; `node_avail` is
useful when modelling explicit reliability SLOs (e.g., "fleet must maintain
P99 TTFT even if 2% of nodes are simultaneously being repaired").

**API usage:**

```python
from fleet_sim import FleetOptimizer, node_availability

# RSC-1 failure rate, 48-hour GPU swap MTTR
A = node_availability(r_f_per_node_day=0.0065, mttr_hours=48)  # → 0.9871

opt = FleetOptimizer(
    gpu_short=A100_80GB,
    B_short=8192,
    t_slo_ms=500,
    node_avail=A,
)
```

### Assumptions

| Assumption | When it breaks down |
|---|---|
| Poisson arrivals (M/G/c) | Bursty traffic (over-dispersed), correlated arrivals (time-of-day patterns), or batch job arrivals |
| Service times are i.i.d. across requests | Mixed workloads where long requests cluster (e.g., batch jobs); consider separate pools |
| P99 wait has exponential tail | Heavy-tailed service distributions (very long documents); P99 may be underestimated |
| Slot-level queuing (M/G/c, c = n_gpus × n_slots) | Inter-GPU load imbalance; real systems may not spread requests evenly across all GPU slots |

---

## 6. Routing Algorithms

### LengthRouter

Routes each request to the pool whose `max_ctx` is the smallest value ≥ the
request's total token count (`l_in + l_out`).

```
if total ≤ threshold: → short pool
else:                 → long pool
```

**Use when**: production deployment, tight SLOs.  Zero overhead.

### SpilloverRouter

Length-based primary routing with load-aware overflow from short to long pool.

```
if total > threshold:                     → long pool  (context constraint)
elif short_pool_pressure ≥ spill_thr:     → long pool  (overflow)
else:                                     → short pool
```

`pool_pressure = (active + queued) / n_gpus`.  Default `spill_threshold = 2.0`
req/GPU.

**Key parameter:** `spill_threshold`

| Value | Behaviour |
|---|---|
| 1.0 | Aggressive spillover — spills as soon as queue forms |
| 2.0 | Moderate — spills only under sustained load |
| 5.0 | Conservative — short pool must be heavily congested before spilling |

**When to use:** workloads with < 5 % long requests (e.g. LMSYS).  Allows the
long pool to be sized only for the long traffic (1–2 GPUs), using idle long-pool
capacity for short-request bursts.

### CompressAndRouteRouter (C+R)

Requests with `threshold < total ≤ γ × threshold` are compressed (prompt
shortened) and routed to the short pool.  Compression is modeled as:
- Effective output length reduced by `1/γ`.
- The short pool's effective traffic fraction α increases by the compressed
  borderline fraction.

**Use when:** offline sizing / optimizer γ sweep only.  In live simulation it
adds 33 % to P99 TTFT compared to LengthRouter.  Never deploy C+R as a runtime
router for latency-sensitive workloads.

### LeastLoadedRouter

Routes to the pool with the lowest `(active + queued) / total_slots` ratio.

**Use when:** homogeneous multi-pool fleet, load balancing across identical
pools.

---

## 7. Workload-Derived Threshold Selection (Pareto Sweep)

Choosing `B_short` by gut-feel is unreliable because the optimal split
depends jointly on the workload CDF shape, arrival rate, and SLO target.
`threshold_pareto()` (exposed as `run_sim.py pareto`) automates this.

### Algorithm

```
candidates = all CDF breakpoints where 1% ≤ α(B_short) ≤ 99.9%

for each candidate B_short:
    size fleet analytically at gamma=1 (pure length routing)
    record: n_s, n_l, total_gpus, cost, P99-short, P99-long

homo_baseline = FleetOptimizer(B_short=long_max_ctx)   # single pool

for each result r:
    r.savings = (homo_cost - r.cost) / homo_cost * 100
    r.worst_p99 = max(r.p99_short, r.p99_long)
    r.pareto = not any(other.cost < r.cost and other.worst_p99 < r.worst_p99)
```

A point is Pareto-dominated when another threshold achieves **both** strictly
lower fleet cost and strictly lower worst-case P99.  The Pareto-optimal set
defines the trade-off curve between cost and latency.

### Recommended threshold

`print_threshold_pareto()` recommends the cheapest SLO-meeting Pareto-optimal
threshold.  This is not always the absolute minimum-cost point — it may be a
cheaper point whose worst-case P99 is _also_ lower than even cheaper
(but Pareto-dominated) points.

### Calibration fix for accurate savings

`_calibrate()` uses `gpu.service_time(l_in, l_out, pool_max)` (seq-len-aware)
rather than a fixed `iter_t`.  This ensures the M/G/c service-time distribution
correctly reflects the attention cost of the actual sequence lengths in each CDF
slice, not just the worst-case `max_ctx`.

---

## 8. Disaggregated Fleet Sizing

`DisaggFleetOptimizer` (in `optimizer/disagg.py`) models a system where
prefill (context-ingestion) and decode (token-generation) run on separate,
independently-scaled GPU pools.  It is exposed via `run_sim.py disagg`.

### Model

For a fleet of `nP` prefill workers and `nD` decode workers:

```
r_pre  = thru_prefill_single × nP × α_pre   (effective prefill capacity, req/s)
r_dec  = thru_decode_single  × nD × α_dec   (effective decode capacity, req/s)
r_sys  = min(r_pre, r_dec)                  (bottleneck-bound system rate)

TTFT_eff = prefill_base_ms × β_TTFT         (includes KV-transfer overhead)
TPOT     = decode_profile.iter_latency(n_slots) × 1000
```

The optimizer sweeps all `(nP, nD)` pairs and returns the Pareto-efficient set
of configurations that satisfy `r_sys ≥ λ` and `TTFT_eff ≤ SLO_TTFT` and
`TPOT ≤ SLO_TPOT`.  The minimum-cost configuration is recommended.

### Empirical constants

| Constant | Default | Meaning |
|---|---|---|
| `α_pre` | 0.90 | Prefill throughput degradation from pipeline interference |
| `α_dec` | 0.92 | Decode throughput degradation from pipeline interference |
| `β_TTFT` | 1.80 | TTFT multiplier for KV-cache transfer overhead |

Override when you have measured values: `DisaggFleetOptimizer(alpha_pre=..., beta_ttft=...)`.

### When disagg is worth the complexity

Disagg reduces cost by ~35–46% vs aggregated serving at λ ≥ 100 req/s
(Azure workload, max_ctx=8192), at the expense of higher TTFT (~1.8× raw
prefill time) and additional operational complexity (two scaling policies,
KV-transfer networking).  Below ~50 req/s the cost saving is too small to
justify the overhead.

---

## 9. DES Event Flow

```
PoissonWorkload.generate()
    → list of (arrival_time, Request)

Fleet.run(arrivals):
    for each arrival in time order:
        router.route(req)  → pool_id
        pool.accept(req)   → Instance (least-loaded in pool)

    # Advance simulation time:
    heap-merge all Instance.next_event_time()
    advance to next event:
        Instance.advance_to(t):
            while queue not empty and slots available:
                _start_next(now)           # compute S_raw, fire completion event
            process all completions at this time:
                release slot + KV blocks
                fill freed slot from queue (_start_next)
```

### Warmup period

The DES discards requests arriving in the first `warmup_fraction` (default 20 %)
of the simulation time.  This removes the cold-start transient where the queue
is building up from empty.

Recommended: `n_sim_req ≥ 15 000` to leave at least 12 000 post-warmup requests
for stable P99 estimates.

---

## 10. Profile Tuning

### Calibrating W and H from benchmarks

Run a benchmark that measures iteration latency at varying batch sizes and
two context lengths, then fit:

```python
# Two-point fit for H at calibration_ctx
iter_t_n1 = measured at n_active = n1
iter_t_n2 = measured at n_active = n2

H = (iter_t_n2 - iter_t_n1) / (n2 - n1)
W = iter_t_n1 - H * n1
```

Set `calibration_ctx` to the context length at which the benchmark was run.

### Choosing `blk_size`

vLLM defaults to 16 tokens per block.  Larger block sizes reduce block-table
overhead but waste memory for short sequences.  Most production deployments
use 16–32; set `ManualProfile(blk_size=...)` to match.

### Choosing `max_slots`

`max_slots` is the attention-bandwidth saturation limit at `calibration_ctx`.
It corresponds to the vLLM `--max-num-seqs` setting you plan to deploy with,
measured (or estimated) when the GPU begins saturating at the calibration
context length.

The simulator automatically scales this limit for shorter contexts
(`compute_cap = max_slots × calibration_ctx / max_ctx`), so you only need
to provide the single calibration-point value.  A100-80GB default: 128 at
8 192 tokens; H100-80GB default: 256 at 8 192 tokens.

### Disaggregated serving constants

`DisaggFleetOptimizer` uses empirical degradation factors.  Override defaults
when you have measured values for your deployment:

| Constant | Default | Meaning | Override via |
|---|---|---|---|
| `alpha_pre` | 0.90 | Prefill throughput fraction vs monolithic | `DisaggFleetOptimizer(alpha_pre=...)` |
| `alpha_dec` | 0.92 | Decode throughput fraction vs monolithic | `DisaggFleetOptimizer(alpha_dec=...)` |
| `beta_ttft` | 1.80 | KV-transfer TTFT multiplier | `DisaggFleetOptimizer(beta_ttft=...)` |

These reflect the overhead of KV cache transfer between prefill and decode
workers.  Measure them with a representative trace using your specific
network fabric and KV compression settings.

### ProfileBuilder constants (`computed.py`)

`ProfileBuilder` derives `W` and `H` from first principles:

```
W = model_bytes_per_gpu / (mem_bw × alpha_bw) + n_layers × layer_overhead_s
H = kv_bytes_per_token_per_gpu / (mem_bw × alpha_bw) × calibration_ctx
```

| Constant | Default | Meaning |
|---|---|---|
| `alpha_bw` | 0.80 | Effective bandwidth fraction (sustained vs peak) |
| `layer_overhead_s` | 3 × 10⁻⁶ s | Per-layer kernel launch + sync overhead |
| `gpu_memory_utilization` | 0.90 | Fraction of GPU memory committed to model + KV cache |

`gpu_memory_utilization` matches vLLM's `--gpu-memory-utilization` parameter
(default 0.90).  The remaining 10 % reserves headroom for:
- **Peak activation memory** (2–8 GB for large models): vLLM profiles this with
  an actual forward pass; we must estimate it statically.
- **CUDA graph capture buffers** (~0.5–2 GB).
- Other runtime allocations not in the static roofline model.

The formula used:
```
committed   = mem_capacity × gpu_memory_utilization   (e.g. 0.90 × 80 GiB = 72 GiB)
kv_budget   = committed − model_bytes_per_gpu − nccl_overhead
kv_blks     = kv_budget ÷ kv_bytes_per_blk
```

Note that `HardwareSpec.free_vram()` uses the full `mem_capacity` minus a
fixed `other_mem = 3.5 GiB` safety buffer; `ProfileBuilder` overrides this
with the utilization-based cap above.

These are embedded in `builder.py` and reflect typical measured values for
production transformer inference.  Adjust them if your deployment achieves
different sustained bandwidth (e.g. FP8 kernels often reach 85–90 % of peak).

---

## 11. Comparison with vLLM and Vidur

This section documents what we learned from reviewing the vLLM v0.7+ source
(`vllm/v1/`) and the Vidur simulator (Microsoft Research, 2024), and how those
insights shaped — or were already reflected in — this simulator.

### vLLM v1 (production inference engine)

| vLLM behaviour | Our model | Gap / notes |
|---|---|---|
| `gpu_memory_utilization = 0.90` applied before KV allocation | `ServingConfig.gpu_memory_utilization = 0.90` added to `ProfileBuilder` | ✅ Now aligned |
| KV blocks = `free_vram // (page_size × n_layers)` | Same formula; `page_size = blk_size × kv_bytes_per_token` | ✅ Equivalent |
| Activation memory profiled live (~2–8 GB) | Covered by 10 % utilization reserve | Approximate — use `gpu_memory_utilization` to tune |
| CUDA graph memory included in non-KV overhead | Included in utilization reserve | Same caveat |
| NCCL workspace by TP size (hard-coded bytes) | `HardwareSpec.nccl_mem` dict, same values | ✅ Aligned |
| 150 MiB extra OOM-avoidance buffer | Our `other_mem = 3.5 GiB` covers more conservatively | Slightly more conservative |
| Separate prefill attention (`attn_prefill`) and decode attention (`attn_decode`) FLOPs | `prefill_iter_latency()` uses roofline check: `max(compute_bound, mem_bound)` | ✅ Now aligned for `ComputedProfile` |
| TP all_reduce latency (`2*(tp-1)/tp * hidden * batch / nvlink_bw`) | Not modeled explicitly; absorbed into empirical `W` for `ManualProfile` | Gap for `ComputedProfile` at TP>1 — see §11.1 |
| CPU scheduling overhead (1–5 ms/iteration) | Not modeled | Gap — see §11.2 |
| GQA batching overhead (+10% when batch > 1, from Vidur) | Not modeled | Gap — see §11.3 |
| Prefix caching / LRU eviction | Not modeled | Fleet sizing impact: KV capacity appears larger if cache hit rate is high |

### Vidur (Microsoft Research DES simulator)

Vidur takes an **empirical profiling** approach rather than a roofline model:
it trains sklearn ML models on measured CSV kernel-timing tables
(`mlp.csv`, `attention.csv`, `all_reduce.csv`) for each GPU × model combination.

| Aspect | Vidur | Our simulator | Comparison |
|---|---|---|---|
| Decode W | Empirical MLP table (per batch size) | `model_bytes / eff_bw + n_layers × 3 µs` | Vidur is more accurate for non-linear batch effects; our model is portable without profiling |
| Decode H | Empirical attention table `f(batch, kv_size)` | `kv_bytes / eff_bw` (linear) | Vidur captures non-linear GQA overhead; we approximate |
| Prefill attention | Empirical `f(kv_cache_size, chunk²)` | `max(attn_flops / tflops, W+H*n)` | Both model the quadratic cost; Vidur uses measured data |
| Memory bandwidth | **Not modeled** (only TFLOPS + capacity) | ✅ Explicit `mem_bw` roofline | Our model is more physically grounded for memory-bound decode |
| MoE models | **Not supported** | ✅ `_MOE_TABLE` in `builder.py` | Clear advantage over Vidur |
| Portability | Requires real hardware profiles (CSV files) | ✅ Works from hardware spec alone | Better for planning new hardware before deployment |
| Accuracy | Higher (measured kernels) | Lower (analytical model) | Trade-off: accuracy vs portability |

### 11.1 TP all_reduce overhead (gap)

For TP > 1, each decode step requires an all_reduce of the output activations
across TP ranks.  The latency scales with batch size and hidden dimension:

```
t_allreduce_per_step ≈ 2 × (tp−1)/tp × n_active × hidden_size × dtype_bytes × n_layers / nvlink_bw
```

For H100 TP=8, `n_active=128`, LLaMA-3-70B: ≈ 1.3 ms per step.  Against a
W of 6.5 ms, this is ~20 % overhead that `ComputedProfile` currently omits.

**Workaround**: lower `gpu_memory_utilization` or increase `W` manually to
account for all_reduce.  A future `ProfileBuilder._compute_H_allreduce()`
method could add this term to `H` automatically.

### 11.2 CPU scheduling overhead (gap)

Vidur optionally models scheduling, sampling, and `prepare_inputs` CPU time,
typically 1–3 ms per iteration.  This is ~10–30 % of W for H100 at small
batch sizes.

**Workaround**: add the expected CPU overhead to `ManualProfile.W` when
calibrating from production traces.

### 11.3 GQA batching overhead (gap)

Vidur applies a +10 % overhead (`attention_decode_batching_overhead_fraction`)
for GQA decode when batch size > 1.  This captures the extra indexing work
when Q-heads are grouped and K/V heads are shared.  Most modern models
(LLaMA-3, Qwen-3, Mistral) use GQA.

**Impact**: ~10 % systematic underestimate of `H` for GQA models, resulting
in slight under-sizing when using `ComputedProfile`.

**Workaround**: multiply the computed `H` by 1.10 when using `ManualProfile`:
```python
H_calibrated = H_measured × 1.10   # apply for GQA models
```

---

## 12. Known Limitations

| Limitation | Effect | Workaround |
|---|---|---|
| `ManualProfile.prefill_iter_latency` falls back to memory-bound cost | Overestimates TTFT for large-model prompt-heavy workloads | Use `ComputedProfile` (needs HardwareSpec + ModelSpec) for accurate prefill modeling |
| TP all_reduce not modeled in `ComputedProfile` | W underestimated by ~10–20 % at TP ≥ 4 | Add measured all_reduce overhead to `ManualProfile.W` |
| CPU scheduling overhead absent | iter_t underestimated by 1–5 ms | Add expected CPU overhead to `ManualProfile.W` |
| GQA batching overhead absent | H underestimated by ~10 % for GQA models | Multiply measured H by 1.10 for GQA models |
| Activation memory estimated by utilization fraction | KV capacity may be slightly off for unusual batch sizes | Tune `ServingConfig.gpu_memory_utilization` from profiling |
| Static traffic (Poisson) | Real traffic has diurnal patterns and correlated bursts | Run `whatif` with peak-hour λ; use `SageServe` for 24h scaling |
| Poisson sub-stream approximation (pool routing / C&R) | The optimizer splits λ into `λ_s = α·λ` and `λ_l = (1−α)·λ` and treats both as independent Poisson streams.  Poisson thinning only preserves the Poisson property under *independent* per-arrival coin-flips; length-based routing is deterministic on `L_total`, so the sub-streams are correlated and not strictly Poisson.  Analytical queue lengths and P99 TTFT are an approximation.  The DES verification step provides an empirical check; for workloads with strong temporal autocorrelation in request length (e.g., sessions that alternate between very short and very long requests), the error can be material. | Use the DES path (`_run_simulate_pool`) instead of the analytical path; or apply an autocorrelation correction to the effective arrival rate per pool. |
| No speculative decoding | Throughput underestimated by 20–50 % for short outputs | Scale μ upward by your measured speculative-decode speedup |
| No continuous batching across requests | Each request's service time computed at arrival | Impact is small at >50 % utilisation where batches are consistently full |
| Single-pool Erlang-C | Inter-pool load balancing (LeastLoadedRouter) not captured analytically | Use DES for topologies with more than 2 pools |
| Preemption restarts from scratch | Overstates preemption latency vs vLLM swap-in | Acceptable for planning; measure preemption rate in production |

---

## 13. GPU Power Model, Energy Efficiency, and Grid Flexibility

### 13.1 Two power model approaches

The simulator uses two complementary power models:

#### 13.1.1 First-principles model (`ComputedProfile.power_at_concurrency`)

Derived from the same W / H roofline decomposition used for latency.
No additional fitting parameters beyond two empirical TDP fractions.

**Formula:**
```
kv_frac(n)      = n × H_eff / (W + n × H_eff)
                  KV-cache fraction of per-iteration HBM traffic

compute_frac(n) = 2 × active_params/tp × n / (iter_latency(n) × fp16_tc_flops)
                  Fraction of peak tensor-core FLOP throughput

activity(n)     = min(1.0,  kv_frac(n) + compute_frac(n))
power(n)        = hw.power × (0.43 + (0.86 − 0.43) × activity(n))
```

The two constants (0.43 idle fraction, 0.86 active fraction) are from ML.ENERGY
v3.0 (H100-SXM5, batch=1 → 300 W and batch=128 → 600 W). They transfer to other
HBM GPUs in the same bandwidth-bound decode regime.

**Why two terms?** `kv_frac` captures increasing DRAM activity as KV-cache reads
grow with batch. `compute_frac` captures increasing SM activity from larger
weight matmuls per second. Together they account for the full measured power
range with ~15–20% accuracy at high batch (remaining error: NVLink all-reduce
power and dynamic voltage scaling not in the FLOP count).

**Model dependency:** W, H, and `active_params` all depend on the served model.
The same GPU has a different power curve — and different tok/W — for different
model sizes. See `POWER_MODEL_METHODOLOGY.md` §Part 1 for the full derivation
and validation table.

#### 13.1.2 Empirical logistic model (`ManualProfile.power_at_concurrency`)

Each `ManualProfile` stores four logistic parameters fitted to silicon data:
```
P(b) = P_range / (1 + exp(-k × (log₂(b) − x0))) + P_idle
```

| Parameter | H100-SXM5 | A100-SXM4 | A10G | Quality |
|---|---|---|---|---|
| `power_idle_w`    | 300 W | 175 W | 75 W  | H=meas / F=proj / L=proj |
| `power_nominal_w` | 600 W | 385 W | 120 W | H=meas / F=derived / L=proj |
| `power_logistic_k` | 1.0  | 1.0   | 0.7  | H=fitted / F=proj / L=proj |
| `power_logistic_x0` | 4.2 | 2.5   | 3.7  | H=fitted / F=scaled / L=proj |

Quality: **HIGH** (H100, measured ML.ENERGY + G2G data) /
**FAIR** (A100, one anchor + FLOPS-scaling derivation) /
**LOW** (A10G, projection only, no published LLM batch-vs-power data).

See `POWER_MODEL_METHODOLOGY.md` §Part 2 for full derivation of each value.

#### 13.1.3 When to use which model

| Scenario | Recommended model |
|---|---|
| New GPU, no published measurements | `ComputedProfile` first-principles |
| New model on existing GPU | `ComputedProfile` first-principles |
| H100 + any Llama-class model | Either; logistic is more accurate |
| A100 or A10G | Logistic (FAIR/LOW) — but verify in-house |
| Production DR contract (±5% matters) | Logistic with your own calibration |
| Paper / planning comparison | `ComputedProfile` (portable, auditable) |

### 13.2 `grid_flex_analysis()` — power–latency trade-off

`grid_flex_analysis(cdf, lam, n_gpus, gpu, t_slo_ms, max_ctx, flex_pcts)` models
the GPU-to-Grid (G2G) batch-size control mechanism from Hassan et al.
(arXiv:2602.05116v1, 2025):

**Mechanism:** When the grid signals high demand, the serving engine caps
`max_num_seqs` (≈ `n_max_cap` in the simulator). Fewer concurrent requests
reduces memory-bandwidth pressure and therefore GPU power. The trade-off is
longer queuing delays.

**Algorithm for each `flex_pct` in the sweep:**

1. Compute target power: `P_target = power_nominal_w × (1 − flex_pct/100)`,
   clamped to `power_idle_w`.
2. Invert linear power model: `n_max_cap = max_slots × (P_target − P_idle) / (P_nom − P_idle)`.
3. Run M/G/c analytical P99 TTFT with `c_slots = n_gpus × n_max_cap`.
4. Report `(flex_pct, n_max_cap, power_per_gpu_w, power_fleet_kw, p99_ttft_ms, slo_met)`.

**Key finding from representative workload (30 H100s, λ=300 req/s, SLO=200 ms):**

| Flex | n_max | W/GPU | Fleet kW | P99 TTFT | SLO |
|---|---|---|---|---|---|
| 0% | 128 | 450 W | 13.5 kW | 7.9 ms | ✓ |
| 20% | 77 | 390 W | 11.7 kW | 7.9 ms | ✓ |
| 25% | 64 | 375 W | 11.2 kW | 1058 ms | ✗ |

The safe flex depth has a hard cliff at the Erlang-C saturation threshold
(`arrival_load ≥ n_max_cap`). The simulator locates this cliff analytically
before any hardware change.

### 13.3 CLI usage

```bash
python3 run_sim.py grid-flex \
    --cdf data/azure_cdf.json \
    --lam 300 --n-gpus 30 --gpu h100 --slo 200
```

---

## 14. Tokens-per-Watt Analysis

### 14.1 `DecodeEfficiencyPoint` — the full derivation chain

`ComputedProfile.decode_efficiency(n_active, mean_ctx)` returns a
`DecodeEfficiencyPoint` that exposes every intermediate quantity from hardware
datasheet → tok/W:

```
Step 1: iter_latency_s  = W + H_eff × n_active
Step 2: tokens_per_s    = n_active / iter_latency_s
Step 3: mem_bytes/iter  = model_bytes_per_gpu           ← weight streaming
                        + n × kv_bytes_per_token/tp × ctx  ← KV scanning
Step 4: kv_frac         = KV_bytes / mem_bytes_per_iter
Step 5: flops_per_iter  = 2 × active_params/tp × n      ← weight matmuls
                        + 4 × n_kv × head_dim × ctx × n × n_layers / tp  ← attention
Step 6: arith_intensity = flops / mem_bytes  [fl/byte]
        roofline_bound  = "memory" when AI < hw.fp16_tc_flops / hw.mem_bw
Step 7: compute_frac    = flops / (iter_latency × hw.fp16_tc_flops)
Step 8: activity        = min(1.0, kv_frac + compute_frac)
        power_w         = hw.power × (0.43 + 0.43 × activity)
Step 9: tokens_per_watt = tokens_per_s / power_w   [tok/J]
```

All inputs are from `HardwareSpec` (datasheet values) and `ModelSpec` (architecture
parameters).  No curve-fitting is performed.

**Usage:**
```python
from fleet_sim.hardware import H100_SXM
from fleet_sim.models import LLAMA_3_1_70B
from fleet_sim.gpu_profiles import ProfileBuilder, ServingConfig

profile = ProfileBuilder().build(
    hw=H100_SXM,
    model=LLAMA_3_1_70B,
    cfg=ServingConfig(tp=8, dtype_bytes=2.0, mean_ctx_tokens=4096),
)
pt = profile.decode_efficiency(n_active=32, mean_ctx=4096)
print(pt.show())   # prints the full derivation chain
```

### 14.2 `tpw_analysis()` — single-pool GPU comparison

Correct **only when comparing the same model on different hardware** (all GPUs receive
the same CDF and arrival rate, N cancels within each pool):

```
tok/W (single pool) = mu_gpu × ρ × mean_L_out / power(n_active)
                    = decode_tps(n) / power(n)        [independent of N]
```

```bash
python3 run_sim.py tok-per-watt \
    --cdf data/azure_cdf.json --lam 100 --slo 500 \
    --gpus h100 a100 --rho-sweep
```

**Warning:** pre-built `ManualProfile` instances (h100, a100, a10g) are calibrated for
different models (70B for h100/a100, 7B-class for a10g).  Comparing them with
`tpw_analysis()` mixes two different served models; the tok/W ratio reflects model-size
difference, not GPU efficiency alone.  Use `ComputedProfile` with a consistent `ModelSpec`
for a clean hardware comparison.

### 14.3 `fleet_tpw_analysis()` — multi-pool fleet

For hetero-pool routing (different GPU types per pool) or semantic-router topologies
(different models per pool), **N does not cancel across pools**:

```
fleet_tok/W = Σ_i(λ_i × mean_L_out_i) / Σ_i(N_i × power_i(n_i))
```

The CDF for each pool must be the normalised sub-CDF for that pool's traffic.
`_split_cdf(full_cdf, b_short)` produces the short and long sub-CDFs with
correct normalisation.

**CLI (two-pool routing comparison):**
```bash
# Compare homo H100 vs short-to-A10G + long-to-H100 routing
python3 run_sim.py tok-per-watt \
    --cdf data/azure_cdf.json --lam 100 --slo 500 \
    --b-short 4096 --gpu-short a10g --gpu-long h100
```

Output shows per-pool breakdowns and the correct fleet-level aggregate, plus
the routing benefit (Δtok/W and Δ$/1M tok vs homo baseline).

### 14.4 Correctness checklist

| Question | Correct approach |
|---|---|
| Which GPU is more energy-efficient for the same 70B model? | `tpw_analysis()` with `ComputedProfile` for both GPUs and the same `ModelSpec` |
| Does routing short requests to A10G (7B) + long to H100 (70B) save energy? | `fleet_tpw_analysis()` — N does not cancel, sub-CDFs must be used |
| What context length am I most efficient at? | `decode_efficiency(n, mean_ctx=X)` for a range of X |
| Is A10G better than H100 for my workload? | Cannot compare directly — different models. Use `fleet_tpw_analysis()` with matched routing scenario |
| How much does utilisation affect my energy bill? | `tpw_analysis(rho_sweep=[0.2, 0.4, 0.6, 0.8])` and plot the curve |
