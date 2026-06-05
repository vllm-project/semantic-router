---
title: Capacity Planning Scenarios
---

# Capacity Planning Scenarios

`vllm-sr-sim` answers fleet-planning questions that cannot be resolved
from first principles alone: where to set a split threshold, whether a fleet
will actually meet SLO under real queue dynamics, which GPU type is cheapest
for a given workload, and when to pre-provision the next tier.

GPU unit costs used throughout:

| GPU | $/hr | $/yr |
|---|---|---|
| A10G 24 GB | $1.01 | $8.85 K |
| A100 80 GB | $2.21 | $19.4 K |
| H100 80 GB | $4.02 | $35.2 K |

> **P99 TTFT** = P99(KV-slot queue wait) + mean prefill time.
> Each KV-cache slot is modelled as a server in an M/G/c queue.

---

## When to split pools — the short version

Before reaching for the simulator, apply this filter:

```
Heavy-tail service times (agent / long-context)?
  → Split required. Homo cannot meet SLO regardless of GPU count.

ctx ratio R = long_max_ctx / B_short  and  long-request fraction f:

  R ≤ 2×  or  f > 30%   →  homo usually cheaper; split for latency isolation only
  R ≥ 4×  and  f &lt; 10%  →  split cheaper at high traffic (λ > ~100 req/s)
  R ≥ 16× and  f &lt; 5%   →  split cheaper at any meaningful traffic level
```

Everything below is a puzzle the rule of thumb cannot solve on its own.

---

## Puzzle 1 — Where exactly should I split?

**Question:** the rule says "split" — but at which token threshold?

The optimal threshold depends entirely on the shape of your CDF.  Too low and
the long pool handles too much traffic; too high and the short pool's slot
advantage evaporates.  The `pareto` command sweeps every CDF breakpoint and
finds the cost–latency frontier.

```bash
vllm-sr-sim pareto \
  --cdf data/lmsys_cdf.json --lam 100 --slo 500 --long-max-ctx 65536
```

**LMSYS result** (λ=100, A100, homo baseline = $271K / 14 GPUs):

```
 B_short   α-short   n_s  n_l   GPUs      $/yr   saving    P99-s    P99-l   SLO  Pareto
---------------------------------------------------------------------------------------
     512    63.8%     2   13     15  $   290K    -7.1%       9ms      25ms     ✓       ★
   1,024    83.1%     2   10     12  $   232K   +14.3%      10ms      36ms     ✓       ★
   2,048    94.8%     2    7      9  $   174K   +35.7%      12ms      63ms     ✓       ★
   4,096    98.4%     3    5      8  $   155K   +42.9%      13ms     108ms     ✓       ★  ← optimal
   8,192    99.7%     4    4      8  $   155K   +42.9%      14ms     212ms     ✓       ★
  12,288    99.9%     5    3      8  $   155K   +42.9%      14ms     319ms     ✓       ★
```

**Insight:** B_short=4096 is optimal — 98% of LMSYS traffic fits below it, so
the short pool (256 KV slots at max_ctx=4096) is 16× more slot-efficient than
the homo pool (16 slots at max_ctx=65536).  Result: 14 GPUs → 8 GPUs, **−43%
cost**.  Choosing B_short=512 instead would route only 64% of traffic short,
leaving most requests in the expensive long pool and costing 7% *more* than homo.

**Azure result** (λ=200, A100, homo baseline = $465K / 24 GPUs):

The entire Azure CDF fits within 8192 tokens, so the max ctx ratio is only 2×.
Best Pareto point (B_short=3072) saves just 4% — the slot gain is too small to
overcome Erlang fragmentation.  The value here is **latency isolation**: the
short pool's P99 drops from 26 ms (homo) to 19 ms, useful for tiered SLAs.

**Agent result** (λ=200, A100, homo baseline = $9293K / 480 GPUs):

```
  B_short   saving    P99-s    P99-l   SLO
  16,384    +13.3%     69ms    339ms    ✓   ← optimal
  32,768    +12.9%     86ms    593ms    ✗   ← SLO FAIL: long-pool prefill dominates
```

B_short=16384 (64 KV slots vs 16 for homo) saves 64 GPUs.  Above 32768 the
long pool receives requests with 300–600 ms prefill times that bust the 500 ms
SLO budget entirely — **the P99 failure is caused by prefill cost, not queue
wait**, something only visible in a full simulation.

---

## Puzzle 2 — Why is my agent fleet failing SLO?

**Question:** 24 H100 GPUs at λ=20 req/s is only ~30% utilisation.
The analytics say the fleet is fine.  The DES says it is not.

```bash
# Homo baseline — looks feasible analytically
vllm-sr-sim optimize \
  --cdf data/agent_heavy_cdf.json --lam 20 --slo 1000 \
  --gpu-short h100 --gpu-long h100 \
  --b-short 65536 --long-max-ctx 65536

# Two-pool — the fix
vllm-sr-sim optimize \
  --cdf data/agent_heavy_cdf.json --lam 20 --slo 1000 \
  --gpu-short h100 --gpu-long h100 \
  --b-short 4096 --long-max-ctx 65536 --verify-top 3
```

| Config | GPUs | Cost/yr | P99 TTFT | SLO 1000ms |
|---|---|---|---|---|
| Homo 65K ctx | 24 | $845 K | **1 052 ms** | **✗ FAIL** |
| Two-pool 4K/65K | 25 | $880 K | 17ms / 147ms | ✓ |

**Why analytics miss this:** the M/G/c model assumes service times are drawn
i.i.d. from a distribution with finite variance.  Agent requests span
10–300 seconds of service time — a coefficient of variation cv²>>1.  A single
long request holds a KV slot for minutes, causing other requests to queue
behind it even when GPU utilisation appears low.  The DES replays the actual
arrival sequence and exposes these spikes; Erlang-C does not.

**Two-pool solves it** by routing the 46% of long requests (>4K tokens) to a
dedicated pool where their slow service time cannot block short requests.
Cost premium: +4% — essentially free insurance against SLO failure.

---

## Puzzle 3 — Which GPU type is actually cheapest?

**Question:** A10G is cheap per card but slow.  H100 is expensive per card
but fast.  Which is cheapest for a given workload?

```bash
for gpu in a10g a100 h100; do
  # homo baseline
  vllm-sr-sim optimize --cdf data/azure_cdf.json --lam 100 --slo 500 \
    --gpu-short $gpu --gpu-long $gpu --b-short 8192 --long-max-ctx 8192
  # two-pool
  vllm-sr-sim optimize --cdf data/azure_cdf.json --lam 100 --slo 500 \
    --gpu-short $gpu --gpu-long $gpu --long-max-ctx 8192 --verify-top 3
done
```

**Result (Azure, λ=100, SLO=500ms):**

| GPU | Layout | GPUs | Cost/yr | P99 TTFT |
|---|---|---|---|---|
| **A10G** | **Two-pool** | **19** | **$168 K ← cheapest** | 155ms / 335ms |
| H100 | Homo | 6 | $211 K | 26 ms |
| A100 | Two-pool | 12 | $232 K | 52ms / 112ms |
| H100 | Two-pool | 7 | $247 K | 13ms / 30ms |

**The non-obvious result:** A10G two-pool ($168K) is cheaper than H100 homo
($211K) — a slower GPU wins by using two-pool routing to compensate.  This
happens because the Azure ctx ratio (8192/4096 = 2×) doubles A10G's KV slots
from 64 to 128 per GPU, enough to offset its lower throughput.

**The decision depends on your constraint:**

| Priority | Choice |
|---|---|
| Minimum cost | A10G two-pool ($168K) |
| Minimum rack space / power | H100 homo (6 GPUs) |
| Best latency | H100 two-pool (13ms P99 short) |
| Long-context / agent | H100 or A100 — A10G's 24GB VRAM limits KV cache |

---

## Puzzle 4 — When do I need to add GPUs?

**Question:** traffic is growing — at which exact λ do I need to provision
the next GPU tier to avoid a reactive SLO violation?

```bash
vllm-sr-sim whatif \
  --cdf data/azure_cdf.json --slo 500 \
  --gpu-short h100 --gpu-long h100 --long-max-ctx 8192 \
  --lam-range 25 50 100 150 200 300 400
```

**GPU step thresholds (two-pool, H100):**

| λ (req/s) | GPUs | Cost/yr | Add GPUs before reaching… |
|---|---|---|---|
| 25 | 4 | $141 K | λ = 65 |
| 50 | 5 | $176 K | λ = 90 |
| 100 | 7 | $247 K | λ = 130 |
| 150 | 10 | $352 K | λ = 185 |
| 200 | 12 | $423 K | λ = 270 |
| 300 | 18 | $634 K | λ = 370 |
| 400 | 23 | $810 K | — |

**Insight:** GPU scaling is sub-linear — traffic grows 16× (25→400) but GPUs
grow only 5.75× (4→23).  Use this table to pre-provision *before* the step,
not after.  Waiting until SLO is already violated means at least one traffic
tier with degraded P99.

---

## Puzzle 5 — Which router causes SLO violations?

**Question:** you have sized the fleet correctly.  Does the router matter?

```bash
# Agent fleet — where the choice is consequential
vllm-sr-sim compare-routers \
  --cdf data/agent_heavy_cdf.json --lam 20 --slo 1000 \
  --gpu-short h100 --gpu-long h100 \
  --n-s 2 --n-l 23 --long-max-ctx 65536 --n-req 5000
```

**Agent fleet (λ=20, n_s=2, n_l=23):**

| Router | P99 TTFT | SLO 1000ms |
|---|---|---|
| LengthRouter | 495 ms | ✓ 99.98% |
| **CompressAndRoute** | **534 ms** | **✗ 99.94%** |
| RandomRouter | 292 ms | ✓ 100% |

**Two surprises:**

1. **CompressAndRoute violates SLO** despite being designed to reduce fleet
   size.  It compresses borderline-length requests and routes them to the short
   pool; when several arrive together they overwhelm the 2-GPU short pool and
   spike P99.  It is a *planning* tool — use it to discover a lower GPU count
   at sizing time, then deploy LengthRouter for production.

2. **RandomRouter passes SLO** at this fleet size because it spreads load
   across all 25 GPUs uniformly, diluting the heavy-tail requests across more
   KV slots.  Its P99 is actually lowest (292 ms), but this is fragile:
   short requests share slots with long ones, so a traffic-mix shift can
   cause unpredictable latency degradation.

For chatbot workloads (Azure, low utilisation) all three routers pass SLO —
the difference only matters for agent or near-saturation fleets.

---

## Puzzle 6 — Does mixing GPU types in the two-pool fleet save money?

**Question:** short requests are memory-bandwidth bound and cheap to serve;
long requests need large KV caches and fast prefill.  Can I cut costs by
putting cheap GPUs in the short pool and premium GPUs only where they are
needed?

```bash
# Azure: A10G short + H100/A100 long
vllm-sr-sim optimize \
  --cdf data/azure_cdf.json --lam 100 --slo 500 \
  --gpu-short a10g --gpu-long h100 --long-max-ctx 8192

# LMSYS 65K-ctx: test whether A100 can meet SLO on the long pool
vllm-sr-sim optimize \
  --cdf data/lmsys_cdf.json --lam 100 --slo 500 \
  --gpu-short a10g --gpu-long h100 --long-max-ctx 65536
vllm-sr-sim optimize \
  --cdf data/lmsys_cdf.json --lam 100 --slo 500 \
  --gpu-short a10g --gpu-long a100 --long-max-ctx 65536
```

**Azure λ=100 result:**

| Config | GPUs | Cost/yr | P99 short | P99 long |
|---|---|---|---|---|
| All-A100 (baseline) | 12 | $232 K | 52 ms | 112 ms |
| **A10G short + H100 long** | **12** | **$212 K** | 155 ms | 30 ms |
| A10G short + A100 long | 15 | $206 K | 155 ms | 112 ms |

**LMSYS λ=100 result (max_ctx=65536):**

| Config | GPUs | Cost/yr | P99 short | P99 long |
|---|---|---|---|---|
| All-A100 (baseline) | 8 | $155 K | 43 ms | **2 822 ms ✗** |
| **A10G short + H100 long** | **7** | **$141 K** | 129 ms | 181 ms ✓ |
| A10G short + A100 long | 9 | $132 K | 129 ms | **2 822 ms ✗** |

**Two insights:**

1. **Azure (short-context):** A10G+H100 saves 9% vs all-A100 with the same 12
   GPUs.  Expensive GPUs land only in the long pool, where context length
   justifies them; cheap A10Gs handle the 98% short traffic.

2. **LMSYS (long-context):** A10G+A100 is the wrong pairing — it is 11%
   cheaper on paper but the A100 long pool **cannot meet 500 ms SLO**.
   For requests up to 65536 tokens the A100 prefill takes ~700–2800 ms;
   H100 halves that with its larger chunk size (1024 vs 512) and lower W.
   H100 is not a luxury here — it is a correctness requirement for the long
   pool.  Mixing A10G (short) + H100 (long) saves 9% vs all-A100 *and*
   fixes the SLO that all-A100 cannot meet.

---

## Puzzle 7 — When should I switch to disaggregated prefill/decode?

**Question:** prefill is compute-bound (FLOP-intensive); decode is
memory-bandwidth-bound (weight-streaming per token).  If I run them on
separate pools sized independently, which pairing minimises cost?

```bash
# Find optimal nP × nD ratio for H100 prefill + A100 decode
vllm-sr-sim disagg \
  --cdf data/azure_cdf.json --lam 100 \
  --slo-ttft 500 --slo-tpot 100 \
  --gpu-prefill h100 --gpu-decode a100 --max-ctx 8192

# Compare: A100 prefill + H100 decode
vllm-sr-sim disagg \
  --cdf data/azure_cdf.json --lam 100 \
  --slo-ttft 500 --slo-tpot 100 \
  --gpu-prefill a100 --gpu-decode h100 --max-ctx 8192
```

**Azure λ=100 result** (mean ISL≈1450 tok, mean OSL≈483 tok, TTFT includes 1.8× KV-transfer overhead):

| Config | GPUs | Cost/yr | TTFT | TPOT |
|---|---|---|---|---|
| All-A100 aggregated | 12 | $232 K | 26 ms | — |
| All-H100 aggregated | 6 | $211 K | 8 ms | — |
| H100P + A100D | 7 (1P+6D) | $151 K | 162 ms | 91 ms |
| H100P + H100D | 4 (1P+3D) | $141 K | 162 ms | 45 ms |
| **A100P + H100D** | **4 (1P+3D)** | **$125 K ← cheapest** | **492 ms** | **45 ms** |

**Insights:**

1. **Disagg saves 35–46% vs aggregated** — at the cost of higher TTFT
   (KV-transfer overhead adds 1.8× to raw prefill time).

2. **A100P + H100D beats H100P + A100D** despite H100 costing 1.82× more
   than A100.  The reason: H100 decode workers each handle 2.5× more
   requests/second than A100 (lower W and faster per-token iteration), so
   you need far fewer of them (3 vs 6).  One cheap A100 handles all prefill
   at λ=100 req/s.  Counter-intuitively, the *decode* pool is where the
   premium GPU earns back its cost.

3. **When disagg is worth it:** high throughput targets where cost efficiency
   matters more than latency.  Disagg cuts per-GPU cost by ~46%, but TTFT
   climbs to ~500 ms (KV transfer + A100 prefill).  If your SLO is ≤ 200 ms,
   use H100P (TTFT drops to 162 ms) at $141 K — still 33% cheaper than all-H100.

4. **When to stay aggregated:** low-traffic or latency-critical fleets (&lt; 50
   req/s, TTFT SLO ≤ 100 ms).  Disagg adds operational complexity (two
   separate scaling policies, KV-transfer networking) that is not justified
   below ~100 req/s.

---

## Summary

### Cost vs homo at a glance

| Workload | Config | Cost delta | SLO |
|---|---|---|---|
| Azure chat, λ=50 | Two-pool H100 | **+25%** | ✓ |
| Azure chat, λ=200 | Two-pool H100 | **tied** | ✓ |
| LMSYS chat, λ=100 | Two-pool H100 | **−38%** | ✓ |
| Agent-heavy, λ=20 | Homo H100 | — | **✗ FAIL** |
| Agent-heavy, λ=20 | Two-pool H100 | +4% | ✓ |
| Azure, λ=100 | A10G two-pool | **−21% vs H100 homo** | ✓ |

### Rules of thumb

| Question | Answer |
|---|---|
| Should I split? | Yes if heavy-tail (required) or ctx ratio ≥4× and λ is high |
| Where to split? | Run `pareto` — not derivable from the CDF shape alone |
| Which GPU? | Run `whatif --gpu-compare` — A10G two-pool can beat H100 homo |
| Mix GPU types? | Yes — cheap GPU for short pool, match long-pool GPU to max_ctx demands |
| Long-context GPU? | A100 fails SLO above ~20K-token P99; H100 required for 65K-ctx fleets |
| Disaggregate P/D? | At λ ≥ 100 req/s: A100P + H100D saves ~46% vs all-A100 (TTFT rises ~20×) |
| When to provision? | Run `whatif` for step thresholds; pre-provision before the step |
| Which router in production? | LengthRouter always; CompressAndRoute for sizing only |
| Do I need DES? | Yes for agent / heavy-tail — analytics will say "feasible" when it is not |
| Reliability margin? | See below — add `node_avail` on top of SLO sizing |

---

## Reliability sizing — converting SLO counts to production counts

> **All GPU counts above assume `node_avail=1.0`** (every node healthy).
> For production deployments, inflate by the inverse of steady-state node availability.

GPU and NVLink hardware fails at a measurable rate. Published measurements
(Kokolis et al. 2024, arXiv:2410.21680; Cui et al. 2025, arXiv:2503.11901):

| GPU | Failure rate r_f | MTTR (failure-type driven, not GPU-model driven) | Steady-state availability |
|---|---|---|---|
| A100 (Meta RSC-1) | 0.0065 / node-day | ~4h soft (driver reset) → **A ≈ 99.89%** | `A100_AVAIL_RSC1_FAST` |
| A100 (Meta RSC-1) | 0.0065 / node-day | ~48h hard (GPU/NVLink swap) → **A ≈ 98.71%** | `A100_AVAIL_RSC1_SLOW` |
| H100 at scale | — | 5% overprovisioning rule (Cui 2025) → **A = 95%** | `H100_AVAIL_5PCT` |

**MTTR is the same for A100 and H100** for equivalent failure categories — both
use vendor-swapped SXM modules. What differs is the failure rate: H100 has
3.2× more memory MTBE (more soft/ECC events, short MTTR) but better critical
hardware resilience (fewer hard failures, long MTTR).

### How to apply

```python
from fleet_sim import FleetOptimizer, node_availability, A100_AVAIL_RSC1_SLOW, H100_AVAIL_5PCT

# Option A: use a pre-computed constant
opt = FleetOptimizer(gpu_short=A100_80GB, B_short=4096, t_slo_ms=500,
                     node_avail=A100_AVAIL_RSC1_SLOW)   # ≈ 0.987

# Option B: compute from your own fleet telemetry
A = node_availability(r_f_per_node_day=0.0065, mttr_hours=48)   # → 0.9871
opt = FleetOptimizer(..., node_avail=A)

# Option C: H100 rule-of-thumb
opt = FleetOptimizer(..., node_avail=H100_AVAIL_5PCT)   # 0.95
```

### Effect on the puzzle numbers

The SLO-sized counts above are the *minimum under no failures*. With a reliability
margin applied, each count rounds up by `ceil(n / A) - n` extra GPUs:

| Scenario | SLO-sized | `A100_AVAIL_RSC1_SLOW` (A=0.987) | `H100_AVAIL_5PCT` (A=0.95) |
|---|---|---|---|
| LMSYS two-pool (8 GPUs) | 8 | **9** (+1) | **9** (+1) |
| Agent two-pool (25 H100) | 25 | 26 (+1) | **27** (+2) |
| Azure disagg A100P+H100D (4) | 4 | 5 (+1) | 5 (+1) |

For most scenarios **one extra GPU covers reliability** at realistic A100 failure
rates. The H100 5% rule costs one to two GPUs more, reflecting the higher memory
error rate at scale.

> **Note:** the existing `ρ_max=0.85` utilisation cap provides queuing-stability
> headroom (15% over-provision), not reliability headroom. The two concerns are
> independent and multiplicative — both apply in production.

---

## Puzzle 8 — How much grid power can I shed without an SLO breach?

**Scenario.** Your data center is enrolled in a demand-response (DR) program.
The grid operator can send a curtailment signal asking the site to reduce power
consumption by X% for 15–60 minutes.  You want to know: what is the *maximum*
curtailment you can safely commit to, without your P99 TTFT SLO slipping?

**The mechanism.** The GPU-to-Grid (G2G) control framework (Hassan et al.,
arXiv:2602.05116v1) shows that capping the serving engine's maximum in-flight
batch size (`vLLM max_num_seqs`) is the most effective software knob for
modulating GPU power.  A lower cap reduces the number of sequences processed
concurrently, which directly reduces GPU memory-bandwidth pressure and,
therefore, power draw.  The trade-off is that requests queue for longer when
fewer slots are available, increasing P99 TTFT.

**What the simulator does.** `grid_flex_analysis()` sweeps power-reduction
targets, inverts the GPU power model to find the implied batch cap, and
recomputes P99 TTFT with that reduced concurrency.  Two verification levels:

- **Analytical** (fast, ~1 s): M/G/c approximation, recalibrated at each
  batch-cap level so the service rate reflects the actual concurrent-sequence
  count at that curtailment depth.
- **DES verification** (`--verify-des N`): runs a discrete-event simulation
  with `N` requests at each flex level; directly verifies the latency claim.
  Recommended for final sign-off before committing a DR contract.

**Power model.** Each profile now uses a **logistic curve** (Hassan et al.
arXiv:2602.05116v1, Eq. 2) fitted to ML.ENERGY Benchmark v3.0 data:

```
P(b) = P_range / (1 + exp(-k * (log2(b) - x0))) + P_idle
```

For H100-SXM5 (k=1.0, x0=4.2): P(1)≈304W, P(128)≈583W (vs measured ~600W).
This is more accurate than the previous linear model, especially at deep
curtailment (low batch), where the logistic curve correctly shows that power
drops sub-linearly as batch decreases from the saturation plateau.

**Key G2G finding reproduced.** The logistic fit reveals a critical insight
from the G2G paper: at production load (batch=128), the H100 is already at
~97% of nominal power.  Reducing batch by half (64→32) only saves ~13W (~2%).
Meaningful power savings require cutting batch well below 50, which approaches
the Erlang-C saturation threshold and raises P99 TTFT.  The DES confirms
this picture precisely.

### Numerical example (with DES verification)

40 H100s, λ = 200 req/s, SLO = 500 ms:

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

**Insight.** For a short-burst DR event (15000-request window ≈ 75 seconds
at 200 req/s), this fleet can commit up to **40% power reduction** while
keeping P99 TTFT under 500 ms.  The analytical model flags 40% as
steady-state unstable (inf), but DES shows it is safe during a time-limited
event.  At 50% (n_max=1), the queue collapses catastrophically (449 s P99).

**Sustained vs. event curtailment.** The analytical model gives the limit for
*sustained* power reduction (hours): if the recalibrated ρ ≥ 1 in steady
state, the queue diverges eventually.  The DES gives the *event* limit
(minutes): how deep you can go before latency breaches SLO during the DR
window.  Both are useful:

| Use case | Model to use | Max safe flex (example above) |
|---|---|---|
| Long-term steady-state reduction | Analytical | 30% |
| Short DR event (15–60 min) | DES (15 000–50 000 req) | 40% |

**What differs from over-provisioning.** Reliability sizing (Puzzle 7) adds
GPUs to cover failures.  Grid flex throttles the serving batch at *fixed GPU
count* — no hardware changes required.  The two controls are independent and
can be combined.

### How to apply

```python
from fleet_sim import grid_flex_analysis, print_grid_flex_table, H100_80GB
import json

cdf = json.load(open("data/azure_cdf.json"))
cdf = [(int(t), float(f)) for t, f in (cdf["cdf"] if isinstance(cdf, dict) else cdf)]

# Analytical only (fast)
results = grid_flex_analysis(
    cdf=cdf, lam=200, n_gpus=40, gpu=H100_80GB,
    t_slo_ms=500, max_ctx=8192,
)

# With DES verification (recommended before signing a DR contract)
results = grid_flex_analysis(
    cdf=cdf, lam=200, n_gpus=40, gpu=H100_80GB,
    t_slo_ms=500, max_ctx=8192,
    n_sim_requests=15000,    # requests to simulate per flex level
    verbose=True,
)
print_grid_flex_table(results, t_slo_ms=500, n_gpus=40, lam=200)
```

Or via the CLI:

```bash
# Analytical only
vllm-sr-sim grid-flex \
    --cdf data/azure_cdf.json \
    --lam 200 --n-gpus 40 --gpu h100 --slo 500

# With DES verification
vllm-sr-sim grid-flex \
    --cdf data/azure_cdf.json \
    --lam 200 --n-gpus 40 --gpu h100 --slo 500 \
    --verify-des 15000
```

> **Power model accuracy.** The logistic curve is accurate to within ~3% at
> batch ≥ 16 (operationally relevant range).  At batch &lt; 8 the curve is less
> well-constrained; re-fit `power_logistic_k` and `power_logistic_x0` from
> your own `vllm serve` profiling runs for tighter accuracy at deep curtailment.

---

## Puzzle 9 — Which GPU is most energy-efficient for my workload?

**Question:** GPU power bills are a growing fraction of inference costs.  Which
GPU type delivers the most output tokens per watt, and how much does energy
efficiency degrade when the fleet is under-utilised?

### Important correctness caveat: tok/W is (GPU, model) pair property

The pre-built `ManualProfile` instances are calibrated for **different models**:
`H100-80GB` and `A100-80GB` are calibrated for Llama-3-70B on 8-GPU TP;
`A10G` is calibrated for 7B-class models on a single GPU.  Running
`tok-per-watt --gpus h100 a100 a10g` applies the same workload CDF to three
profiles that represent different models — the tok/W ratio reflects
model-size difference as much as GPU efficiency.

For a **clean GPU comparison** (same model, different hardware), use `ComputedProfile`:

```python
from fleet_sim.hardware import H100_SXM, A100_SXM
from fleet_sim.models import LLAMA_3_1_70B
from fleet_sim.gpu_profiles import ProfileBuilder, ServingConfig

for hw in [H100_SXM, A100_SXM]:
    p = ProfileBuilder().build(hw=hw, model=LLAMA_3_1_70B,
                               cfg=ServingConfig(tp=8, dtype_bytes=2.0,
                                                 mean_ctx_tokens=4096))
    pt = p.decode_efficiency(n_active=32, mean_ctx=4096)
    print(f"{hw.name:12s}  iter={pt.iter_latency_s*1000:.1f}ms  "
          f"kv_frac={pt.kv_frac:.2f}  P={pt.power_w:.0f}W  tok/W={pt.tokens_per_watt:.3f}")
```

The tok/W formula derivation is transparent — every step from FLOPs to power is
exposed in `DecodeEfficiencyPoint.show()`.  See `SIM_ALGORITHMS.md` §14 for the
full derivation chain.

### Single-pool per-GPU comparison (same model assumed)

```bash
vllm-sr-sim tok-per-watt \
  --cdf data/azure_cdf.json --lam 100 --slo 500 \
  --gpus h100 a100 a10g --rho-sweep
```

**Result (Azure CDF, λ=100 req/s, SLO=500 ms) — SLO-optimal fleet:**

```
Pool / GPU               GPUs    ρ   n_act  Tok/W  P/GPU(W)  $/1M tok  P99(ms)  PwrQ
------------------------------------------------------------------------------------
H100-80GB                   6  0.82  104.4   9.38    577 W   $  0.21      7.9   HIGH
A100-80GB                  12  0.83  106.1   7.09    382 W   $  0.23     25.9   FAIR
A10G                       21  0.82   52.3  13.56    114 W   $  0.18     71.1   LOW *
```

*LOW quality power model: projection only; no published batch-vs-power data for A10G.
 A10G also serves a different model (7B-class) than H100/A100 (70B) in these profiles.*

**Tok/W vs utilisation (H100 and A100 only — HIGH/FAIR quality):**

```
GPU           ρ=0.20  ρ=0.40  ρ=0.60  ρ=0.80   @ SLO-opt
----------------------------------------------------------
H100-80GB      2.69    4.63    6.43    8.12      9.38 ★
A100-80GB      1.79    3.45    5.03    6.55      7.09 ★
```

**What the numbers say:**

- **H100 leads on verified tok/W** (9.38 vs 7.09 for A100 at ρ≈0.82). Despite drawing
  more watts per GPU (577 W vs 382 W), the H100 delivers nearly 2× the throughput,
  so the *ratio* is better.

- **Utilisation dominates.** Running at ρ=0.20 cuts tok/W by 3–4× vs ρ=0.80.
  The idle-power floor (H100: 300 W; A100: 175 W) is paid regardless of load;
  efficiency only improves as you fill the GPU.  Auto-scaling matters as much
  as GPU selection.

- **Workload shape changes the ranking.** With `--cdf data/lmsys_cdf.json`
  (shorter contexts, fewer GPUs needed):

  ```
  H100-80GB   3 GPUs  ρ=0.46   8.20 tok/W
  A100-80GB   4 GPUs  ρ=0.70   8.92 tok/W  ← beats H100 at lower utilisation
  ```

  The H100 fleet is over-provisioned (ρ≈0.46) and burns idle power, flipping
  the ranking.  This is the most important non-obvious insight from the analysis.

**Decision guide:**

| Priority | Choice |
|---|---|
| Lowest energy per token at tight SLO + long context | H100 — HIGH quality model |
| Lowest cost per token | A10G or A100 — verify with `--rho-sweep` |
| Hardware comparison for a specific model | Use `ComputedProfile.decode_efficiency()` |
| Production DR contract | Calibrate your own logistic curve first |

> **Do not over-interpret absolute tok/W values.** The SHAPE of the tok/W vs
> utilisation curve and the RELATIVE ranking between H100 and A100 (same model)
> are reliable.  The A10G absolute number is a projection only.

---

## Puzzle 10 — Does routing short requests to a small model save energy?

**Question:** If a semantic router sends short or simple requests to a small model
(A10G + 7B) and complex ones to a large model (H100 + 70B), how does fleet-level
energy efficiency compare to a homogeneous H100 fleet?

This is the multi-pool tok/W problem.  **N does not cancel across pools:**

```
fleet_tok/W = Σ_i(λ_i × mean_L_out_i) / Σ_i(N_i × power_i(n_i))
```

Each pool has its own model-specific W, H, and power curve.  The `fleet_tpw_analysis()`
function computes this correctly using per-pool normalised sub-CDFs.

```bash
# Compare homo H100 (70B everywhere) vs two-pool A10G short + H100 long
vllm-sr-sim tok-per-watt \
    --cdf data/azure_cdf.json --lam 100 --slo 500 \
    --b-short 4096 --gpu-short a10g --gpu-long h100
```

**What happens at B_short = 4096:**

- Short pool (≤4096 tokens, α≈0.72 of traffic) → A10G + 7B model
- Long pool  (>4096 tokens, 1-α≈0.28) → H100 + 70B model
- N_short and N_long are sized independently to the same SLO

**Why this matters structurally:**

- The short pool handles majority traffic with a fast, cheap, low-power GPU
- The long pool handles the tail with the high-power but efficient GPU
- Fleet-level power = sum of both pools' power; tok/W is the aggregate ratio
- The result depends on the CDF shape (what fraction of requests are short)
  and the SLO (which determines N per pool via Erlang-C sizing)

**Routing tok/W caveats:**

1. The A10G and H100 profiles serve different models — the quality and capability
   of responses differs.  tok/W measures energy, not quality.
2. The semantic router's classification accuracy affects the actual split;
   this analysis assumes perfect classification.
3. For semantic-router routing by complexity (not just length), model the
   routing fraction α explicitly using `--b-short` to represent the fraction
   of traffic going to each pool.

**To compute for your own pools using Python:**

```python
from fleet_sim.optimizer import fleet_tpw_analysis, _split_cdf, print_fleet_tpw
from fleet_sim.gpu_profiles import A10G, H100_80GB
import json

cdf = json.load(open("data/azure_cdf.json"))
short_cdf, long_cdf, alpha = _split_cdf(cdf, b_short=4096)

result = fleet_tpw_analysis(
    pools=[
        {"gpu": A10G,     "cdf": short_cdf, "lam": alpha * 100,
         "max_ctx": 4096, "label": "short-7B-A10G"},
        {"gpu": H100_80GB,"cdf": long_cdf,  "lam": (1-alpha) * 100,
         "max_ctx": 8192, "label": "long-70B-H100"},
    ],
    lam_total=100,
    t_slo_ms=500,
    topology="two-pool A10G(7B)+H100(70B)",
)
print_fleet_tpw(result)
```

---

## Reproducing all results

```bash
# Puzzle 6: Mixed GPU type pools
vllm-sr-sim optimize --cdf data/azure_cdf.json  --lam 100 --slo 500 \
  --gpu-short a10g --gpu-long h100 --long-max-ctx  8192
vllm-sr-sim optimize --cdf data/lmsys_cdf.json --lam 100 --slo 500 \
  --gpu-short a10g --gpu-long h100 --long-max-ctx 65536
vllm-sr-sim optimize --cdf data/lmsys_cdf.json --lam 100 --slo 500 \
  --gpu-short a10g --gpu-long a100 --long-max-ctx 65536

# Puzzle 7: Disaggregated prefill/decode
vllm-sr-sim disagg --cdf data/azure_cdf.json --lam 100 \
  --slo-ttft 500 --slo-tpot 100 --gpu-prefill h100 --gpu-decode a100 --max-ctx 8192
vllm-sr-sim disagg --cdf data/azure_cdf.json --lam 100 \
  --slo-ttft 500 --slo-tpot 100 --gpu-prefill a100 --gpu-decode h100 --max-ctx 8192

# Puzzle 1: Pareto threshold selection
vllm-sr-sim pareto --cdf data/lmsys_cdf.json       --lam 100 --slo 500 --long-max-ctx 65536
vllm-sr-sim pareto --cdf data/azure_cdf.json       --lam 200 --slo 500 --long-max-ctx  8192
vllm-sr-sim pareto --cdf data/agent_heavy_cdf.json --lam 200 --slo 500 --long-max-ctx 65536

# Puzzle 2: Agent SLO failure
vllm-sr-sim optimize --cdf data/agent_heavy_cdf.json --lam 20 --slo 1000 \
  --gpu-short h100 --gpu-long h100 --b-short 65536 --long-max-ctx 65536
vllm-sr-sim optimize --cdf data/agent_heavy_cdf.json --lam 20 --slo 1000 \
  --gpu-short h100 --gpu-long h100 --b-short 4096  --long-max-ctx 65536 --verify-top 3

# Puzzle 3: GPU type selection
for gpu in a10g a100 h100; do
  vllm-sr-sim optimize --cdf data/azure_cdf.json --lam 100 --slo 500 \
    --gpu-short $gpu --gpu-long $gpu --b-short 8192 --long-max-ctx 8192
  vllm-sr-sim optimize --cdf data/azure_cdf.json --lam 100 --slo 500 \
    --gpu-short $gpu --gpu-long $gpu --long-max-ctx 8192 --verify-top 3
done

# Puzzle 4: Traffic growth step thresholds
vllm-sr-sim whatif --cdf data/azure_cdf.json --slo 500 \
  --gpu-short h100 --gpu-long h100 --long-max-ctx 8192 \
  --lam-range 25 50 100 150 200 300 400

# Puzzle 5: Router comparison
vllm-sr-sim compare-routers --cdf data/azure_cdf.json --lam 200 --slo 500 \
  --gpu-short h100 --gpu-long h100 --n-s 5 --n-l 7 --long-max-ctx 8192 --n-req 5000
vllm-sr-sim compare-routers --cdf data/agent_heavy_cdf.json --lam 20 --slo 1000 \
  --gpu-short h100 --gpu-long h100 --n-s 2 --n-l 23 --long-max-ctx 65536 --n-req 5000

# Puzzle 9: GPU energy efficiency comparison (single-pool, same-model assumption)
vllm-sr-sim tok-per-watt --cdf data/azure_cdf.json --lam 100 --slo 500 \
  --gpus h100 a100 a10g --rho-sweep
vllm-sr-sim tok-per-watt --cdf data/lmsys_cdf.json --lam 100 --slo 500 \
  --gpus h100 a100

# Puzzle 10: Multi-pool routing tok/W (fleet_tpw_analysis, N does not cancel)
vllm-sr-sim tok-per-watt --cdf data/azure_cdf.json --lam 100 --slo 500 \
  --b-short 4096 --gpu-short a10g --gpu-long h100
vllm-sr-sim tok-per-watt --cdf data/lmsys_cdf.json --lam 100 --slo 500 \
  --b-short 2048 --gpu-short a10g --gpu-long h100

# Puzzle 8: Grid flexibility (demand response) — analytical + DES verified
vllm-sr-sim grid-flex --cdf data/azure_cdf.json \
  --lam 200 --n-gpus 40 --gpu h100 --slo 500
vllm-sr-sim grid-flex --cdf data/azure_cdf.json \
  --lam 200 --n-gpus 40 --gpu h100 --slo 500 --verify-des 15000
vllm-sr-sim grid-flex --cdf data/azure_cdf.json \
  --lam 100 --n-gpus 15 --gpu a100 --slo 500 --verify-des 15000
```
