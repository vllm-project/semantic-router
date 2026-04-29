# Context-Length Power Efficiency Study

**Title:** *The 1/W Law: Why Context-Length Routing Topology Outperforms GPU Upgrades for LLM Inference Energy Efficiency*

---

## Overview

This paper analyzes LLM inference energy efficiency (tokens per watt, tok/W) across GPU generations (H100, H200, B200, GB200), model architectures (dense and MoE), and fleet routing topologies (homogeneous, two-pool, FleetOpt).

All quantitative results are computed analytically using the local
`fleet_sim` package in this repository.
No GPU hardware is required to reproduce the tables.

---

## Repository layout

```
studies/context-length-power-efficiency/
├── main.tex            # LaTeX source
├── Makefile            # Convenience targets for tables/PDF/cleanup
├── refs.bib            # Bibliography
├── scripts/
│   ├── profiles.py         # Canonical GPU profiles (H100 empirical + B200 projected)
│   ├── table1_ctx_nmax.py  # Table 1: n_max and tok/W vs context window
│   ├── table2_arch_tpw.py  # Table 2: single-GPU tok/W by model architecture
│   ├── table3_fleet_tpw.py # Table 3: fleet tok/W by topology and GPU generation
│   ├── table4_routing.py   # Table 4: context-window vs semantic routing
│   ├── table5_gen_compare.py# Table 5: GPU generation comparison
│   └── reproduce_all.py    # Run all five scripts in sequence
└── README.md
```

---

## Prerequisites

### 1. Python packages

Install the simulator package from the `fleet-sim` root:

```bash
cd src/fleet-sim
pip install -e .          # or: pip install -e .[dev]
```

No additional packages are needed.

### 2. LaTeX (to recompile the PDF)

```bash
# Debian/Ubuntu:
sudo apt-get install texlive-full
# macOS (Homebrew):
brew install --cask mactex
```

---

## Reproducing all tables

From this study directory:

```bash
make tables
```

Expected runtime: ~10 seconds (fleet analysis dominates).

To run a single table:

```bash
python3 scripts/table1_ctx_nmax.py   # Table 1
python3 scripts/table2_arch_tpw.py   # Table 2
python3 scripts/table3_fleet_tpw.py  # Table 3
python3 scripts/table4_routing.py    # Table 4
python3 scripts/table5_gen_compare.py # Table 5
```

---

## Recompiling the PDF

```bash
make pdf
```

If you prefer the raw LaTeX commands:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex    # third pass to resolve cross-references
```

---

## GPU profile methodology

### H100-SXM5 (HIGH quality)
Empirically calibrated `ManualProfile` from
[Hassan et al. 2026 (G2G)](https://arxiv.org/abs/...) and
[ML.ENERGY Benchmark v3.0](https://ml.energy/leaderboard):

| Parameter | Value | Source |
|-----------|-------|--------|
| W (weight-streaming latency) | 4.0 ms | ML.ENERGY / G2G Fig. 2 |
| H (KV-scan overhead @ 8K ctx) | 0.32 ms/seq | calibrated to H100 |
| n_max @ 8K context | 128 | empirical (TP-sharded GQA KV) |
| P_idle | 300 W | ML.ENERGY |
| P_nominal (saturated) | 600 W | ML.ENERGY |

**KV-storage assumption:** TP=8 with Llama-3.1-70B's 8 GQA heads → 1 KV head per GPU → ~55 KB/token. This matches the empirical calibration and is consistent with vLLM's default TP-sharded attention.

### B200-SXM (FAIR quality, ±20%)
Projected `ManualProfile` anchored to the H100 calibration:

1. Build `ComputedProfile` for both H100 and B200 from first principles.
2. KV-budget ratio: `R = cp_b200.total_kv_blks / cp_h100.total_kv_blks ≈ 2.62`.
3. Scale H100 empirical `total_kv_blks` and `max_slots` by `R`.
4. Scale H (per-seq KV latency) by the bandwidth ratio (3.35/8.0 TB/s ≈ 0.42).
5. Use `cp_b200.W = 2.955 ms` from the roofline model.
6. Power: `P_idle = 0.43 × TDP = 430 W`, `P_nom = 0.86 × TDP = 860 W`.

No direct silicon power measurements for B200 are available as of March 2026.

### H200 and GB200 (FAIR quality)
First-principles `ComputedProfile` only; used for the GPU generation
comparison table (Table 5).

---

## Workload traces

| File | Description |
|------|-------------|
| `../../data/azure_cdf.json` | Azure Conversations (Arch. I: 89.8% ≤ 4K tokens) |
| `../../data/lmsys_multiturn_cdf.json` | LMSYS multi-turn chat (90.9% ≤ 1.5K tokens) |

---

## MoE models (Table 2)

`ComputedProfile` uses **total** parameter bytes to compute `W`, which
overestimates latency for MoE models where only a fraction of experts
is active per token. The scripts apply a correction:

```
W_MoE = active_param_bytes_per_gpu / hw.mem_bw
```

This is a **lower bound** on true `W` because it ignores MoE dispatch
overhead (all-to-all token routing). The reported tok/W values for
Qwen3-235B-A22B and DeepSeek-V3 are upper bounds.

| Model | Active params | Assumed TP |
|-------|--------------|------------|
| Qwen3-235B-A22B | 22B (exact) | 8 |
| DeepSeek-V3 | ~37B (estimated) | 8 |

---

## Key results summary

| Finding | Value |
|---------|-------|
| tok/W ratio (64K vs 4K context, H100) | ~12× |
| Topology gain (FleetOpt vs homogeneous) | ~2.5× |
| Generation gain (B200 vs H100) | ~1.7× |
| Combined gain (B200 FleetOpt vs H100 Homo) | ~4.25× |
| Azure short-context fraction (≤4K) | 89.8% |
| LMSYS short-context fraction (≤1.5K) | 90.9% |
