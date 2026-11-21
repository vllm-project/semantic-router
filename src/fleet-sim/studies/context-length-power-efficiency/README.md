# Measured B200 Context-Length Sweep

This directory archives the measured B200 results contributed for the
context-length power-efficiency study. It contains the experiment summary and
a small standard-library script that validates and renders the measured table.

It intentionally does not contain or modify the paper's LaTeX or PDF. The
paper authors can integrate these measurements into the manuscript separately.

## Measurement setup

- Date: April 29, 2026
- System: one node with 8x NVIDIA B200 GPUs
- Model: Llama-3.1-70B, tensor parallelism 8, fp16
- Software: vLLM 0.20.0, PyTorch 2.11.0+cu130, CUDA 13.0
- Context windows: 2K, 4K, 8K, 16K, 32K, 64K, and 128K tokens

The archived results are:

| Context | Measured nmax | Decode tok/s | Mean GPU power (W) | Paper-normalized tok/W | 8-GPU system tok/W |
|---:|---:|---:|---:|---:|---:|
| 2K | 1725 | 29486.22 | 794.99 | 37.09 | 4.64 |
| 4K | 872 | 21620.98 | 712.54 | 30.34 | 3.79 |
| 8K | 439 | 13451.24 | 654.55 | 20.55 | 2.57 |
| 16K | 220 | 7757.27 | 596.00 | 13.02 | 1.63 |
| 32K | 110 | 4654.67 | 590.21 | 7.89 | 0.99 |
| 64K | 55 | 2734.58 | 636.53 | 4.30 | 0.54 |
| 128K | 28 | 822.49 | 723.43 | 1.14 | 0.14 |

## Artifacts

- `data/b200_table1_measurements/final_results.csv`: measured table and
  calculation columns.
- `data/b200_table1_measurements/B200_env.txt`: software and GPU environment.
- `data/b200_table1_measurements/nmax.txt`: vLLM KV-cache allocation excerpts
  used to obtain maximum concurrency.
- `scripts/render_b200_table.py`: validates the archived calculations and
  prints a Markdown table.

## Metric definitions

`decode_tok_s` is aggregate decode throughput for the TP=8 serving instance.
`per_gpu_decode_power_w` is the mean power of one GPU.

The CSV preserves two tok/W conventions:

1. `measured_decode_tok_per_w` divides aggregate TP=8 throughput by mean
   per-GPU power. This is the normalization used by the study's paper table,
   but it is not physical whole-system energy efficiency.
2. `measured_8gpu_decode_tok_per_w` divides the same throughput by estimated
   total GPU power (`8 * per_gpu_decode_power_w`). This is the physical
   8-GPU serving-instance metric under balanced per-GPU power.

The `paper_projected_*` columns are retained only as the analytical reference
available when the measurements were collected. They are not inputs to the
measured calculations.

## Reproduce the archived table

From this directory:

```bash
python3 scripts/render_b200_table.py
```

The script reads `final_results.csv`, recomputes both tok/W columns from
throughput and power, and fails if a reported value differs by more than the
CSV's two-decimal precision.

## Scope and limitations

These files reproduce the table from the archived experiment summary; they do
not rerun the GPU experiment from scratch. The original load-generation
scripts, raw power traces, and repeat-run samples are not part of the archived
contribution. Consequently:

- the measurements should be treated as single-node evidence;
- the archive does not quantify run-to-run or system-to-system variance; and
- claims or manuscript changes based on the table should be reviewed by the
  paper authors.
