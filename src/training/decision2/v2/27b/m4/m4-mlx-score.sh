#!/usr/bin/env bash
# Score the ~27B M4 mlx-diag collections on node A, where the mlx-diag gold lives (run on the workstation). The
# F2 scorer (m3f2/f2-mlx-score.sh) with MLX_ROOT=/data/dev2/runs/27b/m4-mlx, where m4-tail.sh mlx collects: per
# NAME it streams node B's gold-free receipts, logs and predictions (no cache) to the same path on node A, checks
# the predictions' SHA-256 on both sides and runs v2.eval.multilingual_panel score from node A's mirror c8a5504b5
# against /data/dev2/private/panels/mlx-diag-v1 (type-macro, English / non-English, per language). Smokes are not
# scored. Internal diagnostic only (XNLI is CC BY-NC); never a release score.
# Usage: m4-mlx-score.sh NAME...   (M4-A20-soup, M4-A20r-soup, M4-Ar-soup)
set -euo pipefail
MLX_ROOT=${MLX_ROOT:-/data/dev2/runs/27b/m4-mlx} exec bash "$(dirname "$0")/../m3f2/f2-mlx-score.sh" "$@"
