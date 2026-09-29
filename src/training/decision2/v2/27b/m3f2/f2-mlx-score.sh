#!/usr/bin/env bash
# Score ~27B mlx-diag collections on node A, where the mlx-diag gold lives (run on the workstation, since
# the nodes do not reach each other). Per NAME: stream node B MLX_ROOT/NAME's gold-free receipts, logs and
# predictions (no cache) to the same path on node A, check the predictions' SHA-256 on both sides, then run
# v2.eval.multilingual_panel score against /data/dev2/private/panels/mlx-diag-v1 (gold 71515a41) from node A's
# mirror c8a5504b5 (multilingual_panel.py, same_panel.py, benchmark/score.py and generate.py byte-identical to
# 35fa052d2) and print type-macro, English / non-English and per-language accuracy.
# Usage: f2-mlx-score.sh NAME...
set -euo pipefail
nodes=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
A=$(grep '^node-a=' "$nodes" | cut -d= -f2-)
B=$(grep '^node-b=' "$nodes" | cut -d= -f2-)
D=${MLX_ROOT:-/data/dev2/runs/27b/m3-f2/mlx-diag}
M=/data/dev2/src/c8a5504b5b39049390fccfbe3bcac5ef65e1e710-src_training_decision2/src/training/decision2
PANEL=/data/dev2/private/panels/mlx-diag-v1
PRED=output/mlx-diag.predictions.jsonl
on_a() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$A" "$@"; }
on_b() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$B" "$@"; }

for name in "$@"; do
  [[ "$name" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "bad NAME $name" >&2; exit 2; }
  hb=$(on_b "cd '$D/$name' && test -f COLLECT.json && sha256sum $PRED") \
    || { echo "$name: no finished collection on node B" >&2; exit 1; }
  on_a "test ! -e '$D/$name'" || { echo "$name: $D/$name exists on node A" >&2; exit 1; }
  on_a "mkdir -p '$D/$name'"
  on_b "tar -C '$D/$name' -cf - COLLECT.json GPU-TIME.json output logs triton-cache.copy.json triton-cache.post.json" \
    | on_a "tar -C '$D/$name' -xf -"
  [ "$(on_a "cd '$D/$name' && sha256sum $PRED")" = "$hb" ] || { echo "$name: relayed predictions differ" >&2; exit 1; }
  on_a "cd '$M' && PYTHONPATH='$M' PYTHONDONTWRITEBYTECODE=1 python3 -m v2.eval.multilingual_panel score \
    --panel '$PANEL' --predictions '$D/$name/$PRED' --output '$D/$name/mlx-diag.score.json'"
  on_a "python3 - '$D/$name/mlx-diag.score.json'" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
keys = ("type_macro_accuracy", "english_type_macro_accuracy", "non_english_type_macro_accuracy",
        "cross_language_consistency", "invalid_or_missing")
print(json.dumps({k: s[k] for k in keys}))
print("per language", json.dumps({k: round(v, 4) for k, v in s["per_language_mean_accuracy"].items()}))
for kind, t in s["by_type"].items():
    cells = {lang: f"{v['correct']}/{v['n']}" for lang, v in t["languages"].items()}
    print(kind, json.dumps(cells), "non-English minus English", round(t["gap_vs_english"], 4))
EOF
done
