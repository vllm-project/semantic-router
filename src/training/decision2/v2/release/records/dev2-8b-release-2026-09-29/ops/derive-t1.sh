#!/usr/bin/env bash
# DEV2.0-8B (9B K-a13) T = 1 scored bindings (coordinator calibration rule 2026-09-28 23:15: CAL698 rejected on the
# development panels because typed-DEV and CSS-pilot Brier worsen; receipt devcal/dev2-8b.json). The sealed formal run
# formal-m4/K-a13-16k and its mlx-diag run were scored with the CAL698 temperatures; undo them offline
# (softmax(log q * T); answers unchanged), adopt, seal, report, pair against the same comparator runs, and score
# mlx-diag. Node A, CPU only, host python3, run from the exact mirror that contains this file.
set -euo pipefail
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
G=/data/dev2/private/panels/goldfree
R=/data/dev2/runs/9b
RUN=$R/formal-m4/K-a13-16k
MLX=$R/formal-m4/K-a13-16k-mlx
CKPT=$R/m4/K-a13-build/soup
CAL=$R/m4/K-a13-cal/calibration.json
I=/data/dev2/runs/release/inputs/dev2-8b-t1/derived
D=/data/dev2/runs/release/dev2-8b-t1-derived
DM=/data/dev2/runs/release/dev2-8b-t1-derived-mlx
LABEL="DEV2.0-8B (K-a13) at T = 1 (derived from the CAL698 run; post-key same-panel)"

echo "mirror: $S"
sha256sum "$CAL" | cut -c1-64 | grep -qx 65297c6d72b8121174c4994da49e17dc8ff760a3d2101b7b418223c47f3ba7cc
sha256sum "$R/m4/K-a13-build/SHA256SUMS" | cut -c1-64 | grep -qx 6913eb61836afa9cefe28722cb712bd2b3f06b594e8f45da3007264e7a1ef3be
(cd "$R/m4" && sha256sum -c --quiet K-a13-build/SHA256SUMS)
echo "node copy verified: $(wc -l < "$R/m4/K-a13-build/SHA256SUMS") files"
python3 - "$RUN/SEAL.json" "$RUN/output" "$MLX/output" <<'PY'
import hashlib, json, sys
seal = json.load(open(sys.argv[1]))
for panel, v in seal["panels"].items():
    got = hashlib.sha256(open(f"{sys.argv[2]}/{panel}.predictions.jsonl", "rb").read()).hexdigest()
    assert got == v["predictions_sha256"], (panel, got)
    print("sealed predictions match:", panel, got[:12])
mlx = hashlib.sha256(open(f"{sys.argv[3]}/mlx-diag.predictions.jsonl", "rb").read()).hexdigest()
print("mlx-diag predictions:", mlx)
PY

mkdir -p "$(dirname "$I")"; mkdir "$I"
cd "$S"; export PYTHONPATH="$S"
for p in typed-final css15 public231; do
  python3 -B -m v2.release.retemper_predictions --predictions "$RUN/output/$p.predictions.jsonl" \
    --prompts "$G/$p.prompts.jsonl" --calibration "$CAL" --undo \
    --output "$I/$p.predictions.jsonl" --receipt "$I/$p.retemper.json"
done
python3 -B -m v2.release.retemper_predictions --predictions "$MLX/output/mlx-diag.predictions.jsonl" \
  --prompts "$G/mlx-diag.prompts.jsonl" --calibration "$CAL" --undo \
  --output "$I/mlx-diag.predictions.jsonl" --receipt "$I/mlx-diag.retemper.json"

mkdir "$D"
python3 -B -m v2.eval.same_panel adopt --run-dir "$D" \
  --typed-final "$I/typed-final.predictions.jsonl" --css15 "$I/css15.predictions.jsonl" \
  --public231 "$I/public231.predictions.jsonl" \
  --prior-receipt "$RUN/COLLECT.json" --prior-receipt "$RUN/SEAL.json" --prior-receipt "$CAL" \
  --prior-receipt "$I/typed-final.retemper.json" --prior-receipt "$I/css15.retemper.json" \
  --prior-receipt "$I/public231.retemper.json" \
  --reason "T = 1 (uncalibrated) predictions of DEV2.0-8B (9B K-a13) derived offline from the sealed CAL698 run formal-m4/K-a13-16k by undoing its per-type temperatures (calibration 65297c6d; softmax(log q * T)); identical answers, only probabilities differ; for the coordinator 23:15 calibration rule"
python3 -B -m v2.eval.same_panel seal --run-dir "$D"
python3 -B -m v2.eval.same_panel report --run-dir "$D" --label "$LABEL" --tier 9B --family decision2 \
  --count-safetensors "$CKPT"
while read -r dir name; do
  python3 -B -m v2.eval.same_panel compare --run-dir "$D" --comparator-run-dir "$dir" \
    --left-name "DEV2.0-8B (T = 1)" --right-name "$name" > "$D/compare-$name.log"
done <<'EOF'
/data/dev2/runs/eval/m1/d1-lux1-autotune-cache adopted-1.0
/data/dev2/runs/9b/formal-m3/lux1-16k-shared same-renderer-16k
/data/dev2/runs/eval/m2/q6-nimble2 nimble2
/data/dev2/runs/eval/m1-adopt/jpt9b jpt9b
EOF
mkdir "$DM"
python3 -B -m v2.eval.multilingual_panel score --panel /data/dev2/private/panels/mlx-diag-v1 \
  --predictions "$I/mlx-diag.predictions.jsonl" --output "$DM/mlx-diag.score.json"
sha256sum "$I"/* "$D"/*.json "$DM/mlx-diag.score.json"
