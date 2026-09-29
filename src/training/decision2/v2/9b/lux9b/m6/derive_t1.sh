#!/usr/bin/env bash
# usage: derive_t1.sh RUNNER_SHA NAME [LABEL]
# CPU, node A, host python3: T = 1 scored bindings of a Milestone 6 finalist when the 23:15 rule
# (ship_cal.sh) rejects CAL698, exactly as the DEV2.0-8B release derived its T = 1 run (release
# records ops/derive-t1.sh), paths adapted. From formal-m6/NAME-16k (+ -mlx) and the checkpoint
# and calibration recorded in formal-m6/NAME.inputs.json: checks the calibration hash and the
# sealed predictions, undoes the CAL698 temperatures offline with v2.release.retemper_predictions
# --undo (softmax(log q * T); answers unchanged) into formal-m6/NAME-16k-t1.inputs, then
# same_panel adopt / seal / report (--count-safetensors) into formal-m6/NAME-16k-t1, compares
# against the release's four comparators (adopted-1.0, same-renderer-16k, nimble2, jpt9b) and
# scores mlx-diag into formal-m6/NAME-16k-t1-mlx. M5 / M6 additions: gates paired vs the released
# DEV2.0-9B T = 1 run and vs Nimble v2, gates types, and successor.json (T = 1 vs T = 1) into
# formal-m6/NAME-16k-t1.gates; the mlx paired and Score level files are the CAL698 run's (same
# answers). All code from the mirror RUNNER_SHA.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
rsha=$1; name=$2; label=${3:-"$name at T = 1 (derived from the CAL698 run; post-key same-panel)"}
S=$(code_dir "$rsha")
F=$RUNS/formal-m6
G=$DATA/dev2/private/panels/goldfree
RUN=$F/$name-16k
MLX=$F/$name-16k-mlx
I=$F/$name-16k-t1.inputs
D=$F/$name-16k-t1
DM=$F/$name-16k-t1-mlx
GT=$F/$name-16k-t1.gates
T1=$DATA/dev2/runs/release/dev2-8b-t1-derived
NIMBLE=$DATA/dev2/runs/eval/m2/q6-nimble2
[ -f "$S/v2/release/retemper_predictions.py" ] || { echo "no v2.release.retemper_predictions in $S" >&2; exit 2; }
[ -f "$F/$name.inputs.json" ] || { echo "no $F/$name.inputs.json (formal.sh writes it)" >&2; exit 2; }
[ -f "$RUN/SEAL.json" ] || { echo "$RUN is not sealed" >&2; exit 2; }
read -r CKPT CALDIR CALSHA < <(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["checkpoint"], d["calibration_dir"], d["calibration_sha256"])' "$F/$name.inputs.json")
CAL=$CALDIR/calibration.json
echo "mirror: $S"
sha256sum "$CAL" | cut -c1-64 | grep -qx "$CALSHA" || { echo "calibration $CAL changed since the formal run" >&2; exit 2; }
python3 - "$RUN/SEAL.json" "$RUN/output" "$MLX/output" "$CALSHA" <<'PY'
import hashlib, json, sys
seal = json.load(open(sys.argv[1]))
for panel, v in seal["panels"].items():
    path = f"{sys.argv[2]}/{panel}.predictions.jsonl"
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert got == v["predictions_sha256"], (panel, got)
    print("sealed predictions match:", panel, got[:12])
mlx = hashlib.sha256(open(f"{sys.argv[3]}/mlx-diag.predictions.jsonl", "rb").read()).hexdigest()
print("mlx-diag predictions:", mlx)
PY

mkdir "$I"
cd "$S"; export PYTHONPATH="$S"
for p in typed-final css15 public231; do
  dry python3 -B -m v2.release.retemper_predictions --predictions "$RUN/output/$p.predictions.jsonl" \
    --prompts "$G/$p.prompts.jsonl" --calibration "$CAL" --undo \
    --output "$I/$p.predictions.jsonl" --receipt "$I/$p.retemper.json"
done
dry python3 -B -m v2.release.retemper_predictions --predictions "$MLX/output/mlx-diag.predictions.jsonl" \
  --prompts "$G/mlx-diag.prompts.jsonl" --calibration "$CAL" --undo \
  --output "$I/mlx-diag.predictions.jsonl" --receipt "$I/mlx-diag.retemper.json"

mkdir "$D"
dry python3 -B -m v2.eval.same_panel adopt --run-dir "$D" \
  --typed-final "$I/typed-final.predictions.jsonl" --css15 "$I/css15.predictions.jsonl" \
  --public231 "$I/public231.predictions.jsonl" \
  --prior-receipt "$RUN/COLLECT.json" --prior-receipt "$RUN/SEAL.json" --prior-receipt "$CAL" \
  --prior-receipt "$I/typed-final.retemper.json" --prior-receipt "$I/css15.retemper.json" \
  --prior-receipt "$I/public231.retemper.json" \
  --reason "T = 1 (uncalibrated) predictions of 9B M6 $name derived offline from the sealed CAL698 run formal-m6/$name-16k by undoing its per-type temperatures (calibration ${CALSHA:0:8}; softmax(log q * T)); identical answers, only probabilities differ; for the coordinator 23:15 calibration rule"
dry python3 -B -m v2.eval.same_panel seal --run-dir "$D"
dry python3 -B -m v2.eval.same_panel report --run-dir "$D" --label "$label" --tier 9B --family decision2 \
  --count-safetensors "$CKPT"
while read -r dir cname; do
  dry python3 -B -m v2.eval.same_panel compare --run-dir "$D" --comparator-run-dir "$dir" \
    --left-name "$name (T = 1)" --right-name "$cname" > "$D/compare-$cname.log"
done <<EOF
$DATA/dev2/runs/eval/m1/d1-lux1-autotune-cache adopted-1.0
$RUNS/formal-m3/lux1-16k-shared same-renderer-16k
$NIMBLE nimble2
$DATA/dev2/runs/eval/m1-adopt/jpt9b jpt9b
EOF
mkdir "$DM"
dry python3 -B -m v2.eval.multilingual_panel score --panel "$DATA/dev2/private/panels/mlx-diag-v1" \
  --predictions "$I/mlx-diag.predictions.jsonl" --output "$DM/mlx-diag.score.json"
mkdir "$GT"
dry python3 -B -m v2.eval.gates paired --left "$D" --right "$T1" --left-name "$name (T = 1)" \
  --right-name DEV2.0-9B-T1 --output "$GT/PAIRED-vs-DEV2.0-9B-T1.json"
dry python3 -B -m v2.eval.gates paired --left "$D" --right "$NIMBLE" --left-name "$name (T = 1)" \
  --right-name Nimble2 --output "$GT/PAIRED-vs-Nimble2.json"
dry python3 -B -m v2.eval.gates types --run "$D" --label "$name (T = 1)" --output "$GT/types.json"
successor_summary "$GT/successor.json" "$name (T = 1)" "$D" "$GT" "$D/PAIRED-vs-adopted-1.0.json" \
  "$F/$name.gates/MLX-PAIRED-vs-DEV2.0-9B-T1.json" "$F/$name.gates/score-levels.json"
sha256sum "$I"/* "$D"/*.json "$DM/mlx-diag.score.json" "$GT"/*.json 2>/dev/null || true
