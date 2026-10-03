#!/usr/bin/env bash
# Decision-2.0-Lux-9B Index-first successor of KIB4-a40 (9B M10 amendment 7): release inputs of one candidate on node A,
# CPU only, from the exact mirror holding this file (as dev2-9b-m10-2026-10-02/ops/prep_m10.sh, with the KIB4-a40 weights
# as the comparator; the current revision is their runtime-only switch revision 214ffa43).
#   current  the current revision's sealed gate receipt and final decision (the Lux switch release, record
#            dev2-runtime-a-2026-10-02 9b/switch) -> $IN/current/, checked against make_m10c.py's digests
#   derive   the sealed CAL698 formal run formal-m9/M10-CAND-16k and its mlx-diag run returned to T = 1 offline
#            (softmax(log q * T); answers unchanged), then adopt, seal, report, compare (the current revision's T = 1
#            run, adopted Lux 1.0, the 16K Lux 1.0 control, Nimble v2, JPT-9B) and mlx-diag score; gates types,
#            public 231 and card-eligible mlx-diag vs the current revision
#   paired   v2.eval.gates paired files (vs the current revision, adopted Lux 1.0, Nimble v2), each checked equal to
#            the same_panel compare
# Usage (node A): bash <mirror>/v2/release/records/dev2-9b-m10c-2026-10-03/ops/prep_m10c.sh CAND current|derive|paired
set -euo pipefail
CAND=${1:?CAND} STAGE=${2:?STAGE}
[[ "$CAND" =~ ^[A-Za-z0-9]+(-[A-Za-z0-9]+)*$ ]] || { echo "bad CAND $CAND" >&2; exit 2; }
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
R=$S/v2/release/records/dev2-9b-m10c-2026-10-03
RRA=$S/v2/release/records/dev2-runtime-a-2026-10-02
G=/data/dev2/private/panels/goldfree
MLXP=/data/dev2/private/panels/mlx-diag-v1
F=/data/dev2/runs/9b/formal-m9
RUN=$F/M10-$CAND-16k
MLX=$F/M10-$CAND-16k-mlx
CAL=$F/M10-$CAND-cal/calibration.json
CKPT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$F/M10-$CAND.inputs.json" 2> /dev/null || true)
IN=/data/dev2/runs/release/inputs/dev2-9b-m10c-$CAND
I=$IN/t1
D=/data/dev2/runs/release/dev2-9b-m10c-$CAND-t1-derived
DM=$D-mlx
CUR=/data/dev2/runs/release/dev2-9b-m10-KIB4-a40-t1-derived
CURM=/data/dev2/runs/release/dev2-9b-m10-KIB4-a40-t1-derived-mlx
GATES=$IN/gates
NAME="Decision-2.0-Lux-9B M10 $CAND (T = 1)"
LABEL="Decision-2.0-Lux-9B candidate 9B M10 $CAND at T = 1 (derived from the CAL698 run; post-key same-panel)"
COMPARATORS="$CUR lux-9b
/data/dev2/runs/eval/m1/d1-lux1-autotune-cache adopted-1.0
/data/dev2/runs/9b/formal-m3/lux1-16k-shared same-renderer-16k
/data/dev2/runs/eval/m2/q6-nimble2 nimble2
/data/dev2/runs/eval/m1-adopt/jpt9b jpt9b"
export PYTHONPATH=$S:$S/v2/9b
cd "$S"

case "$STAGE" in
current)
  mkdir -p "$IN/current"
  for pair in "$RRA/9b/switch/release/receipts/gate.json:ras-gate.json" \
    "$RRA/Decision-2.0-Lux-9B.decision.ras.json:Decision-2.0-Lux-9B.decision.ras.json"; do
    src=${pair%%:*} dst=$IN/current/${pair#*:}
    if [[ -e "$dst" ]]; then cmp "$src" "$dst"; else cp "$src" "$dst"; chmod 444 "$dst"; fi
  done
  python3 - "$IN/current" "$R/ops/make_m10c.py" <<'PY'
import hashlib, re, sys
from pathlib import Path
text = Path(sys.argv[2]).read_text()
for name, key in (("ras-gate.json", "gate_sha256"), ("Decision-2.0-Lux-9B.decision.ras.json", "decision_sha256")):
    want = re.search(rf'"{key}": "([0-9a-f]{{64}})"', text).group(1)
    got = hashlib.sha256((Path(sys.argv[1]) / name).read_bytes()).hexdigest()
    assert got == want, (name, got)
    print(name, got[:12])
PY
  ;;
derive)
  [[ -n "$CKPT" && -f "$CKPT/decision_config.json" ]] || { echo "no formal inputs for M10-$CAND" >&2; exit 1; }
  for f in "$RUN/SEAL.json" "$MLX/output/mlx-diag.predictions.jsonl" "$CAL" "$CUR/REPORT.json" \
    "$CURM/output/mlx-diag.predictions.jsonl"; do
    [[ -f "$f" ]] || { echo "missing $f" >&2; exit 1; }
  done
  python3 - "$RUN/SEAL.json" "$RUN/output" <<'PY'
import hashlib, json, sys
seal = json.load(open(sys.argv[1]))
for panel, v in seal["panels"].items():
    got = hashlib.sha256(open(f"{sys.argv[2]}/{panel}.predictions.jsonl", "rb").read()).hexdigest()
    assert got == v["predictions_sha256"], (panel, got)
    print("sealed predictions match:", panel, got[:12])
PY
  mkdir -p "$IN"; mkdir "$I"
  for p in typed-final css15 public231; do
    python3 -B -m v2.release.retemper_predictions --predictions "$RUN/output/$p.predictions.jsonl" \
      --prompts "$G/$p.prompts.jsonl" --calibration "$CAL" --undo \
      --output "$I/$p.predictions.jsonl" --receipt "$I/$p.retemper.json"
  done
  python3 -B -m v2.release.retemper_predictions --predictions "$MLX/output/mlx-diag.predictions.jsonl" \
    --prompts "$G/mlx-diag.prompts.jsonl" --calibration "$CAL" --undo \
    --output "$I/mlx-diag.predictions.jsonl" --receipt "$I/mlx-diag.retemper.json"
  calsha=$(sha256sum < "$CAL" | cut -c1-8)
  mkdir "$D"
  python3 -B -m v2.eval.same_panel adopt --run-dir "$D" \
    --typed-final "$I/typed-final.predictions.jsonl" --css15 "$I/css15.predictions.jsonl" \
    --public231 "$I/public231.predictions.jsonl" \
    --prior-receipt "$RUN/COLLECT.json" --prior-receipt "$RUN/SEAL.json" --prior-receipt "$CAL" \
    --prior-receipt "$I/typed-final.retemper.json" --prior-receipt "$I/css15.retemper.json" \
    --prior-receipt "$I/public231.retemper.json" \
    --reason "T = 1 (uncalibrated) predictions of the 9B M10 continuation candidate $CAND derived offline from the sealed CAL698 run formal-m9/M10-$CAND-16k by undoing its per-type temperatures (calibration $calsha; softmax(log q * T)); identical answers, only probabilities differ; every Decision 2.0 model keeps T = 1"
  python3 -B -m v2.eval.same_panel seal --run-dir "$D"
  python3 -B -m v2.eval.same_panel report --run-dir "$D" --label "$LABEL" --tier 9B --family decision2 \
    --count-safetensors "$CKPT"
  while read -r dir cname; do
    python3 -B -m v2.eval.same_panel compare --run-dir "$D" --comparator-run-dir "$dir" \
      --left-name "$NAME" --right-name "$cname" > "$D/compare-$cname.log"
  done <<< "$COMPARATORS"
  mkdir -p "$DM/output"
  cp "$I/mlx-diag.predictions.jsonl" "$DM/output/"
  python3 -B -m v2.eval.multilingual_panel score --panel $MLXP \
    --predictions "$DM/output/mlx-diag.predictions.jsonl" --output "$DM/mlx-diag.score.json"
  mkdir "$GATES"
  python3 -B -m v2.eval.gates types --run "$D" --label "$NAME" --output "$GATES/types.json"
  python3 -B -m v2.eval.gates public231 --left "$D" --right "$CUR" --left-name "$NAME" \
    --right-name "Decision-2.0-Lux-9B (current, T = 1)" --output "$GATES/public231-vs-lux-9b.json"
  python3 -B -m lux9b.mlx_paired --left "$DM" --right "$CURM" --panel $MLXP --left-name "$NAME" \
    --right-name "Decision-2.0-Lux-9B (current, T = 1)" --output "$GATES/mlx-paired-vs-lux-9b.json"
  sha256sum "$I"/* "$D"/*.json "$DM/mlx-diag.score.json" "$DM"/output/* "$GATES"/*
  ;;
paired)
  while read -r dir cname; do
    [[ "$cname" == same-renderer-16k || "$cname" == jpt9b ]] && continue
    python3 -B -m v2.eval.gates paired --left "$D" --right "$dir" --left-name "$NAME" \
      --right-name "$cname" --output "$GATES/paired-vs-$cname.json"
    python3 - "$GATES/paired-vs-$cname.json" "$D/PAIRED-vs-$cname.json" <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
assert a["point"] == b["point"] and a["ci95"] == b["ci95"] and a["axis_ci95"] == b["axis_ci95"], "paired differs"
print("paired equal to same_panel compare:", sys.argv[1])
PY
  done <<< "$COMPARATORS"
  sha256sum "$GATES"/paired-vs-*.json
  ;;
*) echo "usage: prep_m10c.sh CAND current|derive|paired" >&2; exit 2 ;;
esac
