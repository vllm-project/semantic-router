#!/usr/bin/env bash
# Decision-2.0-Lux-9B successor K-a12IB (Index sweep; user Index-first rule 2026-10-02 09:55): release inputs on
# node A, CPU only. Adapted from dev2-9b-ka13ib-2026-10-02/ops/prep.sh; the current revision is K-a13IB.
# derive: the sealed CAL698 formal run formal-m9/K-a12IB-16k and its mlx-diag run returned to T = 1 offline
#   (softmax(log q * T); answers unchanged), then adopt, seal, report, compare (the current revision's T = 1 run
#   dev2-9b-ka13ib-t1-derived, adopted Lux 1.0, the 16K Lux 1.0 control, Nimble v2, JPT-9B) and mlx-diag score;
#   reference evidence on that run (types = the R3 integrity item; public 231 and card-eligible mlx-diag vs the
#   current revision).
# paired: v2.eval.gates paired files (each checked equal to the same_panel compare).
# bf16: the release weights = the Index sweep's v2.release.bf16_copy of the FP32 soup (models/ix1/index-sweep/bf16/
#   IS-K-a12IB, the copy its IX1 run IS-K-a12IB-bf16 scored), hard-linked into the release inputs after a SHA-256
#   check against its receipt's source identity.
# Usage (node A, from the exact mirror holding this file): bash <mirror>/.../ops/prep.sh derive|paired|bf16
set -euo pipefail
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
G=/data/dev2/private/panels/goldfree
MLXP=/data/dev2/private/panels/mlx-diag-v1
F=/data/dev2/runs/9b/formal-m9
CKPT=/data/dev2/runs/9b/m9/soup/K-a12IB/build/K-a12IB
BUILT=/data/dev2/runs/9b/m9/soup/K-a12IB/build/K-a12IB.built.json
FP32_IDENTITY=68fed4cb24ecefb9499f4b532a4a34df621ab89138bc03f2086965f5dc0c35d0
SWEEP_BF16=/data/dev2/models/ix1/index-sweep/bf16/IS-K-a12IB
CAL=$F/K-a12IB-cal/calibration.json
RUN=$F/K-a12IB-16k
MLX=$F/K-a12IB-16k-mlx
IN=/data/dev2/runs/release/inputs/dev2-9b-ka12ib
I=$IN/t1
D=/data/dev2/runs/release/dev2-9b-ka12ib-t1-derived
DM=/data/dev2/runs/release/dev2-9b-ka12ib-t1-derived-mlx
CUR=/data/dev2/runs/release/dev2-9b-ka13ib-t1-derived
CURM=/data/dev2/runs/release/dev2-9b-ka13ib-t1-derived-mlx
GATES=$IN/gates
NAME="Decision-2.0-Lux-9B K-a12IB (T = 1)"
LABEL="Decision-2.0-Lux-9B successor K-a12IB at T = 1 (derived from the CAL698 run; post-key same-panel)"
COMPARATORS="$CUR dev2-9b
/data/dev2/runs/eval/m1/d1-lux1-autotune-cache adopted-1.0
/data/dev2/runs/9b/formal-m3/lux1-16k-shared same-renderer-16k
/data/dev2/runs/eval/m2/q6-nimble2 nimble2
/data/dev2/runs/eval/m1-adopt/jpt9b jpt9b"
export PYTHONPATH=$S:$S/v2/9b
cd "$S"

verify_ckpt() {  # the soup's content manifest (sorted per-file SHA-256 list) equals its build record's
  local want got
  want=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_manifest"])' "$BUILT")
  got=$(cd "$CKPT" && find . -type f | LC_ALL=C sort | xargs -r -P 8 -n 4 sha256sum | LC_ALL=C sort -k2 | sha256sum | cut -c1-64)
  [ "$got" = "$want" ] || { echo "soup content manifest $got != build record $want" >&2; exit 1; }
  echo "soup verified: content manifest ${got:0:12} = $(basename "$BUILT")"
}

case "${1:-}" in
derive)
  verify_ckpt
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["checkpoint"]==sys.argv[2], d' \
    "$F/K-a12IB.inputs.json" "$CKPT"
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
    python3 -B -m v2.release.retemper_predictions --predictions $RUN/output/$p.predictions.jsonl \
      --prompts $G/$p.prompts.jsonl --calibration $CAL --undo \
      --output $I/$p.predictions.jsonl --receipt $I/$p.retemper.json
  done
  python3 -B -m v2.release.retemper_predictions --predictions $MLX/output/mlx-diag.predictions.jsonl \
    --prompts $G/mlx-diag.prompts.jsonl --calibration $CAL --undo \
    --output $I/mlx-diag.predictions.jsonl --receipt $I/mlx-diag.retemper.json
  calsha=$(sha256sum < "$CAL" | cut -c1-8)
  mkdir "$D"
  python3 -B -m v2.eval.same_panel adopt --run-dir $D \
    --typed-final $I/typed-final.predictions.jsonl --css15 $I/css15.predictions.jsonl \
    --public231 $I/public231.predictions.jsonl \
    --prior-receipt $RUN/COLLECT.json --prior-receipt $RUN/SEAL.json --prior-receipt $CAL \
    --prior-receipt $I/typed-final.retemper.json --prior-receipt $I/css15.retemper.json \
    --prior-receipt $I/public231.retemper.json \
    --reason "T = 1 (uncalibrated) predictions of the 9B successor K-a12IB derived offline from the sealed CAL698 run formal-m9/K-a12IB-16k by undoing its per-type temperatures (calibration $calsha; softmax(log q * T)); identical answers, only probabilities differ; every Decision 2.0 model keeps T = 1"
  python3 -B -m v2.eval.same_panel seal --run-dir $D
  python3 -B -m v2.eval.same_panel report --run-dir $D --label "$LABEL" --tier 9B --family decision2 \
    --count-safetensors $CKPT
  while read -r dir cname; do
    python3 -B -m v2.eval.same_panel compare --run-dir $D --comparator-run-dir "$dir" \
      --left-name "$NAME" --right-name "$cname" > "$D/compare-$cname.log"
  done <<< "$COMPARATORS"
  mkdir -p $DM/output
  cp $I/mlx-diag.predictions.jsonl $DM/output/
  python3 -B -m v2.eval.multilingual_panel score --panel $MLXP \
    --predictions $DM/output/mlx-diag.predictions.jsonl --output $DM/mlx-diag.score.json
  mkdir "$GATES"
  python3 -B -m v2.eval.gates types --run $D --label "$NAME" --output $GATES/types.json
  python3 -B -m v2.eval.gates public231 --left $D --right $CUR --left-name "$NAME" \
    --right-name "Decision-2.0-Lux-9B (current K-a13IB, T = 1)" --output $GATES/public231-vs-dev2-9b.json
  python3 -B -m lux9b.mlx_paired --left $DM --right $CURM --panel $MLXP --left-name "$NAME" \
    --right-name "Decision-2.0-Lux-9B (current K-a13IB, T = 1)" --output $GATES/mlx-paired-vs-dev2-9b.json
  sha256sum $I/* $D/*.json $DM/mlx-diag.score.json $DM/output/* $GATES/*
  ;;
paired)
  while read -r dir cname; do
    [[ "$cname" == same-renderer-16k || "$cname" == jpt9b ]] && continue
    python3 -B -m v2.eval.gates paired --left $D --right "$dir" --left-name "$NAME" \
      --right-name "$cname" --output "$GATES/paired-vs-$cname.json"
    python3 - "$GATES/paired-vs-$cname.json" "$D/PAIRED-vs-$cname.json" <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
assert a["point"] == b["point"] and a["ci95"] == b["ci95"] and a["axis_ci95"] == b["axis_ci95"], "paired differs"
print("paired equal to same_panel compare:", sys.argv[1])
PY
  done <<< "$COMPARATORS"
  sha256sum $GATES/paired-vs-*.json
  ;;
bf16)
  verify_ckpt
  out=$IN/bf16
  python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["source_model_sha256"]==sys.argv[2], r["source_model_sha256"]' \
    "$SWEEP_BF16.receipt/bf16-copy.json" "$FP32_IDENTITY"
  mkdir -p "$IN"; mkdir "$out"
  cp -al "$SWEEP_BF16" "$out/checkpoint"
  cp -p "$SWEEP_BF16.receipt/bf16-copy.json" "$out/bf16-copy.json"
  echo "receipt $(sha256sum < "$out/bf16-copy.json" | cut -c1-64) bytes $(du -sb "$out/checkpoint" | cut -f1)"
  ;;
*) echo "usage: prep.sh derive|paired|bf16" >&2; exit 2 ;;
esac
