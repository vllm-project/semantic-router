#!/usr/bin/env bash
# Decision-2.0-Lux-9B successor K-a13IB (9B Milestone 9; coordinator decision 2026-10-02 02:05, the Index path):
# release inputs on node A, CPU only.
# derive: the sealed CAL698 formal run formal-m9/K-a13IB-16k and its mlx-diag run returned to T = 1 offline
#   (softmax(log q * T); answers unchanged), then adopt, seal, report, compare (the current revision's T = 1 run,
#   adopted Lux 1.0, the 16K Lux 1.0 control, Nimble v2, JPT-9B) and mlx-diag score; successor evidence on that run
#   (types, public 231 and card-eligible mlx-diag vs the current revision).
# paired: v2.eval.gates paired files (the release gate's paired files name their runs), each checked equal to the
#   same_panel compare.
# bf16: the v2.release.bf16_copy of the FP32 soup (scored image, no network).
# Usage (node A, from the exact mirror holding this file): bash <mirror>/.../ops/prep.sh derive|paired|bf16
set -euo pipefail
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
G=/data/dev2/private/panels/goldfree
MLXP=/data/dev2/private/panels/mlx-diag-v1
F=/data/dev2/runs/9b/formal-m9
CKPT=/data/dev2/runs/9b/m9/soup/K-a13IB/build/K-a13IB
SUMS=/data/dev2/runs/9b/m9/soup/K-a13IB/SHA256SUMS
CAL=$F/K-a13IB-cal/calibration.json
RUN=$F/K-a13IB-16k
MLX=$F/K-a13IB-16k-mlx
IN=/data/dev2/runs/release/inputs/dev2-9b-ka13ib
I=$IN/t1
D=/data/dev2/runs/release/dev2-9b-ka13ib-t1-derived
DM=/data/dev2/runs/release/dev2-9b-ka13ib-t1-derived-mlx
CUR=/data/dev2/runs/release/dev2-8b-t1-derived
CURM=$IN/current-mlx
GATES=$IN/gates
NAME="Decision-2.0-Lux-9B K-a13IB (T = 1)"
LABEL="Decision-2.0-Lux-9B successor K-a13IB at T = 1 (derived from the CAL698 run; post-key same-panel)"
COMPARATORS="$CUR dev2-9b
/data/dev2/runs/eval/m1/d1-lux1-autotune-cache adopted-1.0
/data/dev2/runs/9b/formal-m3/lux1-16k-shared same-renderer-16k
/data/dev2/runs/eval/m2/q6-nimble2 nimble2
/data/dev2/runs/eval/m1-adopt/jpt9b jpt9b"
export PYTHONPATH=$S:$S/v2/9b
cd "$S"

verify_ckpt() {
  (cd "$CKPT" && sha256sum -c --quiet "$SUMS")
  echo "soup verified: $(wc -l < "$SUMS") files (SHA256SUMS $(sha256sum < "$SUMS" | cut -c1-12))"
}

case "${1:-}" in
derive)
  verify_ckpt
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["checkpoint"]==sys.argv[2], d' \
    "$F/K-a13IB.inputs.json" "$CKPT"
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
    --reason "T = 1 (uncalibrated) predictions of the 9B M9 successor K-a13IB derived offline from the sealed CAL698 run formal-m9/K-a13IB-16k by undoing its per-type temperatures (calibration $calsha; softmax(log q * T)); identical answers, only probabilities differ; every Decision 2.0 model keeps T = 1"
  python3 -B -m v2.eval.same_panel seal --run-dir $D
  python3 -B -m v2.eval.same_panel report --run-dir $D --label "$LABEL" --tier 9B --family decision2 \
    --count-safetensors $CKPT
  while read -r dir cname; do
    python3 -B -m v2.eval.same_panel compare --run-dir $D --comparator-run-dir "$dir" \
      --left-name "$NAME" --right-name "$cname" > "$D/compare-$cname.log"
  done <<< "$COMPARATORS"
  mkdir -p $DM/output $CURM/output
  cp $I/mlx-diag.predictions.jsonl $DM/output/
  python3 -B -m v2.eval.multilingual_panel score --panel $MLXP \
    --predictions $DM/output/mlx-diag.predictions.jsonl --output $DM/mlx-diag.score.json
  cp /data/dev2/runs/release/inputs/dev2-8b-t1/derived/mlx-diag.predictions.jsonl $CURM/output/
  cp /data/dev2/runs/release/dev2-8b-t1-derived-mlx/mlx-diag.score.json $CURM/
  mkdir "$GATES"
  python3 -B -m v2.eval.gates types --run $D --label "$NAME" --output $GATES/types.json
  python3 -B -m v2.eval.gates public231 --left $D --right $CUR --left-name "$NAME" \
    --right-name "Decision-2.0-Lux-9B (current, T = 1)" --output $GATES/public231-vs-dev2-9b.json
  python3 -B -m lux9b.mlx_paired --left $DM --right $CURM --panel $MLXP --left-name "$NAME" \
    --right-name "Decision-2.0-Lux-9B (current, T = 1)" --output $GATES/mlx-paired-vs-dev2-9b.json
  sha256sum $I/* $D/*.json $DM/mlx-diag.score.json $DM/output/* $CURM/output/* $GATES/*
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
  IMAGE=decision20-train-fast:host2
  test "$(docker image inspect -f '{{.Id}}' $IMAGE)" = sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
  cpu=(docker run --rm --network none -e HIP_VISIBLE_DEVICES= -e CUDA_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES=
    -e PYTHONPATH="$S" -v "$S:$S:ro" -w "$S" --entrypoint python3)
  "${cpu[@]}" "$IMAGE" -B -m unittest v2.release.tests.test_bf16_copy
  verify_ckpt
  out=$IN/bf16
  mkdir -p "$IN"; mkdir "$out"
  "${cpu[@]}" -v "$CKPT:$CKPT:ro" -v "$out:$out" "$IMAGE" \
    -B -m v2.release.bf16_copy --source "$CKPT" --output "$out/checkpoint" --receipt "$out/bf16-copy.json"
  echo "receipt $(sha256sum < "$out/bf16-copy.json" | cut -c1-64) bytes $(du -sb "$out/checkpoint" | cut -f1)"
  ;;
*) echo "usage: prep.sh derive|paired|bf16" >&2; exit 2 ;;
esac
