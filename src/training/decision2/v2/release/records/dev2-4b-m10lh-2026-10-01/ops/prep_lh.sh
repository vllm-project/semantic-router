#!/usr/bin/env bash
# DEV2.0-4B successor m10-4b-LH (decoder M10; user request 2026-10-01): release inputs on node A, CPU only.
# 1. the frozen package files against the decoder's hash list; 2. T = 1 scored bindings (COORDINATION 2026-10-01
# 13:40 "keep T = 1 everywhere"): the sealed CAL698 formal run and its mlx-diag run returned to T = 1 offline
# (softmax(log q * T); answers unchanged), checked equal to the decoder's own T = 1 derivation, then adopt, seal,
# report, compare (the current revision's T = 1 run, adopted Nox 1.0, the 16K control, Decider 4B, Jet v6.2) and
# mlx-diag score; 3. successor evidence on that run (types, public 231 and card-eligible mlx-diag vs the current
# revision); 4. the v2.release.bf16_copy of the FP32 checkpoint (scored image, no network).
# Usage (node A, from the exact mirror holding this file): bash <mirror>/.../ops/prep_lh.sh derive|bf16
set -euo pipefail
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
G=/data/dev2/private/panels/goldfree
MLXP=/data/dev2/private/panels/mlx-diag-v1
F=/data/dev2/runs/dec/formal/m10
PKG=$F/pkg/m10-4b-LH
P=$PKG/m6/m10-4b-LH
CAL=$P/cal698-16k/calibration.json
RUN=$F/m10-4b-LH
MLX=$F/m10-4b-LH-mlx
DEC_T1=$F/m10-4b-LH-t1-derived-inputs
IN=/data/dev2/runs/release/inputs/dev2-4b-lh
I=$IN/t1
D=/data/dev2/runs/release/dev2-4b-lh-t1-derived
DM=/data/dev2/runs/release/dev2-4b-lh-t1-derived-mlx
CUR=/data/dev2/runs/release/dev2-4b-t1-derived
CURM=$IN/current-mlx
GATES=$IN/gates
LABEL="DEV2.0-4B successor m10-4b-LH at T = 1 (derived from the CAL698 run; post-key same-panel)"
export PYTHONPATH=$S:$S/v2/9b
cd "$S"

case "${1:-}" in
derive)
  sha256sum "$CAL" | cut -c1-64 | grep -qx 8d88e163630da9a8aeba91bb1333ce1f84019fdf082082d7e9a2fe3dc4524e8f
  (cd "$P/checkpoint" && sha256sum -c --quiet ../checkpoint.files.sha256)
  echo "package verified: $(wc -l < "$P/checkpoint.files.sha256") files"
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
    cmp $I/$p.predictions.jsonl $DEC_T1/$p.predictions.jsonl
    echo "$p equals the decoder's T = 1 derivation"
  done
  python3 -B -m v2.release.retemper_predictions --predictions $MLX/output/mlx-diag.predictions.jsonl \
    --prompts $G/mlx-diag.prompts.jsonl --calibration $CAL --undo \
    --output $I/mlx-diag.predictions.jsonl --receipt $I/mlx-diag.retemper.json
  mkdir "$D"
  python3 -B -m v2.eval.same_panel adopt --run-dir $D \
    --typed-final $I/typed-final.predictions.jsonl --css15 $I/css15.predictions.jsonl \
    --public231 $I/public231.predictions.jsonl \
    --prior-receipt $RUN/COLLECT.json --prior-receipt $RUN/SEAL.json --prior-receipt $CAL \
    --prior-receipt $I/typed-final.retemper.json --prior-receipt $I/css15.retemper.json \
    --prior-receipt $I/public231.retemper.json \
    --reason "T = 1 (uncalibrated) predictions of the DEV2.0-4B successor m10-4b-LH derived offline from the sealed CAL698 run m10-4b-LH by undoing its per-type temperatures (calibration 8d88e163; softmax(log q * T)); identical answers, only probabilities differ; COORDINATION 2026-10-01 13:40 keep T = 1"
  python3 -B -m v2.eval.same_panel seal --run-dir $D
  python3 -B -m v2.eval.same_panel report --run-dir $D --label "$LABEL" --tier 4B --family decision2 \
    --count-safetensors $P/checkpoint
  for cmp in "$CUR dev2-4b" "/data/dev2/runs/eval/m1-adopt/nox1 adopted-1.0" \
             "/data/dev2/runs/dec/formal/m3/nox1-16k same-limit-16k" \
             "/data/dev2/runs/eval/m1-adopt/decider4b decider4b" "/data/dev2/runs/eval/m2/q5b-jet62 jet62"; do
    # shellcheck disable=SC2086
    set -- $cmp
    python3 -B -m v2.eval.same_panel compare --run-dir $D --comparator-run-dir "$1" \
      --left-name "DEV2.0-4B m10-4b-LH (T = 1)" --right-name "$2" > "$D/compare-$2.log"
  done
  mkdir -p $DM/output $CURM/output
  cp $I/mlx-diag.predictions.jsonl $DM/output/
  python3 -B -m v2.eval.multilingual_panel score --panel $MLXP \
    --predictions $DM/output/mlx-diag.predictions.jsonl --output $DM/mlx-diag.score.json
  cp /data/dev2/runs/release/inputs/dev2-4b-t1/derived/mlx-diag.predictions.jsonl $CURM/output/
  cp /data/dev2/runs/release/dev2-4b-t1-derived-mlx/mlx-diag.score.json $CURM/
  mkdir "$GATES"
  python3 -B -m v2.eval.gates types --run $D --label "DEV2.0-4B m10-4b-LH (T = 1)" --output $GATES/types.json
  python3 -B -m v2.eval.gates public231 --left $D --right $CUR --left-name "m10-4b-LH (T = 1)" \
    --right-name "DEV2.0-4B (current, T = 1)" --output $GATES/public231-vs-dev2-4b.json
  python3 -B -m lux9b.mlx_paired --left $DM --right $CURM --panel $MLXP --left-name "m10-4b-LH (T = 1)" \
    --right-name "DEV2.0-4B (current, T = 1)" --output $GATES/mlx-paired-vs-dev2-4b.json
  sha256sum $I/* $D/*.json $DM/mlx-diag.score.json $DM/output/* $CURM/output/* $GATES/*
  ;;
bf16)
  IMAGE=decision20-train-fast:host2
  test "$(docker image inspect -f '{{.Id}}' $IMAGE)" = sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
  cpu=(docker run --rm --network none -e HIP_VISIBLE_DEVICES= -e CUDA_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES=
    -e PYTHONPATH="$S" -v "$S:$S:ro" -w "$S" --entrypoint python3)
  "${cpu[@]}" "$IMAGE" -B -m unittest v2.release.tests.test_bf16_copy
  (cd "$P/checkpoint" && sha256sum -c --quiet ../checkpoint.files.sha256)
  out=$IN/bf16
  mkdir "$out"
  "${cpu[@]}" -v "$P/checkpoint:$P/checkpoint:ro" -v "$out:$out" "$IMAGE" \
    -B -m v2.release.bf16_copy --source "$P/checkpoint" --output "$out/checkpoint" --receipt "$out/bf16-copy.json"
  echo "receipt $(sha256sum < "$out/bf16-copy.json" | cut -c1-64) bytes $(du -sb "$out/checkpoint" | cut -f1)"
  ;;
*) echo "usage: prep_lh.sh derive|bf16" >&2; exit 2 ;;
esac
