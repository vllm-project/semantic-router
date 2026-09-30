#!/usr/bin/env bash
# usage: m6-htdev2.sh SHA GPU NAME   (after m6-formal.sh NAME; or via m6-post.sh)
# 9B Milestone 6 HT-DEV v2 diagnostic for one finalist NAME (COORDINATION 04:10: M6's development rule was frozen
# before HT-DEV v2 passed, so it is reported only, never a selection criterion). ht-dev2 is collected like the
# 9B reference 9b-m4-K-a13 (eval htdev2 job): the eval collection runtime (mirror bc0a12d70; the reference ran
# 8097e859b, identical collection code), v2.dec.infer_dec at 16,384 tokens with NAME's CAL698 temperatures, and
# a fresh copy of formal-m6/triton-cache; then v2.eval.dev_readout against the reference predictions into
# formal-m6/NAME-htdev2/readout.json (H_dev2, per-task macro-F1, paired delta, CI and verdict). Stops at the
# first failed step.
set -uo pipefail
SHA=${1:?mirror sha}; G=${2:?gpu}; NAME=${3:?finalist}
L=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2/v2/9b/lux9b/m6
[ -f "$L/chain-step.sh" ] || { echo "mirror $L missing" >&2; exit 2; }
. "$L/lib.sh"
C=m6-htdev2-$NAME
HT=bc0a12d70472c65c6ee68e16227f8173e63f007c
S=$(code_dir "$HT")
F=$RUNS/formal-m6
RUN=$F/$NAME-htdev2
TC=$F/$NAME-htdev2.triton-cache
REF=$DATA/dev2/runs/eval/htdev2/collect/9b-m4-K-a13/output/ht-dev2.predictions.jsonl
LUX=$DATA/decision20-20260926/models/Decision-1.0-Lux-9B
CKPT=$M6/$NAME-build/soup
CAL=$M6/$NAME-cal
step() { bash "$L/chain-step.sh" "$SHA" "$G" "$C" "$@"; }
[ -d "$S" ] || { echo "collection mirror $S missing" >&2; exit 2; }
[ -f "$F/$NAME-16k/SEAL.json" ] || { echo "no sealed formal run for $NAME" >&2; exit 2; }
[ -f "$CKPT/decision_config.json" ] && [ -f "$CAL/calibration.json" ] && [ -s "$REF" ] \
  || { echo "missing checkpoint, calibration or reference for $NAME" >&2; exit 2; }
[ ! -e "$RUN" ] && [ ! -e "$TC" ] || { echo "$RUN or $TC exists" >&2; exit 66; }
budget_ok 0.1 || exit 1
cp -a "$F/triton-cache" "$TC" || exit 1
printf '{"source": "%s", "copied_utc": "%s", "files": %s, "tree_sha256": "%s"}\n' "$F/triton-cache" \
  "$(date -u +%FT%TZ)" "$(find "$TC" -type f | wc -l)" "$(tree_sha "$TC")" > "$F/$NAME-htdev2.cache.json"
step "$NAME-htdev2" 20 -- bash "$S/v2/eval/run_same_panel.sh" --gpu "$G" --track "$TRACK" \
  --src "$HT-src_training_decision2" --run-dir "$RUN" --model-dir "$CKPT" --mount "$LUX" --mount "$CAL" \
  --env TRITON_CACHE_AUTOTUNING=1 --env "TRITON_CACHE_DIR=$TC" --mount-rw "$TC" \
  --purpose "9B M6 HT-DEV v2 diagnostic $NAME" -- \
  --adapter-spec "$S/v2/dec/adapter-spec-infer-dec.json" --model-path "$CKPT" --revision "m6-$NAME-soup" \
  --extra source="$LUX" --extra model_id="decision2-9b-m6-$NAME" --extra max_length=16384 \
  --extra calibration="$CAL/calibration.json" --panels ht-dev2 || exit 1
(cd "$S" && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$S" dry python3 -m v2.eval.dev_readout --run-dir "$RUN" \
  --label "$NAME" --output "$RUN/readout.json" --htdev2-reference "$REF") > "$RUN/readout.console" || exit 1
python3 -c 'import json,sys; d=json.load(open(sys.argv[1]))["htdev2"]; v=d.get("vs_reference", {}); print("htdev2", sys.argv[2], "H_dev2 %.4f" % d["H_dev2"], {k: v.get(k) for k in ("delta", "ci95", "verdict")})' \
  "$RUN/readout.json" "$NAME"
echo "chain $C done"
