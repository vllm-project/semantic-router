#!/usr/bin/env bash
# Decoder M4 formal run for one finalist on node A GPU5 (prereg dec-m4-prereg-2026-09-29.md): download the
# staged folder at its commit, verify every file against node B's SHA-256 list, collect once at 16,384
# tokens (smoke first) with the M3 formal procedure (image of the adopted Nox 1.0 run, per-run persisted
# autotune cache), score against the adopted Nox 1.0 run (the bar), the 16K Nox 1.0 control, Decider 4B,
# Jet v6.2 and the M3 N4LKr soup, then mlx-diag.
# usage: m4-formal.sh <mirror-dir> <name e.g. N4XA-soup> <staging commit> <node-B sha256 list (node A path)>
set -u
SRC=$1 NAME=$2 COMMIT=$3 LIST=$4
S=/data/dev2/src/$SRC/src/training/decision2
# shellcheck source=src/training/decision2/v2/dec/ops/m4/m4-formal-lib.sh
. "$S/v2/dec/ops/m4/m4-formal-lib.sh"
D=/data/dev2/runs/dec/m4/formal-candidates/staging-${COMMIT:0:8}
mkdir -p "$D"
export HF_HUB_CACHE=/data/dev2/hf-cache
/data/dev2/tools/hf-cli/bin/hf download llm-semantic-router/dev2-dec-staging --revision "$COMMIT" --include "m4/$NAME/*" \
  --local-dir "$D" > "$D/$NAME.download.log" 2>&1 || { flog "$NAME download FAILED"; exit 1; }
if (cd "$D" && sha256sum -c --quiet "$LIST" > "$D/$NAME.verify.log" 2>&1); then
  flog "$NAME download verified ($(wc -l < "$LIST") files, staging $COMMIT)"
else
  flog "$NAME hash verification FAILED"
  exit 1
fi
P=$D/m4/$NAME
NOX=/data/dev2/hf-cache/models--llm-semantic-router--Decision-1.0-Nox-4B/snapshots/cde2a68dbaa557ea65dc458104d410a0802ee259
spec=$S/v2/dec/adapter-spec-infer-dec.json
common=(--extra "source=$NOX" --extra "model_id=decision2-dec-m4-$NAME" --extra max_length=16384
  --extra "calibration=$P/cal698-16k/calibration.json" --mount "$D")
collect "$SRC" "m4-$NAME-nodeA" "$P/checkpoint" "m4-$NAME-$COMMIT" "$spec" "M4 formal post-key collection $NAME (prereg 0b8ebfc54)" "${common[@]}" || exit 1
E=/data/dev2/runs/eval
score "$SRC" "m4-$NAME-nodeA" "decision2-dec-m4-$NAME" 4B decision2 "$P/checkpoint" \
  "$E/m1-adopt/nox1" adopted-1.0 /data/dev2/runs/dec/formal/m3/nox1-16k same-limit-16k \
  "$E/m1-adopt/decider4b" decider4b "$E/m2/q5b-jet62" jet62 /data/dev2/runs/dec/formal/m3/m3-N4LKr-soup-nodeA m3-n4lkr
mlx "$SRC" "m4-$NAME" "$P/checkpoint" "m4-$NAME-$COMMIT" "$spec" "${common[@]}"
flog "$NAME formal chain done"
