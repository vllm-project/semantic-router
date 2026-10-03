#!/usr/bin/env bash
# ~27B release-form evidence of a frozen soup without an M6 chain (node B host side): a cross-arm soup M6-IBxIB2-mNN
# (seeds of M6-IB and M6-IB2, m6-tail.sh lsoup) or an M7 arm soup. The Index-first rule (COORDINATION 2026-10-02
# 09:55) gates on the Index run alone; the release still needs the frozen formal package and its references, so this
# runs the M6 chain's per-finalist steps without the development gates, on one node B GPU (0 / 1 / 5, leased for
# track 27b first):
#   readout  CAL698 kernel fit, typed DEV + CSS pilot + HT-DEV v2 (the formal run's adoption input)
#   formal   package (T = 1 unless CAL698 is adopted), typed FINAL / CSS15 / public 231, seal, report, paired compares
#            (also vs M6-IB and M6-IB2); LOADED_PARAMETERS from the soup's LoRA rank
#   mlx      the mlx-diag collection, pushed to node A over the M6 node link (node A's m6-mlx-watch.sh NAME scores and
#            pairs it), then pulled back
#   gates    m6-gates.sh gates NAME (types = the release gate's R3, paired vs A20r / AutoJev-27B / peers, public 231);
#            gates/contrast.json, which m4_contrast refuses to overwrite, is moved to contrast-NAME-<UTC>.json under a lock
#   current  the references against the current release CURRENT (default M6-IB, Vega-27B 781b2b24): paired v3 and
#            public 231 vs its formal run, and node A's mlx-diag pairing vs its collection (MLX_ALSO of the watcher)
#            pulled to gates/mlx/NAME-vs-CURRENT.json
# Each stage is skipped when its output exists. Usage: m6-xarm.sh MIRROR_SHA NAME GPU (detached; log on stdout)
set -euo pipefail
echo "m6 xarm $*: start $(date -u +%FT%TZ)"
SHA=${1:?MIRROR_SHA} NAME=${2:?NAME} GPU=${3:?GPU}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
[[ "$NAME" =~ ^(M6-IBxIB2-m[0-9]{2}|M7-(IB124ML|IB14ML)|M8-(IB14|IB124)|X7-(IBxIB2xIB14ML|4ARM)|X8-(IBxIB2-8|ML)|X9-(LRH2|LRH2xM50|LRH|LRHxM50|ML0|IBxIB2-10))$ ]] ||
  { echo "bad NAME $NAME" >&2; exit 2; }
case "$GPU" in 0 | 1 | 5) ;; *) echo "GPU$GPU is not an M6 node B GPU (0, 1, 5)" >&2; exit 2 ;; esac
S=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2
TAIL=$S/v2/27b/m6/m6-tail.sh GATES=$S/v2/27b/m6/m6-gates.sh
[ -f "$TAIL" ] && [ -f "$GATES" ] || { echo "missing mirror $SHA" >&2; exit 2; }
R=/data/dev2/runs/27b/m6 KEY=/data/dev2/tmp/27b-m6-xfer
CKPT=$R/$NAME/checkpoint
[ -f "$CKPT/soup_manifest.json" ] || { echo "no soup $CKPT" >&2; exit 2; }
X="ssh -i $KEY/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=$KEY/known_hosts -o StrictHostKeyChecking=yes -o BatchMode=yes"
stamp() { date -u +%FT%TZ; }
on_a() { rsync --list-only -e "$X" "root@$(cat "$KEY/peer"):$1" > /dev/null 2>&1; }
to_a() { rsync -a --mkpath -e "$X" "$1" "root@$(cat "$KEY/peer"):$2"; }
rank=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["lora"]["rank"])' "$CKPT/soup_manifest.json")
[[ "$rank" =~ ^[1-9][0-9]{1,3}$ ]] || { echo "bad LoRA rank $rank" >&2; exit 2; }
LOADED=$((25629863936 + 7295488 * rank))
echo "$(stamp) $NAME: rank $rank, loaded parameters $LOADED, node B GPU$GPU"
owner=/data/dev2/leases/gpu$GPU.lock/owner
if ! { [ -s "$owner" ] && grep -qx 'track=27b' "$owner"; }; then bash "$TAIL" lease "$SHA" "$GPU"; fi
[ -f "$R/readouts/$NAME/READOUT-M4B.json" ] || bash "$TAIL" readout "$SHA" "$NAME" "$CKPT" "$GPU"
extra=()
for other in M6-IB M6-IB2; do [ -f "$R/$other/formal/SEAL.json" ] && extra+=("$other=$R/$other/formal"); done
[ -f "$R/$NAME/formal/SEAL.json" ] ||
  LOADED_PARAMETERS=$LOADED EXTRA_COMPARATOR="${extra[*]}" bash "$TAIL" formal "$SHA" "$NAME" "$CKPT" "$GPU"
[ -f "$R/mlx-diag/$NAME/COLLECT.json" ] || bash "$TAIL" mlx "$SHA" "$NAME" "$GPU"
if ! on_a "mlx/$NAME-vs-A20r.json"; then
  bash "$TAIL" mlx-push "$SHA" "$NAME"
  stamp > "$R/logs/$NAME.PUSHED"
  to_a "$R/logs/$NAME.PUSHED" "mlx/$NAME.PUSHED"
  echo "$(stamp) $NAME: mlx-diag pushed; waiting for node A's pairing"
  until on_a "mlx/$NAME-vs-A20r.json"; do sleep 60; done
  sleep 30
fi
[ -f "$R/gates/mlx/$NAME-vs-A20r.json" ] || bash "$TAIL" mlx-pull "$SHA" "$NAME"
if [ ! -f "$R/gates/$NAME/types.json" ]; then
  exec 9> "$R/gates/.contrast.lock"
  flock 9
  [ ! -e "$R/gates/contrast.json" ] || mv "$R/gates/contrast.json" "$R/gates/contrast-before-$NAME-$(date -u +%Y%m%dT%H%M%SZ).json"
  bash "$GATES" "$SHA" gates "$NAME"
  t=$(date -u +%Y%m%dT%H%M%SZ)
  for f in contrast.json contrast.log; do [ ! -e "$R/gates/$f" ] || mv "$R/gates/$f" "$R/gates/${f%%.*}-$NAME-$t.${f#*.}"; done
  flock -u 9
fi
CURRENT=${CURRENT:-M6-IB}
G=$R/gates/$NAME
[ -f "$R/$CURRENT/formal/SEAL.json" ] || { echo "no sealed formal run of the current release $CURRENT" >&2; exit 2; }
(cd "$S" && export PYTHONPATH=$S PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/dev2/tmp &&
  { [ -f "$G/paired-vs-$CURRENT.json" ] || python3 -m v2.eval.gates paired --left "$R/$NAME/formal" --left-name "$NAME" \
    --right "$R/$CURRENT/formal" --right-name "$CURRENT" --output "$G/paired-vs-$CURRENT.json" > "$G/paired-vs-$CURRENT.log"; } &&
  { [ -f "$G/public231-vs-$CURRENT.json" ] || python3 -m v2.eval.gates public231 --left "$R/$NAME/formal" \
    --right "$R/$CURRENT/formal" --left-name "$NAME" --right-name "$CURRENT" --output "$G/public231-vs-$CURRENT.json" \
    > "$G/public231-vs-$CURRENT.log"; })
if [ ! -f "$R/gates/mlx/$NAME-vs-$CURRENT.json" ]; then
  if on_a "mlx/$NAME-vs-$CURRENT.json"; then
    rsync -a -e "$X" "root@$(cat "$KEY/peer"):mlx/$NAME-vs-$CURRENT.json" "$R/gates/mlx/$NAME-vs-$CURRENT.json"
  else
    echo "$(stamp) $NAME: node A has no mlx-diag pairing vs $CURRENT (start its watcher with MLX_ALSO)" >&2
  fi
fi
python3 -c 'import json,sys; t=json.load(open(sys.argv[1])); print("types", json.dumps(t.get("verdicts", t)))' \
  "$G/types.json" | cut -c1-400
echo "m6 xarm $NAME complete: $(stamp)"
