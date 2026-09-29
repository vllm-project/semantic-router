#!/usr/bin/env bash
# JevArena-C1 v1.2 post-key successor guard (successor-rule item 8), node A.
# C1 is post-key after its three scoring events: it is never training data and never a selection
# criterion (development or siblings), and a card may report it only as "JevArena-C1 v1.2,
# post-key (not an independent validation)".
# Usage (on node A from an exact mirror; the key arrives once on stdin, never in argv or a file):
#   c1-postkey.sh collect --gpu N --src MIRROR --spec SPEC [--lease-name owner.eval] [--shared]
#                         [--approval TEXT] [--verify-only | --preflight-only]
#   c1-postkey.sh gate --src MIRROR --left RUN --right RUN --left-name A --right-name B --output OUT
#   c1-postkey.sh reproduce --src MIRROR --output-dir DIR [--event-dir DIR]
#   c1-postkey.sh rescore --src MIRROR --run RUN --label NAME --output OUT
# collect: SPEC (dev2-c1-postkey-spec/1, absolute or relative to the mirror's src/training/decision2)
#   is one frozen release package with its formal runtime. The plan resolves it against
#   v2/eval/sealed/c1-postkey-baselines.json: a successor is gated against its tier's registered
#   baseline (item 8), a current revision builds its tier's baseline. Then verification (CPU), the
#   smoke with exact parity against the stored formal run, and the ledger rule (one successor per
#   baseline, no second scoring of the same weights, unless --approval records the coordinator's
#   approval). Only then is the key read: prompts are decrypted to the panel root, the package is
#   collected, its predictions are sealed with the post-key label and the prompts removed, the gold
#   is decrypted to a private temp dir, the run is scored on v1.2, v2.eval.gates c1 runs against
#   every comparison run, the gold is removed, SUMMARY.json is written and the custodial ledger
#   ($C1/postkey/LEDGER.jsonl) gets one line. --verify-only and --preflight-only never read the key.
#   Job dir: /data/dev2/runs/eval/c1-postkey/<tier>/<name>-<UTC>; the run is <job>/cand.
# gate: v2.eval.gates c1 between two sealed C1 runs; the gold exists only during the call.
# reproduce: v2.eval.gates c1 on every pair of the event-3 plan, compared with the event's files.
# rescore: a stored sealed C1 run (e.g. an earlier event's baseline) scored on v1.2 with the post-key label;
#   OUT (a REPORT-C1.json) goes outside the run directory.
# One post-key process at a time (lock). Plaintext is removed on any exit, each key-reading step is
# logged in C1's ACCESS.log, and each runner job writes only the named lease entry.
set -uo pipefail
umask 077

MODE=${1:-}
[ $# -gt 0 ] && shift
case $MODE in
collect | gate | reproduce | rescore) ;;
*) echo "usage: c1-postkey.sh collect|gate|reproduce|rescore ... (see the header)" >&2; exit 2 ;;
esac
GPU="" SRC="" SPEC="" LEASE_NAME=owner.eval SHARED=0 APPROVAL="" PHASE=run J=""
LEFT="" RIGHT="" LEFT_NAME="" RIGHT_NAME="" OUTPUT="" OUT_DIR="" RUN="" LABEL=""
EVENT_DIR=/data/dev2/runs/eval/m4/c1-event3
while [ $# -gt 0 ]; do
  case $1 in
  --gpu) GPU=$2; shift 2 ;;
  --src) SRC=$2; shift 2 ;;
  --spec) SPEC=$2; shift 2 ;;
  --lease-name) LEASE_NAME=$2; shift 2 ;;
  --shared) SHARED=1; shift ;;
  --approval) APPROVAL=$2; shift 2 ;;
  --verify-only) PHASE=verify; shift ;;
  --preflight-only) PHASE=preflight; shift ;;
  --left) LEFT=$2; shift 2 ;;
  --right) RIGHT=$2; shift 2 ;;
  --left-name) LEFT_NAME=$2; shift 2 ;;
  --right-name) RIGHT_NAME=$2; shift 2 ;;
  --output) OUTPUT=$2; shift 2 ;;
  --output-dir) OUT_DIR=$2; shift 2 ;;
  --event-dir) EVENT_DIR=$2; shift 2 ;;
  --run) RUN=$2; shift 2 ;;
  --label) LABEL=$2; shift 2 ;;
  *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
# No child process may read the key: stdin becomes /dev/null; only a key-reading run keeps it on fd 3.
if [ "$PHASE" = run ]; then
  exec 3<&0 0</dev/null
else
  exec 0</dev/null
fi
[ -n "$SRC" ] || { echo "--src MIRROR is required" >&2; exit 2; }
case $MODE in
collect)
  [ -n "$SPEC" ] || { echo "collect needs --spec" >&2; exit 2; }
  if [ "$PHASE" != verify ]; then
    [[ "$GPU" =~ ^[0-7]$ ]] || { echo "--gpu 0-7 is required" >&2; exit 2; }
  fi
  [[ "$LEASE_NAME" =~ ^owner\.[a-z0-9-]+$ ]] || { echo "lease name must be a named entry owner.NAME" >&2; exit 2; }
  ;;
gate)
  [ "$PHASE" = run ] || { echo "gate has no --verify-only / --preflight-only" >&2; exit 2; }
  [ -n "$LEFT" ] && [ -n "$RIGHT" ] && [ -n "$LEFT_NAME" ] && [ -n "$RIGHT_NAME" ] && [ -n "$OUTPUT" ] ||
    { echo "gate needs --left --right --left-name --right-name --output" >&2; exit 2; }
  ;;
reproduce)
  [ "$PHASE" = run ] || { echo "reproduce has no --verify-only / --preflight-only" >&2; exit 2; }
  [ -n "$OUT_DIR" ] || { echo "reproduce needs --output-dir" >&2; exit 2; }
  ;;
rescore)
  [ "$PHASE" = run ] || { echo "rescore has no --verify-only / --preflight-only" >&2; exit 2; }
  [ -n "$RUN" ] && [ -n "$LABEL" ] && [ -n "$OUTPUT" ] || { echo "rescore needs --run --label --output" >&2; exit 2; }
  case $OUTPUT in "${RUN%/}"/*) echo "--output must be outside the run directory" >&2; exit 2 ;; esac
  ;;
esac
for path in "$LEFT" "$RIGHT" "$OUTPUT" "$OUT_DIR" "$EVENT_DIR" "$RUN"; do
  case $path in "" | /*) ;; *) echo "paths must be absolute: $path" >&2; exit 2 ;; esac
done
[ -f "/data/dev2/src/$SRC/.dev2-mirror.json" ] || { echo "no verified mirror $SRC" >&2; exit 1; }

S="/data/dev2/src/$SRC/src/training/decision2"
C1=/data/dev2/private/sealed/c1
PK=$C1/postkey
LEDGER=$PK/LEDGER.jsonl
PANEL=/data/dev2/private/panels/goldfree/sealed-c1.prompts.jsonl
PROMPTS_SHA=0b29686f60c980f3fbc8a03b88537fc0bf90ee967afa67d4fe0c958b1bfde16a
GOLD_SHA=c02777713c0e58b40cf602947433420744b2765465252ce22d692eb676ca4fe1
BUNDLE_SHA=d924389a7ffc3d852d9204c3dfca1c12c32f82685a7e5c65277ba9980110534f
RETIRED=$C1/v1_2/RETIRED-v1_2.json
RETIRED_SHA=bbf095c70917f028d691fce570b114725e4f22fd6a1987456f6ae2c30f22990a
REGISTRY="$S/v2/eval/sealed/c1-postkey-baselines.json"
JOBS=/data/dev2/runs/eval/c1-postkey
LEASE="/data/dev2/leases/gpu$GPU.lock/$LEASE_NAME"
TC="$S/v2/27b/triton_cache.py"
UTC=$(date -u +%Y%m%dT%H%M%SZ)
export PYTHONPATH="$S" PYTHONDONTWRITEBYTECODE=1
cd "$S" || exit 1
mkdir -p "$PK"
exec 9>>"$PK/.lock"
flock -n 9 || { echo "another post-key C1 process holds $PK/.lock" >&2; exit 1; }

DECRYPTED=0 GT="" LEASED=0 KEY="" LOG="" PLAINTEXT=""
log() {
  local t
  t=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if [ -n "$LOG" ]; then echo "$t $*" >>"$LOG"; fi
  if [ "$PHASE" = run ]; then echo "$t POSTKEY $MODE $*" >>"$C1/ACCESS.log"; fi
  echo "$t $*"
}
abort() {
  log "ABORT: $*"
  exit 1
}
cleanup() {
  local code=$?
  if [ "$DECRYPTED" = 1 ]; then rm -f "$PANEL"; fi
  if [ -n "$GT" ]; then rm -rf "$GT"; fi
  KEY=""
  unset KEY
  if [ "$LEASED" = 1 ]; then
    if [ -f "$J/lease-before.owner" ]; then
      cp -p "$J/lease-before.owner" "$LEASE"
    else
      printf 'track=eval\nstatus=idle-released (C1 post-key %s done)\nlast_job_end_utc=%s\n' "$PHASE" "$(date -u +%FT%TZ)" >"$LEASE"
    fi
  fi
  local what="nothing was decrypted" lease="no GPU lease taken"
  if [ -n "$PLAINTEXT" ]; then what="$PLAINTEXT decrypted and removed"; fi
  if [ "$LEASED" = 1 ]; then lease="GPU$GPU entry $LEASE_NAME restored"; fi
  log "cleanup (exit $code): $what; $lease"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
decrypt() {
  openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -pass fd:4 -in "$C1/c1-v1-bundle.tar.enc" 4<<<"$KEY" | tar -xOf - "$1"
}
helper() { python3 -m v2.eval.sealed.postkey "$@" 3<&-; }
field() { helper field --plan "$PLAN" --field "$1"; }
read_key() {
  [ ! -e "$PANEL" ] || abort "C1 prompts already sit on the panel root (another C1 process or an unclean exit); key not read"
  IFS= read -r KEY <&3 || [ -n "$KEY" ] || abort "no key on stdin"
  exec 3<&-
  [ -n "$KEY" ] || abort "empty key"
  [ "$(sha256sum <"$C1/c1-v1-bundle.tar.enc" | cut -c1-64)" = "$BUNDLE_SHA" ] || abort "encrypted bundle hash mismatch"
}
gold_in() {
  GT=$(mktemp -d "$C1/.gold.XXXXXX")
  PLAINTEXT="${PLAINTEXT:+$PLAINTEXT and }gold"
  decrypt v1/build-5/gold.jsonl >"$GT/gold.jsonl"
  [ "$(sha256sum <"$GT/gold.jsonl" | cut -c1-64)" = "$GOLD_SHA" ] || abort "gold hash mismatch"
  [ "$(sha256sum <"$RETIRED" | cut -c1-64)" = "$RETIRED_SHA" ] || abort "the v1.2 retired list differs from ${RETIRED_SHA:0:12}"
  log "gold decrypted to a private temp dir"
}
gold_out() {
  rm -rf "$GT"
  GT=""
  log "gold removed"
}
drain() {
  local i v=""
  for ((i = 0; i < 60; i++)); do
    v=$({ rocm-smi -d "$GPU" --showmemuse 2>/dev/null | awk -F': ' '/VRAM%/ {v = $NF} END {print v}'; } 3<&-)
    [ "$v" = 0 ] && break
    sleep 2 3<&-
  done
  if [ "$i" -gt 0 ]; then log "GPU$GPU VRAM ${v:-unknown}% before $1; waited $((2 * i)) s"; fi
}
# phase(smoke|collect) run-dir: one runner job, with a fresh copy of the frozen autotune cache if any
job() {
  local phase=$1 d=$2 frozen cache="" rc
  local -a extra=() argv=()
  frozen=$(field model.cache.frozen)
  if [ -n "$frozen" ]; then
    cache="$d-triton"
    python3 "$TC" copy --frozen "$frozen" --dest "$cache" --expect "$(field model.cache.sha256)" >>"$LOG" 2>&1 3<&- || return 1
    extra=(--cache-dir "$cache")
  fi
  if [ "$SHARED" = 1 ]; then extra+=(--shared); else drain "$phase"; fi
  helper argv --plan "$PLAN" --phase "$phase" --run-dir "$d" --gpu "$GPU" --src "$SRC" --src-root "$S" \
    --lease-name "$LEASE_NAME" "${extra[@]}" >"$d.argv" || return 1
  mapfile -d '' -t argv <"$d.argv"
  bash "$S/v2/eval/run_same_panel.sh" "${argv[@]}" >"$d.log" 2>&1 3<&-
  rc=$?
  if [ -n "$cache" ]; then python3 "$TC" finish --dest "$cache" >>"$LOG" 2>&1 3<&-; fi
  return "$rc"
}

if [ "$MODE" = gate ]; then
  LOG="$OUTPUT.log"
  log "start: gates c1 $LEFT_NAME ($LEFT) vs $RIGHT_NAME ($RIGHT); mirror $SRC"
  read_key
  gold_in
  python3 -m v2.eval.gates c1 --left "$LEFT" --right "$RIGHT" --left-name "$LEFT_NAME" --right-name "$RIGHT_NAME" \
    --gold "$GT/gold.jsonl" --retired "$RETIRED" --retired-sha "$RETIRED_SHA" --output "$OUTPUT" 3<&- | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  gold_out
  [ "$rc" = 0 ] || abort "gates c1 failed (exit $rc)"
  log "end: $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$OUTPUT" 3<&-) ($OUTPUT)"
  exit 0
fi

if [ "$MODE" = reproduce ]; then
  mkdir -p "$OUT_DIR"
  LOG="$OUT_DIR/REPRODUCE.log"
  log "start: gates c1 on every pair of $EVENT_DIR/PLAN.json; mirror $SRC"
  read_key
  gold_in
  helper reproduce --event-dir "$EVENT_DIR" --gold "$GT/gold.jsonl" --output-dir "$OUT_DIR" | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  gold_out
  [ "$rc" = 0 ] || abort "a pair differs from the event's paired file, or a gate failed (exit $rc)"
  log "end: every event-3 pair reproduced ($OUT_DIR/REPRODUCE.json)"
  exit 0
fi

if [ "$MODE" = rescore ]; then
  LOG="$OUTPUT.log"
  log "start: $LABEL ($RUN) rescored on v1.2 with the post-key label; mirror $SRC"
  read_key
  gold_in
  python3 -m v2.eval.sealed.score score --post-key --gold "$GT/gold.jsonl" \
    --predictions "$RUN/output/sealed-c1.predictions.jsonl" --seal "$RUN/SEAL-C1.json" --label "$LABEL" \
    --retired "$RETIRED" --retired-sha "$RETIRED_SHA" --output "$OUTPUT" 3<&- | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  gold_out
  [ "$rc" = 0 ] || abort "scoring failed (exit $rc)"
  log "end: $OUTPUT"
  exit 0
fi

case $SPEC in /*) ;; *) SPEC="$S/$SPEC" ;; esac
PLAN="$PK/.plan-$UTC-$$.json"
helper plan --spec "$SPEC" --registry "$REGISTRY" --output "$PLAN" >/dev/null || {
  echo "the plan was refused ($SPEC)" >&2
  exit 1
}
TIER=$(field tier) NAME=$(field name) ROLE=$(field role)
J="$JOBS/$TIER/$NAME-$UTC"
if [ "$PHASE" != run ]; then J="$J.$PHASE"; fi
mkdir -p "$J" || exit 1
mv "$PLAN" "$J/PLAN.json"
PLAN="$J/PLAN.json"
LOG="$J/POSTKEY.log"
log "start: C1 v1.2 post-key $PHASE, $ROLE $NAME ($TIER; GPU${GPU:-none} entry $LEASE_NAME; mirror $SRC; job $J)"
helper show --plan "$PLAN" >>"$LOG" 2>&1
helper verify --plan "$PLAN" --src-root "$S" --output "$J/VERIFY.json" >>"$LOG" 2>&1 ||
  abort "verification failed (see $J/VERIFY.json); key not read"
log "verified the image, mirror module, paths, package manifest and identity, pinned files, frozen cache and comparison seals"
APPROVAL_ARGS=()
if [ -n "$APPROVAL" ]; then APPROVAL_ARGS=(--approval "$APPROVAL"); fi
helper ledger-check --plan "$PLAN" --ledger "$LEDGER" "${APPROVAL_ARGS[@]}" >>"$LOG" 2>&1 ||
  abort "refused by the non-selection ledger (see $LOG); key not read"
log "ledger: no earlier post-key run of these weights or of a sibling successor${APPROVAL:+ (or approved: $APPROVAL)}"
if [ "$PHASE" = verify ]; then
  log "verify-only: done"
  exit 0
fi

if [ -f "$LEASE" ]; then cp -p "$LEASE" "$J/lease-before.owner"; fi
LEASED=1
printf 'track=eval\npurpose=C1 v1.2 post-key %s (%s)\nstart_utc=%s\n' "$PHASE" "$NAME" "$(date -u +%FT%TZ)" >"$LEASE"
job smoke "$J/smoke" || abort "the smoke failed (see $J/smoke.log); key not read"
helper parity --plan "$PLAN" --run-dir "$J/smoke" --output "$J/smoke/PARITY.json" >>"$LOG" 2>&1 ||
  abort "the smoke is not identical to the stored formal run (see $J/smoke/PARITY.json); key not read"
log "smoke identical to the stored formal run"
if [ "$PHASE" = preflight ]; then
  log "preflight-only: stopped before the key"
  exit 0
fi

read_key
DECRYPTED=1 PLAINTEXT=prompts
decrypt v1/build-5/prompts.jsonl >"$PANEL"
chmod 644 "$PANEL"
[ "$(sha256sum <"$PANEL" | cut -c1-64)" = "$PROMPTS_SHA" ] || abort "prompts hash mismatch"
log "prompts decrypted to the panel root for the collection"
job collect "$J/cand"
rc=$?
if [ "$rc" != 0 ] && [ ! -s "$J/cand/output/sealed-c1.predictions.jsonl" ]; then
  for f in "$J/cand" "$J/cand.log" "$J/cand.argv" "$J/cand-triton"; do
    if [ -e "$f" ]; then mv "$f" "$f.failed-launch"; fi
  done
  log "collection failed before any prediction (exit $rc); moved aside, one relaunch"
  job collect "$J/cand"
  rc=$?
fi
[ "$rc" = 0 ] || abort "collection failed (exit $rc; see $J/cand.log)"
log "collected"
python3 -m v2.eval.sealed.score seal --post-key --prompts "$PANEL" \
  --predictions "$J/cand/output/sealed-c1.predictions.jsonl" --output "$J/cand/SEAL-C1.json" >>"$LOG" 2>&1 3<&- ||
  abort "seal failed"
log "sealed ($(sha256sum <"$J/cand/SEAL-C1.json" | cut -c1-12)) before any gold"
rm -f "$PANEL"
DECRYPTED=0
log "prompts removed from the panel root"

gold_in
python3 -m v2.eval.sealed.score score --post-key --gold "$GT/gold.jsonl" \
  --predictions "$J/cand/output/sealed-c1.predictions.jsonl" --seal "$J/cand/SEAL-C1.json" \
  --label "$(field model.label)" --retired "$RETIRED" --retired-sha "$RETIRED_SHA" \
  --output "$J/cand/REPORT-C1.json" >>"$LOG" 2>&1 3<&- || abort "scoring failed"
log "scored on v1.2"
mapfile -d '' -t GATES < <(helper gates --plan "$PLAN" --job-dir "$J")
for ((i = 0; i < ${#GATES[@]}; i += 3)); do
  python3 -m v2.eval.gates c1 --left "$J/cand" --right "${GATES[i]}" --left-name "$(field model.label) $(field model.revision)" \
    --right-name "${GATES[i + 1]}" --gold "$GT/gold.jsonl" --retired "$RETIRED" --retired-sha "$RETIRED_SHA" \
    --output "${GATES[i + 2]}" >"${GATES[i + 2]}.log" 2>&1 3<&- &
done
wait
gold_out
helper finish --plan "$PLAN" --job-dir "$J" --ledger "$LEDGER" "${APPROVAL_ARGS[@]}" --output "$J/SUMMARY.json" |
  tee -a "$LOG"
rc=${PIPESTATUS[0]}
[ -f "$J/SUMMARY.json" ] || abort "no summary (exit $rc)"
[ "$rc" = 0 ] || abort "a gate output is missing (see $J/GATE-C1-*.log); the ledger line is appended"
log "end: $NAME scored post-key on v1.2 (summary $(sha256sum <"$J/SUMMARY.json" | cut -c1-12)); ledger line appended"
