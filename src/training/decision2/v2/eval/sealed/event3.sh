#!/usr/bin/env bash
# JevArena-C1 v1.1 scoring event 3 (the last of three), node A.
# Usage (on node A, from an exact mirror; in event mode the key arrives once on stdin):
#   event3.sh --gpu N --src MIRROR [--lease-name owner.eval] [--shared] [--models K,K,...]
#             [--c27 f1|f2] [--peers27 K,K] [--c27-package DIR --c27-manifest SHA --c27-repo ID
#             --c27-revision REV] [--allow-deviation ID]... [--verify-only | --preflight-only]
# The model table is v2/eval/sealed/event3-models.json; `python3 -m v2.eval.sealed.event3 plan`
# resolves it and refuses anything outside the C1 limits or with a PARENT-FILLS field.
# --verify-only: images, mirror modules, paths, release manifests, tree digests, frozen caches (CPU).
# --preflight-only: also every model's typed-FINAL smoke on the GPU with its parity check against
#   the stored formal run; stdin is closed, the key is never read, and neither the sealed directory
#   nor the event directory is touched.
# Event mode: the same verification and smokes (a failure leaves C1 untouched and the event unused),
# the stored event-2 seals are checked, the key is read, the event directory is created (event 3 is
# used from here), prompts are decrypted, every model is collected once, all predictions are sealed,
# prompts are removed, gold is decrypted to a private temp dir, reports and paired comparisons are
# written. Plaintext is removed on any exit. Each runner job writes only the named lease entry.
set -uo pipefail
umask 077

GPU="" SRC="" LEASE_NAME=owner.eval SHARED=0 MODE=event C27=f1
PLAN_ARGS=()
while [ $# -gt 0 ]; do
  case $1 in
  --gpu) GPU=$2; shift 2 ;;
  --src) SRC=$2; shift 2 ;;
  --lease-name) LEASE_NAME=$2; shift 2 ;;
  --shared) SHARED=1; shift ;;
  --c27) C27=$2; shift 2 ;;
  --models | --peers27 | --c27-package | --c27-manifest | --c27-repo | --c27-revision | --allow-deviation)
    PLAN_ARGS+=("$1" "$2"); shift 2 ;;
  --verify-only) MODE=verify; shift ;;
  --preflight-only) MODE=preflight; shift ;;
  *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
# No child process may read the key: stdin becomes /dev/null; only event mode keeps it on fd 3.
if [ "$MODE" = event ]; then
  exec 3<&0 0</dev/null
else
  exec 0</dev/null
fi
[ -n "$SRC" ] || { echo "--src MIRROR is required" >&2; exit 2; }
if [ "$MODE" != verify ]; then
  [[ "$GPU" =~ ^[0-7]$ ]] || { echo "--gpu 0-7 is required" >&2; exit 2; }
fi
[[ "$LEASE_NAME" =~ ^owner\.[a-z0-9-]+$ ]] || { echo "lease name must be a named entry owner.NAME" >&2; exit 2; }
[ -f "/data/dev2/src/$SRC/.dev2-mirror.json" ] || { echo "no verified mirror $SRC" >&2; exit 1; }

S="/data/dev2/src/$SRC/src/training/decision2"
C1=/data/dev2/private/sealed/c1
PANEL=/data/dev2/private/panels/goldfree/sealed-c1.prompts.jsonl
PROMPTS_SHA=0b29686f60c980f3fbc8a03b88537fc0bf90ee967afa67d4fe0c958b1bfde16a
GOLD_SHA=c02777713c0e58b40cf602947433420744b2765465252ce22d692eb676ca4fe1
BUNDLE_SHA=d924389a7ffc3d852d9204c3dfca1c12c32f82685a7e5c65277ba9980110534f
E=/data/dev2/runs/eval/m4/c1-event3
P="/data/dev2/runs/eval/m4/c1-event3-preflight/$MODE-$(date -u +%Y%m%dT%H%M%SZ)"
PLAN="$P/PLAN.json"
LEASE="/data/dev2/leases/gpu$GPU.lock/$LEASE_NAME"
TC="$S/v2/27b/triton_cache.py"
export PYTHONPATH="$S" PYTHONDONTWRITEBYTECODE=1
cd "$S" || exit 1
if [ "$MODE" = event ] && [ -e "$E" ]; then
  echo "event 3 directory already exists: the event is used" >&2
  exit 1
fi
mkdir -p "$P"

DECRYPTED=0 GT="" LEASED=0 KEY=""
log() {
  local t
  t=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "$t $*" >>"$P/PREFLIGHT.log"
  if [ "$MODE" = event ]; then
    echo "$t EVENT3 $*" >>"$C1/ACCESS.log"
    if [ -d "$E" ]; then echo "$t $*" >>"$E/EVENT.log"; fi
  fi
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
    if [ -f "$P/lease-before.owner" ]; then
      cp -p "$P/lease-before.owner" "$LEASE"
    else
      printf 'track=eval\nstatus=idle-released (C1 event 3 %s done)\nlast_job_end_utc=%s\n' "$MODE" "$(date -u +%FT%TZ)" >"$LEASE"
    fi
  fi
  local what="nothing was decrypted" lease="no GPU lease taken"
  if [ "$DECRYPTED" = 1 ]; then what="prompts and gold plaintext removed"; fi
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
helper() { python3 -m v2.eval.sealed.event3 "$@" 3<&-; }
field() { helper field --plan "$PLAN" --key "$1" --field "$2"; }

# key phase(smoke|collect) run-dir: one runner job, with a fresh copy of the frozen autotune cache if any
job() {
  local k=$1 phase=$2 d=$3 frozen cache="" rc
  local -a extra=() argv=()
  frozen=$(field "$k" cache.frozen)
  if [ -n "$frozen" ]; then
    cache="$d-triton"
    python3 "$TC" copy --frozen "$frozen" --dest "$cache" --expect "$(field "$k" cache.sha256)" \
      >>"$P/PREFLIGHT.log" 2>&1 3<&- || return 1
    extra=(--cache-dir "$cache")
  fi
  if [ "$SHARED" = 1 ]; then extra+=(--shared); fi
  helper argv --plan "$PLAN" --key "$k" --phase "$phase" --run-dir "$d" --gpu "$GPU" --src "$SRC" \
    --src-root "$S" --lease-name "$LEASE_NAME" "${extra[@]}" >"$d.argv" || return 1
  mapfile -d '' -t argv <"$d.argv"
  bash "$S/v2/eval/run_same_panel.sh" "${argv[@]}" >"$d.log" 2>&1 3<&-
  rc=$?
  if [ -n "$cache" ]; then python3 "$TC" finish --dest "$cache" >>"$P/PREFLIGHT.log" 2>&1 3<&-; fi
  return "$rc"
}

log "start: C1 v1.1 event 3 ($MODE; GPU${GPU:-none} entry $LEASE_NAME; mirror $SRC; 27B candidate $C27; preflight dir $P)"
helper plan --table "$S/v2/eval/sealed/event3-models.json" --c27 "$C27" "${PLAN_ARGS[@]}" --site node-a \
  --output "$PLAN" >>"$P/PREFLIGHT.log" 2>&1 || abort "the plan was refused (see $P/PREFLIGHT.log)"
helper show --plan "$PLAN" >>"$P/PREFLIGHT.log" 2>&1
mapfile -d '' -t COLLECT < <(helper keys --plan "$PLAN" --collected)
[ "${#COLLECT[@]}" -gt 0 ] || abort "the plan collects no model"
log "plan $(sha256sum <"$PLAN" | cut -c1-12): collect ${COLLECT[*]}"
helper verify --plan "$PLAN" --src-root "$S" --output "$P/VERIFY.json" >>"$P/PREFLIGHT.log" 2>&1 ||
  abort "verification failed before any smoke (see $P/VERIFY.json; C1 not touched)"
log "verified images, mirror modules, paths, release manifests, tree digests and frozen caches"
if [ "$MODE" = verify ]; then
  log "verify-only: done"
  exit 0
fi

if [ -f "$LEASE" ]; then cp -p "$LEASE" "$P/lease-before.owner"; fi
LEASED=1
printf 'track=eval\npurpose=C1 scoring event 3 (%s)\nstart_utc=%s\n' "$MODE" "$(date -u +%FT%TZ)" >"$LEASE"
for k in "${COLLECT[@]}"; do
  if ! job "$k" smoke "$P/smoke-$k"; then
    abort "preflight smoke failed for $k before decryption (C1 not decrypted; event 3 not used)"
  fi
  if ! helper parity --plan "$PLAN" --key "$k" --run-dir "$P/smoke-$k" --output "$P/smoke-$k/PARITY.json" >>"$P/PREFLIGHT.log" 2>&1; then
    abort "preflight parity failed for $k (see $P/smoke-$k/PARITY.json; C1 not decrypted; event 3 not used)"
  fi
  log "preflight passed: $k"
done
log "preflight passed for all ${#COLLECT[@]} collected models"
if [ "$MODE" = preflight ]; then
  helper summary --plan "$PLAN" --event-dir "$P/no-event" --preflight-dir "$P" --output "$P/PREFLIGHT-SUMMARY.json" >>"$P/PREFLIGHT.log" 2>&1
  log "preflight-only: stopped before the key; C1 not touched"
  exit 0
fi

helper stored --plan "$PLAN" --output "$P/STORED-C1.json" >>"$P/PREFLIGHT.log" 2>&1 ||
  abort "stored event-2 predictions do not match their seals (C1 not decrypted; event 3 not used)"
log "stored event-2 predictions match their seals (hashes only)"
IFS= read -r KEY <&3 || [ -n "$KEY" ] || abort "no key on stdin (C1 not decrypted; event 3 not used)"
exec 3<&-
[ -n "$KEY" ] || abort "empty key (C1 not decrypted; event 3 not used)"
mkdir "$E" || abort "cannot create the event directory"
cp -p "$PLAN" "$P/VERIFY.json" "$P/STORED-C1.json" "$E/"
for k in "${COLLECT[@]}"; do cp -p "$P/smoke-$k/PARITY.json" "$E/PREFLIGHT-PARITY-$k.json"; done
log "event 3 used from here: decryption starts (preflight $P passed)"
[ "$(sha256sum <"$C1/c1-v1-bundle.tar.enc" | cut -c1-64)" = "$BUNDLE_SHA" ] || abort "encrypted bundle hash mismatch"
DECRYPTED=1
decrypt v1/build-5/prompts.jsonl >"$PANEL"
chmod 644 "$PANEL"
[ "$(sha256sum <"$PANEL" | cut -c1-64)" = "$PROMPTS_SHA" ] || abort "prompts hash mismatch"
log "prompts decrypted to the panel root for collection (sha verified)"
for k in "${COLLECT[@]}"; do
  job "$k" collect "$E/$k"
  rc=$?
  if [ "$rc" != 0 ] && [ ! -s "$E/$k/output/sealed-c1.predictions.jsonl" ]; then
    # As event 1's Kev: a launch that failed before any prediction is relaunched once, logged.
    for f in "$E/$k" "$E/$k.log" "$E/$k.argv" "$E/$k-triton" "$E/$k-triton.copy.json" "$E/$k-triton.post.json"; do
      if [ -e "$f" ]; then mv "$f" "$f.failed-launch"; fi
    done
    log "collection of $k failed before any prediction (exit $rc); moved aside, one relaunch"
    job "$k" collect "$E/$k"
    rc=$?
  fi
  log "collected $k (exit $rc)"
done
for k in "${COLLECT[@]}"; do
  if python3 -m v2.eval.sealed.score seal --prompts "$PANEL" --predictions "$E/$k/output/sealed-c1.predictions.jsonl" \
    --output "$E/$k/SEAL-C1.json" >>"$E/EVENT.log" 2>&1; then
    log "sealed predictions $k ($(sha256sum <"$E/$k/SEAL-C1.json" | cut -c1-12))"
  else
    log "SEAL FAILED $k"
  fi
done
rm -f "$PANEL"
log "prompts removed from the panel root"

GT=$(mktemp -d "$C1/.gold.XXXXXX")
decrypt v1/build-5/gold.jsonl >"$GT/gold.jsonl"
[ "$(sha256sum <"$GT/gold.jsonl" | cut -c1-64)" = "$GOLD_SHA" ] || abort "gold hash mismatch"
log "gold decrypted to a private temp dir after all predictions were sealed"
for k in "${COLLECT[@]}"; do
  [ -f "$E/$k/SEAL-C1.json" ] || continue
  python3 -m v2.eval.sealed.score score --gold "$GT/gold.jsonl" --predictions "$E/$k/output/sealed-c1.predictions.jsonl" \
    --seal "$E/$k/SEAL-C1.json" --label "$(field "$k" label)" --output "$E/$k/REPORT-C1.json" >>"$E/EVENT.log" 2>&1 &&
    log "scored $k"
done
mapfile -d '' -t PAIRS < <(helper pairs --plan "$PLAN" --event-dir "$E" 2>>"$E/EVENT.log")
for ((i = 0; i < ${#PAIRS[@]}; i += 5)); do
  python3 -m v2.eval.sealed.score compare --gold "$GT/gold.jsonl" --left "${PAIRS[i]}" --right "${PAIRS[i + 1]}" \
    --left-name "${PAIRS[i + 2]}" --right-name "${PAIRS[i + 3]}" --output "${PAIRS[i + 4]}" >"${PAIRS[i + 4]}.log" 2>&1 &
done
wait
for ((i = 0; i < ${#PAIRS[@]}; i += 5)); do
  if [ -f "${PAIRS[i + 4]}" ]; then log "paired ${PAIRS[i + 4]##*/}"; else log "PAIRED FAILED ${PAIRS[i + 4]##*/}"; fi
done
rm -rf "$GT"
GT=""
log "gold removed"
helper summary --plan "$PLAN" --event-dir "$E" --preflight-dir "$P" --output "$E/EVENT3-SUMMARY.json" >>"$E/EVENT.log" 2>&1
log "end: scoring complete (3 of 3 events used; JevArena-C1 v1.1 is now post-key)"
echo DONE >>"$E/EVENT.log"
