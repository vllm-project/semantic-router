#!/usr/bin/env bash
# JevArena-C1 v1.1 scoring event 2 (DEV2.0-0.6B + comparators), node A.
# Usage: event2.sh <gpu> <mirror-dir-name> [lease-name, default owner.eval]   (the key arrives once on stdin)
# The GPU is shared with its owner track: every job writes only the named lease entry and skips the idle check.
# Every model first passes a 20-item typed-final smoke; only then are the prompts decrypted.
# Predictions are sealed before the gold is decrypted; plaintext is removed on any exit.
set -uo pipefail
umask 077
GPU="$1"
SRC="$2"
LEASE_NAME="${3:-owner.eval}"
[[ "$LEASE_NAME" =~ ^owner\.[a-z0-9-]+$ ]] || { echo "lease name must be a named entry owner.NAME" >&2; exit 2; }
IFS= read -r KEY
S="/data/dev2/src/$SRC/src/training/decision2"
C1=/data/dev2/private/sealed/c1
PANEL=/data/dev2/private/panels/goldfree/sealed-c1.prompts.jsonl
PROMPTS_SHA=0b29686f60c980f3fbc8a03b88537fc0bf90ee967afa67d4fe0c958b1bfde16a
GOLD_SHA=c02777713c0e58b40cf602947433420744b2765465252ce22d692eb676ca4fe1
E=/data/dev2/runs/eval/m4/c1-event2
H=/data/dev2/hf-cache
M=/data/decision20-20260926/models
K=/data/decision20-20260926/competitors
KL=/data/dev2/tools/envs/kai-lex
# Snapshot files are symlinks into the repo's blobs/; the large ones continue into the cache-wide $H/blobs.
PKG_REPO="$H/models--llm-semantic-router--DEV2.0-0.6B"
PKG_REV=e61b2b4419383672cb6a92d63699f7974e5f81ac
PKG="$PKG_REPO/snapshots/$PKG_REV"
LEASE="/data/dev2/leases/gpu$GPU.lock/$LEASE_NAME"
GT=""
[ -e "$E/EVENT.log" ] && {
  echo "event 2 directory already used" >&2
  exit 1
}
mkdir -p "$E"
log() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) EVENT2 $*" >>"$C1/ACCESS.log"
  echo "$(date -u +%FT%TZ) $*" >>"$E/EVENT.log"
}
if [ -f "$LEASE" ]; then cp -p "$LEASE" "$E/lease-before.owner"; fi
cleanup() {
  rm -f "$PANEL"
  if [ -n "$GT" ]; then rm -rf "$GT"; fi
  if [ -f "$E/lease-before.owner" ]; then
    cp -p "$E/lease-before.owner" "$LEASE"
  else
    printf 'track=eval\nstatus=idle-released (C1 event 2 done)\nlast_job_end_utc=%s\n' "$(date -u +%FT%TZ)" >"$LEASE"
  fi
  unset KEY
  log "cleanup: prompts and gold plaintext removed; GPU$GPU lease entry $LEASE_NAME restored"
}
trap cleanup EXIT
decrypt() {
  openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -pass fd:4 -in "$C1/c1-v1-bundle.tar.enc" 4<<<"$KEY" | tar -xOf - "$1"
}

MODELS="cand kai1 lex bosun06 gliner25"
declare -A LABEL=([cand]="DEV2.0-0.6B" [kai1]="Decision 1.0 Kai" [lex]="Decision 1.0 Lex"
  [bosun06]="Bosun v3.1 0.6B" [gliner25]="GLiNER2.5-Decide")
# Runner options, then "--", then collect options (without --panels).
args() {
  case $1 in
  cand) echo "--model-dir $PKG_REPO --mount $H/blobs -- --adapter-spec $S/v2/06b/records/adapters/dev2-06b-causal-8k.json --model-path $PKG --revision $PKG_REV --extra model_id=llm-semantic-router/DEV2.0-0.6B" ;;
  kai1) echo "--model-dir $M/Decision-1.0-Kai-0.6B --mount $KL -- --adapter kai --model-path $M/Decision-1.0-Kai-0.6B --revision 7185f514f54b8f93c55998b1e8f9c5cc67f0d029 --extra model_id=llm-semantic-router/Decision-1.0-Kai-0.6B" ;;
  lex) echo "--model-dir $M/Decision-1.0-Lex-0.6B --mount $KL -- --adapter lex --model-path $M/Decision-1.0-Lex-0.6B --revision ee8e74d912fca8328a353c11d174b44da3f91781 --extra model_id=llm-semantic-router/Decision-1.0-Lex-0.6B" ;;
  bosun06) echo "--model-dir $K/bosun-v31-06b-r1 --mount $K/qwen3-06b-bosun-base-r1 -- --adapter bosun06 --model-path $K/bosun-v31-06b-r1 --revision 1d8b6f9611f9b64b514ce8b57cd86398fbc31a3b --extra base=$K/qwen3-06b-bosun-base-r1 --extra model_id=Hanno-Labs/bosun-v3.1-0.6b" ;;
  gliner25) echo "--image decision20-gliner25:host2 --model-dir $M/GLiNER2.5-Decide --mount $H --env HF_HUB_CACHE=$H --env HF_HUB_OFFLINE=1 -- --adapter gliner25 --model-path $M/GLiNER2.5-Decide --revision 7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6 --extra model_id=fastino/GLiNER2.5-Decide --extra variant=english" ;;
  esac
}
# name run-dir purpose panels [max-items]
run() {
  local name=$1 dir=$2 purpose=$3 panels=$4 a
  local -a runner collect
  a=$(args "$name")
  read -r -a runner <<<"${a%% -- *}"
  read -r -a collect <<<"${a#* -- }"
  if [ -n "${5:-}" ]; then collect+=(--max-items "$5"); fi
  bash "$S/v2/eval/run_same_panel.sh" --gpu "$GPU" --track eval --lease-name "$LEASE_NAME" --shared --src "$SRC" \
    --run-dir "$dir" --purpose "$purpose" "${runner[@]}" -- "${collect[@]}" --panels "$panels" >"$dir.log" 2>&1
}

log "start: C1 v1.1 event 2 (GPU$GPU shared, entry $LEASE_NAME, source $SRC); candidate DEV2.0-0.6B@$PKG_REV (manifest a5cdabed, identity 5b30b7e2, T=1, 8192 tokens); independence recheck2 vs ba848147/38c2db3c/b1df84c4/30e0a1f7 + local pools m3a2/m3b + node-B lux-xl-w2 + derived files: no source name, 0 OVERLAP, the same 9 weak REVIEW (names 41fe708e, overlap c0f92ce9), no v1.2"
printf 'track=eval\npurpose=C1 scoring event 2 (DEV2.0-0.6B + comparators)\nstart_utc=%s\n' "$(date -u +%FT%TZ)" >"$LEASE"
for m in $MODELS; do
  run "$m" "$E/smoke-$m" "C1 event 2 preflight: $m" typed-final 20
  n=$(cat "$E/smoke-$m"/smoke/typed-final.predictions.jsonl 2>/dev/null | wc -l)
  if [ "$n" -lt 1 ]; then
    log "ABORT before decryption: preflight failed for $m (no C1 access used)"
    exit 1
  fi
done
log "preflight passed for all models (typed-final 20 items)"

decrypt v1/build-5/prompts.jsonl >"$PANEL"
chmod 644 "$PANEL"
if [ "$(sha256sum <"$PANEL" | cut -c1-64)" != "$PROMPTS_SHA" ]; then
  log "ABORT: prompts hash mismatch"
  exit 1
fi
log "prompts decrypted to the panel root for collection (sha verified)"
for m in $MODELS; do
  run "$m" "$E/$m" "C1 event 2: $m" sealed-c1
  log "collected $m (exit $?)"
done
cd "$S" || exit 1
export PYTHONPATH="$S" PYTHONDONTWRITEBYTECODE=1
for m in $MODELS; do
  if python3 -m v2.eval.sealed.score seal --prompts "$PANEL" --predictions "$E/$m/output/sealed-c1.predictions.jsonl" \
    --output "$E/$m/SEAL-C1.json" >>"$E/EVENT.log" 2>&1; then
    log "sealed predictions $m"
  else
    log "SEAL FAILED $m"
  fi
done
rm -f "$PANEL"
log "prompts removed from the panel root"

GT=$(mktemp -d "$C1/.gold.XXXXXX")
decrypt v1/build-5/gold.jsonl >"$GT/gold.jsonl"
if [ "$(sha256sum <"$GT/gold.jsonl" | cut -c1-64)" != "$GOLD_SHA" ]; then
  log "ABORT: gold hash mismatch"
  exit 1
fi
log "gold decrypted to a private temp dir after all predictions were sealed"
for m in $MODELS; do
  [ -f "$E/$m/SEAL-C1.json" ] || continue
  python3 -m v2.eval.sealed.score score --gold "$GT/gold.jsonl" --predictions "$E/$m/output/sealed-c1.predictions.jsonl" \
    --seal "$E/$m/SEAL-C1.json" --label "${LABEL[$m]}" --output "$E/$m/REPORT-C1.json" >>"$E/EVENT.log" 2>&1 &&
    log "scored $m"
done
for m in kai1 lex bosun06 gliner25; do
  [ -f "$E/$m/SEAL-C1.json" ] && [ -f "$E/cand/SEAL-C1.json" ] || continue
  python3 -m v2.eval.sealed.score compare --gold "$GT/gold.jsonl" --left "$E/cand/output/sealed-c1.predictions.jsonl" \
    --right "$E/$m/output/sealed-c1.predictions.jsonl" --left-name "${LABEL[cand]}" --right-name "${LABEL[$m]}" \
    --output "$E/PAIRED-C1-cand-vs-$m.json" >>"$E/EVENT.log" 2>&1 && log "paired cand vs $m"
done
log "end: scoring complete (2 of 3 events used)"
echo DONE >>"$E/EVENT.log"
