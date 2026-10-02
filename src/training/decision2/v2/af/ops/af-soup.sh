#!/usr/bin/env bash
# Arm factory soups and interpolation points (prereg "Candidates"), node side: a uniform FP32 soup (v2.dec.soup, CPU)
# of a member list; a member listed k times carries weight k / n.
#
# usage: AF_NODE=a|b|c|f af-soup.sh <mirror-dir> <NAME> <gpu|-> <member>...
#   run:<RUN>    a finished seed on this node. 4B: its BEST LoRA checkpoint, merged first into a full FP32 checkpoint
#                on <gpu> (M10's m10_merge.py with its SELECT agreement check; cached in merged/<RUN>). 9B: its BEST
#                full checkpoint.
#   soup:<NAME>  an arm-factory soup or point on this node.
#   ext:<path>   an owner's frozen full checkpoint on this node under /data/dev2/runs/dec or /data/dev2/runs/9b
#                (mounted read-only as /dec or /r9b).
#   lux          9B only: the Lux 1.0 zero-step member (this node's 9b-KIB4-s4 zero-step checkpoint, byte-checked
#                against K-a13IB's Lux member list, M10's lux-zero-m9-KIB-s1.sha256).
# A run without a DONE marker stops the build unless AF_SKIP_UNFINISHED=1 (then it is left out and logged). With
# AF_WAIT=1 the build first waits (<= 6 h) until every run: and soup: member has a terminal marker; a failed or
# stopped seed is then left out (disclosed in members.txt and the log). Merges hold <gpu>'s arm-factory lease as busy
# (only a lease its chain released; it is released again after the merges).
# Output /data/dev2/runs/af/<size>/soup/<NAME>/{build/<NAME>, members.txt, DONE, MODEL_SHA256}; FAILED is never
# rebuilt.
set -u
SRC=$1 NAME=$2 GPU=$3
shift 3
NODE=${AF_NODE:?set AF_NODE}
case $NODE in a) SIZE=9b ;; *) SIZE=4b ;; esac
M=/data/dev2/runs/af/$SIZE
OUT=$M/soup/$NAME
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/af/ops
L=$OPS/af-launch.sh
REV_4B=1001bb4d826a52d1f399e183466143f4da7b741b
LUXRUN=9b-KIB4-s4
LUXHOST=$M/arms/pre/$LUXRUN-zero/checkpoint-0000000
LUXCK=/runs/9b/arms/pre/$LUXRUN-zero/checkpoint-0000000
LUXSUMS=/data/dev2/runs/9b/m10/inputs/lux-zero-m9-KIB-s1.sha256
mkdir -p "$M/soup" "$M/merged"
log() { echo "$(date -u +%FT%TZ) soup-$NAME $*" | tee -a "$M/OPERATIONS.log"; }
fail() { mkdir -p "$OUT" && echo "$*" > "$OUT/FAILED"; log "FAILED: $*"; exit 1; }
[ -f "$OUT/DONE" ] && { log "already built"; exit 0; }
[ -f "$OUT/FAILED" ] && { log "failed earlier; not rebuilt"; exit 1; }
[ ! -e "$OUT" ] || fail "$OUT exists without DONE / FAILED"
if [ "${AF_WAIT:-0}" = 1 ]; then
  n=0
  for spec in "$@"; do
    case $spec in
      run:*) t=$M/status/${spec#run:}; ends=("$t.DONE" "$t.FAILED" "$t.STOPPED") ;;
      soup:*) t=$M/soup/${spec#soup:}; ends=("$t/DONE" "$t/FAILED") ;;
      *) continue ;;
    esac
    until [ -f "${ends[0]}" ] || [ -f "${ends[1]}" ] || [ -f "${ends[2]:-/nonexistent}" ]; do
      [ $((n % 30)) = 0 ] && log "waits for $spec"
      n=$((n + 1))
      [ $n -gt 360 ] && fail "$spec did not finish within 6 h"
      sleep 60
    done
  done
  AF_SKIP_UNFINISHED=1
fi
LEASE=/data/dev2/leases/gpu$GPU.lock/owner
held=0
hold() {  # a chain writes its seed's DONE marker just before it releases its lease: wait <= 15 min for the release
  local w=0
  [ "$GPU" != - ] && [ "$held" = 0 ] || return 0
  until grep -qs '^track=arm-factory' "$LEASE" && grep -qsE '^status=(released|idle)' "$LEASE"; do
    w=$((w + 1))
    [ $w -gt 90 ] && fail "GPU$GPU's lease is not a released arm-factory lease"
    sleep 10
  done
  printf 'track=arm-factory\nstatus=busy\npurpose=arm factory LoRA merges for %s\nstart_utc=%s\n' "$NAME" \
    "$(date -u +%FT%TZ)" > "$LEASE"
  held=1
}
unhold() {
  [ "$held" = 1 ] || return 0
  printf 'track=arm-factory\nstatus=released\npurpose=arm factory merges for %s done\nlast_job_end_utc=%s\n' "$NAME" \
    "$(date -u +%FT%TZ)" > "$LEASE"
  held=0
}
trap unhold EXIT

paths=() lines=()
for spec in "$@"; do
  case $spec in
    run:*)
      run=${spec#run:}
      if [ ! -f "$M/status/$run.DONE" ]; then
        [ "${AF_SKIP_UNFINISHED:-0}" = 1 ] || fail "$run has no DONE marker"
        log "$run is not DONE; left out (AF_SKIP_UNFINISHED=1)"
        lines+=("$spec left out (no DONE marker)")
        continue
      fi
      best=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$M/arms/full/$run/BEST.json")
      if [ "$SIZE" = 9b ]; then
        [ -f "$M/arms/full/$run/$best/decision_config.json" ] || fail "$run BEST $best is not a checkpoint"
        p=/runs/9b/arms/full/$run/$best
      else
        if [ ! -f "$M/merged/$run/merge_check.json" ]; then
          [ "$GPU" != - ] || fail "$run needs a LoRA merge and no GPU was given"
          hold
          AF_NODE=$NODE AF_CACHE=4b-read bash "$L" "merge-$run" "$SRC" "$M/merged/.job-$run" --gpu "$GPU" -- \
            v2/dec/ops/m10/m10_merge.py --checkpoint "/runs/4b/arms/full/$run/$best" \
            --source-path "/models/Qwen--Qwen3.5-4B-Base/$REV_4B" --select /data/select.jsonl --output "/out/$run" \
            || fail "merge of $run failed (see $M/merged/.job-$run.stderr.log)"
          mv -T "$M/merged/.job-$run/$run" "$M/merged/$run"
          log "$run merged ($best): $(tail -c 240 "$M/merged/.job-$run.stdout.log" | tr '\n' ' ')"
        fi
        p=/runs/4b/merged/$run
      fi
      lines+=("$spec $best") ;;
    soup:*)
      s=${spec#soup:}
      [ -f "$M/soup/$s/DONE" ] || fail "no soup $s"
      p=/runs/$SIZE/soup/$s/build/$s
      lines+=("$spec $(cat "$M/soup/$s/MODEL_SHA256")") ;;
    ext:/data/dev2/runs/dec/* | ext:/data/dev2/runs/9b/*)
      h=${spec#ext:}
      [ -f "$h/decision_config.json" ] || fail "no checkpoint at $h"
      case $h in /data/dev2/runs/dec/*) p=/dec/${h#/data/dev2/runs/dec/} ;; *) p=/r9b/${h#/data/dev2/runs/9b/} ;; esac
      lines+=("$spec") ;;
    lux)
      [ "$SIZE" = 9b ] || fail "lux is a 9B member"
      (cd "$LUXHOST" && sha256sum -c --quiet "$LUXSUMS") || fail "the Lux zero-step member differs from K-a13IB's"
      p=$LUXCK
      lines+=("lux $LUXRUN-zero") ;;
    *) fail "unknown member $spec" ;;
  esac
  paths+=(--member "$p")
done
unhold
[ ${#paths[@]} -ge 4 ] || fail "fewer than two members"
mkdir -p "$OUT"
printf '%s\n' "${lines[@]}" > "$OUT/members.txt"
AF_NODE=$NODE bash "$L" "soup-$NAME" "$SRC" "$OUT/build" --cpu -- -m v2.dec.soup "${paths[@]}" --output "/out/$NAME" \
  || fail "soup build failed (see $OUT/build.stderr.log)"
python3 -c 'import json,sys; print(json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])["model_sha256"])' \
  "$OUT/build.stdout.log" > "$OUT/MODEL_SHA256" || fail "no model_sha256 in the soup output"
echo "$OUT/build/$NAME" > "$OUT/DONE"
log "built $NAME ($((${#paths[@]} / 2)) members): model $(cut -c1-12 "$OUT/MODEL_SHA256")"
