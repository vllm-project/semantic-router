#!/usr/bin/env bash
# Decoder M7 line watcher (prereg dec-m7-prereg-2026-09-30.md, "Candidates", "Diagnostics", "Development gates"):
# as each arm's chain marker appears (status/<ARM>.DONE | .FAILED), runs the arm's line readouts and the arm soup's
# diagnostics on the given GPU as a co-tenant (m7-lines.sh line / diag); once every arm is handled it writes the
# tier's finalists (m7_rules.py; arms without a soup are declared dropped with the chain's reason). Development only;
# no formal job starts here.
# usage: m7-watch.sh launch|run <mirror-dir> <tier 4b|2b> <gpu> <ARM> [<ARM> ...]
set -u
MODE=$1 SRC=$2 TIER=$3 GPU=$4
shift 4
ARMS=("$@")
M=/data/dev2/runs/dec/m7
S=/data/dev2/src/$SRC/src/training/decision2
OPS=$S/v2/dec/ops/m7
mkdir -p "$M/logs" "$M/select" "$M/chains"
if [ "$MODE" = launch ]; then
  mkdir "$M/chains/watch-$TIER.lock" 2>/dev/null || { echo "M7 watcher for $TIER already launched"; exit 0; }
  setsid nohup bash "$0" run "$SRC" "$TIER" "$GPU" "${ARMS[@]}" > "$M/logs/watch-$TIER.log" 2>&1 < /dev/null &
  echo $! > "$M/chains/watch-$TIER.pid"
  echo "$(date -u +%FT%TZ) M7 watcher $TIER launched from $SRC (pid $(cat "$M/chains/watch-$TIER.pid"))" | tee -a "$M/OPERATIONS.log"
  exit 0
fi
[ "$MODE" = run ] || { echo "unknown mode $MODE" >&2; exit 2; }
log() { echo "$(date -u +%FT%TZ) watch-$TIER $*" | tee -a "$M/OPERATIONS.log"; }
dropped=()
handle() {  # <ARM>
  local g=$1
  if [ -f "$M/status/$g.FAILED" ]; then
    dropped+=(--dropped "L-$g=$(tr '\n' ' ' < "$M/status/$g.FAILED")")
    log "$g has no soup ($(cat "$M/status/$g.FAILED")); line dropped"
    return 0
  fi
  if ! M7_GPU=$GPU bash "$OPS/m7-lines.sh" "$TIER" line "$g"; then
    dropped+=(--dropped "L-$g=line readout failed")
    log "L-$g readout FAILED; line dropped (not rerun)"
    return 0
  fi
  log "L-$g read"
  if M7_GPU=$GPU bash "$OPS/m7-lines.sh" "$TIER" diag "$TIER-$g-b1"; then
    log "$TIER-$g-b1 diagnostics done"
  else
    log "$TIER-$g-b1 diagnostics FAILED (report only)"
  fi
}
log "watching ${ARMS[*]} on GPU$GPU from $SRC"
pending=("${ARMS[@]}")
n=0
while [ ${#pending[@]} -gt 0 ]; do
  left=()
  for g in "${pending[@]}"; do
    if [ -f "$M/status/$g.DONE" ] || [ -f "$M/status/$g.FAILED" ]; then handle "$g"; else left+=("$g"); fi
  done
  pending=("${left[@]}")
  [ ${#pending[@]} -eq 0 ] && break
  [ $((n % 30)) = 0 ] && log "waiting for ${pending[*]}"
  n=$((n + 1))
  sleep 60
done
if python3 -B "$OPS/m7_rules.py" --tier "$TIER" --lines-root "$M/lines/$TIER" \
  --rules-module "$S/v2/9b/lux9b/m4_rules.py" --output "$M/select/$TIER-finalists.json" "${dropped[@]}" \
  > "$M/select/$TIER-rules.log" 2>&1; then
  log "finalists: $(tail -1 "$M/select/$TIER-rules.log")"
else
  log "rules FAILED: $(tail -3 "$M/select/$TIER-rules.log" | tr '\n' ' ')"
fi
