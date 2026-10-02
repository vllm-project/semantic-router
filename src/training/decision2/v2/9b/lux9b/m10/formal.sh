#!/usr/bin/env bash
# 9B M10 formal typed-FINAL path of one FP32 point, started speculatively next to its Index run (COORDINATION
# 2026-10-02 17:25): M9's formal.sh (CAL698 fit, gold-free smoke, typed FINAL + CSS15 + public 231, mlx-diag, seal,
# report, compares, gates) on one node A GPU under lease track=9b-m10. It stays on node A: the comparators
# (formal-m3, the DEV2.0-9B T = 1 run, Nimble v2, K-a13's mlx run), the CAL698 inputs, the frozen formal autotune
# cache and Lux 1.0's package are node A paths. The runner (training/model, infer_dec, the eval runner, formal.sh)
# is unchanged since K-a13IB's formal mirror 787abdc54. Runs land in /data/dev2/runs/9b/formal-m9/M10-NAME-*.
#
# usage (node A, from an exact mirror): formal.sh launch|status NAME [GPU]
#   launch  the point is soup/NAME (built on node A: KX) or models/ix1/9b-m10/ckpt/NAME (ix.sh ckcopy); GPU (1-7) must
#           be idle with its owner file absent, idle or released; runs detached and releases the lease at the end
#   status  exit code, log tail, run directories
set -uo pipefail
MODE=${1:?launch|status} NAME=${2:?NAME} GPU=${3:-}
OPS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
B9=/data/dev2/runs/9b/m10
F=/data/dev2/runs/9b/formal-m9
LUX=/data/decision20-20260926/models/Decision-1.0-Lux-9B
LOG=$B9/logs/formal-$NAME
lease=/data/dev2/leases/gpu$GPU.lock/owner
point() { if [ -f "$B9/soup/$NAME/DONE" ]; then cat "$B9/soup/$NAME/DONE"; else echo "/data/dev2/models/ix1/9b-m10/ckpt/$NAME"; fi; }
case "$MODE" in
  launch)
    [[ "$GPU" =~ ^[1-7]$ ]] || { echo "node A GPU1-7" >&2; exit 2; }
    ck=$(point)
    [ -f "$ck/decision_config.json" ] || { echo "no point at $ck" >&2; exit 3; }
    mkdir -p "$B9/logs"
    if [ -e "$F/M10-$NAME-16k" ] || ! mkdir "$LOG.lock" 2> /dev/null; then
      echo "formal M10-$NAME already launched" >&2
      exit 3
    fi
    if [ -s "$lease" ] && ! grep -q '^status=\(released\|idle\)' "$lease"; then
      rmdir "$LOG.lock"; echo "gpu$GPU is leased: $(head -1 "$lease")" >&2; exit 1
    fi
    used=$(rocm-smi -d "$GPU" --showmeminfo vram --json | python3 -c \
      'import json,sys; d=json.load(sys.stdin); print(int(list(d.values())[0]["VRAM Total Used Memory (B)"]) >> 30)')
    [ "$used" -le 2 ] || { rmdir "$LOG.lock"; echo "gpu$GPU holds $used GiB" >&2; exit 1; }
    mkdir -p "$(dirname "$lease")"
    [ -f "$lease" ] && mv "$lease" "$lease.prev-$(date -u +%Y%m%dT%H%M%SZ)"
    printf 'track=9b-m10\nstatus=busy\npurpose=9B M10 formal typed-FINAL path of %s (worker 7e1c9ce8)\nstart_utc=%s\nexpected_end_utc=%s\n' \
      "$NAME" "$(date -u +%FT%TZ)" "$(date -u -d '+150 min' +%FT%TZ)" > "$lease"
    echo "$(date -u +%FT%TZ) formal M10-$NAME launched on node A GPU$GPU (point $ck)" | tee -a "$B9/OPERATIONS.log"
    setsid nohup bash "$0" run "$NAME" "$GPU" > "$LOG.log" 2>&1 < /dev/null &
    ;;
  run)
    ck=$(point)
    M9_FORMAL_GPUS=$GPU M9_FORMAL_TRACK=9b-m10 M9_A_GPUS=$GPU M9_LEASE_TRACK=9b-m10 \
      bash "$OPS/../m9/formal.sh" "$GPU" "M10-$NAME" "$ck" "$LUX" "9B M10 $NAME"
    rc=$?
    echo "$rc" > "$LOG.exit"
    printf 'track=9b-m10\nstatus=released (9B M10 formal %s exit %s)\nlast_job_end_utc=%s\n' "$NAME" "$rc" \
      "$(date -u +%FT%TZ)" > "$lease"
    echo "$(date -u +%FT%TZ) formal M10-$NAME exit $rc" >> "$B9/OPERATIONS.log"
    ;;
  status)
    echo "formal M10-$NAME exit: $(cat "$LOG.exit" 2> /dev/null || echo running)"
    tail -n 4 "$LOG.log" 2> /dev/null
    find "$F" -maxdepth 1 -name "M10-$NAME*" -printf '%f ' 2> /dev/null
    echo ;;
  *) echo "usage: formal.sh launch|status NAME [GPU]" >&2; exit 2 ;;
esac
