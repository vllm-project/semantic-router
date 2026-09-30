# shellcheck shell=bash disable=SC2034
# Shared host-side settings of the decoder M8-small scripts (prereg dec-m8s-prereg-2026-09-30.md); sourced.
# The caller runs from an exact mirror under /data/dev2/src; S / SRC are derived from this file's location.
S=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
MIRROR=${S%/src/training/decision2}
SRC=${MIRROR##*/}
[ -f "$MIRROR/.dev2-mirror.json" ] || { echo "run from an exact mirror under /data/dev2/src" >&2; exit 2; }
OPS=$S/v2/dec/ops/m8s
LAUNCH=$S/v2/dec/launch.sh
R=/data/dev2/runs/dec
M=$R/m8s
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
declare -A RENDER=([2]=/dev/dri/renderD145 [5]=/dev/dri/renderD169 [6]=/dev/dri/renderD177 [7]=/dev/dri/renderD185)
SELCAL=$R/m3/data-sel700-cal698
TCACHE=$M/triton-cache/dbe5f32b2263
declare -A START=([2b]=$M/start/dev2-2b-a53cf66a [08b]=$M/start/dev2-0p8b-bede7938)
declare -A START_REPO=([2b]=llm-semantic-router/DEV2.0-2B [08b]=llm-semantic-router/DEV2.0-0.8B)
declare -A START_REV=([2b]=a53cf66a0d9d492a84b6617b61e7ce35fcd03af0 [08b]=bede7938a8c209c09f27400b79eed57948d6b75e)
declare -A START_ID=([2b]=32872f2968e38aa99901797aaab5e12e32e281bf67bb924037d443394411170d
  [08b]=3f02f0e5fc68a1fd985ce48bf3a3bb2d4ed16d1a1705c2258317b78a4d6514f4)
SEEDS=(20260930 20260931 20260932)
LAMBDA=${M8S_LAMBDA:-1.0}
declare -A CKL=([2b]=0.5 [08b]="")
declare -A CAP=([2b-D1]=1.2 [2b-D2]=1.2 [2b-C]=1.2 [08b-D1]=0.9 [08b-D2]=0.9 [08b-C]=0.9)
GOLD=/data/dev2/private/panels/gold
HT_GOLD=$GOLD/ht-dev2.gold.jsonl HT_GOLD_SHA=659c92b45d2a5b37e250ccb728cdbf5580ba361b3fc4b2f2ab505a13e0a556cc
HT_PROMPTS_SHA=90cd409a1e091a623362c0e5b227d13b7301bf13fe266f7905e09233cf815f74
TRAIN_STOP_GPUH=21
mkdir -p "$M/logs" "$M/status" "$M/arms" "$M/soup" "$M/early"

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$M/OPERATIONS.log" >&2; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
incontainer() { echo "/runs/${1#"$R"/}"; }
vram_free_gb() {
  rocm-smi -d "$1" --showmeminfo vram | awk -F': ' '/Total Memory/ {t=$NF} /Total Used Memory/ {u=$NF} END {printf "%d\n", (t-u)/1e9}'
}
dec_env() {  # <gpu>: launch.sh environment for node B GPU <gpu>
  [ -n "${RENDER[$1]:-}" ] || { echo "GPU$1 is not an M8-small GPU (node B 2, 5, 6, 7)" >&2; return 2; }
  export DEC_IMAGE=$IMAGE DEC_RENDER=${RENDER[$1]} DEC_GPU_LABEL="node B GPU$1" DEC_DATA=$SELCAL DEC_TRITON_CACHE=$TCACHE
}
lease() {  # <gpu> <status> <purpose>: this track's shared lease entry (never another track's owner file)
  printf 'track=dec-small\nstatus=%s\npurpose=decoder M8-small %s\nupdated_utc=%s\nnote=shared lease; GPU owner file belongs to the ~27B track\n' \
    "$2" "$3" "$(date -u +%FT%TZ)" > "/data/dev2/leases/gpu$1.lock/owner.dec-m8s"
}
wait_gpu() {  # <gpu> <min free GB>: the owner track has no running job and enough VRAM is free
  local n=0 owner=/data/dev2/leases/gpu$1.lock/owner
  while grep -qsE '^status=(running|busy)' "$owner" || [ "$(vram_free_gb "$1")" -lt "$2" ]; do
    [ $((n % 15)) = 0 ] && log "GPU$1 waiting (owner: $(grep -s '^status=' "$owner" | head -1); free $(vram_free_gb "$1") GB < $2?)"
    n=$((n + 1))
    sleep 60
  done
}
gpu_seconds() {  # <launch receipt> <item> <purpose>
  python3 - "$1" "$2" "$3" >> "$M/GPU-SECONDS.jsonl" <<'EOF'
import datetime as dt, json, sys
r = json.load(open(sys.argv[1]))
t = lambda s: dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
print(json.dumps({"item": sys.argv[2], "purpose": sys.argv[3], "job": r["job"], "gpu": r["gpu"],
                  "start_utc": r["start_utc"], "end_utc": r["end_utc"], "exit_status": r["exit_status"],
                  "gpu_seconds": (t(r["end_utc"]) - t(r["start_utc"])).total_seconds(), "receipt": sys.argv[1]}))
EOF
}
gpuh_total() { python3 -c 'import json,sys; print(round(sum(json.loads(l)["gpu_seconds"] for l in open(sys.argv[1]))/3600, 4))' "$M/GPU-SECONDS.jsonl" 2>/dev/null || echo 0; }
gpuh_item() { python3 -c 'import json,sys; print(round(sum(r["gpu_seconds"] for r in map(json.loads, open(sys.argv[1])) if r["item"].startswith(sys.argv[2]))/3600, 4))' "$M/GPU-SECONDS.jsonl" "$1" 2>/dev/null || echo 0; }
