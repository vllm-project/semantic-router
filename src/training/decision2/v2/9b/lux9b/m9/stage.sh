#!/usr/bin/env bash
# 9B M9 input staging (prereg "Stage 1 arms" / "Autotune cache"), from an exact mirror. Hashes are the prereg's.
#
#   stage.sh push-c <user@node-c>   on node A: x60 TRAIN + own-Lux targets, SELECT700 / CAL698, Lux 1.0 (the K recipe's
#                                   /model package) and one copy of the 9B training cache m4/triton-cache to node C
#                                   over the temporary transfer key (rsync; sizes and hashes re-checked on node C)
#   stage.sh local-a                on node A: SELECT700 / CAL698 and M9's readout cache (one copy of m4/triton-cache)
#   stage.sh hf-base                on node C: Qwen/Qwen3.5-9B-Base at the pinned revision with the host HF CLI
#   stage.sh ready                  on node C: re-hash every staged input; data/READY.json only if all match
#   stage.sh base-hashes            on any node: SHA-256 of the base snapshot files (cross-node comparison)
set -euo pipefail
M=/data/dev2/runs/9b/m9
BASE_REPO=Qwen/Qwen3.5-9B-Base
BASE_REV=68c46c4b3498877f3ef123c856ecfde50c39f404
BASE_DIR=/data/dev2/models/Qwen--Qwen3.5-9B-Base/$BASE_REV
LUX_REV=bd45a30aee8c84032791c245c70f86dee5389cc8
LUX_C=/data/dev2/models/Decision-1.0-Lux-9B/$LUX_REV
LUX_A=/data/decision20-20260926/models/Decision-1.0-Lux-9B
X60=/data/dev2/runs/9b/m4/data/m4-k-xl-r2-60m/build
SELECT_A=/data/decision20-20260926/data/rights_clean_goemotions_v2/select.jsonl
CAL_A=/data/dev2/hf-cache/datasets--llm-semantic-router--decision-2.0-training-data/snapshots/ed87a03ab80ca5b9560780bba51a83a77ff47d14/m2/cal/CAL698/cal.jsonl
TRAIN_SHA=a66131b1165513128e5a51af78587c4ddefcc836773e724c4514a592e33fc0e0
TEACHER_SHA=cdcd99c1550d35531df0982ae85116fe3ea5035bc9a09c1ef7b5b3063a32f2c9
SELECT_SHA=32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6
CAL_SHA=19cc1a8c4ebe6fd13031079f6b7d131046ce503f435f894e3ea672f91c2ed41f
CACHE_SRC=/data/dev2/runs/9b/m4/triton-cache
tree_sha() { (cd "$1" && find . -type f ! -path './.cache/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1); }
log() { echo "$(date -u +%FT%TZ) stage $*" | tee -a "$M/OPERATIONS.log"; }
mkdir -p "$M"

case ${1:-} in
  push-c)
    dest=${2:?user@node-c}
    rs() { rsync -a -e 'ssh -i /root/.ssh/d2_temp_cd -o BatchMode=yes' "$@"; }
    on() { ssh -i /root/.ssh/d2_temp_cd -o BatchMode=yes "$dest" "$@"; }
    on "mkdir -p $M/data/x60 $M/inputs/sel700-cal698 $M/triton-cache $LUX_C /data/dev2/runs/dec/panels"
    rs "$X60/train.jsonl" "$X60/teacher.jsonl" "$X60/manifest.json" "$dest:$M/data/x60/"
    rs "$SELECT_A" "$dest:$M/inputs/sel700-cal698/select.jsonl"
    rs -L "$CAL_A" "$dest:$M/inputs/sel700-cal698/cal.jsonl"
    rs --exclude .cache "$LUX_A/" "$dest:$LUX_C/"
    on "test -d $M/triton-cache/f83b1d10" || rs "$CACHE_SRC/" "$dest:$M/triton-cache/f83b1d10/"
    printf '{"lux_tree_sha256": "%s", "cache_tree_sha256": "%s", "cache_files": %s}\n' "$(tree_sha "$LUX_A")" \
      "$(tree_sha "$CACHE_SRC")" "$(find "$CACHE_SRC" -type f | wc -l)" | on "cat > $M/inputs/node-a-trees.json"
    log "pushed x60, SELECT/CAL, Lux 1.0 and the training cache to node C"
    ;;
  local-a)
    mkdir -p "$M/inputs/sel700-cal698" "$M/triton-cache"
    [ -f "$M/inputs/sel700-cal698/select.jsonl" ] || cp "$SELECT_A" "$M/inputs/sel700-cal698/select.jsonl"
    [ -f "$M/inputs/sel700-cal698/cal.jsonl" ] || cp -L "$CAL_A" "$M/inputs/sel700-cal698/cal.jsonl"
    [ "$(sha256sum < "$M/inputs/sel700-cal698/select.jsonl" | cut -d' ' -f1)" = "$SELECT_SHA" ] || { log "SELECT hash differs"; exit 1; }
    [ "$(sha256sum < "$M/inputs/sel700-cal698/cal.jsonl" | cut -d' ' -f1)" = "$CAL_SHA" ] || { log "CAL hash differs"; exit 1; }
    if [ ! -d "$M/triton-cache/f83b1d10" ]; then
      cp -a "$CACHE_SRC" "$M/triton-cache/f83b1d10.pending"
      printf '{"source": "%s", "copied_utc": "%s", "files": %s, "tree_sha256": "%s"}\n' "$CACHE_SRC" \
        "$(date -u +%FT%TZ)" "$(find "$M/triton-cache/f83b1d10.pending" -type f | wc -l)" \
        "$(tree_sha "$M/triton-cache/f83b1d10.pending")" > "$M/triton-cache/f83b1d10.copy.json"
      mv "$M/triton-cache/f83b1d10.pending" "$M/triton-cache/f83b1d10"
    fi
    log "node A inputs ready (SELECT/CAL, readout cache $(cat "$M/triton-cache/f83b1d10.copy.json"))"
    ;;
  hf-base)
    mkdir -p "$BASE_DIR"
    HF_HUB_CACHE=/data/dev2/hf-cache hf download "$BASE_REPO" --revision "$BASE_REV" --local-dir "$BASE_DIR" \
      > "$M/inputs/hf-base.log" 2>&1
    log "downloaded $BASE_REPO@$BASE_REV: $(find "$BASE_DIR" -name '*.safetensors' | wc -l) safetensors files"
    ;;
  base-hashes)
    dir=${2:-$BASE_DIR}
    (cd "$dir" && find . -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.json' -o -name '*.txt' \) -print0 \
      | LC_ALL=C sort -z | xargs -0 sha256sum)
    ;;
  ready)
    python3 - "$M" "$TRAIN_SHA" "$TEACHER_SHA" "$SELECT_SHA" "$CAL_SHA" "$LUX_C" "$BASE_DIR" "$BASE_REV" << 'EOF'
import hashlib, json, os, subprocess, sys
m, train, teacher, select, cal, lux, base, rev = sys.argv[1:]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()
def tree(d):
    out = subprocess.run(f"cd {d} && find . -type f ! -path './.cache/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum",
                         shell=True, capture_output=True, text=True, check=True).stdout.split()[0]
    return out
checks = {
    "train": (sha(f"{m}/data/x60/train.jsonl"), train),
    "teacher": (sha(f"{m}/data/x60/teacher.jsonl"), teacher),
    "select": (sha(f"{m}/inputs/sel700-cal698/select.jsonl"), select),
    "cal": (sha(f"{m}/inputs/sel700-cal698/cal.jsonl"), cal),
}
trees = json.load(open(f"{m}/inputs/node-a-trees.json"))
checks["lux_tree"] = (tree(lux), trees["lux_tree_sha256"])
checks["cache_tree"] = (tree(f"{m}/triton-cache/f83b1d10"), trees["cache_tree_sha256"])
bad = {k: v for k, v in checks.items() if v[0] != v[1]}
if bad:
    print(json.dumps({"status": "FAIL", "mismatch": bad}))
    sys.exit(1)
if not any(n.endswith(".safetensors") for n in os.listdir(base)):
    print(json.dumps({"status": "FAIL", "reason": "base snapshot missing"}))
    sys.exit(1)
lock = {"schema": "lux9b-m9-ready/1", "train_sha256": train, "teacher_sha256": teacher, "select_sha256": select,
        "cal_sha256": cal, "lux_tree_sha256": checks["lux_tree"][0], "cache_tree_sha256": checks["cache_tree"][0],
        "base_dir": base, "base_revision": rev}
with open(f"{m}/data/READY.json", "x") as f:
    json.dump(lock, f, indent=2)
print(json.dumps({"status": "READY", **lock}))
EOF
    log "data/READY.json written"
    ;;
  *) sed -n '2,10p' "$0"; exit 2 ;;
esac
