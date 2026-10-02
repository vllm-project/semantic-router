#!/usr/bin/env bash
# 9B M10 arm KSW (amendment 1) on node B: the 4B swap recipe at 9B.
#   ksw.sh teacher <mirror-dir> <node-C address>   (CPU) pull node C's M9 KIB soup over the node key (SHA-256 lists
#        equal), rebuild K-a13IB = [KIB soup, Lux zero-step, Lux zero-step] (v2.dec.soup) in teacher/; its model
#        SHA-256 must be K-a13IB's
#   ksw.sh data <mirror-dir>                        (CPU) m9_data.py --cut-language en: IB1-r3 minus `sentfin` + IB2 at
#        K-a13's 60,183,732 native tokens (K-a13IB's cut seed); x60-kept.jsonl = the kept x60 rows (TRAIN order)
#   ksw.sh teach launch|run <mirror-dir>            (GPU) after the phase-1 chains of GPU2 / 4 release their flocks: K-a13IB's T = 1 targets on x60-kept.jsonl (v2.dec.teacher_label --teacher-kind dec --uncalibrated,
#        source Lux 1.0; a 1/1000 pre-warm shard, then two shards), joined in TRAIN order into teacher-sd.jsonl
#        (every kept x60 id once, input hashes equal); KSW enters data/READY-m10.json; then the phase-2 chains
#        (chains.sh, M10_PHASE=2) start on GPU2 / 4 / 6 (GPU6's after its phase-1 chain).
# A failed step writes ksw/FAILED and is never rerun.
set -euo pipefail
MODE=$1
M=/data/dev2/runs/9b/m10
K=$M/ksw
I=$M/inputs/m9
LOCK=$M/data/READY-m10.json
KA13IB=4701ba41c70636b5e215cf1f296bc920a9ee39960d4d28a2338e20b2b81e0d91
LUXSUMS=$M/inputs/lux-zero-m9-KIB-s1.sha256
LUXHOST=$M/arms/pre/m10-KUP-s1-zero/checkpoint-0000000
LUXCK=/runs/m10/arms/pre/m10-KUP-s1-zero/checkpoint-0000000
GPUS=(2 4)
mkdir -p "$K" "$M/logs"
log() { echo "$(date -u +%FT%TZ) ksw $*" | tee -a "$M/OPERATIONS.log"; }
fail() { echo "$*" > "$K/FAILED"; log "FAILED: $*"; exit 1; }
[ ! -f "$K/FAILED" ] || { log "arm KSW stopped earlier ($(cat "$K/FAILED")); not rerun"; exit 1; }
sums() { (cd "$1" && find . -type f | LC_ALL=C sort | xargs -r sha256sum); }

case $MODE in
  teacher)
    SRC=$2 CADDR=$3
    L=/data/dev2/src/$SRC/src/training/decision2/v2/9b/lux9b/m10/launch.sh
    [ ! -e "$K/teacher" ] || { log "teacher exists"; exit 0; }
    soup=/data/dev2/runs/9b/m9/soup/KIB/build/KIB-soup
    dst=$M/inputs/m9-KIB-soup
    if [ ! -d "$dst" ]; then
      rsync -a -e "ssh -i /root/.ssh/d2_temp_cd -o BatchMode=yes" "$CADDR:$soup/" "$dst.part/"
      a=$(ssh -i /root/.ssh/d2_temp_cd -o BatchMode=yes "$CADDR" "cd $soup && find . -type f | LC_ALL=C sort | xargs -r sha256sum")
      [ -n "$a" ] && [ "$a" = "$(sums "$dst.part")" ] || fail "KIB soup copy differs from node C's"
      mv -T "$dst.part" "$dst"
      log "KIB soup pulled from node C ($(wc -l <<< "$a") files, SHA-256 lists equal)"
    fi
    (cd "$LUXHOST" && sha256sum -c --quiet "$LUXSUMS") || fail "Lux zero-step member differs from K-a13IB's"
    # decision_config.json records the member paths, so the members sit (hard links) at M9's container paths
    m9=/data/dev2/runs/9b/m9
    if [ ! -d "$m9/soup/KIB/build/KIB-soup" ]; then
      mkdir -p "$m9/soup/KIB/build" "$m9/arms/pre/m9-KIB-s1-zero"
      cp -al "$dst" "$m9/soup/KIB/build/KIB-soup"
      cp -al "$LUXHOST" "$m9/arms/pre/m9-KIB-s1-zero/checkpoint-0000000"
      echo "node B: hard links of M10's KIB soup copy and Lux zero-step member at M9's paths (9B M10 KSW teacher)" \
        > "$m9/README.node-b"
    fi
    M10_NODE=b bash "$L" ksw-teacher "$SRC" "$K/teacher" --cpu -- -m v2.dec.soup \
      --member /runs/m9/soup/KIB/build/KIB-soup --member /runs/m9/arms/pre/m9-KIB-s1-zero/checkpoint-0000000 \
      --member /runs/m9/arms/pre/m9-KIB-s1-zero/checkpoint-0000000 --output /out/K-a13IB \
      || fail "teacher soup build failed"
    got=$(grep -o '"model_sha256": *"[0-9a-f]\{64\}"' "$K/teacher.stdout.log" | tail -1 | grep -o '[0-9a-f]\{64\}')
    [ "$got" = "$KA13IB" ] || fail "rebuilt teacher identity $got is not K-a13IB's"
    log "teacher rebuilt: K-a13IB identity ${got:0:12} (equal to the release's source)"
    ;;
  data)
    SRC=$2
    L=/data/dev2/src/$SRC/src/training/decision2/v2/9b/lux9b/m10/launch.sh
    [ ! -e "$K/build" ] || { log "KSW data exists"; exit 0; }
    M10_NODE=b bash "$L" ksw-data "$SRC" "$K/build" --cpu -- v2/9b/lux9b/m9_data.py \
      --x60-dir /runs/m10/inputs/m9/data/x60 \
      --ib1 /runs/m10/inputs/m9/inputs/ib/ib1-31b200a3/m6/ib1/ib1.train.jsonl \
      --ib2 /runs/m10/inputs/m9/inputs/ib/ib2-c5dbdd0a/m6/ib2/ib2.train.jsonl --exclude-family sentfin \
      --match-tokens 60183732 --x60-ids /runs/m10/inputs/m9/inputs/x60-ids/mx-xl-full-r2.ids.jsonl \
      --keep-seed 20261001:m9-s3:keep --cut-language en --output /out/ksw || fail "KSW data build failed"
    n=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["x60_rows_kept"])' "$K/build/ksw/manifest.json")
    head -n "$n" "$K/build/ksw/train.jsonl" > "$K/x60-kept.jsonl"
    log "KSW data: $(python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); k=m["x60_keep"]; print("rows", m["rows"], "x60 kept", m["x60_rows_kept"], "fixed groups", k["fixed_groups"], "tokens", m["train_native_tokens"], "IB share", m["ib_token_share"], "train", m["train_sha256"][:12])' "$K/build/ksw/manifest.json")"
    ;;
  teach)
    SUB=$2 SRC=$3
    OPS=/data/dev2/src/$SRC/src/training/decision2/v2/9b/lux9b/m10
    if [ "$SUB" = launch ]; then
      mkdir "$M/chains/ksw-teach.lock" 2> /dev/null || { echo "KSW labeling already launched"; exit 0; }
      setsid nohup bash "$0" teach run "$SRC" > "$M/logs/ksw-teach.log" 2>&1 < /dev/null &
      log "labeling chain launched from $SRC (pid $!)"
      exit 0
    fi
    [ "$SUB" = run ] || { echo "unknown teach mode $SUB" >&2; exit 2; }
    [ -f "$K/x60-kept.jsonl" ] && [ -f "$K/teacher/K-a13IB/decision_config.json" ] || fail "no KSW data or teacher"
    fd=20
    for g in "${GPUS[@]}"; do
      eval "exec $fd>\"$M/chains/gpu$g.flock\""
      log "waiting for GPU$g's phase-1 chain"
      flock "$fd"
      fd=$((fd + 1))
    done
    lease() {  # <gpu> <status> <purpose> <minutes>
      mkdir -p "/data/dev2/leases/gpu$1.lock"
      printf 'track=9b-m10\nstatus=%s\npurpose=9B M10 %s (worker 7e1c9ce8)\nstart_utc=%s\nexpected_end_utc=%s\n' \
        "$2" "$3" "$(date -u +%FT%TZ)" "$(date -u -d "+$4 min" +%FT%TZ)" > "/data/dev2/leases/gpu$1.lock/owner"
    }
    for g in "${GPUS[@]}"; do
      grep -qs '^track=9b-m10' "/data/dev2/leases/gpu$g.lock/owner" || fail "GPU$g lease is not 9b-m10's"
      lease "$g" busy "KSW self-distillation targets" 90
    done
    label() {  # <job> <gpu> <shard> <count>
      M10_NODE=b bash "$OPS/launch.sh" "$1" "$SRC" "$K/teach/$1" --gpu "$2" -- -m v2.dec.teacher_label \
        --teacher-kind dec --uncalibrated --teacher-path /runs/m10/ksw/teacher/K-a13IB --teacher-source-path /lux \
        --teacher-repo m10-ksw-sd-K-a13IB --teacher-revision "$KA13IB" --train /runs/m10/ksw/x60-kept.jsonl \
        --output /out/labels.jsonl --shard-index "$3" --shard-count "$4"
    }
    mkdir -p "$K/teach"
    log "pre-warm shard on GPU${GPUS[0]}"
    label ksw-teach-warm "${GPUS[0]}" 0 1000 || fail "pre-warm labeling failed"
    pids=()
    for i in "${!GPUS[@]}"; do
      label "ksw-teach-s$i" "${GPUS[$i]}" "$i" "${#GPUS[@]}" &
      pids+=($!)
    done
    ok=1
    for p in "${pids[@]}"; do wait "$p" || ok=0; done
    [ "$ok" = 1 ] || fail "a labeling shard failed"
    python3 - "$K/x60-kept.jsonl" "$K/teach" "${#GPUS[@]}" "$K/teacher-sd.jsonl" << 'EOF' || fail "label join failed"
import json, sys
train, teach, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
rows = [json.loads(line) for line in open(train, encoding="utf-8")]
shards = [open(f"{teach}/ksw-teach-s{i}/labels.jsonl", encoding="utf-8").read().splitlines(True) for i in range(n)]
assert sum(map(len, shards)) == len(rows), "label count differs from the kept x60 rows"
agree = {}
with open(out, "x", encoding="utf-8") as f:
    for p, row in enumerate(rows):
        line = shards[p % n][p // n]
        rec = json.loads(line)
        assert rec["id"] == row["id"] and rec["input_sha256"] == row["input_sha256"], row["id"]
        assert set(rec["teacher_probs"]) == {o["key"] for o in row["options"]}, row["id"]
        f.write(line)
        a = agree.setdefault(row["task_type"], [0, 0])
        a[0] += int(max(rec["teacher_probs"], key=rec["teacher_probs"].get) == row["label"])
        a[1] += 1
print(json.dumps({"rows": len(rows), "agreement": {k: round(c / t, 4) for k, (c, t) in sorted(agree.items())}}))
EOF
    log "teacher targets joined: $(sha256sum < "$K/teacher-sd.jsonl" | cut -c1-12)"
    python3 - "$LOCK" "$M" << 'EOF' || fail "data lock entry failed"
import hashlib, json, os, sys
lock_path, m = sys.argv[1:]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()
lock = json.load(open(lock_path))
if "KSW" in lock["arms"]:
    sys.exit("KSW is already in the lock")
files = ["ksw/build/ksw/train.jsonl", "ksw/teacher-sd.jsonl"]
lock["arms"]["KSW"] = {"args": ["--train", "/runs/m10/ksw/build/ksw/train.jsonl", "--teacher", "/runs/m10/ksw/teacher-sd.jsonl"],
                       "files": {rel: sha(f"{m}/{rel}") for rel in files}}
with open(lock_path + ".tmp", "w") as f:
    json.dump(lock, f, indent=2)
os.replace(lock_path + ".tmp", lock_path)
print(json.dumps(lock["arms"]["KSW"]))
EOF
    log "KSW locked"
    for fd in 20 21; do eval "exec $fd>&-"; done
    for g in 2 4 6; do M10_NODE=b M10_PHASE=2 bash "$OPS/chains.sh" launch "$SRC" "$g"; done
    log "phase-2 chains launched"
    ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
