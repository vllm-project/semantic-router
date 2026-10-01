#!/usr/bin/env bash
# 9B M9 stage-3 inputs on node C (amendment 3), host python3 from an exact mirror. The x60 ids file (per-row native
# tokens and pools; research & data's XL r2 ids, copied from node A), x60 and the stage-2 IB TRAIN files are
# hash-checked, then lux9b/m9_data.py --match-tokens builds, at K-a13's 60,183,732 native tokens per seed,
#   data/kib   x60 cut to (60,183,732 - IB tokens) in whole groups, stratified as the x60 recipe, + IB1-r3 + IB2
#   data/kibx  the same without the IB in-distribution families isarc, w2c, hover, gsm2 (so x60 is cut less)
# with one cut seed for both (the kib x60 rows are a subset of the kibx ones), and data/READY3.json records both
# builds' TRAIN / teacher hashes and token counts (the stage-3 chains re-hash against it).
#
# usage: stage3.sh build
set -euo pipefail
M=/data/dev2/runs/9b/m9
D=$M/inputs/ib
OWN=$(cd "$(dirname "$0")/../../../.." && pwd)
IB1=$D/ib1-31b200a3/m6/ib1/ib1.train.jsonl
IB2=$D/ib2-c5dbdd0a/m6/ib2/ib2.train.jsonl
IDS=$M/inputs/x60-ids/mx-xl-full-r2.ids.jsonl
MATCH=60183732
SEED=20261001:m9-s3:keep
declare -A WANT=(
  ["$IB1"]=1e1b08f3d37f9051ffe2e4b99fd7be673f315d05cc74a27a8d350eae2bb706c5
  ["$IB2"]=ee137efa8bbf86e5c62574f3b8fbd6204063e095a8da514f32600ba3d51d1cfa
  ["$IDS"]=7843afb7b2bbb315902b6748d6559532f384283d8efd836b935daac5f55dbd4a
  ["$M/data/x60/train.jsonl"]=a66131b1165513128e5a51af78587c4ddefcc836773e724c4514a592e33fc0e0
  ["$M/data/x60/teacher.jsonl"]=cdcd99c1550d35531df0982ae85116fe3ea5035bc9a09c1ef7b5b3063a32f2c9
)
log() { echo "$(date -u +%FT%TZ) stage3 $*" | tee -a "$M/OPERATIONS.log"; }
[ "${1:-}" = build ] || { sed -n '2,10p' "$0"; exit 2; }
for f in "${!WANT[@]}"; do
  [ "$(sha256sum < "$f" | cut -d' ' -f1)" = "${WANT[$f]}" ] || { log "hash differs: $f"; exit 1; }
done
build() {  # <name> [m9_data args...]
  local name=$1
  shift
  PYTHONPATH="$OWN:$OWN/v2/9b" python3 -B "$OWN/v2/9b/lux9b/m9_data.py" --x60-dir "$M/data/x60" --ib1 "$IB1" \
    --ib2 "$IB2" --match-tokens "$MATCH" --x60-ids "$IDS" --keep-seed "$SEED" "$@" --output "$M/data/$name" \
    > "$M/data/$name.build.log"
}
build kib
build kibx --exclude-family isarc --exclude-family w2c --exclude-family hover --exclude-family gsm2
python3 - "$M" "$MATCH" "$IB1" "$IB2" << 'EOF'
import hashlib, json, sys
m, match, ib1, ib2 = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()
def x60_ids(name, n):
    with open(f"{m}/data/{name}/train.jsonl") as f:
        return {json.loads(next(f))["id"] for _ in range(n)}
lock = {"schema": "lux9b-m9-ready3/1", "match_tokens": match,
        "ib_tokens_files": {p: sha(p.replace(".jsonl", ".tokens.jsonl")) for p in (ib1, ib2)}}
kept = {}
for name in ("kib", "kibx"):
    man = json.load(open(f"{m}/data/{name}/manifest.json"))
    if man["x60_keep"]["x60_native_tokens"] != match:
        sys.exit(f"{name}: x60 native tokens {man['x60_keep']['x60_native_tokens']} != {match}")
    lock[name] = {k: man[k] for k in ("train_sha256", "teacher_sha256", "rows", "x60_rows_kept", "ib_rows_kept",
                                      "ib_native_tokens_kept", "train_native_tokens", "ib_token_share")}
    lock[name]["x60_keep"] = {k: man["x60_keep"][k] for k in ("budget_tokens", "native_tokens", "overshoot_tokens",
                                                              "groups", "rows", "strata", "seed", "tolerance")}
    kept[name] = x60_ids(name, man["x60_rows_kept"])
lock["kib_x60_subset_of_kibx"] = kept["kib"] <= kept["kibx"]
with open(f"{m}/data/READY3.json", "x") as f:
    json.dump(lock, f, indent=2)
print(json.dumps(lock))
EOF
log "stage-3 TRAIN built at matched tokens: data/READY3.json"
