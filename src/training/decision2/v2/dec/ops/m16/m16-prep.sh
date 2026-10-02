#!/usr/bin/env bash
# Decoder M16 preparation on node A / B (prereg dec-m16-prereg-2026-10-01.md, "Points", "Development readouts"), CPU
# only, from an exact mirror. Idempotent; a failed step is not rerun. Sources are pulled read-only over the temporary
# transfer key from node E (M16_FROM_E) and node F (M16_FROM_F, node B only); the addresses come from the operator's
# environment and are not logged.
#   1. leases: node A GPU3-5 / node B GPU2-4 owner files -> track=dec-m16 idle (old files kept as owner.prev-<UTC>);
#   2. arms: the M12 / M13 soups copied whole into m16/inputs/arms/<ARM> (node A 08b-RASD, 08b-RA from node E; node B
#      2b-RA from node E, 2b-RASD and 4b-LHA10SD from node F), content manifest equal on both sides;
#   3. Triton read caches: cp -a of this node's M14 <tier>-read caches;
#   4. references: M14's same-node readouts of the tier references (08b-C0-a; 2b-C0-b, 4b-LH-b) and of the M14 arms
#      (as <ARM>-a100, report only) copied into m16/lines (predictions and manifests; receipts to inputs/m14-receipts);
#   5. lineage: m16_interp.py lineage per release / arm pair -> m16/lineage/<ARM>/<ARM>.json (CPU container);
#   6. (node B) MLX-DEV panels per tier: m16_mlxpanel.py on the decoder MLX-DEV panel minus the tier's M12 arm TRAIN
#      -> m16/mlxdev/<tier>; the index hash must equal M15's lock (0.8B / 4B 6ffa4b84..., 2B 99283f94...).
#
# usage: M16_NODE=a|b M16_FROM_E=<user@node-e> [M16_FROM_F=<user@node-f>] m16-prep.sh <mirror-dir>
set -euo pipefail
SRC=$1
NODE=${M16_NODE:?set M16_NODE=a or b}
FROM_E=${M16_FROM_E:?set M16_FROM_E}
R=/data/dev2/runs/dec
M=$R/m16
CODE=/data/dev2/src/$SRC/src/training/decision2
OPS=$CODE/v2/dec/ops/m16
mkdir -p "$M/inputs/arms" "$M/inputs/m14-receipts" "$M/triton-cache" "$M/lines" "$M/lineage" "$M/status"
log() { echo "$(date -u +%FT%TZ) prep-$NODE $*" | tee -a "$M/OPERATIONS.log"; }
KEY=(-i /root/.ssh/d2_temp_cd -o BatchMode=yes -o ConnectTimeout=20)
manifest='find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | cut -d" " -f1'
# shellcheck disable=SC2029  # the remote paths are expanded here on purpose
pull_dir() {  # <host> <remote dir> <local dir>: whole directory, content manifest equal on both sides
  local want got
  want=$(ssh "${KEY[@]}" "$1" "cd '$2' && $manifest")
  if [ ! -d "$3" ]; then
    mkdir -p "$(dirname "$3")"
    rsync -a -e "ssh ${KEY[*]}" "$1:$2/" "$3.part/" && mv "$3.part" "$3"
  fi
  got=$(cd "$3" && eval "$manifest")
  [ "$got" = "$want" ] || { log "FAILED: $3 manifest $got != source $want"; exit 1; }
  log "pulled $(basename "$3") (manifest $(echo "$got" | cut -c1-16), $(find "$3" -type f | wc -l) files)"
}

case $NODE in
  a) GPUS="3 4 5" TIERS="08b" REFS="08b-C0-a" UP="08b-RAUP"
    ARMS=("08b-RASD=$FROM_E:$R/m13/soup/08b-RASD/build/08b-RASD-soup" "08b-RA=$FROM_E:$R/m12/soup/08b-RA/build/08b-RA-soup") ;;
  b) FROM_F=${M16_FROM_F:?set M16_FROM_F on node B}
    GPUS="2 3 4" TIERS="2b 4b" REFS="2b-C0-b 4b-LH-b" UP="2b-RAUP 4b-LHA10UP"
    ARMS=("2b-RA=$FROM_E:$R/m12/soup/2b-RA/build/2b-RA-soup" "2b-RASD=$FROM_F:$R/m13/soup/2b-RASD/build/2b-RASD-soup"
      "4b-LHA10SD=$FROM_F:$R/m13/soup/4b-LHA10SD/build/4b-LHA10SD-soup") ;;
  *) echo "unknown node $NODE" >&2; exit 2 ;;
esac

stamp=$(date -u +%Y%m%dT%H%M%SZ)
for g in $GPUS; do
  d=/data/dev2/leases/gpu$g.lock
  mkdir -p "$d"
  if ! grep -qs '^track=dec-m16' "$d/owner"; then
    [ -f "$d/owner" ] && mv "$d/owner" "$d/owner.prev-$stamp"
    printf 'track=dec-m16\nstatus=idle\npurpose=decoder M16 (interpolation readouts; staging)\nstart_utc=%s\nexpected_end_utc=%s\n' \
      "$(date -u +%FT%TZ)" "$(date -u -d '+6 hours' +%FT%TZ)" > "$d/owner"
    log "lease gpu$g -> dec-m16"
  fi
done

for spec in "${ARMS[@]}"; do
  arm=${spec%%=*} rest=${spec#*=}
  pull_dir "${rest%%:*}" "${rest#*:}" "$M/inputs/arms/$arm"
done

for t in $TIERS; do
  c=$t-read
  [ -d "$M/triton-cache/$c" ] && continue
  [ -d "$R/m14/triton-cache/$c" ] || { log "FAILED: no M14 cache $c on this node"; exit 1; }
  cp -a "$R/m14/triton-cache/$c" "$M/triton-cache/$c.tmp" && mv "$M/triton-cache/$c.tmp" "$M/triton-cache/$c"
  log "Triton cache $c copied from m14/triton-cache/$c ($(find "$M/triton-cache/$c" -type f | wc -l) files)"
done

for p in $REFS $UP; do
  dst=$M/lines/$p
  case $p in *-C0-* | *-LH-b) ;; *) dst=$M/lines/$p-a100 ;; esac
  [ -d "$dst" ] && continue
  rsync -a --include='*/' --include='*.predictions.jsonl' --include='*.manifest.json' --exclude='*' \
    "$R/m14/lines/$p/" "$dst/"
  rsync -a --include='*/' --include='*.launch.json' --exclude='*' "$R/m14/lines/$p/" "$M/inputs/m14-receipts/$p/"
  log "staged M14 readouts of $p as $(basename "$dst"): $(find "$dst" -name '*.predictions.jsonl' | wc -l) panels"
done

ref_ck() {
  case $1 in
    08b) echo "/runs/m14/inputs/refs/DEV2.0-0.8B/bede7938a8c209c09f27400b79eed57948d6b75e" ;;
    2b) echo "/runs/m14/inputs/refs/DEV2.0-2B/a53cf66a0d9d492a84b6617b61e7ce35fcd03af0" ;;
    4b) echo "/runs/m14/inputs/refs/LH-soup" ;;
  esac
}
for arm in $UP $(for s in "${ARMS[@]}"; do echo "${s%%=*}"; done); do
  out=$M/lineage/$arm
  [ -e "$out.launch.json" ] && continue
  case $arm in *UP) x=/runs/m14/soup/$arm/build/$arm-soup ;; *) x=/runs/m16/inputs/arms/$arm ;; esac
  if M16_NODE=$NODE bash "$OPS/m16-launch.sh" "lineage-$arm" "$SRC" "$out" --cpu -- v2/dec/ops/m16/m16_interp.py \
    lineage --release "$(ref_ck "${arm%%-*}")" --arm "$x" --output "/out/$arm.json"; then
    log "lineage $arm: PASS ($(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["release_model_sha256"][:16], d["arm_model_sha256"][:16])' "$out/$arm.json"))"
  else
    log "lineage $arm: FAILED $(tail -c 300 "$out.stdout.log" "$out.stderr.log" | tr '\n' ' ')"
  fi
done

if [ "$NODE" = b ]; then
  declare -A TRAIN=([08b]="m14/inputs/m12/08b/08b-RA/train.jsonl=12bd63d8b215877b6f471c12652e534e0f0be794c15bff74ad5ac50e98018bf3"
    [2b]="m14/inputs/m12/2b/2b-RA/train.jsonl=08140409a6afc439e35bc4e0aefc9b5812162c4a2972e5e1da5b137a99d0e592"
    [4b]="m14/inputs/m12/4b/4b-LHA10/train.jsonl=d41cdd1aa1e63b47394ed8fa61a87bf461cb4d2cea08cc7ff12f2e58b36dc9d5")
  declare -A WANT=([08b]=6ffa4b84 [2b]=99283f94 [4b]=6ffa4b84)
  for t in 08b 2b 4b; do
    out=$M/mlxdev/$t
    [ -f "$out/OK" ] && continue
    [ ! -e "$out.launch.json" ] || { log "MLX-DEV $t build failed earlier; not rerun"; exit 1; }
    mkdir -p "$M/mlxdev"
    M16_NODE=$NODE bash "$OPS/m16-launch.sh" "mlxpanel-$t" "$SRC" "$out.build" --cpu -- v2/dec/ops/m16/m16_mlxpanel.py \
      --panel /runs/m5/mlxdev/build/panel.jsonl --panel-sha 100ae4e770973640e00b79115e50c7fcca7d4837c22e596690ed6023530652a7 \
      --index /runs/m5/mlxdev/build/panel.jsonl.index.jsonl \
      --index-sha 6ffa4b84bca0ba76c8b10517c4dcfb86cd68b06a8a4031af9eb0cdebc70dc7a8 --train "/runs/${TRAIN[$t]}" \
      --name "mlxdev-m16-$t" --output /out/panel || { log "FAILED: MLX-DEV $t (see $out.build.stderr.log)"; exit 1; }
    got=$(sha256sum "$out.build/panel/panel.jsonl.index.jsonl" | cut -c1-8)
    [ "$got" = "${WANT[$t]}" ] || { log "FAILED: MLX-DEV $t index $got != M15's ${WANT[$t]}"; exit 1; }
    mv "$out.build/panel" "$out" && echo ok > "$out/OK"
    log "MLX-DEV $t: $(tail -1 "$out.build.stdout.log" | cut -c1-300)"
  done
fi
log "prep finished"
