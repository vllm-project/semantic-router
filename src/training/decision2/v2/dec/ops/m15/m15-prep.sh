#!/usr/bin/env bash
# Decoder M15 data preparation on node E / F (prereg dec-m15-prereg-2026-10-01.md, "Data" and "Development readouts"),
# CPU only (host python3, standard library), from an exact mirror. Idempotent; a failed build is not rerun.
#   1. inputs, each against its hash: M12's arm TRAIN files (4b-LHA10, 2b-RA, 08b-RA), the released TRAIN files, IB1-r3 /
#      IB2 TRAIN, M13's SD targets (m15/inputs/teacher-<tier>.jsonl, copied from the labeling node), the decoder MLX-DEV
#      panel (m15/mlxdev/src, from node B);
#   2. Triton caches <tier>-train / <tier>-read: cp -a copies of this node's M13 caches;
#   3. TRAIN + teacher of the five arms (m15_data.py, seed 20261002), on both nodes (the data lock compares the builds);
#   4. MLX-DEV-M15 panels per tier (m15_mlxpanel.py);
#   5. reference readouts (08b-C0-e on E; 4b-LH-f, 2b-C0-f on F) and the M13 arms as <arm>-m13, copied from M13;
#   6. (node E) the M12 probe-overlap hits, valid for every M15 arm (every M15 TRAIN row is a released or IB row).
#
# usage: M15_NODE=e|f m15-prep.sh <mirror-dir>
set -euo pipefail
SRC=$1
NODE=${M15_NODE:?set M15_NODE=e or f}
R=/data/dev2/runs/dec
M=$R/m15
CODE=/data/dev2/src/$SRC/src/training/decision2
OPS=$CODE/v2/dec/ops/m15
mkdir -p "$M/data/4b" "$M/data/2b" "$M/data/08b" "$M/triton-cache" "$M/probes" "$M/lines" "$M/mlxdev"
log() { echo "$(date -u +%FT%TZ) prep-$NODE $*" | tee -a "$M/OPERATIONS.log"; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
check() { [ "$(sha "$1")" = "$2" ] || { log "FAILED: $1 is not $2"; exit 1; }; }
IB1=$R/m11/inputs/ib1/ib1.train.jsonl IB1_SHA=1e1b08f3d37f9051ffe2e4b99fd7be673f315d05cc74a27a8d350eae2bb706c5
IB2=$R/m11/inputs/ib2/ib2.train.jsonl IB2_SHA=ee137efa8bbf86e5c62574f3b8fbd6204063e095a8da514f32600ba3d51d1cfa
BASE_4b=$R/m10/data/m10-4b-base/train.jsonl BSHA_4b=c385406e8f78a2ae257cf2479b7088a523be18009e55e16caf59327c34260f09
BASE_2b=$R/m11/inputs/m4-v2m-ret-r2/train.jsonl BSHA_2b=1527b38b1ba888695fe48dd43e92827d1719d57674009cfc29d5ab759b08dd2c
BASE_08b=$R/m11/inputs/m6-e8f-r2clean/train.jsonl BSHA_08b=f9f3c0229551357ad1cb8e6923c64c636ee697e71d162f059548ae223e78d8ae
ARM_4b=$R/m12/data/4b/4b-LHA10 ASHA_4b=d41cdd1aa1e63b47394ed8fa61a87bf461cb4d2cea08cc7ff12f2e58b36dc9d5
ARM_2b=$R/m12/data/2b/2b-RA ASHA_2b=08140409a6afc439e35bc4e0aefc9b5812162c4a2972e5e1da5b137a99d0e592
ARM_08b=$R/m12/data/08b/08b-RA ASHA_08b=12bd63d8b215877b6f471c12652e534e0f0be794c15bff74ad5ac50e98018bf3
TSHA_4b=7639fab17c719bb3ed7a18bd16397130f109c8bd204d17c45f6743d6f0c17496
TSHA_2b=2b9858d89fb650db9bc3e7fa608755135b81d7b20ff7b1b6fa186c86bf2e49f2
TSHA_08b=18827abcc49ffa73fb06ffbda68661848f373417d279f544d6c255fadbade4f6
MLX=$M/mlxdev/src MLX_SHA=100ae4e770973640e00b79115e50c7fcca7d4837c22e596690ed6023530652a7
MLXI_SHA=6ffa4b84bca0ba76c8b10517c4dcfb86cd68b06a8a4031af9eb0cdebc70dc7a8
SEED=20261002

check "$IB1" $IB1_SHA
check "$IB2" $IB2_SHA
for t in 4b 2b 08b; do
  b=BASE_$t bs=BSHA_$t a=ARM_$t as=ASHA_$t ts=TSHA_$t
  check "${!b}" "${!bs}"
  check "${!a}/train.jsonl" "${!as}"
  check "$M/inputs/teacher-$t.jsonl" "${!ts}"
done
check "$MLX/panel.jsonl" $MLX_SHA
check "$MLX/panel.jsonl.index.jsonl" $MLXI_SHA
log "M15 inputs verified"

for t in 4b 2b 08b; do
  for k in train read; do
    c=$t-$k
    [ -d "$M/triton-cache/$c" ] && continue
    [ -d "$R/m13/triton-cache/$c" ] || { log "FAILED: no M13 cache $c on this node"; exit 1; }
    cp -a "$R/m13/triton-cache/$c" "$M/triton-cache/$c.tmp" && mv "$M/triton-cache/$c.tmp" "$M/triton-cache/$c"
    log "Triton cache $c copied from m13/triton-cache/$c ($(find "$M/triton-cache/$c" -type f | wc -l) files)"
  done
done

arm() {  # <tier> <ARM> [--ib-share X]
  local t=$1 name=$2 a=ARM_$1 as=ASHA_$1 ts=TSHA_$1
  shift 2
  [ -f "$M/data/$t/$name/report.json" ] && return 0
  [ ! -e "$M/data/$t/$name.FAILED" ] || { log "$name build failed earlier; not rerun"; exit 1; }
  if python3 -B "$OPS/m15_data.py" --arm-train "${!a}/train.jsonl" --arm-sha "${!as}" --arm-ids "${!a}/train.ids.jsonl" \
    --teacher "$M/inputs/teacher-$t.jsonl" --teacher-sha "${!ts}" --name "$name" --seed $SEED --output "$M/data/$t" "$@" \
    > "$M/data/$t/$name.log" 2>&1; then
    log "TRAIN $name: $(tail -1 "$M/data/$t/$name.log" | cut -c1-400)"
  else
    touch "$M/data/$t/$name.FAILED"
    log "FAILED: $name build (see $M/data/$t/$name.log)"
    exit 1
  fi
}
arm 4b 4b-LHA10SDML
arm 2b 2b-RASDML
arm 2b 2b-RA10SDML --ib-share 0.1
arm 08b 08b-RASDML
arm 08b 08b-RA10SDML --ib-share 0.1

for t in 4b 2b 08b; do
  out=$M/mlxdev/$t
  [ -f "$out/report.json" ] && continue
  [ ! -e "$out.FAILED" ] || { log "MLX-DEV-M15 $t build failed earlier; not rerun"; exit 1; }
  b=BASE_$t bs=BSHA_$t
  if python3 -B "$OPS/m15_mlxpanel.py" --panel "$MLX/panel.jsonl" --panel-sha $MLX_SHA \
    --index "$MLX/panel.jsonl.index.jsonl" --index-sha $MLXI_SHA --train "${!b}=${!bs}" --train "$IB1=$IB1_SHA" \
    --train "$IB2=$IB2_SHA" --name "mlxdev-m15-$t" --output "$out" > "$out.log" 2>&1; then
    log "MLX-DEV-M15 $t: $(tail -1 "$out.log")"
  else
    touch "$out.FAILED"
    log "FAILED: MLX-DEV-M15 $t (see $out.log)"
    exit 1
  fi
done

case $NODE in
  e) refs="08b-C0-e:08b-C0-e 08b-RASD:08b-RASD-m13" ;;
  f) refs="4b-LH-f:4b-LH-f 2b-C0-f:2b-C0-f 4b-LHA10SD:4b-LHA10SD-m13 2b-RASD:2b-RASD-m13" ;;
esac
for ref in $refs; do
  from=${ref%%:*} to=${ref#*:}
  [ -d "$M/lines/$to" ] && continue
  cp -a "$R/m13/lines/$from" "$M/lines/$to.tmp" && mv "$M/lines/$to.tmp" "$M/lines/$to"
  log "readouts $to copied from m13/lines/$from ($(find "$M/lines/$to" -name '*.predictions.jsonl' | wc -l) panels)"
done

[ "$NODE" = e ] || { log "prep finished (node F: no probe step)"; exit 0; }
for t in 4b 2b 08b; do
  [ -f "$M/probes/hits-$t.json" ] || cp "$R/m12/probes/hits-$t.json" "$M/probes/hits-$t.json"
done
log "probe hits copied from m12/probes (4b / 2b / 08b)"
log "prep finished"
