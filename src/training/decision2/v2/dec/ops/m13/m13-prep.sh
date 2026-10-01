#!/usr/bin/env bash
# Decoder M13 data preparation on node E / F (prereg dec-m13-prereg-2026-10-01.md, "Data"), CPU only, from an exact
# mirror. Idempotent; a failed build is not rerun.
#   1. inputs: the released TRAIN files (as M12), IB1-r3 / IB2 TRAIN and the three M12 arm TRAIN files that M13 reuses,
#      each against its M12 data-lock hash;
#   2. M13 Triton caches <tier>-train / <tier>-read: cp -a copies of this node's M12 caches;
#   3. TRAIN: 4b-LHA10SD / 2b-RASD / 08b-RASD are hard links of M12's 4b-LHA10 / 2b-RA / 08b-RA files; 4b-LHA5 is
#      M12's m12_data.py with 4b-LHA5=all:0.05 (same seed, so its IB groups are drawn exactly as M12's arms); 08b-RAAG
#      is m13_data.py over M12's 08b-RA (the eligibility-gate Choice families at weight 3); built on both nodes (the
#      data lock compares the builds);
#   4. reference readouts: node E 08b-C0-e and node F 4b-LH-f from M12 (all eight panels); node F 2b-C0-f from M11
#      (no ib-dev panel; the 2b post chain reads it);
#   5. (node E) the M12 probe-overlap hits, valid for every M13 arm (every M13 TRAIN row is a released TRAIN or IB row).
#
# usage: M13_NODE=e|f m13-prep.sh <mirror-dir>
set -euo pipefail
SRC=$1
NODE=${M13_NODE:?set M13_NODE=e or f}
R=/data/dev2/runs/dec
M=$R/m13
CODE=/data/dev2/src/$SRC/src/training/decision2
OPS=$CODE/v2/dec/ops/m13
mkdir -p "$M/data/4b" "$M/data/2b" "$M/data/08b" "$M/triton-cache" "$M/probes" "$M/lines"
log() { echo "$(date -u +%FT%TZ) prep-$NODE $*" | tee -a "$M/OPERATIONS.log"; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
check() { [ "$(sha "$1")" = "$2" ] || { log "FAILED: $1 is not $2"; exit 1; }; }
IB1=/runs/m11/inputs/ib1/ib1.train.jsonl IB1_SHA=1e1b08f3d37f9051ffe2e4b99fd7be673f315d05cc74a27a8d350eae2bb706c5
IB2=/runs/m11/inputs/ib2/ib2.train.jsonl IB2_SHA=ee137efa8bbf86e5c62574f3b8fbd6204063e095a8da514f32600ba3d51d1cfa
BASE_4b=/runs/m10/data/m10-4b-base/train.jsonl SHA_4b=c385406e8f78a2ae257cf2479b7088a523be18009e55e16caf59327c34260f09
BASE_08b=/runs/m11/inputs/m6-e8f-r2clean/train.jsonl SHA_08b=f9f3c0229551357ad1cb8e6923c64c636ee697e71d162f059548ae223e78d8ae
TOK_4b=/models/Qwen--Qwen3.5-4B-Base/1001bb4d826a52d1f399e183466143f4da7b741b
LHA10=$R/m12/data/4b/4b-LHA10/train.jsonl LHA10_SHA=d41cdd1aa1e63b47394ed8fa61a87bf461cb4d2cea08cc7ff12f2e58b36dc9d5
RA2=$R/m12/data/2b/2b-RA/train.jsonl RA2_SHA=08140409a6afc439e35bc4e0aefc9b5812162c4a2972e5e1da5b137a99d0e592
RA08=$R/m12/data/08b/08b-RA/train.jsonl RA08_SHA=12bd63d8b215877b6f471c12652e534e0f0be794c15bff74ad5ac50e98018bf3
AG_FAMILIES=scoped_sources,stage3_evidence_scope,stage4_replay_stage3_evidence_scope,stage4_scope,authorization,stage4_replay_authorization,stage4_replay_policy
AG_WEIGHT=3
host() { echo "$R/${1#/runs/}"; }

check "$(host $IB1)" $IB1_SHA
check "$(host $IB2)" $IB2_SHA
check "$(host $BASE_4b)" $SHA_4b
check "$(host $BASE_08b)" $SHA_08b
check "$LHA10" $LHA10_SHA
check "$RA2" $RA2_SHA
check "$RA08" $RA08_SHA
log "M13 inputs verified"

for t in 4b 2b 08b; do
  for k in train read; do
    c=$t-$k
    [ -d "$M/triton-cache/$c" ] && continue
    [ -d "$R/m12/triton-cache/$c" ] || { log "FAILED: no M12 cache $c on this node"; exit 1; }
    cp -a "$R/m12/triton-cache/$c" "$M/triton-cache/$c.tmp" && mv "$M/triton-cache/$c.tmp" "$M/triton-cache/$c"
    log "Triton cache $c copied from m12/triton-cache/$c ($(find "$M/triton-cache/$c" -type f | wc -l) files)"
  done
done

link() {  # <tier> <ARM> <M12 file>
  [ -f "$M/data/$1/$2/train.jsonl" ] && return 0
  mkdir -p "$M/data/$1/$2"
  ln "$3" "$M/data/$1/$2/train.jsonl"
  log "TRAIN $2 = hard link of ${3#"$R"/}"
}
link 4b 4b-LHA10SD "$LHA10"
link 2b 2b-RASD "$RA2"
link 08b 08b-RASD "$RA08"

if [ ! -f "$M/data/4b/4b-LHA5/train.jsonl" ]; then
  [ ! -e "$M/data/lha5-build.launch.json" ] || { log "4b-LHA5 build failed earlier; not rerun"; exit 1; }
  M13_NODE=$NODE bash "$OPS/m13-launch.sh" "data-lha5" "$SRC" "$M/data/lha5-build" --cpu -- v2/dec/ops/m12/m12_data.py \
    --tier 4b --base $BASE_4b --base-sha $SHA_4b --ib1 $IB1 --ib1-sha $IB1_SHA --ib2 $IB2 --ib2-sha $IB2_SHA \
    --tokenizer $TOK_4b --indist w2c,isarc,hover,gsm2 --arm 4b-LHA5=all:0.05 --output /out/4b --workers 32 \
    || { log "4b-LHA5 build FAILED (see $M/data/lha5-build.stderr.log)"; exit 1; }
  mv "$M/data/lha5-build/4b/4b-LHA5" "$M/data/4b/4b-LHA5"
  cp "$M/data/lha5-build/4b/report.json" "$M/data/4b/4b-LHA5/report.json"
  log "TRAIN 4b-LHA5: $(tail -1 "$M/data/lha5-build.stdout.log" | cut -c1-400)"
fi

if [ ! -f "$M/data/08b/08b-RAAG/train.jsonl" ]; then
  [ ! -e "$M/data/raag-build.launch.json" ] || { log "08b-RAAG build failed earlier; not rerun"; exit 1; }
  M13_NODE=$NODE bash "$OPS/m13-launch.sh" "data-raag" "$SRC" "$M/data/raag-build" --cpu -- v2/dec/ops/m13/m13_data.py \
    --arm-train /runs/m12/data/08b/08b-RA/train.jsonl --arm-sha $RA08_SHA --released $BASE_08b \
    --released-sha $SHA_08b --families $AG_FAMILIES --weight $AG_WEIGHT --name 08b-RAAG --output /out \
    || { log "08b-RAAG build FAILED (see $M/data/raag-build.stderr.log)"; exit 1; }
  mv "$M/data/raag-build/08b-RAAG" "$M/data/08b/08b-RAAG"
  log "TRAIN 08b-RAAG: $(tail -1 "$M/data/raag-build.stdout.log" | cut -c1-400)"
fi

case $NODE in
  e) refs="m12:08b-C0-e" ;;
  f) refs="m12:4b-LH-f m11:2b-C0-f" ;;
esac
for ref in $refs; do
  ms=${ref%%:*} p=${ref#*:}
  [ -d "$M/lines/$p" ] && continue
  cp -a "$R/$ms/lines/$p" "$M/lines/$p.tmp" && mv "$M/lines/$p.tmp" "$M/lines/$p"
  log "reference readouts $p copied from $ms/lines/$p ($(find "$M/lines/$p" -name '*.predictions.jsonl' | wc -l) panels)"
done

[ "$NODE" = e ] || { log "prep finished (node F: no probe step)"; exit 0; }
for t in 4b 2b 08b; do
  [ -f "$M/probes/hits-$t.json" ] || cp "$R/m12/probes/hits-$t.json" "$M/probes/hits-$t.json"
done
log "probe hits copied from m12/probes (4b / 2b / 08b)"
log "prep finished"
