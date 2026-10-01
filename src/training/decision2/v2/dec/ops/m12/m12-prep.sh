#!/usr/bin/env bash
# shellcheck disable=SC2034  # the per-tier values (NAME_<tier>) are read by name through tv()
# Decoder M12 data preparation on node E / F (prereg dec-m12-prereg-2026-10-01.md, "Data"), CPU only, from an exact
# mirror. Idempotent; a failed build is not rerun.
#   1. input hashes: each tier's released TRAIN (as M11), IB1-r3 / IB2 TRAIN, the 4B own-Lux and 2B own-Sol teachers;
#   2. M12 Triton caches <tier>-train / <tier>-read, each a cp -a copy of this node's M11 cache of the same name;
#   3. the TRAIN files (m12_data.py in the decoder image) -> m12/data/<tier>/<ARM>/train.jsonl + report.json, every
#      tier on both nodes (the data lock compares the two builds);
#   4. reference readouts copied from M11 (cp -a): node E 2b-C0-e / 08b-C0-e, node F 4b-LH-f;
#   5. (node E) retention-probe overlap (m10_probes.py overlap --kind train, 13-grams): the 4B released TRAIN and the
#      whole IB1-r3 / IB2 pool, joined with M11's 2B / 0.8B base hits -> m12/probes/hits-<tier>.json (the panel ids hit
#      by the tier's base TRAIN or by any IB row; a superset of every arm's hits).
#
# usage: M12_NODE=e|f m12-prep.sh <mirror-dir>
set -euo pipefail
SRC=$1
NODE=${M12_NODE:?set M12_NODE=e or f}
R=/data/dev2/runs/dec
M=$R/m12
CODE=/data/dev2/src/$SRC/src/training/decision2
OPS=$CODE/v2/dec/ops/m12
mkdir -p "$M/data" "$M/triton-cache" "$M/probes" "$M/lines"
log() { echo "$(date -u +%FT%TZ) prep-$NODE $*" | tee -a "$M/OPERATIONS.log"; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
check() { [ "$(sha "$1")" = "$2" ] || { log "FAILED: $1 is not $2"; exit 1; }; }
IB1=/runs/m11/inputs/ib1/ib1.train.jsonl IB1_SHA=1e1b08f3d37f9051ffe2e4b99fd7be673f315d05cc74a27a8d350eae2bb706c5
IB2=/runs/m11/inputs/ib2/ib2.train.jsonl IB2_SHA=ee137efa8bbf86e5c62574f3b8fbd6204063e095a8da514f32600ba3d51d1cfa
BASE_4b=/runs/m10/data/m10-4b-base/train.jsonl SHA_4b=c385406e8f78a2ae257cf2479b7088a523be18009e55e16caf59327c34260f09
BASE_2b=/runs/m11/inputs/m4-v2m-ret-r2/train.jsonl SHA_2b=1527b38b1ba888695fe48dd43e92827d1719d57674009cfc29d5ab759b08dd2c
BASE_08b=/runs/m11/inputs/m6-e8f-r2clean/train.jsonl SHA_08b=f9f3c0229551357ad1cb8e6923c64c636ee697e71d162f059548ae223e78d8ae
TEACHER_4b=$R/m10/inputs/n4xf-teacher/m4-xl-full-29m/lux-teacher.jsonl
TEACHER_4b_SHA=e2ff27ce2fc397c8c203e37133dcfac52ac3d1f172c6f3e9412ec28c2ffe296c
TEACHER_2b=$R/m11/data/2b/teacher.jsonl TEACHER_2b_SHA=3f92e8c108c5a9949fda4b08638907fcc80175f68efec18bece912d0edc8eea1
TOK_4b=/models/Qwen--Qwen3.5-4B-Base/1001bb4d826a52d1f399e183466143f4da7b741b
TOK_2b=/models/Qwen--Qwen3.5-2B-Base/b1485b2fa6dfa1287294f269f5fb618e03d52d7c
TOK_08b=/models/Qwen--Qwen3.5-0.8B-Base/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68
ARMS_4b="4b-LHA=all:0.25 4b-LHA10=all:0.10 4b-LHAx=transfer:0.25 4b-LHA10x=transfer:0.10"
ARMS_2b="2b-RA=all:0.25"
ARMS_08b="08b-RA=all:x3"
host() { echo "$R/${1#/runs/}"; }
tv() { local n=$1_$2; echo "${!n}"; }

check "$(host $IB1)" $IB1_SHA
check "$(host $IB2)" $IB2_SHA
for t in 4b 2b 08b; do check "$(host "$(tv BASE $t)")" "$(tv SHA $t)"; done
check "$TEACHER_4b" $TEACHER_4b_SHA
check "$TEACHER_2b" $TEACHER_2b_SHA
log "M12 inputs verified"

for t in 4b 2b 08b; do
  for k in train read; do
    c=$t-$k
    [ -d "$M/triton-cache/$c" ] && continue
    [ -d "$R/m11/triton-cache/$c" ] || { log "FAILED: no M11 cache $c on this node"; exit 1; }
    cp -a "$R/m11/triton-cache/$c" "$M/triton-cache/$c.tmp" && mv "$M/triton-cache/$c.tmp" "$M/triton-cache/$c"
    log "Triton cache $c copied from m11/triton-cache/$c ($(find "$M/triton-cache/$c" -type f | wc -l) files)"
  done
done

for t in 4b 2b 08b; do
  [ -f "$M/data/$t/report.json" ] && continue
  [ ! -e "$M/data/$t-build.launch.json" ] || { log "$t build failed earlier; not rerun"; exit 1; }
  arms=()
  for a in $(tv ARMS $t); do arms+=(--arm "$a"); done
  M12_NODE=$NODE bash "$OPS/m12-launch.sh" "data-$t" "$SRC" "$M/data/$t-build" --cpu -- v2/dec/ops/m12/m12_data.py \
    --tier $t --base "$(tv BASE $t)" --base-sha "$(tv SHA $t)" --ib1 $IB1 --ib1-sha $IB1_SHA --ib2 $IB2 --ib2-sha $IB2_SHA \
    --tokenizer "$(tv TOK $t)" --indist w2c,isarc,hover,gsm2 "${arms[@]}" --output "/out/$t" --workers 32 \
    || { log "$t build FAILED (see $M/data/$t-build.stderr.log)"; exit 1; }
  mv "$M/data/$t-build/$t" "$M/data/$t"
  log "TRAIN $t: $(tail -1 "$M/data/$t-build.stdout.log" | cut -c1-500)"
done

case $NODE in
  e) refs="2b-C0-e 08b-C0-e" ;;
  f) refs="4b-LH-f" ;;
esac
for p in $refs; do
  [ -d "$M/lines/$p" ] && continue
  cp -a "$R/m11/lines/$p" "$M/lines/$p.tmp" && mv "$M/lines/$p.tmp" "$M/lines/$p"
  log "reference readouts $p copied from m11/lines/$p ($(find "$M/lines/$p" -name '*.predictions.jsonl' | wc -l) panels)"
done

[ "$NODE" = e ] || { log "prep finished (node F: no overlap step)"; exit 0; }
cand=$R/m10/probes/candidates/candidates.jsonl
for name in ib1 ib2 4b-base; do
  out=$M/probes/overlap-train-$name.json
  [ -f "$out" ] && continue
  case $name in ib1) corpus=$(host $IB1) ;; ib2) corpus=$(host $IB2) ;; 4b-base) corpus=$(host $BASE_4b) ;; esac
  python3 "$CODE/v2/dec/ops/m10/m10_probes.py" overlap --candidates "$cand" --corpus "$corpus" --kind train \
    --output "$out" > "$M/probes/overlap-train-$name.log" 2>&1 || { log "overlap $name FAILED"; exit 1; }
done
for t in 4b 2b 08b; do
  [ -f "$M/probes/hits-$t.json" ] && continue
  case $t in 4b) basehits=$M/probes/overlap-train-4b-base.json ;; *) basehits=$R/m11/probes/hits-$t.json ;; esac
  python3 - "$R/panels/m10-probes.prompts.jsonl" "$M/probes/hits-$t.json" "$basehits" \
    "$M/probes/overlap-train-ib1.json" "$M/probes/overlap-train-ib2.json" << 'EOF'
import hashlib, json, sys
panel_path, out, *sources = sys.argv[1:]
panel = {json.loads(line)["id"] for line in open(panel_path)}
hits, files = set(), {}
for path in sources:
    d = json.load(open(path))
    ids = d["panel_hits"] if "panel_hits" in d else d["hit_ids"]
    hits |= set(ids) & panel
    files[path] = hashlib.sha256(open(path, "rb").read()).hexdigest()
json.dump({"sources": files, "panel_items": len(panel), "panel_hits": sorted(hits)}, open(out, "x"), indent=1)
EOF
  log "probe overlap $t: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(len(d["panel_hits"]), "of", d["panel_items"], "panel items hit")' "$M/probes/hits-$t.json")"
done
log "prep finished"
