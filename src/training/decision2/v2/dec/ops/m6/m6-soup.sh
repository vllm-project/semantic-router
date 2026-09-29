#!/usr/bin/env bash
# Decoder M6 arm soup (prereg dec-m6-prereg-2026-09-29.md): the uniform FP32 soup (v2.dec.soup, CPU) of the BEST
# checkpoints of the arm's finished seeds (status/<ARM>-s<i>.DONE; at least two, a two-seed soup is disclosed),
# to /data/dev2/runs/dec/m6/soup/<ARM>/build/<ARM>-soup, plus a per-file SHA-256 list. No readouts here (the
# line / readout tooling reads the soup). Writes soup/<ARM>/DONE or soup/<ARM>/FAILED; a started soup is not rebuilt.
# usage: m6-soup.sh <mirror-dir> <ARM> a|b
set -u
SRC=$1 G=$2 NODE=$3
M=/data/dev2/runs/dec/m6
A=$M/arms O=$M/soup/$G ST=$M/status
L=/data/dev2/src/$SRC/src/training/decision2/v2/dec/launch.sh
log() { echo "$(date -u +%FT%TZ) soup $G $*" | tee -a "$M/OPERATIONS.log"; }
case $NODE in
  a) export DEC_IMAGE=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54 DEC_DATA=/data/decision20-20260926/data/hf-private-decision20-clean-v2 ;;
  b) export DEC_IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1 DEC_DATA=/data/dev2/runs/dec/m3/data-sel700-cal698 ;;
  *) echo "unknown node $NODE" >&2; exit 2 ;;
esac
mkdir -p "$O"
mkdir "$O/started" 2>/dev/null || { log "already started; not rebuilt"; exit 0; }
members=() seeds=()
: > "$O/members.txt"
for s in 1 2 3; do
  [ -f "$ST/m6-$G-s$s.DONE" ] || [ -f "$ST/$G-s$s.DONE" ] || continue
  b=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['checkpoint'])" "$A/full/m6-$G-s$s/BEST.json") || continue
  members+=(--member "/runs/m6/arms/full/m6-$G-s$s/$b")
  seeds+=("s$s")
  echo "m6-$G-s$s $b" >> "$O/members.txt"
done
if [ ${#seeds[@]} -lt 2 ]; then
  log "fewer than two finished seeds (${seeds[*]:-none}); no soup"
  echo "fewer than two finished seeds: ${seeds[*]:-none}" > "$O/FAILED"
  exit 1
fi
if ! bash "$L" "m6-$G-soup-build" "$SRC" "$O/build" --cpu -- -m v2.dec.soup "${members[@]}" --output "/out/$G-soup"; then
  log "soup build FAILED: $(tail -c 300 "$O/build.stderr.log" | tr '\n' ' ')"
  echo "soup build failed" > "$O/FAILED"
  exit 1
fi
(cd "$O/build/$G-soup" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > "$O/$G-soup.sha256"
list=$(sha256sum "$O/$G-soup.sha256" | cut -d' ' -f1)
printf 'soup=%s\nseeds=%s\nmembers=%s\nsha256_list=%s\nsha256_list_sha256=%s\nbuilt_utc=%s\n' "$O/build/$G-soup" \
  "${seeds[*]}" "$(tr '\n' ';' < "$O/members.txt")" "$O/$G-soup.sha256" "$list" "$(date -u +%FT%TZ)" > "$O/DONE"
log "built from ${seeds[*]}: $(tail -1 "$O/build.stdout.log" | cut -c1-300); file list $list"
