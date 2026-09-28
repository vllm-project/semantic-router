#!/usr/bin/env bash
# M3b own-Lux XL targets (prereg §3): publish each finished node-B wave from node A.
#
# Usage: luxxl_publish.sh MIRROR_SHA WAVE...        e.g. luxxl_publish.sh <full-sha> 2 3 4 5
#
# WAVE is 1-5 (lux-xl-w<k>, queue m3b-luxxl-queue.sh) or c1 / c2 (control waves lux-xl-c-w<k>,
# M3b amendment 3 §4, queue luxxl_control_queue.sh from the node-B mirror of MIRROR_SHA; files
# c-w<k>.*; c1 also refreshes README.md and coverage.json) or h1 (lux-xl-h-w1, the H7 / H8 rows
# of the XL r2 recipes, prereg m3b-lux-h-prereg-2026-09-29.md, queue luxxl_gap_queue.sh; waits
# for LUX_XL_H_DONE, adds the repeat check and the Triton cache states to the provenance, and
# uploads README.md and coverage-r2.json, refused unless both r2 release recipes are covered).
#
# Runs on the local machine and reaches the nodes only through dssh ($DSSH, default
# /tmp/m3a/dssh), so no node address is printed. For each wave k, in order:
#   1. wait for LUX_XL_W<k>_DONE in node B's teach.log (every 120 s, at most 6 h);
#   2. read on node B the launcher line, collector log, output SHA-256 and rows, image id,
#      launcher / queue hashes and the Triton cache tree digest; check the prompt file SHA-256
#      equals node A's;
#   3. copy the output node B -> local -> node A (gzip; SHA-256 checked locally and on node A),
#      or reuse a node-A copy whose SHA-256 already matches;
#   4. on node A, from the exact mirror of MIRROR_SHA: v2.data.m3.luxxl provenance (run checks),
#      v2.data.m2.targets (receipt checks, per-row attestation); refuse node paths or private
#      values in the upload;
#   5. hf upload of w<k>.{targets.jsonl,attestation.jsonl,report.json} (wave 1 adds README.md and
#      coverage.json) to m3/teachers/lux1/xl/, then download that revision and compare SHA-256.
# Events (JSON lines) go to /tmp/m3b/luxxl-publish.events.jsonl and to node A's
# /data/dev2/runs/data/m3b/lux-xl/publish.events.jsonl. Any failure stops the script. A wave with
# a published receipt is skipped; a wave with any earlier upload state is never uploaded again.
set -uo pipefail
[[ $# -ge 2 ]] || { sed -n '2,28p' "$0" >&2; exit 2; }
sha="$1"; shift
[[ "$sha" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
DSSH=${DSSH:-/tmp/m3a/dssh}
REPO_DIR=${REPO_DIR:-$(git rev-parse --show-toplevel)}
GUARD="$REPO_DIR/src/training/decision2/v2/common/check_no_private.sh"
LOCAL=/tmp/m3b/luxxl
EVENTS=/tmp/m3b/luxxl-publish.events.jsonl
W=/data/dev2/runs/data/m3b/lux-xl
X=/data/dev2/runs/data/m3b/upload-xl/m3/mixtures/xl
X2=/data/dev2/runs/data/m3b/xl-r2-a4/build
M=/data/dev2/src/$sha-src_training_decision2/src/training/decision2
B=/data/dev2/runs/data/m3b-lux
BP=/data/dev2/private/data/teachers-v2/m3b
BL=/data/dev2/logs/data
TC=/data/dev2/runs/data/triton-cache-lux-nodeB
CQ=$M/v2/data/m3/luxxl_control_queue.sh
HQ=$M/v2/data/m3/luxxl_gap_queue.sh
REPO=llm-semantic-router/decision-2.0-training-data
DEST=m3/teachers/lux1/xl
mkdir -p "$LOCAL"

na() { "$DSSH" node-a "$@"; }
nb() { "$DSSH" node-b "$@"; }
event() { # name wave rc [detail]
  local line
  line=$(printf '{"event":"%s","wave":"%s","rc":%s,"utc":"%s"%s}' "$1" "$2" "$3" \
    "$(date -u +%FT%TZ)" "${4:+,\"detail\":\"$4\"}")
  echo "$line" >> "$EVENTS"
  echo "$line" | na "cat >> $W/publish.events.jsonl" || echo "node A event append failed" >&2
}
fail() { event "$1" "$2" 1 "${3:-}"; exit 1; }
hex() { [[ "$1" =~ ^[0-9a-f]{$2}$ ]]; }
copy_small() { # node-B file, node-A file: gzip through the local machine, SHA-256 checked on node A
  local want got
  want=$(nb "sha256sum < $1 | cut -d' ' -f1")
  hex "${want:-}" 64 || return 1
  got=$(nb "gzip -1 -c $1" | na "umask 077; gunzip -c > $2.part && sha256sum < $2.part | cut -d' ' -f1")
  [[ "$got" == "$want" ]] && na "mv $2.part $2"
}

[[ -f "$GUARD" ]] || fail no_private_guard all
na "cat /data/dev2/src/$sha-src_training_decision2/.dev2-mirror.json" | grep -q "\"commit\":\"$sha\"" \
  || fail mirror_missing all "$sha"
event started all 0 "$sha waves $*"

publish() {
  local k=$1 name stem marker queue_file
  if [[ "$k" == c* ]]; then
    name="lux-xl-c-w${k#c}" stem="c-w${k#c}" marker="LUX_XL_C_W${k#c}_DONE" queue_file=$CQ
  elif [[ "$k" == h* ]]; then
    name="lux-xl-h-w${k#h}" stem="h-w${k#h}" marker="LUX_XL_H_DONE" queue_file=$HQ
  else
    name="lux-xl-w$k" stem="w$k" marker="LUX_XL_W${k}_DONE" queue_file=$BL/m3b-luxxl-queue.sh
  fi
  local P="$W/publish/$stem" U="$W/upload-$stem" L="$LOCAL/$stem"
  local i state out_sha out_rows image launcher queue teach_mirror tc_files tc_sha pa pb have got rev
  mkdir -p "$L"
  state=$(na "if [ -f $P/published.json ]; then echo published; elif [ -e $U ] || [ -e $P/upload.log ]; then echo partial; else echo fresh; fi") \
    || fail node_a_unreachable "$k"
  case "$state" in
    published) event already_published "$k" 0; return 0 ;;
    fresh) ;;
    *) fail earlier_upload_state_needs_review "$k" ;;
  esac
  event waiting "$k" 0
  for ((i = 0; i < 180; i++)); do
    nb "grep -qx $marker $BL/teach.log" && break
    nb "grep -q '^${marker%_DONE}_STOPPED' $BL/teach.log" && fail queue_stopped "$k"
    sleep 120
  done
  ((i < 180)) || fail timeout_waiting "$k"
  event teacher_done "$k" 0

  nb "grep -F '/$name.prompts.jsonl' $BL/teach.log | tail -1" > "$L/teach.json" || fail teach_line "$k"
  nb "cat $B/$name.jsonl.log" > "$L/collector.log" || fail collector_log "$k"
  read -r out_sha out_rows < <(nb "sha256sum < $B/$name.jsonl | cut -d' ' -f1; wc -l < $B/$name.jsonl" | tr '\n' ' ')
  image=$(nb "docker image inspect --format '{{.Id}}' decision20-lux-runtime:latest")
  read -r launcher queue < <(nb "sha256sum $BL/teach.sh $queue_file | cut -d' ' -f1" | tr '\n' ' ')
  teach_mirror=$(nb "grep -o 'src/[0-9a-f]\{40\}-src_training_decision2' $BL/teach.sh | head -1 | cut -c5-44")
  read -r tc_files tc_sha < <(nb "cd $TC && find . -type f | wc -l && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1" | tr '\n' ' ')
  pb=$(nb "sha256sum < $BP/$name.prompts.jsonl | cut -d' ' -f1")
  pa=$(na "sha256sum < $W/$name.prompts.jsonl | cut -d' ' -f1")
  if ! { hex "${out_sha:-}" 64 && hex "${launcher:-}" 64 && hex "${queue:-}" 64 && hex "${tc_sha:-}" 64 \
    && hex "${teach_mirror:-}" 40 && hex "${pa:-}" 64 && [[ "${image:-}" =~ ^sha256:[0-9a-f]{64}$ ]] \
    && [[ "${tc_files:-}" =~ ^[0-9]+$ ]]; }; then
    fail node_b_facts "$k"
  fi
  [[ "$pa" == "$pb" ]] || fail prompts_differ_between_nodes "$k"
  event node_b_checked "$k" 0 "output $out_sha rows $out_rows"

  have=$(na "for f in $W/teacher/$name.jsonl $W/out/$name.jsonl; do [ -f \$f ] && echo \$f \$(sha256sum < \$f | cut -d' ' -f1); done; true")
  if grep -qx "$W/teacher/$name.jsonl $out_sha" <<< "$have"; then
    event copy_present "$k" 0
  elif grep -qx "$W/out/$name.jsonl $out_sha" <<< "$have"; then
    got=$(na "umask 077; mkdir -p $W/teacher && cp $W/out/$name.jsonl $W/teacher/$name.jsonl.part && sha256sum < $W/teacher/$name.jsonl.part | cut -d' ' -f1")
    [[ "$got" == "$out_sha" ]] || fail node_a_copy_mismatch "$k"
    na "mv $W/teacher/$name.jsonl.part $W/teacher/$name.jsonl" || fail node_a_copy "$k"
    event copy_reused "$k" 0 "$out_sha"
  else
    got=""
    for ((i = 1; i <= 3; i++)); do
      if nb "gzip -1 -c $B/$name.jsonl" > "$L/$name.jsonl.gz" \
        && [[ "$(gunzip -c "$L/$name.jsonl.gz" | sha256sum | cut -d' ' -f1)" == "$out_sha" ]]; then
        got=$(na "umask 077; mkdir -p $W/teacher && gunzip -c > $W/teacher/$name.jsonl.part && sha256sum < $W/teacher/$name.jsonl.part | cut -d' ' -f1" < "$L/$name.jsonl.gz")
        [[ "$got" == "$out_sha" ]] && break
      fi
      event copy_attempt_failed "$k" 1 "attempt $i"
      got=""
      sleep 60
    done
    [[ -n "$got" ]] || fail copy_failed "$k"
    na "mv $W/teacher/$name.jsonl.part $W/teacher/$name.jsonl" || fail node_a_copy "$k"
    rm -f "$L/$name.jsonl.gz"
    event copied "$k" 0 "$out_sha"
  fi

  na "umask 077; mkdir -p $P && cat > $P/teach.json" < "$L/teach.json" || fail stage_logs "$k"
  na "cat > $P/collector.log" < "$L/collector.log" || fail stage_logs "$k"
  local extra="" r="$name-r256"
  if [[ "$k" == h* ]]; then
    copy_small "$B/$r.jsonl" "$W/teacher/$r.jsonl" || fail repeat_copy "$k"
    copy_small "$B/lux-xl-h.triton-cache.jsonl" "$P/triton-cache.jsonl" || fail cache_states_copy "$k"
    nb "grep -F '/$r.prompts.jsonl' $BL/teach.log | tail -1" | na "cat > $P/repeat.teach.json" \
      || fail repeat_teach_line "$k"
    nb "cat $B/$r.jsonl.log" | na "cat > $P/repeat.collector.log" || fail repeat_collector_log "$k"
    pb=$(nb "sha256sum < $BP/$r.prompts.jsonl | cut -d' ' -f1")
    pa=$(na "sha256sum < $W/$r.prompts.jsonl | cut -d' ' -f1")
    { hex "${pa:-}" 64 && [[ "$pa" == "$pb" ]]; } || fail repeat_prompts_differ_between_nodes "$k"
    na "cd $M && python3 -m v2.data.m3.luxxl repeat --wave $name --wave-prompts $W/$name.prompts.jsonl \
      --wave-output $W/teacher/$name.jsonl --prompts $W/$r.prompts.jsonl --guard $W/$r.guard.json \
      --output $W/teacher/$r.jsonl --collector-log $P/repeat.collector.log --teach-line $P/repeat.teach.json \
      --out $P/repeat.json > $P/repeat.stdout 2>&1" || fail repeat_check_failed "$k"
    event repeat_checked "$k" 0 "$(na "cat $P/repeat.stdout" | tr -d '"{}' | tr ',' ';')"
    extra="--repeat-check $P/repeat.json --triton-cache-checks $P/triton-cache.jsonl"
  fi
  na "cd $M && python3 -m v2.data.m3.luxxl provenance --wave $name --prompts $W/$name.prompts.jsonl \
    --guard $W/$name.guard.json --collector-log $P/collector.log --teach-line $P/teach.json \
    --image-id $image --launcher-sha256 $launcher --queue-sha256 $queue --teach-mirror $teach_mirror \
    --triton-cache-files $tc_files --triton-cache-sha256 $tc_sha $extra --out $P/provenance.json > $P/provenance.stdout 2>&1" \
    || fail run_check_failed "$k"
  event run_checked "$k" 0
  na "umask 077; mkdir $U && cd $M && python3 -m v2.data.m2.targets --wave $name --rows $W/$name.rows.jsonl \
    --prompts $W/$name.prompts.jsonl --teacher-output $W/teacher/$name.jsonl --teacher lux \
    --model-id llm-semantic-router/Decision-1.0-Lux-9B --revision bd45a30aee8c84032791c245c70f86dee5389cc8 \
    --provenance $P/provenance.json --out $U/$stem.targets.jsonl --attestation $U/$stem.attestation.jsonl \
    --report $U/$stem.report.json > $P/convert.stdout 2>&1" || fail convert_failed "$k"
  if [[ "$k" == 1 || "$k" == c1 ]]; then
    local waves j
    waves=$(for j in 1 2 3 4 5; do printf -- '--wave lux-xl-w%s=%s ' "$j" "$W/lux-xl-w$j.prompts.jsonl"; done)
    [[ "$k" == c1 ]] && waves+=$(for j in 1 2; do printf -- ' --wave lux-xl-c-w%s=%s' "$j" "$W/lux-xl-c-w$j.prompts.jsonl"; done)
    na "cp $M/v2/data/records/hf-lux1-xl-readme.md $U/README.md && cd $M && python3 -m v2.data.m3.luxxl coverage \
      --manifest $X/mx-xl.manifest.json --missing-dir $X $waves --out $U/coverage.json > /dev/null" \
      || fail readme_coverage_files "$k"
  fi
  if [[ "$k" == h1 ]]; then
    na "cp $M/v2/data/records/hf-lux1-xl-readme.md $U/README.md && cd $M && python3 -m v2.data.m3.luxxl coverage \
      --manifest $X2/mx-xl-r2.manifest.json --missing-dir $X2 --wave lux-xl-c-w2=$W/lux-xl-c-w2.prompts.jsonl \
      --wave $name=$W/$name.prompts.jsonl --require-full mx-xl-full-r2 --require-full mx-xl-short-r2 \
      --out $U/coverage-r2.json > /dev/null" || fail readme_coverage_files "$k"
  fi
  event converted "$k" 0 "$(na "cat $P/convert.stdout" | tr -d '"{}' | tr ',' ';')"
  na "! grep -rlq -e /data/ -e /root/ $U" || fail path_leak "$k"
  na "cat $U/$stem.report.json" > "$L/$stem.report.json" || fail fetch_report "$k"
  bash "$GUARD" -- "$L/$stem.report.json" > "$L/private-check.txt" 2>&1 || fail private_value_in_report "$k"

  event upload_started "$k" 0
  local basis="M3b prereg 3 / amendment 3"
  [[ "$k" == h* ]] && basis="H7 / H8 rows of the XL r2 recipes, prereg m3b-lux-h"
  na "export HF_HUB_CACHE=/data/dev2/hf-cache; hf upload $REPO $U $DEST --repo-type dataset \
    --commit-message 'Add own-Lux XL targets wave $stem (node B GPU7, $basis)' > $P/upload.log 2>&1" \
    || fail upload_failed "$k"
  rev=$(na "grep -o 'commit/[0-9a-f]\{40\}' $P/upload.log | tail -1 | cut -d/ -f2")
  hex "${rev:-}" 40 || fail upload_revision_unknown "$k"
  event uploaded "$k" 0 "$rev"

  na "bash -s -- $REPO $U $DEST $rev $P $stem" <<'EOF' || fail readback_mismatch "$k" "$rev"
set -euo pipefail
repo=$1 up=$2 dest=$3 rev=$4 work=$5 wave=$6
export HF_HUB_CACHE=/data/dev2/hf-cache
tmp=$(mktemp -d -p "$work")
files=()
for f in "$up"/*; do files+=("$dest/$(basename "$f")"); done
hf download "$repo" "${files[@]}" --repo-type dataset --revision "$rev" --local-dir "$tmp" > "$work/readback.log" 2>&1
bad=0
: > "$work/readback.txt"
for f in "$up"/*; do
  a=$(sha256sum < "$f" | cut -d' ' -f1)
  b=$(sha256sum < "$tmp/$dest/$(basename "$f")" | cut -d' ' -f1)
  echo "$(basename "$f") $a $b" >> "$work/readback.txt"
  [ "$a" = "$b" ] || bad=1
done
rm -rf "$tmp"
[ "$bad" = 0 ] && printf '{"wave":"%s","revision":"%s"}\n' "$wave" "$rev" > "$work/published.json"
EOF
  na "cat $P/readback.txt" > "$L/readback.txt"
  event published "$k" 0 "$rev"
}

for k in "$@"; do
  [[ "$k" =~ ^([1-5]|c[12]|h1)$ ]] || fail bad_wave "$k"
  publish "$k"
done
event all_done all 0
