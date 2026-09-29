#!/usr/bin/env bash
# M7(b) post-training on node A (prereg sections 2.3 and 2.4): soups, readouts, uncorrected CHK
# devchecks, finalist selection and the matched M7-vs-M6 contrasts. Never a formal run.
# Usage (node A, detached): m7_post.sh <mirror-dir-name>
#   nohup setsid bash .../m7_post.sh <sha>-src_training_decision2 < /dev/null >> /data/dev2/logs/06b/m7-post.log 2>&1 &
# Two lanes run in parallel, one job per GPU at a time:
#   GPU0: m7-mx seeds, GPU1: m7-cx seeds. Each lane polls every 60 s (gives up after 6 h, TIMEOUT
#   marker, nothing further on that lane). A seed whose queue status starts with COMPLETE (and
#   whose full/COMPLETE.json says COMPLETE) gets its CHK probabilities + devcheck at once, as a
#   recorded co-tenant (owner.m7b-post) while the queue still runs. The family is resolved when
#   the GPU's m7_queue.sh process has exited; non-COMPLETE seeds are recorded. With >= 2
#   COMPLETE seeds the family soup (m6_soupspec.py + m6_soup.sh: export, reload parity, SELECT/CAL,
#   typed DEV + CSS pilot readout) and its CHK run on the family's GPU (queue ended, so `owner` is
#   ours); fewer seeds drop the family. Then m7-mxcx-soup (all COMPLETE seeds of both families)
#   runs on whichever lane gets there first once both families are resolved; it needs both.
# Afterwards: m7_select.py, contrast.py (CPU, image, 10,000 draws), m7_contrast chk/summary,
# SUMMARY.json and DONE under /data/dev2/runs/06b/m7/post/. Every step is write-once: finished
# outputs are skipped on a rerun, and a step that started without finishing is a recorded
# failure and is not retried.
set -uo pipefail
sha="$1"
R=/data/dev2/runs/06b
A=$R/m1/arms
Q=$R/m7/queue
P=$R/m7/post
B=$R/m7/baselines
T=$R/m7/timing
S=/data/dev2/src/$sha/src/training/decision2
D=$S/v2/06b
logs=/data/dev2/logs/06b
POLL=${M7_POST_POLL:-60}
LIMIT=${M7_POST_LIMIT:-21600}
[ -d "$D" ] || { echo "missing mirror $S" >&2; exit 2; }
mkdir -p "$P/specs" "$P/chk" "$T" "$logs"
[ -e "$P/DONE" ] && { echo "$P/DONE exists"; exit 0; }
t0=$(date +%s)

utc() { date -u +%FT%TZ; }
say() { echo "$(utc) m7-post $*"; }
host_py() { PYTHONPATH="$S" PYTHONDONTWRITEBYTECODE=1 python3 "$@"; }
json_field() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$1" "$2" 2> /dev/null || echo MISSING; }
queue_running() { pgrep -f "v2/06b/m7_queue.sh $1 " > /dev/null; }
seed_status() { [ -f "$Q/$1.status" ] && cat "$Q/$1.status" || echo MISSING; }
seed_complete() {
  case "$(seed_status "$1")" in COMPLETE*) ;; *) return 1 ;; esac
  [ "$(json_field "$A/$1/full/COMPLETE.json" status)" = COMPLETE ]
}
expired() { [ $(($(date +%s) - t0)) -ge "$LIMIT" ]; }
set_idle() {
  grep -qs '^track=06b-encoder$' "/data/dev2/leases/gpu$1.lock/owner" || return 0
  printf 'track=06b-encoder\nstatus=idle (%s; no job running)\npurpose=allocation retained by the 0.6B track\nlast_job_end_utc=%s\nupdated_utc=%s\n' \
    "$2" "$(utc)" "$(utc)" > "/data/dev2/leases/gpu$1.lock/owner"
}

chk() { # gpu arm mode
  local out=$P/chk/$2
  [ -f "$out/devcheck.json" ] && return 0
  [ -e "$out/FAILED" ] && return 1
  if bash "$D/m7_chk.sh" "$1" "$sha" "$2" "$out" "$3"; then
    say "$2 CHK + devcheck done ($3)"
  else
    say "$2 CHK failed (recorded, not retried)"
    mkdir -p "$out" && utc > "$out/FAILED"
    return 1
  fi
}

copy_timing() { # run-name prefix
  local f
  for f in "$logs/$1".timing.json "$logs/$1"-*.timing.json; do
    [ -f "$f" ] && cp -n "$f" "$T/"
  done
  return 0
}

soup() { # gpu soup-name seed...
  local gpu=$1 name=$2 spec=$P/specs/$2.json
  shift 2
  if [ -f "$A/$name/readout/READOUT.json" ]; then
    say "$name readout exists (skipped)"
  elif [ -e "$P/$name.soup.started" ]; then
    say "$name started earlier without a readout: recorded failure, not retried"
    return 1
  else
    if [ ! -f "$spec" ]; then
      python3 "$D/m6_soupspec.py" "$spec" "$name" "$@" > "$P/$name.spec.log" 2>&1 || {
        say "$name spec refused: $(tail -1 "$P/$name.spec.log")"
        utc > "$P/$name.soup.started"
        echo spec-refused > "$P/$name.soup.exit"
        return 1
      }
    fi
    utc > "$P/$name.soup.started"
    say "$name soup + readout on gpu$gpu over $*"
    DEV2_SOUP_PURPOSE="0.6B M7(b) soup $name and readout" bash "$D/m6_soup.sh" "$gpu" "$sha" "$name" "$spec" \
      > "$logs/$name.driver.log" 2>&1
    local status=$?
    echo "$status" > "$P/$name.soup.exit"
    utc > "$P/$name.soup.ended"
    copy_timing "$name"
    say "$name soup exit $status"
    [ "$status" = 0 ] || return 1
  fi
  [ "$(json_field "$A/$name/full/COMPLETE.json" status)" = COMPLETE ] || return 1
  chk "$gpu" "$name" owner
}

lane() { # gpu family
  local gpu=$1 fam=$2 seed k done_seeds=()
  say "lane gpu$gpu $fam: waiting for the queue (poll ${POLL}s, limit ${LIMIT}s)"
  while :; do
    local running=0
    queue_running "$gpu" && running=1
    for k in 1 2 3; do
      seed=$fam-s$k
      if seed_complete "$seed"; then
        chk "$gpu" "$seed" "$([ "$running" = 1 ] && echo cotenant || echo owner)"
      fi
    done
    [ "$running" = 0 ] && break
    if expired; then
      say "lane gpu$gpu $fam: TIMEOUT (queue still running)"
      utc > "$P/$fam.TIMEOUT"
      return 1
    fi
    sleep "$POLL"
  done
  for k in 1 2 3; do
    seed=$fam-s$k
    if seed_complete "$seed"; then
      done_seeds+=("$seed")
    else
      say "$seed not COMPLETE: $(seed_status "$seed")"
    fi
  done
  if [ ! -f "$P/$fam.seeds.json" ]; then
    python3 - "$P/$fam.seeds.json" "$Q" "$fam" "${done_seeds[@]}" << 'EOF'
import json, sys
from pathlib import Path
out, q, fam, *done = sys.argv[1:]
seeds = {f"{fam}-s{k}": (Path(q) / f"{fam}-s{k}.status").read_text().strip()
         if (Path(q) / f"{fam}-s{k}.status").is_file() else "MISSING" for k in (1, 2, 3)}
with open(out, "x") as f:
    json.dump({"family": fam, "queue_status": seeds, "complete": done,
               "dropped": len(done) < 2}, f, indent=2)
    f.write("\n")
EOF
  fi
  utc > "$P/$fam.resolved"
  if [ "${#done_seeds[@]}" -ge 2 ]; then
    soup "$gpu" "$fam-soup" "${done_seeds[@]}"
  else
    say "$fam dropped: ${#done_seeds[@]} COMPLETE seed(s)"
  fi
  set_idle "$gpu" "0.6B M7(b) post-training: $fam done"
  local other=m7-cx
  [ "$fam" = m7-cx ] && other=m7-mx
  while [ ! -e "$P/$other.resolved" ]; do
    if [ -e "$P/$other.TIMEOUT" ] || expired; then
      say "lane gpu$gpu: $other never resolved; m7-mxcx-soup not built here"
      return 0
    fi
    sleep "$POLL"
  done
  if mkdir "$P/m7-mxcx-soup.claim" 2> /dev/null; then
    echo "gpu$gpu $(utc)" > "$P/m7-mxcx-soup.claim/by"
    local mx cx
    mx=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(' '.join(d['complete']) if not d['dropped'] else '')" "$P/m7-mx.seeds.json")
    cx=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(' '.join(d['complete']) if not d['dropped'] else '')" "$P/m7-cx.seeds.json")
    if [ -n "$mx" ] && [ -n "$cx" ]; then
      # shellcheck disable=SC2086 # seed lists are space-separated arm names
      soup "$gpu" m7-mxcx-soup $mx $cx
    else
      say "m7-mxcx-soup skipped: a family was dropped"
      echo "skipped: a family was dropped" > "$P/m7-mxcx-soup.skipped"
    fi
    set_idle "$gpu" "0.6B M7(b) post-training: m7-mxcx-soup done"
  fi
  return 0
}

lane 0 m7-mx > "$logs/m7-post-gpu0.log" 2>&1 &
p0=$!
lane 1 m7-cx > "$logs/m7-post-gpu1.log" 2>&1 &
p1=$!
wait "$p0"
wait "$p1"
say "lanes finished"

# ---- selection (prereg 2.3 family artifacts, 2.4 eligibility and order)
families=()
for fam in m7-mx m7-cx; do
  [ -f "$P/$fam.seeds.json" ] || continue
  seeds=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(' '.join(d['complete']) if not d['dropped'] else '')" "$P/$fam.seeds.json")
  # shellcheck disable=SC2206 # space-separated arm names
  [ -n "$seeds" ] && families+=(--family "$fam-soup" $seeds)
done
cross=()
[ -f "$A/m7-mxcx-soup/readout/READOUT.json" ] && cross=(--cross m7-mxcx-soup)
if [ ! -f "$P/select.json" ] && [ "${#families[@]}" -gt 0 ]; then
  host_py -m v2.06b.m7_select "${families[@]}" "${cross[@]}" --devchecks "$P/chk" --output "$P/select.json" \
    > "$P/select.txt" 2>&1 || say "m7_select failed: $(tail -1 "$P/select.txt")"
fi

# ---- matched contrasts m7-x - m6-x (development evidence, not gated)
pairs=()
chk_pairs=()
m6_probs() { [ "$1" = m6-mxcx-soup ] && echo "$R/m7/a/probs/PROBS.json" || echo "$B/$1/PROBS.json"; }
for x in mx-s1 mx-s2 mx-s3 cx-s1 cx-s2 cx-s3 mx-soup cx-soup mxcx-soup; do
  if [ -f "$A/m7-$x/readout/READOUT.json" ] && [ -f "$A/m6-$x/readout/READOUT.json" ]; then
    pairs+=("m7-m6:$x=m7-$x=m6-$x")
  fi
  if [ -f "$P/chk/m7-$x/PROBS.json" ] && [ -f "$(m6_probs "m6-$x")" ]; then
    chk_pairs+=(--pair "m7-m6:$x=$P/chk/m7-$x/PROBS.json=$(m6_probs "m6-$x")")
  fi
done
if [ ! -f "$P/contrast-m7-m6.json" ] && [ "${#pairs[@]}" -gt 0 ]; then
  args=()
  for p in "${pairs[@]}"; do
    IFS='=' read -r label t c <<< "$p"
    args+=(--pair "$label=/runs/m1/arms/$t/readout=/runs/m1/arms/$c/readout")
  done
  start=$(utc)
  docker run --rm --network none --name dev2-06b-m7b-contrast \
    -v /data/decision20-20260926:/work:ro -v "/data/dev2/src/$sha":/src:ro -v "$R":/runs \
    -e PYTHONPATH=/src/src/training/decision2:/opt/decision-fla -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 \
    -w /src/src/training/decision2 decision20-train-fast:host2 \
    python3 -m v2.06b.contrast --typed-gold /work/runs/dev.gold.jsonl \
    --css-gold /work/runs/css-transfer-v1/css-pilot.gold.jsonl "${args[@]}" \
    --output /runs/m7/post/contrast-m7-m6.json > "$P/contrast.txt" 2>&1
  status=$?
  printf '{"step":"m7b matched contrasts (contrast.py, CPU only, no GPU mapped)","start_utc":"%s","end_utc":"%s","exit":%s,"gpu_hours":0}\n' \
    "$start" "$(utc)" "$status" > "$T/m7b-contrast.cpu.timing.json"
  say "contrast.py exit $status"
fi
if [ ! -f "$P/chk-delta-m7-m6.json" ] && [ "${#chk_pairs[@]}" -gt 0 ]; then
  host_py -m v2.06b.m7_contrast chk --aho-dir "$R/m7/a/aho" "${chk_pairs[@]}" --output "$P/chk-delta-m7-m6.json" \
    > "$P/chk-delta.txt" 2>&1 || say "m7_contrast chk failed: $(tail -1 "$P/chk-delta.txt")"
fi
if [ ! -f "$P/contrast-summary.json" ] && [ "${#pairs[@]}" -gt 0 ]; then
  sargs=()
  for p in "${pairs[@]}"; do sargs+=(--pair "$p"); done
  [ -f "$P/contrast-m7-m6.json" ] && sargs+=(--contrast "$P/contrast-m7-m6.json")
  [ -f "$P/chk-delta-m7-m6.json" ] && sargs+=(--chk "$P/chk-delta-m7-m6.json")
  host_py -m v2.06b.m7_contrast summary "${sargs[@]}" --output "$P/contrast-summary.json" \
    > "$P/contrast-summary.txt" 2>&1 || say "m7_contrast summary failed: $(tail -1 "$P/contrast-summary.txt")"
fi

# ---- summary + DONE
python3 - "$P" "$A" "$T" "$sha" << 'EOF'
import json, sys
from pathlib import Path
post, arms, timing = (Path(a) for a in sys.argv[1:4])
sha = sys.argv[4]
if (post / "SUMMARY.json").is_file():
    sys.exit(0)
def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None
def text(p):
    return Path(p).read_text().strip() if Path(p).is_file() else None
out = {"schema": "dev2-06b-m7-post-summary/1", "mirror": sha,
       "label": "development readouts (never release scores); no formal run launched",
       "families": {}, "soups": {}, "chk": {}, "timeouts": [], "files": {}}
for fam in ("m7-mx", "m7-cx"):
    out["families"][fam] = load(post / f"{fam}.seeds.json")
    if (post / f"{fam}.TIMEOUT").is_file():
        out["timeouts"].append(fam)
for soup in ("m7-mx-soup", "m7-cx-soup", "m7-mxcx-soup"):
    out["soups"][soup] = {
        "started": text(post / f"{soup}.soup.started"),
        "exit": text(post / f"{soup}.soup.exit"),
        "skipped": text(post / f"{soup}.skipped"),
        "complete": (load(arms / soup / "full" / "COMPLETE.json") or {}).get("status"),
        "readout": (arms / soup / "readout" / "READOUT.json").is_file(),
    }
for d in sorted((post / "chk").glob("*")):
    check = load(d / "devcheck.json")
    probs = load(d / "PROBS.json") or {}
    out["chk"][d.name] = {
        "failed": (d / "FAILED").is_file(),
        "B-D1": check and check["B-D1"],
        "cal_parity": probs.get("cal_parity", {}).get("pass") if probs.get("cal_parity") else None,
        "state_sha256": probs.get("state_sha256"),
    }
select = load(post / "select.json")
out["finalists"] = select and select["finalists"]
out["verdict"] = select and select["verdict"]
for name in ("select.json", "contrast-m7-m6.json", "chk-delta-m7-m6.json", "contrast-summary.json"):
    out["files"][name] = (post / name).is_file()
out["timing_files"] = sorted(p.name for p in timing.glob("*.json"))
with open(post / "SUMMARY.json", "x") as f:
    json.dump(out, f, indent=2, sort_keys=True)
    f.write("\n")
EOF
utc > "$P/DONE"
say "DONE ($P/DONE)"
