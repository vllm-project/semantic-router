#!/usr/bin/env bash
# usage: bash dryrun_test.sh
# Local dry-run harness for the M5 wrappers (no GPU, no docker, no node): a fake /data under a
# temp dir (D2_DATA), fake mirrors that link this checkout, DRY_RUN=1 so docker / eval-runner /
# host-python module calls are printed, not run. Checks mounts, runtime-mirror defaults, the
# --teacher-partial pass-through, lease files, cache receipts, the GPU7 yield, launch/alive and
# the successor summary. Exit 0 if every check passes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEC=$(cd "$HERE/../../../.." && pwd)
T=$(mktemp -d)
if [ -n "${KEEP:-}" ]; then echo "keeping $T"; else trap 'rm -rf "$T"' EXIT; fi
export D2_DATA=$T/data DRY_RUN=1 D2_9B_GPUS="6 7" D2_YIELD_SLEEP=1 D2_LAUNCH_SETTLE=1
SHA=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
RT=3277dec9d81708fa374405ca884043443bab1b49
R=$D2_DATA/dev2/runs/9b
M5=$R/m5
fails=0
ok() { echo "ok   $*"; }
bad() { echo "FAIL $*"; fails=$((fails + 1)); }
check() { local what=$1; shift; if "$@"; then ok "$what"; else bad "$what"; fi; }
has() { grep -qF -- "$2" "$1"; }

for s in $SHA $RT; do
  m=$D2_DATA/dev2/src/$s-src_training_decision2
  mkdir -p "$m/src/training"
  ln -s "$DEC" "$m/src/training/decision2"
  echo "{\"tree\": \"tree-$s\"}" > "$m/.dev2-mirror.json"
done
L=$D2_DATA/dev2/src/$SHA-src_training_decision2/src/training/decision2/v2/9b/lux9b/m5
mkdir -p "$R/m4/triton-cache/a" "$R/formal-m3/triton-cache/b" "$R/m3/pf-D-s1-zero/run/checkpoint-0000000" \
  "$R/m4/K-a13-build/soup" "$R/m4/K-a13-cal" "$M5/data/x/build" "$D2_DATA/dev2/leases/gpu7.lock"
echo 1 > "$R/m4/triton-cache/a/f"; echo 2 > "$R/formal-m3/triton-cache/b/g"
touch "$R/m3/pf-D-s1-zero/run/checkpoint-0000000/decision_config.json" "$R/m4/K-a13-build/soup/decision_config.json"
echo '{}' > "$M5/data/x/build/manifest.json"

# arm with preflight, KL and --teacher-partial (trainer outputs faked in advance)
mkdir -p "$M5/pf-A1-check" "$M5/A1/run/checkpoint-0000010"
echo '{"status": "PASS"}' > "$M5/pf-A1-check/preflight.json"
echo '{}' > "$M5/A1/run/COMPLETE.json"; echo '{"checkpoint": "checkpoint-0000010"}' > "$M5/A1/run/BEST.json"
touch "$M5/A1/run/checkpoint-0000010/decision_config.json"
"$L/arm.sh" $SHA 6 A1 20260926 x --preflight --kl 1.0 --teacher-partial > "$T/arm.out" 2>&1
check "arm exit 0" has "$T/arm.out" "done A1 checkpoint-0000010"
check "arm passes --teacher-partial" has "$M5/A1/console.log" "--teacher-kl-weight 1.0 --teacher-partial"
check "arm trains from SHA mirror" has "$M5/A1/console.log" "$SHA-src_training_decision2/src/training/decision2,dst=/code"
check "arm keeps M4 full-FT flags" has "$M5/A1/console.log" "--train-mode full --backbone-lr 1e-5 --head-lr 1e-4"
check "arm readout from runtime mirror" has "$M5/A1-dev/console.log" "$RT-src_training_decision2/src/training/decision2,dst=/code"
check "arm model id" has "$M5/A1-css-pilot/console.log" "decision2-9b-m5-A1"
check "container name" has "$M5/A1/console.log" "--name d2-9b-m5-A1-g6"
check "mounts /m5 ro and /out" has "$M5/A1/console.log" "src=$M5,dst=/m5,readonly"
check "lease idle 9b-m5" grep -qx "track=9b-m5" "$D2_DATA/dev2/leases/gpu6.lock/owner"
check "cache receipt" python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["files"]==1 and d["tree_sha256"]==d["source_tree_sha256"]' "$M5/triton-cache.copy.json"
check "cache notes before/after" test "$(grep -c '"job": "A1"' "$M5/triton-cache.jsonl")" = 2
"$L/arm.sh" $SHA 6 A2 1 x --no-teacher --teacher-partial > "$T/arm2.out" 2>&1
check "teacher-partial needs --kl" has "$T/arm2.out" "--teacher-partial needs --kl"

# soup (incumbent + an M5 run), interpolation, readout with an explicit runtime
"$L/soup.sh" $SHA 6 S1 A1 /m4/K-a13-build/soup > "$T/soup.out" 2>&1
check "soup done" has "$T/soup.out" "done S1"
check "soup members" has "$M5/S1-build/members.txt" "/m5/A1/run/checkpoint-0000010"
check "soup runtime mirror" has "$M5/S1-build/console.log" "$RT-src_training_decision2/src/training/decision2,dst=/code"
"$L/interp.sh" $SHA 7 S1-a13 /m5/S1-build/soup 1 3 --runtime $SHA > "$T/interp.out" 2>&1
check "interp done" has "$T/interp.out" "done S1-a13"
check "interp weights 1/3" python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["distinct"][0]["weight"]=="1/3", d' "$M5/S1-a13-build/weights.json"
check "interp --runtime honoured" has "$M5/S1-a13-dev/console.log" "$SHA-src_training_decision2/src/training/decision2,dst=/code"

# score (CPU)
"$L/score.sh" $SHA sc1 inc=m4/K-a13 s=m5/S1 -- inc:s > "$T/score.out" 2>&1
check "score runtime + arms" has "$T/score.out" "--arm s=/runs/m5/S1-dev/dev.predictions.jsonl,/runs/m5/S1-css-pilot/css-pilot.predictions.jsonl"

# chain-step on GPU7 yields to an active eval lease, then runs
printf 'track=eval\nstatus=requested\n' > "$D2_DATA/dev2/leases/gpu7.lock/owner.eval"
( sleep 3; printf 'track=eval\nstatus=idle-released\n' > "$D2_DATA/dev2/leases/gpu7.lock/owner.eval" ) &
t0=$(date +%s)
bash "$L/chain-step.sh" $SHA 7 c7 step1 30 -- true > "$T/step.out" 2>&1
check "chain-step waited for eval" test $(( $(date +%s) - t0 )) -ge 3
check "chain-step yield logged" has "$M5/logs/c7.log" "gpu7 yield: owner.eval status=requested"
check "chain-step logged end" has "$M5/logs/c7.log" "end step=step1"
check "chain-step lease reserved" grep -qx "status=reserved" "$D2_DATA/dev2/leases/gpu7.lock/owner"
check "chain-step kept foreign owner" test -f "$D2_DATA/dev2/leases/gpu7.lock/owner.before-9b-m5" -o -f "$D2_DATA/dev2/leases/gpu7.lock/owner"

# formal (dry), with a faked report / paired / gates to test successor.json
F=$R/formal-m5
mkdir -p "$R/formal-m3/lux1-16k-shared" "$D2_DATA/dev2/runs/release/dev2-8b-t1-derived"
echo '{}' > "$R/formal-m3/lux1-16k-shared/SEAL.json"; echo '{}' > "$D2_DATA/dev2/runs/release/dev2-8b-t1-derived/SEAL.json"
echo '{"temperature_by_type": {}}' > "$R/m4/K-a13-cal/calibration.json"
export D2_FROZEN_CACHE_SHA=wrong
"$L/formal.sh" $RT 6 K5 "$R/m4/K-a13-build/soup" "$R/m4/K-a13-cal" "post-key same-panel" > "$T/formal0.out" 2>&1
check "formal refuses a non-frozen cache" has "$T/formal0.out" "frozen cache tree"
D2_FROZEN_CACHE_SHA=$(cd "$R/formal-m3/triton-cache" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
export D2_FROZEN_CACHE_SHA
"$L/formal.sh" $RT 6 K5 "$R/m4/K-a13-build/soup" "$R/m4/K-a13-cal" "post-key same-panel" > "$T/formal.out" 2>&1
check "formal done" has "$T/formal.out" "done K5"
check "formal runner + track" has "$T/formal.out" "run_same_panel.sh --gpu 6 --track 9b-m5 --src $RT-src_training_decision2"
check "formal smoke" has "$T/formal.out" "--max-items 8"
check "formal model id" has "$T/formal.out" "model_id=decision2-9b-m5-K5"
check "formal report params" has "$T/formal.out" "--loaded-parameters 7940895744"
check "formal paired vs T1" has "$T/formal.out" "dev2-8b-t1-derived --left-name K5 --right-name DEV2.0-9B-T1"
check "formal paired vs Nimble2" has "$T/formal.out" "eval/m2/q6-nimble2 --left-name K5 --right-name Nimble2"
check "formal mlx_paired" has "$T/formal.out" "lux9b.mlx_paired --left $F/K5-16k-mlx --right $R/formal-m4/K-a13-16k-mlx"
check "formal score_levels" has "$T/formal.out" "lux9b.score_levels --run-dir $F/K5-16k"
check "formal cache receipt" has "$F/triton-cache.copy.json" "\"source_tree_sha256\": \"$D2_FROZEN_CACHE_SHA\""
check "formal cache notes" test "$(wc -l < "$F/K5-cache.jsonl")" = 6
check "formal inputs.json" has "$F/K5.inputs.json" "\"runner_sha\": \"$RT\""
check "successor has no verdict" python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert not {"pass","verdict","PASS","decision"} & set(d), d' "$F/K5.gates/successor.json"
mkdir -p "$T/g" "$T/r"
cat > "$T/g/PAIRED-vs-DEV2.0-9B-T1.json" <<'EOF'
{"point": {"delta": {"score": 1.5, "T": 0.01, "H": 0.02}}, "ci95": {"low": -0.5, "high": 3.5},
 "axis_ci95": {"H": {"delta": {"low": -0.01, "high": 0.05}}, "T": {"delta": {"low": 0.0, "high": 0.02}}}, "models": {"right": "DEV2.0-9B-T1"}}
EOF
echo '{"types": {"choice": {"verdict": "OK"}, "score": {"verdict": "COLLAPSED: x"}}}' > "$T/g/types.json"
echo '{"overall": {"delta": -0.01, "ci95": {"low": -0.03, "high": 0.01}}}' > "$T/g/mlx.json"
echo '{"v3": {"score": 68.0}, "panels": {"public231": {"correct": 180, "items": 231, "tiers": {"easy": {"correct": 48}}}}}' > "$T/r/REPORT.json"
bash -c '. "$1/lib.sh"; shift; successor_summary "$@"' _ "$L" "$T/s.json" X "$T/r" "$T/g" \
  "$T/r/PAIRED-vs-Lux1-16K.json" "$T/g/mlx.json" "$T/g/none.json" > /dev/null
check "successor fields" python3 -c '
import json,sys; d=json.load(open(sys.argv[1]))
assert d["vs_T1"]["lb"]==-0.5 and d["vs_T1"]["ub"]==3.5 and d["vs_T1"]["H_delta_high"]==0.05
assert d["types"]=={"choice":"OK","score":"COLLAPSED: x"} and d["mlx_vs_T1"]["ci_high"]==0.01
assert d["public231"]["correct"]==180 and d["v3"]["score"]==68.0 and d["vs_Lux1"]["lb"] is None' "$T/s.json"

# ship_cal / derive_t1 (dry)
mkdir -p "$M5/S1-dev" "$M5/S1-css-pilot" "$M5/S1-cal"
echo '{"temperature_by_type": {}}' > "$M5/S1-cal/calibration.json"
cs=$(sha256sum "$M5/S1-cal/calibration.json" | cut -c1-64)
for p in dev css-pilot; do
  echo '{}' > "$M5/S1-$p/$p.predictions.jsonl"
  echo "{\"calibration\": {\"file_sha256\": \"$cs\"}}" > "$M5/S1-$p/$p.predictions.jsonl.manifest.json"
done
"$L/ship_cal.sh" $RT S1 S1 > "$T/ship.out" 2>&1
check "ship_cal call" has "$T/ship.out" "v2.release.dev_calibration --label S1"
check "ship_cal source = candidate" has "$T/ship.out" "--source-calibration $M5/S1-cal/calibration.json --candidate $M5/S1-cal/calibration.json"
mkdir -p "$F/K5-16k/output" "$F/K5-16k-mlx/output"
echo '{"panels": {"css15": {"predictions_sha256": "'"$(printf x | sha256sum | cut -c1-64)"'"}}}' > "$F/K5-16k/SEAL.json"
printf x > "$F/K5-16k/output/css15.predictions.jsonl"; : > "$F/K5-16k-mlx/output/mlx-diag.predictions.jsonl"
"$L/derive_t1.sh" $RT K5 > "$T/derive.out" 2>&1
check "derive_t1 undo x4" test "$(grep -c 'retemper_predictions.*--undo' "$T/derive.out")" = 4
check "derive_t1 adopt/seal/report" has "$T/derive.out" "--count-safetensors $R/m4/K-a13-build/soup"
check "derive_t1 four compares" test "$(cat "$F"/K5-16k-t1/compare-*.log | grep -c 'same_panel compare')" = 4
check "derive_t1 nimble2 compare" has "$F/K5-16k-t1/compare-nimble2.log" "eval/m2/q6-nimble2 --left-name 'K5 (T = 1)' --right-name nimble2"
check "derive_t1 T1 vs T1" has "$T/derive.out" "gates paired --left $F/K5-16k-t1 --right $D2_DATA/dev2/runs/release/dev2-8b-t1-derived"

# launch / alive (real processes, no docker)
unset DRY_RUN
printf 'echo first-line\nsleep 3\n' > "$T/chain.sh"
size=$(stat -c %s "$T/chain.sh"); sha=$(sha256sum "$T/chain.sh" | cut -c1-64)
bash "$L/launch.sh" demo "$T/chain.sh" "$size" 0000 > "$T/l0.out" 2>&1
check "launch refuses wrong sha" has "$T/l0.out" "refused"
pid=$(bash "$L/launch.sh" demo "$T/chain.sh" "$size" "$sha" 2> "$T/l1.err")
check "launch prints pid" test "$pid" = "$(cat "$M5/logs/demo.pid")"
check "launch session leader" test "$(ps -o sid= -p "$pid" | tr -d ' ')" = "$pid"
DRY_RUN=1 bash "$L/alive.sh" demo > "$T/a1.out" 2>&1
check "alive running" has "$T/a1.out" "chain demo: running pid $pid"
check "alive first line" has "$T/a1.out" "first: first-line"
bash "$L/launch.sh" demo "$T/chain.sh" "$size" "$sha" > "$T/l2.out" 2>&1
check "launch refuses a live chain" has "$T/l2.out" "is running as PID"
sleep 4
DRY_RUN=1 bash "$L/alive.sh" demo > "$T/a2.out" 2>&1
check "alive dead" has "$T/a2.out" "chain demo: dead"

echo "failures: $fails"
[ "$fails" = 0 ]
