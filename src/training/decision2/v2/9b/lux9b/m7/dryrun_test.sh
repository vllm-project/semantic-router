#!/usr/bin/env bash
# usage: bash dryrun_test.sh
# Local dry-run harness for the M7 wrappers (no GPU, no docker, no node): a fake /data under a temp
# dir (D2_DATA), fake mirrors that link this checkout, DRY_RUN=1 so docker / host-python module
# calls are printed, not run. Checks the continuation flags and mounts, the readout options
# (--cal-path, --no-cal, --mlxdev), a line (soup, concurrent interpolation builds, readouts,
# screens), the early rule's inputs and the chain step's lease. Exit 0 if every check passes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEC=$(cd "$HERE/../../../.." && pwd)
T=$(mktemp -d)
if [ -n "${KEEP:-}" ]; then echo "keeping $T"; else trap 'rm -rf "$T"' EXIT; fi
export D2_DATA=$T/data DRY_RUN=1 D2_9B_GPUS="6 7" D2_YIELD_SLEEP=1 D2_LAUNCH_SETTLE=1
SHA=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
RT=3277dec9d81708fa374405ca884043443bab1b49
R=$D2_DATA/dev2/runs/9b
M7=$R/m7
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
L=$D2_DATA/dev2/src/$SHA-src_training_decision2/src/training/decision2/v2/9b/lux9b/m7
mkdir -p "$R/m4/triton-cache/a" "$R/m3/pf-D-s1-zero/run/checkpoint-0000000" "$R/m4/K-s1/run/checkpoint-0001624" \
  "$R/m4/K-a13-build/soup" "$R/m6/ref-ka13-cal" "$M7/data/m7-topup/build/P" "$M7/data/mlxdev/build" \
  "$D2_DATA/dev2/leases/gpu7.lock" "$M7/logs"
echo 1 > "$R/m4/triton-cache/a/f"
touch "$R/m3/pf-D-s1-zero/run/checkpoint-0000000/decision_config.json" "$R/m4/K-a13-build/soup/decision_config.json" \
  "$R/m4/K-s1/run/checkpoint-0001624/decision_config.json" "$R/m6/ref-ka13-cal/calibration.json"
echo '{}' > "$M7/data/m7-topup/build/manifest.json"; echo x > "$M7/data/m7-topup/build/P/train.jsonl"
echo x > "$M7/data/mlxdev/build/panel.jsonl"

# continuation with preflight (trainer outputs faked in advance)
for n in P-m1 C-m1; do
  mkdir -p "$M7/pf-$n-check" "$M7/$n/run/checkpoint-0000300"
  echo '{"status": "PASS"}' > "$M7/pf-$n-check/preflight.json"
  echo '{"step": 300}' > "$M7/$n/run/COMPLETE.json"; echo '{"checkpoint": "checkpoint-0000300"}' > "$M7/$n/run/BEST.json"
  touch "$M7/$n/run/checkpoint-0000300/decision_config.json"
done
bash "$L/cont.sh" $SHA 6 P-m1 20260931 m7-topup:P /m4/K-s1/run/checkpoint-0001624 --preflight > "$T/cont.out" 2>&1
check "cont exit 0" has "$T/cont.out" "done P-m1 checkpoint-0000300"
check "cont inits from the member" has "$M7/P-m1/console.log" "--init decision2 --model-path /m4/K-s1/run/checkpoint-0001624"
check "cont data and teacher" has "$M7/P-m1/console.log" "--train /m7/data/m7-topup/build/P/train.jsonl"
check "cont partial teacher" has "$M7/P-m1/console.log" "--teacher /m7/data/m7-topup/build/P/teacher.jsonl --teacher-kl-weight 1.0 --teacher-partial"
check "cont learning rates" has "$M7/P-m1/console.log" "--backbone-lr 5e-6 --head-lr 5e-5"
check "cont single checkpoint" has "$M7/P-m1/console.log" "--checkpoint-schedule every --save-every 1000000"
check "cont preflight reference" has "$M7/pf-P-m1-check/console.log" "--source-path /m4/K-s1/run/checkpoint-0001624"
check "cont mounts /m7 /m6 /m4" has "$M7/P-m1/console.log" "src=$R/m6,dst=/m6,readonly"
check "cont container name" has "$M7/P-m1/console.log" "--name d2-9b-m7-P-m1-g6"
check "lease track" grep -qx "track=9b-m7" "$D2_DATA/dev2/leases/gpu6.lock/owner"
bash "$L/cont.sh" $SHA 6 X 1 m7-topup /m4/K-s1/run/checkpoint-0001624 > "$T/cont2.out" 2>&1
check "cont refuses BUILD without ARM" has "$T/cont2.out" "data must be BUILD:ARM"

# readouts: explicit calibration, T = 1, MLX-DEV
bash "$L/readout.sh" $SHA 7 ref-ka13 /m4/K-a13-build/soup --cal-path /m6/ref-ka13-cal/calibration.json \
  --panel pn1-dev --mlxdev > "$T/ro.out" 2>&1
check "readout done" has "$T/ro.out" "done ref-ka13"
check "readout no CAL fit with --cal-path" test ! -e "$M7/ref-ka13-cal"
check "readout calibration arg" has "$M7/ref-ka13-pn1-dev/console.log" "--calibration /m6/ref-ka13-cal/calibration.json"
check "readout mlxdev eval_rows" has "$M7/ref-ka13-mlxdev/console.log" "-m v2.dec.eval_rows --checkpoint /m4/K-a13-build/soup"
check "readout mlxdev rows" has "$M7/ref-ka13-mlxdev/console.log" "--rows /m7/data/mlxdev/build/panel.jsonl --tag mlxdev"
check "readout runtime mirror" has "$M7/ref-ka13-pn1-dev/console.log" "$RT-src_training_decision2/src/training/decision2,dst=/code"
bash "$L/readout.sh" $SHA 6 P-m1-e1 /m7/P-m1/run/checkpoint-0000300 --no-cal --panel pn1-dev > "$T/ro2.out" 2>&1
check "readout --no-cal" bash -c "! grep -q -- '--calibration' '$M7/P-m1-e1-pn1-dev/console.log'"

# a line: soup of member runs, concurrent interpolations, readouts and screens
for k in 1 2; do
  mkdir -p "$M7/Q-m$k/run/checkpoint-0000300"; touch "$M7/Q-m$k/run/checkpoint-0000300/decision_config.json"
  echo '{"checkpoint": "checkpoint-0000300"}' > "$M7/Q-m$k/run/BEST.json"
done
bash "$L/line.sh" $SHA 6 Q5 Q-m1 Q-m2 > "$T/line.out" 2>&1
check "line done" has "$T/line.out" "line Q5 done"
check "line soup members" has "$M7/Q5-soup-build/members.txt" "/m7/Q-m2/run/checkpoint-0000300"
check "line soup not read" test ! -e "$M7/Q5-soup-cal"
check "line a23 weights" python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["distinct"][0]["weight"]=="2/3", d' "$M7/Q5-a23-build/weights.json"
check "line a13 base Lux" has "$M7/Q5-a13-build/members.txt" "/m3/pf-D-s1-zero/run/checkpoint-0000000"
for p in a13 a12 a23; do
  check "line $p panels" test -f "$M7/Q5-$p-ht-dev2/console.log" -a -f "$M7/Q5-$p-pn1-dev/console.log" -a -f "$M7/Q5-$p-mlxdev/console.log" -a -f "$M7/Q5-$p-dev/console.log"
done
check "line screens" has "$T/line.out" "screens Q5-a23 done"

# early rule inputs
for n in ref-ka13-pn1-dev P-m1-e1-pn1-dev C-m1-e1-pn1-dev; do mkdir -p "$M7/$n"; echo 0 > "$M7/$n/exit-code.txt"; done
for n in ref-ka13 P-m1-e1 C-m1-e1; do echo '{}' > "$M7/$n-pn1-dev/pn1-dev.predictions.jsonl"; done
bash "$L/early.sh" $SHA > "$T/early.out" 2>&1
check "early uses the final SELECT" has "$M7/rules/early-P.console" "--p-select $M7/P-m1/run/select-step-0000300-metrics.json"

# chain step lease
bash "$L/chain-step.sh" $SHA 7 c7 step1 30 -- true > "$T/step.out" 2>&1
check "chain-step logged end" has "$M7/logs/c7.log" "end step=step1"
check "chain-step lease reserved->idle" grep -qx "track=9b-m7" "$D2_DATA/dev2/leases/gpu7.lock/owner"

echo "failures: $fails"
[ "$fails" = 0 ]
