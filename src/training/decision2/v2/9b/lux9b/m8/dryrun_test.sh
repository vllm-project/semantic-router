#!/usr/bin/env bash
# usage: bash dryrun_test.sh
# Local dry-run harness for the M8 wrappers (no GPU, no docker, no node): a fake /data under a temp
# dir (D2_DATA), fake mirrors that link this checkout, DRY_RUN=1 so docker / host-python module
# calls are printed, not run. Checks the continuation flags, trainer mirror, mounts and recipe
# check, the teacher collection (27B image, scored-runtime arguments, private cache copy), readouts
# of M7 checkpoints into m8/, a line, the early rule's inputs, the shared lease entry (the ~27B
# owner entry is never rewritten) and the refusal of a GPU whose owner entry is busy.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEC=$(cd "$HERE/../../../.." && pwd)
T=$(mktemp -d)
if [ -n "${KEEP:-}" ]; then echo "keeping $T"; else trap 'rm -rf "$T"' EXIT; fi
export D2_DATA=$T/data DRY_RUN=1 D2_YIELD_SLEEP=1 D2_LAUNCH_SETTLE=1
SHA=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
RT=3277dec9d81708fa374405ca884043443bab1b49
TC=7168f08643eb9297c7f457799dc06223056be656
TT=ff660322a76b04c4774fdb6aaa31e5af2a45743e
R=$D2_DATA/dev2/runs/9b
M7=$R/m7
M8=$R/m8
LK=$D2_DATA/dev2/leases
fails=0
ok() { echo "ok   $*"; }
bad() { echo "FAIL $*"; fails=$((fails + 1)); }
check() { local what=$1; shift; if "$@"; then ok "$what"; else bad "$what"; fi; }
has() { grep -qF -- "$2" "$1"; }

for s in $SHA $RT $TC $TT; do
  m=$D2_DATA/dev2/src/$s-src_training_decision2
  mkdir -p "$m/src/training"
  ln -s "$DEC" "$m/src/training/decision2"
  echo "{\"tree\": \"tree-$s\"}" > "$m/.dev2-mirror.json"
done
L=$D2_DATA/dev2/src/$SHA-src_training_decision2/src/training/decision2/v2/9b/lux9b/m8
A=$D2_DATA/dev2/runs/27b/M4-A20r-soup
mkdir -p "$R/m4/triton-cache/a" "$R/m3/pf-D-s1-zero/run/checkpoint-0000000" "$R/m4/K-s1/run/checkpoint-0001624" \
  "$R/m4/K-a13-build/soup" "$R/m6/ref-ka13-cal" "$M8/data/m8-kd/build/D1" "$M8/data/m8-prompts/build" \
  "$M7/data/mlxdev/build" "$M7/C-m1/run/checkpoint-0000166" "$M7/pf-C-m1-zero/run" "$M7/screens/C-m1-e1" \
  "$LK/gpu2.lock" "$LK/gpu3.lock" "$A/soup/checkpoint" "$A/package" "$A/formal/triton-cache/x" "$A/formal/output" \
  "$D2_DATA/decision20-20260926/models/Qwen3.8-27B" "$D2_DATA/dev2/private/panels/goldfree"
echo 1 > "$R/m4/triton-cache/a/f"; echo 1 > "$A/formal/triton-cache/x/f"
touch "$R/m3/pf-D-s1-zero/run/checkpoint-0000000/decision_config.json" "$R/m4/K-a13-build/soup/decision_config.json" \
  "$R/m4/K-s1/run/checkpoint-0001624/decision_config.json" "$R/m6/ref-ka13-cal/calibration.json" \
  "$M7/C-m1/run/checkpoint-0000166/decision_config.json" "$A/soup/checkpoint/decision_config.json" \
  "$A/package/calibration.json" "$D2_DATA/dev2/private/panels/goldfree/typed-final.prompts.jsonl"
echo '{"checkpoint": "checkpoint-0000166"}' > "$M7/C-m1/run/BEST.json"
echo '{"arms": {"D1": {"teacher_rows": 12443}}}' > "$M8/data/m8-kd/build/manifest.json"
echo x > "$M8/data/m8-kd/build/D1/train.jsonl"; echo x > "$M8/data/m8-kd/build/D1/teacher.jsonl"
echo x > "$M8/data/m8-prompts/build/prompts-0.jsonl"; echo x > "$M7/data/mlxdev/build/panel.jsonl"
printf 'track=27b\nstatus=idle\npurpose=m5 done\n' > "$LK/gpu2.lock/owner"
cp "$LK/gpu2.lock/owner" "$T/owner.before"

# continuation with preflight (trainer outputs faked in advance)
n=D1-m1
mkdir -p "$M8/pf-$n-check" "$M8/$n/run/checkpoint-0000166"
echo '{"step": 166}' > "$M8/$n/run/COMPLETE.json"; echo '{"checkpoint": "checkpoint-0000166"}' > "$M8/$n/run/BEST.json"
touch "$M8/$n/run/checkpoint-0000166/decision_config.json"
bash "$L/cont.sh" $SHA 2 D1-m1 20260931 D1 /m4/K-s1/run/checkpoint-0001624 C-m1 --preflight > "$T/cont.out" 2>&1
check "cont exit 0" has "$T/cont.out" "done D1-m1 checkpoint-0000166"
check "cont inits from the member" has "$M8/D1-m1/console.log" "--init decision2 --model-path /m4/K-s1/run/checkpoint-0001624"
check "cont trains on the arm's copy of C" has "$M8/D1-m1/console.log" "--train /m8/data/m8-kd/build/D1/train.jsonl"
check "cont teacher lambda 1 partial" has "$M8/D1-m1/console.log" "--teacher /m8/data/m8-kd/build/D1/teacher.jsonl --teacher-kl-weight 1.0 --teacher-partial"
check "cont learning rates" has "$M8/D1-m1/console.log" "--backbone-lr 5e-6 --head-lr 5e-5"
check "cont single checkpoint" has "$M8/D1-m1/console.log" "--checkpoint-schedule every --save-every 1000000"
check "cont trainer = M7 mirror" has "$M8/D1-m1/console.log" "$TC-src_training_decision2/src/training/decision2,dst=/code"
check "cont mounts /m8 /m7" has "$M8/D1-m1/console.log" "src=$M7,dst=/m7,readonly"
check "cont container name" has "$M8/D1-m1/console.log" "--name d2-9b-m8-D1-m1-g2"
check "cont recipe vs control member" has "$M8/D1-m1/recipe.console" "lux9b.m8_rules recipe --control $M7/C-m1/run/provenance.json --arm $M8/D1-m1/run/provenance.json --teacher-rows 12443"
check "cont preflight recipe vs M7 zero-step" has "$M8/pf-D1-m1-zero/recipe.console" "--control $M7/pf-C-m1-zero/run/provenance.json"
check "lease entry owner.9b-m8" grep -qx "track=9b-m8" "$LK/gpu2.lock/owner.9b-m8"
check "27B owner entry untouched" cmp -s "$LK/gpu2.lock/owner" "$T/owner.before"
bash "$L/cont.sh" $SHA 5 X 1 D1 /m4/K-s1/run/checkpoint-0001624 C-m1 > "$T/cont2.out" 2>&1
check "cont refuses a GPU outside M8" has "$T/cont2.out" "not allocated to 9B M8"

# a busy owner entry refuses the job
printf 'track=27b\nstatus=running\n' > "$LK/gpu3.lock/owner"
bash "$L/cont.sh" $SHA 3 D1-m2 20260932 D1 /m4/K-s1/run/checkpoint-0001624 C-m2 > "$T/busy.out" 2>&1
check "busy owner refused" has "$T/busy.out" "owner entry says status=running"

# teacher collection
bash "$L/teacher.sh" $SHA 2 teacher-0 shard 0 > "$T/teacher.out" 2>&1
check "teacher done" has "$T/teacher.out" "done teacher-0"
check "teacher image" has "$M8/teacher-0/console.log" "sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1 python3 -m v2.27b.typed_collect_kernel"
check "teacher scored args" has "$M8/teacher-0/console.log" "--max-length 32768 --calibration $A/package/calibration.json --input /in/prompts-0.jsonl --output /out/predictions.jsonl"
check "teacher runtime mirror" has "$M8/teacher-0/console.log" "$TT-src_training_decision2/src/training/decision2,dst=/code"
check "teacher kernarg" has "$M8/teacher-0/console.log" "HIP_FORCE_DEV_KERNARG=1"
check "teacher private cache" test -f "$M8/tcache/teacher-0/x/f"
check "teacher cache receipt" has "$M8/teacher-0/cache.json" "\"source\": \"$A/formal/triton-cache\""
bash "$L/teacher.sh" $SHA 2 teacher-parity parity 80 > "$T/parity.out" 2>&1
check "teacher parity smoke" has "$M8/teacher-parity/console.log" "--input /in/typed-final.prompts.jsonl --output /out/predictions.jsonl --max-items 80"

# readout of an M7 checkpoint into m8/
bash "$L/readout.sh" $SHA 2 C-m1-e1 /m7/C-m1/run/checkpoint-0000166 --no-cal --panel dev --panel ht-dev2 > "$T/ro.out" 2>&1
check "readout done" has "$T/ro.out" "done C-m1-e1"
check "readout into m8" test -f "$M8/C-m1-e1-ht-dev2/console.log"
check "readout runtime mirror" has "$M8/C-m1-e1-dev/console.log" "$RT-src_training_decision2/src/training/decision2,dst=/code"
check "readout --no-cal" bash -c "! grep -q -- '--calibration' '$M8/C-m1-e1-dev/console.log'"

# a line: soup of member runs, interpolations, readouts, screens vs M7's reference
for k in 1 2; do
  mkdir -p "$M8/Z-m$k/run/checkpoint-0000300"; touch "$M8/Z-m$k/run/checkpoint-0000300/decision_config.json"
  echo '{"checkpoint": "checkpoint-0000300"}' > "$M8/Z-m$k/run/BEST.json"
done
bash "$L/line.sh" $SHA 2 KDZ Z-m1 Z-m2 > "$T/line.out" 2>&1
check "line done" has "$T/line.out" "line KDZ done"
check "line soup members" has "$M8/KDZ-soup-build/members.txt" "/m8/Z-m2/run/checkpoint-0000300"
check "line a23 weights" python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["distinct"][0]["weight"]=="2/3", d' "$M8/KDZ-a23-build/weights.json"
check "line a13 base Lux" has "$M8/KDZ-a13-build/members.txt" "/m3/pf-D-s1-zero/run/checkpoint-0000000"
check "line panels" test -f "$M8/KDZ-a12-mlxdev/console.log" -a -f "$M8/KDZ-a12-pn1-dev/console.log"
check "line mlxdev rows from M7" has "$M8/KDZ-a12-mlxdev/console.log" "--rows /m7/data/mlxdev/build/panel.jsonl"

# early rule inputs
for s in D1-m1-e1-dev D1-m1-e1-css-pilot D1-m1-e1-ht-dev2 D1-m1-e1-pn1-dev C-m1-e1-dev C-m1-e1-css-pilot C-m1-e1-ht-dev2; do
  mkdir -p "$M8/$s"; echo 0 > "$M8/$s/exit-code.txt"
done
echo '{"step": 166}' > "$M7/C-m1/run/COMPLETE.json"
bash "$L/early.sh" $SHA D1 > "$T/early.out" 2>&1
check "early control PN1 from M7" has "$M8/rules/early-D1.console" "--control-pn1 $M7/screens/C-m1-e1/pn1.json"
check "early final SELECTs" has "$M8/rules/early-D1.console" "--arm-select $M8/D1-m1/run/select-step-0000166-metrics.json --control-select $M7/C-m1/run/select-step-0000166-metrics.json"

# chain step lease
bash "$L/chain-step.sh" $SHA 2 c2 step1 30 -- true > "$T/step.out" 2>&1
check "chain-step logged end" has "$M8/logs/c2.log" "end step=step1"
check "chain-step entry" grep -qx "track=9b-m8" "$LK/gpu2.lock/owner.9b-m8"
check "chain-step owner untouched" cmp -s "$LK/gpu2.lock/owner" "$T/owner.before"

echo "failures: $fails"
[ "$fails" = 0 ]
