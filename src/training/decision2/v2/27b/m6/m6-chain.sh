#!/usr/bin/env bash
# ~27B M6 post-training chain on node B (host side; preregistration "Candidates and development gates" and "Formal,
# successor rule and Index"). For every arm it waits for both seeds:
#   - a node B seed (ARM-SEED=b:DRIVER_PID) is ready when run_lora_arm.sh wrote full/RUN_DIR with a complete, frozen
#     BEST; it failed when its driver ended without that;
#   - a node A seed (ARM-SEED=a) is ready when node A's m6-relay.sh left relay/ARM-SEED/SHA256SUMS on the M6 node link,
#     and failed when it left RELAY-FAILED.txt.
# A ready arm is pulled (node A member, SHA-256 lists compared), souped (exact rank-256 concatenation), read out
# (CAL698 kernel fit, typed DEV + CSS pilot + HT-DEV v2) and sliced (PN1 dev + IB DEV) on its aux GPU (AUX_<ARM> = one of
# its own node B training GPUs, free once its seeds ended). An arm with a failed seed has no candidate (no rerun).
# When every arm is settled: development gates G1-G6 (m6_devgates.py); for each finalist, if the receipts on both nodes
# plus RESERVE stay within CAP, the formal run, the mlx-diag collection and its push to node A (m6-mlx-watch.sh scores
# and pairs it there); non-finalists get mlx/NAME.SKIP. Then mlx-pull and the host-CPU gates (m6-gates.sh gates,
# overlap, verdicts). A failed stage ends the chain with a record; nothing reruns.
# Usage: m6-chain.sh MIRROR_SHA ARM-SEED=b:PID|a ...
# Environment: PN1_ROWS PN1_SHA IB_ROWS IB_SHA (DEV rows and their SHA-256), IN_DIST ("w2c isarc"), AUX_M6_IB,
#   AUX_M6_IBX (node B GPU0 / GPU1 / GPU5), LOADED (formal loaded parameters), CAP (140), RESERVE (formal + mlx, 1.0
#   per finalist), REF_IB (A20r's IB DEV slices name, default A20r-ib1).
set -euo pipefail
echo "m6 chain $*: start $(date -u +%FT%TZ)"
SHA=${1:?MIRROR_SHA}
shift
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
[ $# -ge 2 ] || { echo "at least two ARM-SEED=b:PID|a entries" >&2; exit 2; }
: "${PN1_ROWS:?}" "${PN1_SHA:?}" "${IB_ROWS:?}" "${IB_SHA:?}"
LOADED=${LOADED:-27497508864} CAP=${CAP:-140} RESERVE=${RESERVE:-1.0} REF_IB=${REF_IB:-A20r-ib1}
S=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2
TAIL=$S/v2/27b/m6/m6-tail.sh GATES=$S/v2/27b/m6/m6-gates.sh
[ -f "$TAIL" ] && [ -f "$GATES" ] || { echo "missing mirror $SHA" >&2; exit 2; }
R=/data/dev2/runs/27b/m6 KEY=/data/dev2/tmp/27b-m6-xfer
mkdir -p "$R/logs"
X="ssh -i $KEY/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=$KEY/known_hosts -o StrictHostKeyChecking=yes -o BatchMode=yes"
stamp() { date -u +%FT%TZ; }
on_a() { rsync --list-only -e "$X" "root@$(cat "$KEY/peer"):$1" > /dev/null 2>&1; }
to_a() { rsync -a --mkpath -e "$X" "$1" "root@$(cat "$KEY/peer"):$2"; }
declare -A WHERE=() PID=() STATE=()
ARMS=()
for spec in "$@"; do
  seed=${spec%%=*} where=${spec#*=}
  [[ "$seed" =~ ^(M6-[A-Z0-9]+)-s[12]$ ]] || { echo "bad ARM-SEED $seed" >&2; exit 2; }
  arm=${BASH_REMATCH[1]}
  case "$where" in a) WHERE[$seed]=a ;; b:*) WHERE[$seed]=b PID[$seed]=${where#b:} ;; *) echo "bad $spec" >&2; exit 2 ;; esac
  [[ " ${ARMS[*]} " == *" $arm "* ]] || ARMS+=("$arm")
done
for arm in "${ARMS[@]}"; do
  var=AUX_${arm//-/_}
  [ -n "${!var:-}" ] || { echo "no $var" >&2; exit 2; }
  [ -n "${WHERE[$arm-s1]:-}" ] && [ -n "${WHERE[$arm-s2]:-}" ] || { echo "$arm needs s1 and s2" >&2; exit 2; }
done
best_of() {  # RUN_ROOT: the completed run's frozen BEST checkpoint path
  python3 - "$1/full" <<'EOF'
import json, pathlib, sys
full = pathlib.Path(sys.argv[1])
run = full / (full / "RUN_DIR").read_text().strip()
best = json.loads((run / "BEST.json").read_text())["checkpoint"]
complete = json.loads((run / "COMPLETE.json").read_text())
if complete.get("status") != "complete" or complete.get("best") != best:
    raise SystemExit(f"{run} is not complete with a frozen BEST")
print(run / best)
EOF
}
seed_state() {  # ARM-SEED -> ready | failed | wait
  local seed=$1 run=/data/dev2/runs/27b/$1
  if [ "${WHERE[$seed]}" = b ]; then
    if [ -f "$run/full/RUN_DIR" ]; then echo ready; return; fi
    if ! kill -0 "${PID[$seed]}" 2> /dev/null; then
      sleep 10
      [ -f "$run/full/RUN_DIR" ] && echo ready || echo failed
      return
    fi
    echo wait
  else
    if on_a "relay/$seed/RELAY-FAILED.txt"; then echo failed; return; fi
    on_a "relay/$seed/SHA256SUMS" && echo ready || echo wait
  fi
}
member() {  # ARM-SEED -> checkpoint path on node B (pulls a node A member first)
  if [ "${WHERE[$1]}" = b ]; then
    best_of "/data/dev2/runs/27b/$1"
  else
    [ -f "$R/relay/$1/SHA256SUMS.nodeB" ] || bash "$TAIL" pull "$SHA" "$1" >&2
    echo "$R/relay/$1/checkpoint"
  fi
}
process() {  # ARM
  local arm=$1 var=AUX_${1//-/_} c1 c2 sums=""
  c1=$(member "$arm-s1") c2=$(member "$arm-s2")
  for s in s1 s2; do [ "${WHERE[$arm-$s]}" = a ] && sums+="$R/relay/$arm-$s/checkpoint=$R/relay/$arm-$s/SHA256SUMS.nodeB "; done
  echo "$(stamp) $arm: soup of $c1 and $c2"
  [ -f "$R/$arm/checkpoint/soup_manifest.json" ] || RELAY_SUMS=$sums bash "$TAIL" lsoup "$SHA" "$arm" "$c1" "$c2"
  [ -f "$R/readouts/$arm/READOUT-M4B.json" ] || bash "$TAIL" readout "$SHA" "$arm" "$R/$arm/checkpoint" "${!var}"
  [ -f "$R/slices/$arm/probs/slices.json" ] || bash "$TAIL" slices "$SHA" "$arm" "$R/$arm/checkpoint" "${!var}" \
    "pn1=$PN1_ROWS=$PN1_SHA" "ib=$IB_ROWS=$IB_SHA"
  stamp > "$R/logs/$arm.CANDIDATE"
}
while :; do
  open=0
  for arm in "${ARMS[@]}"; do
    [ -n "${STATE[$arm]:-}" ] && continue
    s1=$(seed_state "$arm-s1") s2=$(seed_state "$arm-s2")
    if [ "$s1" = failed ] || [ "$s2" = failed ]; then
      STATE[$arm]=none
      echo "$(stamp) $arm: a seed ended without a finished run (s1 $s1, s2 $s2): no candidate, nothing reruns" |
        tee "$R/logs/$arm.NO-CANDIDATE"
    elif [ "$s1" = ready ] && [ "$s2" = ready ]; then
      process "$arm"
      STATE[$arm]=candidate
    else
      open=1
    fi
  done
  [ "$open" = 0 ] && break
  sleep 300
done
names=()
for arm in "${ARMS[@]}"; do [ "${STATE[$arm]}" = candidate ] && names+=("$arm"); done
skip() { printf '%s %s\n' "$(stamp)" "$2" > "$R/logs/$1.SKIP"; to_a "$R/logs/$1.SKIP" "mlx/$1.SKIP"; }
if [ ${#names[@]} = 0 ]; then
  for arm in "${ARMS[@]}"; do skip "$arm" "no candidate"; done
  echo "m6 chain complete (no candidate): $(stamp)"
  exit 0
fi
PN1_ROWS=$PN1_ROWS IB_ROWS=$IB_ROWS IN_DIST=${IN_DIST:-} REF_IB=$REF_IB bash "$TAIL" devgates "$SHA" "${names[@]}"
DEVGATES=$(find "$R/readouts" -maxdepth 1 -name 'DEVGATES-*.json' | sort | tail -n 1)
mapfile -t finalists < <(python3 -c "import json,sys; print('\n'.join(json.load(open(sys.argv[1]))['finalists']))" "$DEVGATES" | sed '/^$/d')
echo "$(stamp) development gates $DEVGATES: finalists ${finalists[*]:-none}"
for arm in "${ARMS[@]}"; do
  [[ " ${finalists[*]:-} " == *" $arm "* ]] || skip "$arm" "$arm is not a finalist ($DEVGATES)"
done
[ ${#finalists[@]} -gt 0 ] || { echo "m6 chain complete (no finalist): $(stamp)"; exit 0; }
receipts() {  # GPU-h on node B plus node A's latest relay budget
  python3 - "$R/relay" <<'EOF'
import glob, json, os, sys
seen, total = set(), 0.0
paths = glob.glob("/data/dev2/runs/27b/m6/**/*.json", recursive=True)
paths += glob.glob("/data/dev2/runs/27b/M6-*/**/*.json", recursive=True)
for path in paths:
    real = os.path.realpath(path)
    if real in seen or "/relay/" in path or "/triton-cache" in path:
        continue
    seen.add(real)
    try:
        record = json.load(open(path))
    except Exception:
        continue
    if isinstance(record, dict) and (os.path.basename(path) == "GPU-TIME.json" or (
            os.path.basename(os.path.dirname(path)) == "receipts" and "gpu_hours" in record)):
        total += float(record.get("gpu_hours", 0))
node_a = [json.load(open(p)) for p in glob.glob(os.path.join(sys.argv[1], "*", "BUDGET-nodeA.json"))]
print(round(total + max((b["gpu_hours"] for b in node_a), default=0.0), 3))
EOF
}
formal_gpu=${AUX_M6_IB:-0}
for arm in "${finalists[@]}"; do
  total=$(receipts)
  echo "$(stamp) M6 receipts: $total GPU-h; reserve $RESERVE; cap $CAP"
  if ! python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) + float(sys.argv[2]) <= float(sys.argv[3]) else 1)' \
    "$total" "$RESERVE" "$CAP"; then
    skip "$arm" "budget: $total + $RESERVE passes $CAP"
    echo "$(stamp) budget: $arm's formal run waits for a recorded coordinator decision"
    continue
  fi
  var=AUX_${arm//-/_}
  extra=()
  for other in "${finalists[@]}"; do [ "$other" != "$arm" ] && [ -f "$R/$other/formal/SEAL.json" ] && extra+=("$other=$R/$other/formal"); done
  [ -f "$R/$arm/formal/SEAL.json" ] ||
    LOADED_PARAMETERS=$LOADED EXTRA_COMPARATOR="${extra[*]:-}" bash "$TAIL" formal "$SHA" "$arm" "$R/$arm/checkpoint" "${!var:-$formal_gpu}"
  [ -f "$R/mlx-diag/$arm/COLLECT.json" ] || bash "$TAIL" mlx "$SHA" "$arm" "${!var:-$formal_gpu}"
  bash "$TAIL" mlx-push "$SHA" "$arm"
  stamp > "$R/logs/$arm.PUSHED"
  to_a "$R/logs/$arm.PUSHED" "mlx/$arm.PUSHED"
done
sealed=()
for arm in "${finalists[@]}"; do [ -f "$R/$arm/formal/SEAL.json" ] && sealed+=("$arm"); done
[ ${#sealed[@]} -gt 0 ] || { echo "m6 chain complete (no formal run): $(stamp)"; exit 4; }
for arm in "${sealed[@]}"; do
  until on_a "mlx/$arm-vs-A20r.json"; do sleep 120; done
  sleep 30
  bash "$TAIL" mlx-pull "$SHA" "$arm"
done
bash "$GATES" "$SHA" gates "${sealed[@]}"
bash "$GATES" "$SHA" overlap "${sealed[@]}"
bash "$GATES" "$SHA" verdicts "${sealed[@]}"
echo "m6 chain complete: $(stamp)"
