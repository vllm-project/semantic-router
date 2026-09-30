# shellcheck shell=bash disable=SC2034
# Shared settings of the 9B Milestone 8 node wrappers (sourced, never run). D2_DATA (default
# /data) moves every node path under another root for the dry-run harness; DRY_RUN=1 prints the
# docker / host-python / eval-runner commands instead of running them and skips the GPU probes.
# M8 writes only under m8/ and formal-m8/; M7's runs and data (the control C line, the reference
# re-read ref-ka13, the K-mix top-up TRAIN) are read-only inputs.
DATA=${D2_DATA:-/data}
RUNS=$DATA/dev2/runs/9b
M3=$RUNS/m3
M4=$RUNS/m4
M6=$RUNS/m6
M7=$RUNS/m7
M8=$RUNS/m8
R27=$DATA/dev2/runs/27b
LEASES=$DATA/dev2/leases
TRACK=9b-m8
# GPUs are lent by the ~27B track (node A GPU2-4): M8 writes only its own lease entry
# gpuN.lock/owner.9b-m8 and never rewrites the owner's gpuN.lock/owner.
LEASE_ENTRY=owner.9b-m8
IMAGE=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
# The 27B kernel image of DEV2.0-27B's scored run (teacher targets only).
IMAGE27=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
# Readouts, soups, scoring and formal runs default to the incumbent K-a13's runner mirror:
# training/model/infer.py (hashed into infer_dec's adapter identity) changed after it.
RUNTIME_DEFAULT=${D2_M8_RUNTIME:-3277dec9d81708fa374405ca884043443bab1b49}
# The continuations train with the exact mirror M7's control C trained with (byte-identical trainer).
TRAIN_CODE=${D2_M8_TRAIN_CODE:-7168f08643eb9297c7f457799dc06223056be656}
# DEV2.0-27B (A20r) scored runtime: the mirror, checkpoint, T = 1 calibration and frozen cache of
# its post-key run M4-A20r-soup/formal.
TEACHER_CODE=ff660322a76b04c4774fdb6aaa31e5af2a45743e
TEACHER_CKPT=$R27/M4-A20r-soup/soup/checkpoint
TEACHER_CAL=$R27/M4-A20r-soup/package/calibration.json
TEACHER_CACHE=$R27/M4-A20r-soup/formal/triton-cache
TEACHER_BASE=$DATA/decision20-20260926/models/Qwen3.8-27B
TEACHER_REV=5323310327e52d4eadd119cd10accac9b106c97d
TEACHER_PARITY=$R27/M4-A20r-soup/formal/output/typed-final.predictions.jsonl
CAL=/hfc/datasets--llm-semantic-router--decision-2.0-training-data/snapshots/ed87a03ab80ca5b9560780bba51a83a77ff47d14/m2/cal/CAL698/cal.jsonl
SELECT=/d10/rights_clean_goemotions_v2/select.jsonl
PN1_SNAP=$DATA/dev2/hf-cache/datasets--llm-semantic-router--decision-2.0-training-data/snapshots/27b1d2f130292268b43a618584bebab5d4e4a6b5/m4/pn1
PN1_GOLD=$PN1_SNAP/dev/pn1.dev.gold.jsonl
HTDEV2_REF=$DATA/dev2/runs/eval/htdev2/collect/9b-m4-K-a13/output/ht-dev2.predictions.jsonl
MLXDEV=$M7/data/mlxdev/build
# The five K seeds (SELECT-chosen BEST of M4 K-s1..s3 and M6 K-s4 / K-s5), in member order.
MEMBERS=(/m4/K-s1/run/checkpoint-0001624 /m4/K-s2/run/checkpoint-0001420 /m4/K-s3/run/checkpoint-0001424
  /m6/K-s4/run/checkpoint-0001629 /m6/K-s5/run/checkpoint-0001621)

mirror_dir() { printf '%s\n' "$DATA/dev2/src/$1-src_training_decision2"; }
code_dir() { printf '%s\n' "$DATA/dev2/src/$1-src_training_decision2/src/training/decision2"; }

tree_sha() { (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1); }

dry() {
  if [ "${DRY_RUN:-0}" = 1 ]; then
    local a line=DRY:
    for a in "$@"; do
      case "$a" in *[[:space:]]*|"") line+=" '$a'" ;; *) line+=" $a" ;; esac
    done
    printf '%s\n' "$line"
    return 0
  fi
  "$@"
}

best_of() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$1"; }

# Host path of a container checkpoint path under /m8/, /m7/, /m6/, /m4/ or /m3/.
host_path() {
  case "$1" in
    /m8/*) printf '%s\n' "$M8/${1#/m8/}" ;;
    /m7/*) printf '%s\n' "$M7/${1#/m7/}" ;;
    /m6/*) printf '%s\n' "$M6/${1#/m6/}" ;;
    /m4/*) printf '%s\n' "$M4/${1#/m4/}" ;;
    /m3/*) printf '%s\n' "$M3/${1#/m3/}" ;;
    *) return 1 ;;
  esac
}

# successor_summary OUT NAME RUN GATES_DIR LUX_PAIRED MLX_PAIRED SCORE_LEVELS: numbers the
# preregistered successor rule reads, from the run's REPORT.json, the paired file vs native Lux1,
# the gates directory (PAIRED-vs-DEV2.0-9B-T1.json, PAIRED-vs-Nimble2.json, types.json), the mlx
# paired file and the Score level file (host python3, stdlib only). Missing inputs give null
# fields. Deliberately no pass / fail field.
successor_summary() {
  python3 - "$@" <<'EOF'
import hashlib, json, os, sys
out, name, run, gates, lux_paired, mlx_paired, score_levels = sys.argv[1:]

def load(path):
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError:
        return None, None
    return json.loads(raw), hashlib.sha256(raw).hexdigest()

def dig(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d

inputs = {}
def get(path):
    d, sha = load(path)
    inputs[path] = sha
    return d

def paired(path):
    d = get(path)
    return {
        "point": dig(d, "point", "delta", "score"),
        "lb": dig(d, "ci95", "low"),
        "ub": dig(d, "ci95", "high"),
        "T_delta": dig(d, "point", "delta", "T"),
        "T_delta_low": dig(d, "axis_ci95", "T", "delta", "low"),
        "T_delta_high": dig(d, "axis_ci95", "T", "delta", "high"),
        "H_delta": dig(d, "point", "delta", "H"),
        "H_delta_low": dig(d, "axis_ci95", "H", "delta", "low"),
        "H_delta_high": dig(d, "axis_ci95", "H", "delta", "high"),
        "right": dig(d, "models", "right"),
    }

report = get(os.path.join(run, "REPORT.json"))
types = get(os.path.join(gates, "types.json"))
mlx = get(mlx_paired)
levels = get(score_levels)
pub = dig(report, "panels", "public231") or {}
summary = {
    "schema": "lux9b-m8-successor-summary/1",
    "name": name,
    "run": run,
    "note": "numbers only; the preregistered M8 rule decides",
    "v3": dig(report, "v3"),
    "public231": {"correct": pub.get("correct"), "items": pub.get("items"),
                  "tiers": {k: v.get("correct") for k, v in (pub.get("tiers") or {}).items()}},
    "vs_T1": paired(os.path.join(gates, "PAIRED-vs-DEV2.0-9B-T1.json")),
    "vs_Lux1": paired(lux_paired),
    "vs_Nimble2": paired(os.path.join(gates, "PAIRED-vs-Nimble2.json")),
    "public231_vs_T1": {k: dig(get(os.path.join(gates, "PUBLIC231-vs-DEV2.0-9B-T1.json")), k)
                        for k in ("delta", "mcnemar_exact_p", "verdict")},
    "types": {k: v.get("verdict") for k, v in (dig(types, "types") or {}).items()},
    "mlx_vs_T1": {"delta": dig(mlx, "overall", "delta"), "ci_low": dig(mlx, "overall", "ci95", "low"),
                  "ci_high": dig(mlx, "overall", "ci95", "high")},
    "score_levels": {k: dig(levels, "score", k) for k in
                     ("levels_used", "level0_recall", "largest_answer", "largest_answer_share", "invalid")},
    "inputs_sha256": inputs,
}
with open(out, "w") as f:
    json.dump(summary, f, indent=1, sort_keys=True)
    f.write("\n")
print(json.dumps({"successor": out, "vs_T1": summary["vs_T1"], "types": summary["types"]}))
EOF
}

# GPU7 is shared with the eval track's C1 event 3: never start while its lease entry is active.
# Active = gpu7.lock/owner.eval exists and its status= does not start with idle or released
# (a missing status line, as run_same_panel.sh writes for a running job, counts as active).
yield_gpu7() {
  local gpu=$1 log=${2:-} f=$LEASES/gpu7.lock/owner.eval st waited=0 msg
  [ "$gpu" = 7 ] || return 0
  while [ -f "$f" ]; do
    st=$(sed -n 's/^status=//p' "$f" | head -n 1)
    case "$st" in idle*|released*) break ;; esac
    if [ $((waited % 600)) = 0 ]; then
      msg="$(date -u +%FT%TZ) gpu7 yield: owner.eval status=${st:-<none>}; waiting (${waited}s so far)"
      echo "$msg" >&2
      [ -z "$log" ] || echo "$msg" >> "$log"
    fi
    sleep "${D2_YIELD_SLEEP:-60}"
    waited=$((waited + 60))
  done
  if [ "$waited" -gt 0 ]; then
    msg="$(date -u +%FT%TZ) gpu7 yield: owner.eval free after ${waited}s"
    echo "$msg" >&2
    [ -z "$log" ] || echo "$msg" >> "$log"
  fi
  return 0
}

# GPU-hours used by Milestone 8 so far: every GPU job directory (gpu.txt; wall-clock, running jobs
# up to now) under m8/ and formal-m8/, plus the eval-runner collections' GPU-TIME.json under
# formal-m8/. Soups, data builds and rules are CPU.
m8_gpu_hours() {
  python3 - "$M8" "$RUNS/formal-m8" <<'PY'
import datetime as dt, glob, json, os, sys
fmt = "%Y-%m-%dT%H:%M:%SZ"
now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
total = 0.0
for root in sys.argv[1:]:
    for gpu in glob.glob(os.path.join(root, "*", "gpu.txt")):
        run = os.path.dirname(gpu)
        start = dt.datetime.strptime(open(os.path.join(run, "start-utc.txt")).read().strip(), fmt)
        end_file = os.path.join(run, "end-utc.txt")
        end = dt.datetime.strptime(open(end_file).read().strip(), fmt) if os.path.exists(end_file) else now
        total += (end - start).total_seconds() / 3600
for receipt in glob.glob(os.path.join(sys.argv[2], "*", "GPU-TIME.json")):
    total += float(json.load(open(receipt)).get("gpu_hours") or 0)
print(f"{total:.3f}")
PY
}

# budget_ok NEED: 0 if the GPU-hours used so far plus NEED stay within the 24 GPU-h cap.
budget_ok() {
  local used
  used=$(m8_gpu_hours)
  python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) + float(sys.argv[2]) <= 24.0 else 1)' "$used" "$1" \
    || { echo "budget: $used GPU-h used + $1 projected > 24 cap; not started" >&2; return 1; }
  echo "budget: $used GPU-h used + $1 projected <= 24" >&2
}

# wait_ok RUN MAX_MIN: wait until the M8 GPU job RUN has exit-code 0 (fail on another code or timeout).
wait_ok() {
  local run=$M8/$1 waited=0
  while [ ! -f "$run/exit-code.txt" ]; do
    [ "$waited" -lt "$2" ] || { echo "timeout waiting for $1" >&2; return 1; }
    sleep 60; waited=$((waited + 1))
  done
  [ "$(cat "$run/exit-code.txt")" = 0 ] || { echo "$1 exited $(cat "$run/exit-code.txt")" >&2; return 1; }
}

# early_continue ARM: 0 if rules/early-ARM.json exists and says the arm continues.
early_continue() {
  [ -f "$M8/rules/early-$1.json" ] || return 1
  python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["continue"] else 1)' "$M8/rules/early-$1.json"
}

# arm_decided ARM: 0 once the arm's member-1 outcome is known: its early rule file exists, a
# preflight / recipe check stopped it, or one of its member-1 steps ended with a nonzero exit.
arm_decided() {
  [ -f "$M8/rules/early-$1.json" ] && return 0
  ls "$M8/pf-$1-m1-zero/STOPPED.txt" "$M8/pf-$1-m1-check/STOPPED.txt" "$M8/$1-m1/STOPPED.txt" >/dev/null 2>&1 && return 0
  grep -qE "end step=($1-m1|$1-m1-e1|early-$1) .* exit=[1-9]" "$M8"/logs/m8-gpu*.log 2>/dev/null
}

# parity_ok MAX_MIN: wait for the teacher parity check (m8/teacher-parity/parity.json); 0 on PASS.
parity_ok() {
  local waited=0 f=$M8/teacher-parity/parity.json
  while [ ! -f "$f" ]; do
    if [ -f "$M8/teacher-parity/exit-code.txt" ] && [ "$(cat "$M8/teacher-parity/exit-code.txt")" != 0 ]; then
      echo "teacher parity job failed" >&2; return 1
    fi
    [ "$waited" -lt "$1" ] || { echo "timeout waiting for the teacher parity check" >&2; return 1; }
    sleep 60; waited=$((waited + 1))
  done
  python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["status"] == "PASS" else 1)' "$f" \
    || { echo "teacher parity FAIL" >&2; return 1; }
}

# wait_file PATH MAX_MIN: wait until PATH exists (fail on timeout).
wait_file() {
  local waited=0
  while [ ! -e "$1" ]; do
    [ "$waited" -lt "$2" ] || { echo "timeout waiting for $1" >&2; return 1; }
    sleep 60; waited=$((waited + 1))
  done
}

# lent_ok GPU: 0 if the ~27B owner entry of GPU (gpuN.lock/owner; key=value or JSON) is absent,
# ours, idle or released (a missing status counts as busy unless the entry records a finished job),
# and no other 9B-M8 job holds our entry as running. The owner entry is never rewritten.
lent_ok() {
  local lock=$LEASES/gpu$1.lock st
  if [ -f "$lock/owner" ]; then
    st=$(python3 - "$lock/owner" "$TRACK" <<'PY'
import json, re, sys
text = open(sys.argv[1]).read()
try:
    d = json.loads(text)
    d = {k: str(v) for k, v in d.items()} if isinstance(d, dict) else {}
except ValueError:
    d = dict(re.findall(r"^(\w+)=(.*)$", text, re.M))
if d.get("track") == sys.argv[2]:
    print("ours")
elif "status" in d:
    print(d["status"] or "<empty>")
else:
    print("ended" if "last_job_exit" in d else "<missing>")
PY
)
    case "$st" in ours|ended|idle*|released*) ;; *) echo "gpu$1 owner entry says status=$st" >&2; return 1 ;; esac
  fi
  if grep -q "^status=running" "$lock/$LEASE_ENTRY" 2>/dev/null; then
    echo "gpu$1 $LEASE_ENTRY says running" >&2; return 1
  fi
}
