# shellcheck shell=bash disable=SC2034
# Shared settings of the 9B Milestone 5 node wrappers (sourced, never run). D2_DATA (default
# /data) moves every node path under another root for the dry-run harness; DRY_RUN=1 prints the
# docker / host-python / eval-runner commands instead of running them and skips the GPU probes.
DATA=${D2_DATA:-/data}
RUNS=$DATA/dev2/runs/9b
M3=$RUNS/m3
M4=$RUNS/m4
M5=$RUNS/m5
LEASES=$DATA/dev2/leases
TRACK=9b-m5
IMAGE=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
# Readouts, soups, scoring and formal runs default to the incumbent K-a13's runner mirror:
# training/model/infer.py (hashed into infer_dec's adapter identity) changed after it.
RUNTIME_DEFAULT=${D2_M5_RUNTIME:-3277dec9d81708fa374405ca884043443bab1b49}
CAL=/hfc/datasets--llm-semantic-router--decision-2.0-training-data/snapshots/ed87a03ab80ca5b9560780bba51a83a77ff47d14/m2/cal/CAL698/cal.jsonl
SELECT=/d10/rights_clean_goemotions_v2/select.jsonl

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

# Host path of a container checkpoint path under /m5/, /m4/ or /m3/.
host_path() {
  case "$1" in
    /m5/*) printf '%s\n' "$M5/${1#/m5/}" ;;
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
    "schema": "lux9b-m5-successor-summary/1",
    "name": name,
    "run": run,
    "note": "numbers only; the preregistered M5 rule decides",
    "v3": dig(report, "v3"),
    "public231": {"correct": pub.get("correct"), "items": pub.get("items"),
                  "tiers": {k: v.get("correct") for k, v in (pub.get("tiers") or {}).items()}},
    "vs_T1": paired(os.path.join(gates, "PAIRED-vs-DEV2.0-9B-T1.json")),
    "vs_Lux1": paired(lux_paired),
    "vs_Nimble2": paired(os.path.join(gates, "PAIRED-vs-Nimble2.json")),
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
