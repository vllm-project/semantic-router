#!/usr/bin/env bash
# ~27B M4 post-training tail (node B host), run from its own mirror like m4-launch-arm.sh; one step per subcommand
# (m4-prereg-2026-09-29.md: "Soups, readouts and finalists", "Comparisons, successor rule and attribution",
# mlx-diag, budget). The 27B stages are run_finalist.sh from this mirror; the gate and overlap tools run from the
# eval track's mirrors, as for F2 (m3f2/f2-phase2.sh). ARM is M4-A20, M4-A20r or M4-Ar (ARM-soup accepted).
#   soup ARM              run_finalist.sh STAGES=soup of ARM-soup = ARM-s1 + ARM-s2 (CPU container); both BEST
#                         checkpoints complete, a node-A member only with its relay verified (RELAY.json of
#                         m4-relay-best.sh, rehashed here); then the soup's rank / alpha (r16 a32; M4-A20r r64 a128),
#                         exactness and member bytes (= the relayed ones) are checked
#   readout ARM GPU       M3's soup readout (m3-queue.sh soupreadout): run_finalist.sh STAGES=readout at 32,768 on
#                         fresh copies of the frozen cache 583241fb (CAL698 fit + SELECT700 kernel probabilities,
#                         typed DEV + CSS pilot, v2.eval.dev_readout) -> ARM-soup/readout-kernel-32768
#   guard                 v2.27b.m4_guard over the soup readouts and F1's -> m4-logs/guard.json, prints the
#                         finalists; M4_ABSENT=ARM[,ARM] names arms that have no soup (a recorded stop)
#   prepkg ARM GPU        run_finalist.sh STAGES=cal698,adopt (CAL698 fit at 32,768, 23:15 adoption); finalists only
#   formal ARM GPU        run_finalist.sh STAGES=package,formal from the F1 scored-cache snapshot (03b172f1) with
#                         F1 as EXTRA_COMPARATOR (formal/PAIRED-vs-M3-A-soup.json = ARM-soup - F1); finalists only
#   gates                 per sealed finalist v2.eval.gates paired vs the three peers, F1 (both directions) and F2
#                         (descriptive) and types; the arm contrasts dose (M4-A20 - M4-Ar) and capacity
#                         (M4-A20r - M4-A20); the F1 record's item_cis.py driver (host CPU, eval mirror
#                         c8a5504b5) -> m4-gates/
#   overlap SPEC          v2.eval.overlap_effects exposure of a20 and ar (--expect-sha256) and run on SPEC
#                         restricted to the finalists (host CPU, eval mirror 1bbebc2fe) -> m4-overlap/
#   mlx NAME GPU [smoke]  mlx-diag collection of a sealed finalist (m3f2/f2-mlx.sh: package checkpoint, calibration
#                         binding and limit) on a fresh copy of its own scored post-run formal cache (post hash from
#                         formal/triton-cache.post.json) -> m4-mlx/NAME[-smoke]; the first collection is a smoke
#   contrast              v2.27b.m4_contrast: typed-FINAL families / types (item-level paired bootstrap), Score
#                         levels, successor rule, attribution -> m4-contrast/contrast.json
# Existing outputs are never overwritten (exit 66). Before a GPU stage the expected GPU-hours are printed and
# M4_LEFT (the GPU-hours left under the 70 cap, from m4-budget.sh) must cover them (else exit 3); the optional
# mlx-diag also keeps 0.54 GPU-hours per finalist still without a formal run. CAL698 fits are not optional here:
# run_finalist.sh freezes no package without one (its T = 1 binding is the rejected fit's).
# Test hooks only: M4_RUNS and M4_SRCROOT move the runs and mirror roots (the stages called keep /data/dev2).
set -euo pipefail
S=$(cd "$(dirname "$0")/../../.." && pwd)
SRC=$(basename "$(cd "$S/../../.." && pwd)")
R=${M4_RUNS:-/data/dev2/runs/27b}
SRCROOT=${M4_SRCROOT:-/data/dev2/src}
if [ "$R" != /data/dev2/runs/27b ] || [ "$SRCROOT" != /data/dev2/src ]; then
  [ ! -d /data/dev2/runs/27b ] || { echo "M4_RUNS and M4_SRCROOT are test hooks: unset them on a node" >&2; exit 2; }
fi
L=$R/m4-logs
GUARD=$L/guard.json
F1=$R/M3-A-soup
F2=$R/M3-S-soup
F1_LABEL="DEV2.0-27B (F1)"
F2_LABEL="DEV2.0-27B (F2)"
READOUT_CACHE=$R/m3-warm-32768/triton-cache
READOUT_SHA=583241fbc3bc89e22be51a49722996eab742162356100400d64fb4208cb20daf
SNAP=$R/m3-f2/f1-scored-cache
SNAP_SHA=03b172f1a6adeef6c6a6c491d04389b355c9d8579480008023f408c8659b502b
MLX_ROOT=$R/m4-mlx
MX=/data/dev2/private/27b/m4-data/mixtures-m4-1
OVERLAP_IN=/data/dev2/runs/eval/m5/overlap-effects
EVAL=$SRCROOT/c8a5504b5b39049390fccfbe3bcac5ef65e1e710-src_training_decision2/src/training/decision2
ITEM_CIS=$SRCROOT/5553298a1a7e1c8b192eab1bb7c2699689b7a3dd-src_training_decision2/src/training/decision2/v2/eval/records/m4-dev2-27b-f1-gates/item_cis.py
OVERLAP=$SRCROOT/1bbebc2fe3136858bcda63a12c2cde13c45d9238-src_training_decision2/src/training/decision2
ARMS=(M4-A20 M4-A20r M4-Ar)
# Expected GPU-hours per stage: the larger of the prereg plan and M3's soup receipts (readout 0.25 / 0.23,
# CAL698 fit 0.044 / 0.05, formal smoke + collection 0.53 / 0.45, mlx-diag 0.11, smoke 0.03).
COST_READOUT=0.26 COST_CAL698=0.05 COST_FORMAL=0.54 COST_MLX=0.12 COST_MLX_SMOKE=0.04
export TMPDIR=/data/dev2/tmp PYTHONDONTWRITEBYTECODE=1

stamp() { date -u +%FT%TZ; }
arm_of() {  # ARM or ARM-soup -> ARM
  case "${1%-soup}" in
    M4-A20 | M4-A20r | M4-Ar) echo "${1%-soup}" ;;
    *) echo "unknown arm $1 (M4-A20, M4-A20r or M4-Ar)" >&2; return 2 ;;
  esac
}
gpu_ok() { case "$1" in 5 | 6 | 7) ;; *) echo "GPU$1 is outside node B GPU5-7" >&2; exit 2 ;; esac; }
fresh() {  # PATH...: every output must be new
  local path
  for path in "$@"; do
    [ ! -e "$path" ] || { echo "$path exists: refusing to overwrite" >&2; exit 66; }
  done
}
finalists() {  # [sealed]: the guard's finalists (with sealed: only those whose formal run is sealed)
  [ -f "$GUARD" ] || { echo "no $GUARD: run the guard first" >&2; return 2; }
  python3 - "$GUARD" "$R" "${1:-}" <<'EOF'
import json, os, sys
guard, runs, sealed = sys.argv[1], sys.argv[2], sys.argv[3] == "sealed"
names = json.load(open(guard))["finalists"]
print(" ".join(n for n in names if not sealed or os.path.isfile(f"{runs}/{n}/formal/SEAL.json")))
EOF
}
need_finalist() {  # NAME
  local all
  all=$(finalists)
  case " $all " in *" $1 "*) ;; *) echo "$1 is not a finalist in $GUARD (finalists: $all)" >&2; exit 2 ;; esac
}
all_sealed() {  # -> the finalists; fails unless every finalist's formal run is sealed
  local all sealed
  all=$(finalists)
  sealed=$(finalists sealed)
  [ -n "$all" ] || { echo "the guard left no finalist" >&2; return 2; }
  [ "$all" = "$sealed" ] || { echo "finalists without a sealed formal run: all [$all], sealed [$sealed]" >&2; return 2; }
  echo "$all"
}
budget() {  # STAGE COST [RESERVE]: print the expected GPU-hours; refuse unless M4_LEFT covers them
  local stage=$1 cost=$2 reserve=${3:-0} open=0 need=$2 name
  echo "expected GPU-hours for $stage: $cost"
  if [ "$reserve" != 0 ]; then
    for name in $(finalists); do [ -f "$R/$name/formal/SEAL.json" ] || open=$((open + 1)); done
    need=$(python3 -c 'import sys; c, r, k = map(float, sys.argv[1:]); print(round(c + r * k, 4))' "$cost" "$reserve" "$open")
    echo "optional stage: keeps $reserve GPU-hours free for each of $open formal run(s) to come: needs $need"
  fi
  [ -n "${M4_LEFT:-}" ] || { echo "set M4_LEFT to the GPU-hours left under the 70 cap (m4-budget.sh)" >&2; exit 3; }
  python3 -c 'import sys; sys.exit(0 if float(sys.argv[2]) <= float(sys.argv[1]) else 1)' "$M4_LEFT" "$need" \
    || { echo "running-total check: $stage needs $need GPU-hours, M4_LEFT is $M4_LEFT" >&2; exit 3; }
}
best_of() {  # MEMBER -> its BEST checkpoint (run_finalist.sh's rule: a complete run with a frozen BEST)
  python3 - "$R/$1/full" <<'EOF'
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
relay_ok() {  # MEMBER CHECKPOINT: node-B copy of a node-A arm-seed, rehashed against its RELAY.json
  python3 - "$R/$1" "$2" <<'EOF'
import hashlib, json, pathlib, sys
root, ckpt = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
relay = json.loads((root / "RELAY.json").read_text())
files = relay["files_sha256"]
if ckpt != root / "full" / relay["run_dir"] / relay["best"]:
    raise SystemExit(f"{root}: RELAY.json names another BEST checkpoint")
on_disk = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
if on_disk != sorted([*files, "RELAY.json"]):
    raise SystemExit(f"{root}: files differ from RELAY.json")
for name, digest in files.items():
    sha = hashlib.sha256()
    with open(root / name, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    if sha.hexdigest() != digest:
        raise SystemExit(f"{root / name} differs from RELAY.json")
print(f"relay verified: {root.name} {relay['best']} ({len(files)} files)")
EOF
}
finalist() {  # STAGES GPU ARM [KEY=VALUE]...: run_finalist.sh on ARM-soup, appended to its driver log
  local stages=$1 gpu=$2 arm=$3 status=0
  shift 3
  (cd "$S" && env STAGES="$stages" LABEL="$arm-soup" "$@" \
    bash v2/27b/run_finalist.sh "$arm-soup" "$gpu" "$SRC" "$arm-s1" "$arm-s2") >> "$R/$arm-soup.driver.log" 2>&1 \
    || status=$?
  echo "=== $(stamp) $stages exit=$status" >> "$R/$arm-soup.driver.log"
  [ "$status" = 0 ] || { echo "run_finalist.sh $stages failed (exit $status): $R/$arm-soup.driver.log" >&2; exit "$status"; }
}
paired() {  # LEFT LEFT_NAME RIGHT RIGHT_NAME OUTPUT: v2.eval.gates paired (log beside the output)
  python3 -m v2.eval.gates paired --left "$1" --left-name "$2" --right "$3" --right-name "$4" \
    --output "$5" > "${5%.json}.log"
}

cmd=${1:-}
case "$cmd" in
  soup)
    arm=$(arm_of "${2:?ARM}")
    name=$arm-soup
    fresh "$R/$name/soup"
    for member in "$arm-s1" "$arm-s2"; do
      ckpt=$(best_of "$member")
      case "$member" in M4-A20-s2 | M4-Ar-s1 | M4-Ar-s2) relay_ok "$member" "$ckpt" ;; esac
    done
    echo "expected GPU-hours for soup $name: 0 (CPU-only container)"
    echo "=== $(stamp) soup $name ($arm-s1 + $arm-s2) mirror $SRC" >> "$R/$name.driver.log"
    finalist soup 5 "$arm"
    rank=16 alpha=32
    [ "$arm" = M4-A20r ] && rank=64 alpha=128
    python3 - "$R/$name/soup/checkpoint/soup_manifest.json" "$rank" "$alpha" "$R" <<'EOF' | tee -a "$R/$name.driver.log"
import json, pathlib, sys
path, rank, alpha, runs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), pathlib.Path(sys.argv[4])
manifest = json.load(open(path))
lora, check = manifest["lora"], manifest["verification"]
problems = []
if (lora["members"], lora["rank"], lora["alpha"]) != (2, rank, alpha):
    problems.append(f"soup lora {lora}, expected 2 members, rank {rank}, alpha {alpha}")
if not check["verify_adapter_config"] or check["max_relative_diff"] > check["tolerance_relative"]:
    problems.append(f"soup verification {check['max_relative_diff']} / {check['verify_adapter_config']}")
for member in manifest["members"]:
    ckpt = pathlib.Path(member["path"])
    relay_path = ckpt.parents[2] / "RELAY.json"
    if relay_path.is_file():
        relay = json.loads(relay_path.read_text())
        prefix = ckpt.relative_to(ckpt.parents[2]).as_posix()
        for name, digest in member["files_sha256"].items():
            if relay["files_sha256"].get(f"{prefix}/{name}") != digest:
                problems.append(f"{ckpt}/{name} is not the relayed file")
if problems:
    raise SystemExit("; ".join(problems))
print(json.dumps({"soup": path, "model_sha256": manifest["output"]["model_sha256"], "rank": lora["rank"],
                  "alpha": lora["alpha"], "max_relative_diff": check["max_relative_diff"],
                  "projections": check["projections"]}))
EOF
    ;;
  readout)
    arm=$(arm_of "${2:?ARM}")
    gpu=${3:?GPU}
    gpu_ok "$gpu"
    name=$arm-soup
    [ -f "$R/$name/soup/checkpoint/soup_manifest.json" ] || { echo "no soup for $arm (m4-tail.sh soup $arm)" >&2; exit 2; }
    fresh "$R/$name/readout-kernel-32768"
    budget "soup readout $name" "$COST_READOUT"
    echo "=== $(stamp) readout $name GPU$gpu 32768 cache 583241fb M4_LEFT=$M4_LEFT mirror $SRC" >> "$R/$name.driver.log"
    finalist readout "$gpu" "$arm" FROZEN="$READOUT_CACHE" CACHE_SHA="$READOUT_SHA" LIMIT=32768
    python3 -c 'import json, sys; r = json.load(open(sys.argv[1])); print(json.dumps({"P_dev": r["development_proxy"], "T_dev": r["typed_dev"]["T_dev"], "H_pilot": r["css_pilot"]["H_pilot"]}))' \
      "$R/$name/readout-kernel-32768/READOUT.json"
    ;;
  guard)
    fresh "$GUARD"
    IFS=, read -r -a listed <<< "${M4_ABSENT:-}"
    absent=" "
    for arm in "${listed[@]}"; do absent+="$(arm_of "$arm") "; done
    args=()
    for arm in "${ARMS[@]}"; do
      if [[ "$absent" == *" $arm "* ]]; then
        args+=(--absent "$arm-soup")
        continue
      fi
      readout=$R/$arm-soup/readout-kernel-32768/READOUT.json
      [ -f "$readout" ] || { echo "$arm-soup has no scored readout ($readout); M4_ABSENT names stopped arms" >&2; exit 2; }
      args+=(--candidate "$arm-soup=$readout")
      for seed in s1 s2; do
        args+=(--seed-run "$arm-$seed=$R/$arm-$seed/full/$(cat "$R/$arm-$seed/full/RUN_DIR")")
      done
    done
    mkdir -p "$L"
    echo "=== $(stamp) guard mirror $SRC absent [$absent]" >> "$L/guard.log"
    (cd "$S" && PYTHONPATH=$S python3 -m v2.27b.m4_guard "${args[@]}" \
      --incumbent "$F1_LABEL=$F1/readout-kernel-32768/READOUT.json" --output "$GUARD") 2>&1 | tee -a "$L/guard.log"
    ;;
  prepkg)
    arm=$(arm_of "${2:?ARM}")
    gpu=${3:?GPU}
    gpu_ok "$gpu"
    name=$arm-soup
    need_finalist "$name"
    fresh "$R/$name/cal698" "$R/$name/adopt" "$R/$name/ADOPTION.json"
    budget "CAL698 fit $name" "$COST_CAL698"
    echo "=== $(stamp) cal698,adopt $name GPU$gpu M4_LEFT=$M4_LEFT mirror $SRC" >> "$R/$name.driver.log"
    finalist cal698,adopt "$gpu" "$arm" FROZEN="$READOUT_CACHE" CACHE_SHA="$READOUT_SHA" LIMIT=32768
    python3 -c 'import json, sys; a = json.load(open(sys.argv[1])); print(json.dumps({k: a[k] for k in ("decision", "worsened")}))' \
      "$R/$name/ADOPTION.json"
    ;;
  formal)
    arm=$(arm_of "${2:?ARM}")
    gpu=${3:?GPU}
    gpu_ok "$gpu"
    name=$arm-soup
    need_finalist "$name"
    [ -f "$R/$name/ADOPTION.json" ] || { echo "no $R/$name/ADOPTION.json (m4-tail.sh prepkg $arm GPU)" >&2; exit 2; }
    [ -f "$F1/formal/SEAL.json" ] || { echo "F1's formal run is not sealed" >&2; exit 2; }
    fresh "$R/$name/package" "$R/$name/formal-smoke" "$R/$name/formal"
    budget "formal run $name" "$COST_FORMAL"
    echo "=== $(stamp) package,formal $name GPU$gpu snapshot 03b172f1 M4_LEFT=$M4_LEFT mirror $SRC" >> "$R/$name.driver.log"
    finalist package,formal "$gpu" "$arm" FROZEN="$SNAP" CACHE_SHA="$SNAP_SHA" LIMIT=32768 \
      EXTRA_COMPARATOR="M3-A-soup=$F1/formal"
    python3 - "$R/$name/formal" <<'EOF'
import json, sys
run = sys.argv[1]
report, paired = (json.load(open(f"{run}/{n}")) for n in ("REPORT.json", "PAIRED-vs-M3-A-soup.json"))
print(json.dumps({"v3": report["v3"], "minus_F1": paired["point"]["delta"], "ci95": paired["ci95"],
                  "invalid": {k: v["invalid_or_missing"] for k, v in report["invalid"].items()}}))
EOF
    ;;
  gates)
    G=$R/m4-gates
    fresh "$G"
    list=$(all_sealed)
    read -r -a names <<< "$list"
    mkdir -p "$G"
    cd "$EVAL"
    export PYTHONPATH=$EVAL
    stamp > "$G/run.start"
    python3 -m v2.eval.panels verify --panel typed-final --panel css15 --panel public231 > "$G/panels-verify.json"
    for name in "${names[@]}"; do
      C=$R/$name/formal D=$G/$name
      mkdir -p "$D"
      paired "$C" "$name" "$R/m2-peer-autojev27-nodeB-kernel" AutoJev-27B "$D/paired-vs-autojev27.json"
      paired "$C" "$name" "$R/m3-peer-eikos27-nodeB-kernel" Eikos-27B "$D/paired-vs-eikos27b.json"
      paired "$C" "$name" "$R/m3-peer-jebadiah-nodeB-kernel" Jebadiah-27B "$D/paired-vs-jebadiah27b.json"
      paired "$C" "$name" "$F1/formal" "$F1_LABEL" "$D/paired-vs-F1.json"
      paired "$F1/formal" "$F1_LABEL" "$C" "$name" "$D/paired-F1-minus-cand.json"
      paired "$C" "$name" "$F2/formal" "$F2_LABEL" "$D/paired-vs-F2.json"
      python3 -m v2.eval.gates types --run "$C" --label "$name" --output "$D/types.json" > "$D/types.log"
    done
    for spec in dose:M4-A20-soup:M4-Ar-soup capacity:M4-A20r-soup:M4-A20-soup; do
      IFS=: read -r label left right <<< "$spec"
      if [[ " $list " == *" $left "* && " $list " == *" $right "* ]]; then
        paired "$R/$left/formal" "$left" "$R/$right/formal" "$right" "$G/contrast-$label.json"
      else
        echo "$label contrast ($left - $right) not run: both must be sealed finalists" | tee -a "$G/contrasts.log"
      fi
    done
    cfg=$(python3 - "$R" "$G" "$F1_LABEL" "$F2_LABEL" "${names[@]}" <<'EOF'
import json, sys
runs, out, f1, f2, *names = sys.argv[1:]
peers = {"AutoJev-27B": f"{runs}/m2-peer-autojev27-nodeB-kernel", "Eikos-27B": f"{runs}/m3-peer-eikos27-nodeB-kernel",
         "Jebadiah-27B": f"{runs}/m3-peer-jebadiah-nodeB-kernel"}
models = {**{n: f"{runs}/{n}/formal" for n in names}, f1: f"{runs}/M3-A-soup/formal",
          f2: f"{runs}/M3-S-soup/formal", **peers}
pairs = [[n, other] for n in names for other in (*peers, f1, f2)]
pairs += [[a, b] for a, b in (("M4-A20-soup", "M4-Ar-soup"), ("M4-A20r-soup", "M4-A20-soup")) if a in names and b in names]
print(json.dumps({"models": models, "pairs": pairs, "output": f"{out}/item-cis.json", "jobs": 8}))
EOF
    )
    python3 - "$cfg" > "$G/item-cis.log" < "$ITEM_CIS"
    stamp > "$G/run.end"
    sums=$(cd "$G" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum)
    printf '%s\n' "$sums" > "$G/SHA256SUMS.txt"
    echo "gates for [$list] -> $G"
    ;;
  overlap)
    SPEC=$(cd "$(dirname "${2:?SPEC: v2/27b/m4/overlap-spec-27b-m4.json}")" && pwd)/$(basename "$2")
    [ -f "$SPEC" ] || { echo "no spec $SPEC" >&2; exit 2; }
    V=$R/m4-overlap
    fresh "$V"
    list=$(all_sealed)
    read -r -a names <<< "$list"
    mkdir -p "$V"
    cd "$OVERLAP"
    export PYTHONPATH=$OVERLAP
    stamp > "$V/run.start"
    python3 -m v2.eval.overlap_effects exposure --groups "$OVERLAP_IN/final/excluded-groups.json" \
      --train "$MX/a20.train.jsonl" --expect-sha256 4aa0dc964505682b2840fa5167a7ec14983e5a0cea427480bed1996f1befc0d4 \
      --label "DEV2.0-27B M4 a20.train.jsonl (M4-A20 and M4-A20r soups)" \
      --output "$V/exposure-m4-a20.json" > "$V/exposure-a20.log"
    python3 -m v2.eval.overlap_effects exposure --groups "$OVERLAP_IN/final/excluded-groups.json" \
      --train "$MX/ar.train.jsonl" --expect-sha256 aadeef1a6a9e7aa8cbd7de743ef84c703c90433ac3ffba856edda24b4ad65e81 \
      --label "DEV2.0-27B M4 ar.train.jsonl (M4-Ar soup)" \
      --output "$V/exposure-m4-ar.json" > "$V/exposure-ar.log"
    python3 - "$SPEC" "$V/spec.json" "${names[@]}" <<'EOF'
import hashlib, json, sys
source, target, *names = sys.argv[1:]
spec = json.load(open(source))
tiers = {k: t for k, t in spec["tiers"].items() if t["candidate"] in names}
if sorted(t["candidate"] for t in tiers.values()) != sorted(names):
    raise SystemExit(f"the spec has no tier for some finalists: {names}")
keep = {t["candidate"] for t in tiers.values()}
keep |= {m for t in tiers.values() for role in ("own_1_0", "peers", "internal_peers") for m in t.get(role, [])}
spec["tiers"] = tiers
spec["models"] = {k: v for k, v in spec["models"].items() if k in keep}
spec["reproduce"] = [r for r in spec.get("reproduce", []) if r["left"] in names]
with open(target, "x") as stream:
    json.dump(spec, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps({"spec": source, "sha256": hashlib.sha256(open(source, "rb").read()).hexdigest(),
                  "finalist_tiers": sorted(tiers)}))
EOF
    python3 -m v2.eval.overlap_effects run --spec "$V/spec.json" --flagged "$OVERLAP_IN/final-27b/flagged.json" \
      --output "$V/overlap-effects.json" --jobs 8 > "$V/run.log" 2>&1
    stamp > "$V/run.end"
    python3 -c 'import json, sys; [print(p, json.dumps({k: json.load(open(p))[k] for k in ("groups", "methods_agree")})) for p in sys.argv[1:]]' \
      "$V/exposure-m4-a20.json" "$V/exposure-m4-ar.json"
    ;;
  mlx)
    arm=$(arm_of "${2:?NAME}")
    gpu=${3:?GPU} mode=${4:-full}
    gpu_ok "$gpu"
    name=$arm-soup
    case "$mode" in full) out=$MLX_ROOT/$name cost=$COST_MLX ;; smoke) out=$MLX_ROOT/$name-smoke cost=$COST_MLX_SMOKE ;;
      *) echo "mode is full or smoke" >&2; exit 2 ;; esac
    need_finalist "$name"
    [ -f "$R/$name/formal/SEAL.json" ] || { echo "$name has no sealed formal run" >&2; exit 2; }
    fresh "$out"
    if [ "$mode" = full ] && ! compgen -G "$MLX_ROOT/*-smoke/SMOKE.json" > /dev/null; then
      echo "the first M4 mlx-diag collection is an 8-item smoke (m4-tail.sh mlx NAME GPU smoke)" >&2
      exit 2
    fi
    post=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["post_sha256"])' "$R/$name/formal/triton-cache.post.json")
    budget "mlx-diag $name ($mode)" "$cost" "$COST_FORMAL"
    mkdir -p "$L"
    echo "=== $(stamp) mlx-diag $name $mode GPU$gpu cache $post M4_LEFT=$M4_LEFT mirror $SRC" >> "$L/mlx.log"
    (cd "$S" && SRC=$SRC FROZEN=$R/$name/formal/triton-cache CACHE_SHA=$post MLX_ROOT=$MLX_ROOT \
      bash v2/27b/m3f2/f2-mlx.sh "$name" "$gpu" "$mode") >> "$L/mlx.log" 2>&1
    tail -n 1 "$L/mlx.log"
    ;;
  contrast)
    C=$R/m4-contrast
    fresh "$C"
    list=$(all_sealed)
    read -r -a names <<< "$list"
    [ -f "$R/m4-gates/run.end" ] || { echo "run m4-tail.sh gates first" >&2; exit 2; }
    [ -f "$R/m4-overlap/run.end" ] || { echo "run m4-tail.sh overlap SPEC first" >&2; exit 2; }
    args=(--run "$F1_LABEL=$F1/formal" --run "$F2_LABEL=$F2/formal" --incumbent "$F1_LABEL" --gates "$R/m4-gates")
    for name in "${names[@]}"; do
      exposure="a20"
      if [ "$name" = M4-Ar-soup ]; then exposure="ar"; fi
      args+=(--run "$name=$R/$name/formal" --finalist "$name" --pair "$name:$F1_LABEL"
        --exposure "$name=$R/m4-overlap/exposure-m4-$exposure.json")
    done
    for spec in M4-A20-soup:M4-Ar-soup M4-A20r-soup:M4-A20-soup; do
      if [[ " $list " == *" ${spec%%:*} "* && " $list " == *" ${spec#*:} "* ]]; then
        args+=(--pair "$spec")
      fi
    done
    mkdir -p "$C"
    (cd "$S" && PYTHONPATH=$S python3 -m v2.27b.m4_contrast "${args[@]}" --output "$C/contrast.json") 2>&1 \
      | tee "$C/contrast.log"
    ;;
  *)
    sed -n '2,33p' "$0" >&2
    exit 2
    ;;
esac
