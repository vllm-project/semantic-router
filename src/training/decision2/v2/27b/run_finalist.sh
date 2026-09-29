#!/usr/bin/env bash
# Milestone 3 finalist pipeline of one candidate artifact (node B host side; preregistration "Soups and
# candidate artifacts" and "Finalists and formal runs").
# Usage: run_finalist.sh NAME GPU SRC MEMBER [MEMBER]...
#   NAME    output directory /data/dev2/runs/27b/NAME
#   SRC     mirror directory name under /data/dev2/src
#   MEMBER  an arm-seed run under /data/dev2/runs/27b (full/RUN_DIR and a complete run's BEST.json name its
#           checkpoint). Two or more members: the uniform LoRA soup of their BEST checkpoints. One member (the
#           17:15 rule picked a seed): its BEST checkpoint, no soup and no soup readout; the arm-seed's own
#           kernel readout (MEMBER/readout-kernel-32768) is the development readout.
# Stages (STAGES, default soup,readout,cal698,adopt,package,formal):
#   soup     v2.27b.lora_soup in a CPU-only container (SOUP_CPUS, default 16, so training jobs keep their
#            host CPUs) -> soup/checkpoint (with soup_manifest.json); an existing soup is kept (lora_soup
#            verified it when it was written)
#   readout  run_dev_readout.sh on the soup at LIMIT with a fresh copy of the frozen cache; SELECT700 on the
#            kernel path (no trainer run, no AHO) -> readout-kernel-LIMIT/
#   cal698   v2.release.calibrate_frozen --require-kernels at LIMIT through launch.py, fresh frozen-cache copy
#            -> cal698/calibration.json, cal698/cal.logits.jsonl
#   adopt    v2.release.dev_calibration (host): the readout's typed-DEV / CSS-pilot predictions, re-tempered from
#            the readout's calibration to the CAL698 fit (23:15 rule) -> adopt/dev-calibration.json, ADOPTION.json
#   package  frozen before the formal collection -> package/PACKAGE.json and package/calibration.json: the CAL698
#            fit if adopted, else the T = 1 report (below)
#   formal   run_formal_kernel.sh FORMAL_STAGES (default smoke,collect,score) on the package -> formal/,
#            formal-smoke/; paired comparisons vs AutoJev-27B, Eikos-27B, Jebadiah and EXTRA_COMPARATOR
#            (NAME=RUN_DIR); LOADED_PARAMETERS from the package's safetensors headers
# T = 1: the kernel adapter always passes --calibration to training.model.infer, so a rejected fit is packaged
# as its own binding (model, checkpoint, CAL hashes, limit) with every temperature 1.0 (kernel_readout
# t1-calibration). softmax(z / 1.0) is softmax(z) bit for bit, so the formal answers equal those of a
# qwen-adapter release package without calibration.json (the runtime's temperature 1.0): a later release spec
# sets calibration to null and quotes the rejected temperatures (PACKAGE.json release_calibration).
# GPU-HOURS.json sums the launch receipts and runner GPU-TIME.json files under NAME (and a reused seed readout).
# DRY_RUN=1: see kernel_common.sh; the soup (CPU) still runs, later stages print what their inputs allow.
set -euo pipefail

NAME=$1 GPU=$2 SRC=$3
shift 3
MEMBERS=("$@")
S=/data/dev2/src/$SRC/src/training/decision2
RUNS=/data/dev2/runs/27b
OUT=$RUNS/$NAME
LIMIT=${LIMIT:-32768}
FROZEN=${FROZEN:-$RUNS/m3-warm-32768/triton-cache}
CACHE_SHA=${CACHE_SHA:-583241fbc3bc89e22be51a49722996eab742162356100400d64fb4208cb20daf}
STAGES=${STAGES:-soup,readout,cal698,adopt,package,formal}
LABEL=${LABEL:-$NAME}
COMPARATORS=(
  "AutoJev-27B=$RUNS/m2-peer-autojev27-nodeB-kernel"
  "Eikos-27B=$RUNS/m3-peer-eikos27-nodeB-kernel"
  "Jebadiah=$RUNS/m3-peer-jebadiah-nodeB-kernel"
)
[ -z "${EXTRA_COMPARATOR:-}" ] || COMPARATORS+=("$EXTRA_COMPARATOR")
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "NAME must be one directory name" >&2; exit 2; }
case "$GPU" in 5 | 6 | 7) ;; *) echo "GPU$GPU: launch.py maps only node B GPU5-7" >&2; exit 2 ;; esac
[ ${#MEMBERS[@]} -ge 1 ] || { echo "at least one MEMBER" >&2; exit 2; }
cd "$S"
export PYTHONPATH=$S
source "$S/v2/27b/kernel_common.sh"
has() { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
json() {  # FILE KEY -> value
  python3 -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$1" "$2"
}
best_of() {  # MEMBER -> its BEST checkpoint of a complete run
  python3 - "$RUNS/$1/full" <<'EOF'
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
params() {  # CHECKPOINT -> "LOADED_PARAMETERS<TAB>PARAMETER_SOURCE" from safetensors headers
  python3 - "$1" "$BASE" <<'EOF'
import json, pathlib, struct, sys
from v2.release.build import base_text_parameters
ckpt, base = map(pathlib.Path, sys.argv[1:])
def count(path):
    with path.open("rb") as stream:
        (size,) = struct.unpack("<Q", stream.read(8))
        header = json.loads(stream.read(size))
    total = 0
    for name, meta in header.items():
        if name != "__metadata__":
            n = 1
            for dim in meta["shape"]:
                n *= dim
            total += n
    return total
lora = json.loads((ckpt / "decision_config.json").read_text())["lora"]
backbone = base_text_parameters(base, lora["source_fingerprint"]["files_sha256"], lora["source_kind"])
adapter = count(ckpt / "adapter/adapter_model.safetensors")
head = count(ckpt / "decision_head.safetensors")
print(f"{backbone + adapter + head}\tpinned base text backbone {backbone:,} + LoRA rank {lora['rank']} "
      f"{adapter:,} + head {head:,} (safetensors headers)")
EOF
}

SEEDS=()
for member in "${MEMBERS[@]}"; do SEEDS+=("$(best_of "$member")"); done
if [ ${#MEMBERS[@]} -eq 1 ]; then
  CKPT=${SEEDS[0]}
  READOUT=$RUNS/${MEMBERS[0]}/readout-kernel-$LIMIT
else
  CKPT=$OUT/soup/checkpoint
  READOUT=$OUT/readout-kernel-$LIMIT
fi
need "$BASE" "$FROZEN" "$CAL_FILE"
mkdir -p "$OUT/receipts"
echo "finalist $NAME: checkpoint $CKPT, development readout $READOUT"

if has soup && [ ${#MEMBERS[@]} -ge 2 ]; then
  if [ -f "$CKPT/soup_manifest.json" ]; then
    echo "soup exists: $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['output']['model_sha256'])" "$CKPT/soup_manifest.json")"
  else
    mkdir -p "$OUT/soup"
    args=() mounts=()
    for seed in "${SEEDS[@]}"; do
      args+=(--member "$seed")
      mounts+=(--mount "type=bind,src=$seed,dst=$seed,readonly")
    done
    docker run --rm --name "d2-27b-$NAME-soup" --network none --cpus "${SOUP_CPUS:-16}" \
      -e "OMP_NUM_THREADS=${SOUP_CPUS:-16}" -e HIP_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES= \
      -e PYTHONPATH=/code -e PYTHONDONTWRITEBYTECODE=1 --mount "type=bind,src=$S,dst=/code,readonly" \
      --mount "type=bind,src=$BASE,dst=$BASE,readonly" "${mounts[@]}" \
      --mount "type=bind,src=$OUT/soup,dst=$OUT/soup" -w /code --entrypoint python3 "$IMAGE" \
      -m v2.27b.lora_soup "${args[@]}" --source-path "$BASE" --output "$CKPT" 2>&1 | tee "$OUT/soup/soup.log"
  fi
fi
if has readout && [ ${#MEMBERS[@]} -ge 2 ]; then
  TRAIN_RUN='' AHO='' BASE="$BASE" "$S/v2/27b/run_dev_readout.sh" "$NAME/readout-kernel-$LIMIT" "$GPU" "$SRC" \
    "$CKPT" "$LIMIT" "$FROZEN" "$CACHE_SHA" "$LABEL"
fi
if has cal698; then
  need "$CKPT"
  mkdir -p "$OUT/cal698"
  cache_copy "$FROZEN" "$CACHE_SHA" "$OUT/cal698/triton-cache"
  status=0
  launcher "d2-27b-$NAME-cal698" 0.5 "M3 finalist $NAME CAL698 fit at $LIMIT" "$OUT/receipts/cal698.json" \
    "$OUT/cal698/triton-cache" -- --mount "$CKPT:$CKPT" --mount "$CAL_FILE:/data/cal.jsonl" \
    --mount "$OUT/cal698:$OUT/cal698:rw" -- \
    python3 -m v2.release.calibrate_frozen --checkpoint "$CKPT" --source-path "$BASE" --cal /data/cal.jsonl \
    --cal-sha256 "$CAL_SHA256" --max-length "$LIMIT" --require-kernels \
    --output "$OUT/cal698/calibration.json" --logits "$OUT/cal698/cal.logits.jsonl" || status=$?
  cache_finish "$OUT/cal698/triton-cache"
  [ "$status" = 0 ] || exit "$status"
fi
if has adopt; then
  mkdir -p "$OUT/adopt"
  devcal=(python3 -m v2.release.dev_calibration --label "$LABEL" --panel-root "$PANEL_ROOT"
    --typed-dev "$READOUT/output/typed-dev.predictions.jsonl"
    --css-pilot "$READOUT/output/css-pilot.predictions.jsonl"
    --source-calibration "$READOUT/cal/calibration.json" --candidate "$OUT/cal698/calibration.json"
    --work "$OUT/adopt/work" --output "$OUT/adopt/dev-calibration.json")
  if [ "$DRY_RUN" = 1 ]; then
    dry "${devcal[@]}"
    argcheck "${devcal[@]:2}"
  else
    python3 - "$READOUT" "$OUT/cal698/calibration.json" "$LIMIT" <<'EOF'
import hashlib, json, pathlib, sys
readout, candidate, limit = pathlib.Path(sys.argv[1]), json.load(open(sys.argv[2])), int(sys.argv[3])
source_path = readout / "cal/calibration.json"
source = json.loads(source_path.read_text())
digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
if candidate["model_sha256"] != source["model_sha256"]:
    raise SystemExit("the CAL698 fit and the readout calibration bind different checkpoints")
if candidate["inference"]["max_length"] != limit:
    raise SystemExit("the CAL698 fit is not at the package limit")
for panel in ("typed-dev", "css-pilot"):
    for line in (readout / f"output/{panel}.predictions.jsonl").read_text().splitlines():
        if line.strip() and json.loads(line).get("calibration_sha256") != digest:
            raise SystemExit(f"{panel} predictions were not collected with {source_path}")
EOF
    "${devcal[@]}"
    python3 - "$OUT" "$READOUT" "$CKPT" <<'EOF'
import hashlib, json, pathlib, sys
out, readout, ckpt = sys.argv[1:]
out = pathlib.Path(out)
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
receipt = json.loads((out / "adopt/dev-calibration.json").read_text())
candidate = json.loads((out / "cal698/calibration.json").read_text())
adoption = {
    "schema": "decision2-27b-m3-adoption/1",
    "rule": receipt["rule"],
    "checkpoint": ckpt,
    "model_sha256": candidate["model_sha256"],
    "decision": "CAL698" if receipt["adopt"] else "T = 1",
    "adopt": receipt["adopt"],
    "worsened": receipt["worsened"],
    "criteria": receipt["criteria"],
    "candidate": {"path": str(out / "cal698/calibration.json"), "sha256": receipt["candidate_sha256"],
                  "temperature_by_type": candidate["temperature_by_type"]},
    "source_calibration_sha256": receipt["source_calibration_sha256"],
    "development_readout": readout,
    "receipt": {"path": str(out / "adopt/dev-calibration.json"), "sha256": sha(out / "adopt/dev-calibration.json")},
}
with (out / "ADOPTION.json").open("x") as stream:
    json.dump(adoption, stream, indent=1, sort_keys=True)
    stream.write("\n")
print(json.dumps({k: adoption[k] for k in ("decision", "worsened")}))
EOF
  fi
fi
if has package; then
  if [ -f "$OUT/package/PACKAGE.json" ]; then
    echo "package is frozen: $OUT/package/PACKAGE.json" >&2
  elif [ "$DRY_RUN" = 1 ]; then
    mkdir -p "$OUT/package"
    dry package "$OUT/package/PACKAGE.json" from "$OUT/ADOPTION.json": "$OUT/cal698/calibration.json" or \
      python3 -m v2.27b.kernel_readout t1-calibration --rejected "$OUT/cal698/calibration.json" \
      --adoption "$OUT/adopt/dev-calibration.json" --output "$OUT/package/calibration.json"
  else
    mkdir -p "$OUT/package"
    if [ "$(json "$OUT/ADOPTION.json" adopt)" = True ]; then
      cp "$OUT/cal698/calibration.json" "$OUT/package/calibration.json"
    else
      python3 -m v2.27b.kernel_readout t1-calibration --rejected "$OUT/cal698/calibration.json" \
        --adoption "$OUT/adopt/dev-calibration.json" --output "$OUT/package/calibration.json"
    fi
    counted=$(params "$CKPT")
    IFS=$'\t' read -r loaded source <<< "$counted"
    python3 - "$OUT" "$CKPT" "$LIMIT" "$FROZEN" "$CACHE_SHA" "$loaded" "$source" "${SEEDS[@]}" <<'EOF'
import hashlib, json, pathlib, sys
from datetime import datetime, timezone
out, ckpt, limit, frozen, cache_sha, loaded, source, *seeds = sys.argv[1:]
out = pathlib.Path(out)
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
adoption = json.loads((out / "ADOPTION.json").read_text())
calibration = json.loads((out / "package/calibration.json").read_text())
if calibration["model_sha256"] != adoption["model_sha256"]:
    raise SystemExit("package calibration binds another checkpoint")
package = {
    "schema": "decision2-27b-m3-package/1",
    "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "checkpoint": ckpt,
    "soup": len(seeds) > 1,
    "members": seeds,
    "model_sha256": calibration["model_sha256"],
    "checkpoint_sha256": calibration["checkpoint_sha256"],
    "decision": adoption["decision"],
    "calibration": {"path": str(out / "package/calibration.json"), "sha256": sha(out / "package/calibration.json"),
                    "temperature_by_type": calibration["temperature_by_type"]},
    "release_calibration": (
        {"path": adoption["candidate"]["path"], "sha256": adoption["candidate"]["sha256"]}
        if adoption["adopt"] else None
    ),
    "rejected_temperature_by_type": None if adoption["adopt"] else adoption["candidate"]["temperature_by_type"],
    "max_input_tokens": int(limit),
    "frozen_cache": {"path": frozen, "tree_sha256": cache_sha},
    "adoption_sha256": sha(out / "ADOPTION.json"),
    "loaded_parameters": int(loaded),
    "parameter_source": source,
}
with (out / "package/PACKAGE.json").open("x") as stream:
    json.dump(package, stream, indent=1, sort_keys=True)
    stream.write("\n")
print(json.dumps({k: package[k] for k in ("decision", "model_sha256", "loaded_parameters")}))
EOF
  fi
fi
if has formal; then
  if [ -f "$OUT/package/PACKAGE.json" ]; then
    frozen=$(python3 - "$OUT/package/PACKAGE.json" "$CKPT" <<'EOF'
import hashlib, json, sys
package = json.load(open(sys.argv[1]))
if package["checkpoint"] != sys.argv[2]:
    raise SystemExit("PACKAGE.json names another checkpoint")
path = package["calibration"]["path"]
if hashlib.sha256(open(path, "rb").read()).hexdigest() != package["calibration"]["sha256"]:
    raise SystemExit("package calibration changed after the freeze")
print(path, package["loaded_parameters"], package["parameter_source"], sep="\t")
EOF
    )
    IFS=$'\t' read -r cal loaded source <<< "$frozen"
  elif [ "$DRY_RUN" = 1 ]; then
    mkdir -p "$OUT/package"
    cal=$OUT/package/calibration.json
    loaded=0 source="dry run"
    if [ -d "$CKPT" ]; then
      counted=$(params "$CKPT")
      IFS=$'\t' read -r loaded source <<< "$counted"
    fi
  else
    echo "formal needs a frozen package (package stage)" >&2
    exit 2
  fi
  echo "loaded parameters $loaded: $source"
  LOADED_PARAMETERS=$loaded PARAMETER_SOURCE=$source "$S/v2/27b/run_formal_kernel.sh" "$NAME/formal" "$GPU" \
    "$SRC" "$CKPT" "$cal" "$LIMIT" "$LABEL" "${FORMAL_STAGES:-smoke,collect,score}" "$FROZEN" "$CACHE_SHA" \
    "${COMPARATORS[@]}"
fi

python3 - "$OUT" "$READOUT" <<'EOF'
import json, pathlib, sys
out, readout = map(pathlib.Path, sys.argv[1:])
roots = [out] + ([readout] if not readout.is_relative_to(out) else [])
items = []
for root in roots:
    for path in sorted(root.rglob("*.json")):
        if path.name != "GPU-TIME.json" and path.parent.name != "receipts":
            continue
        record = json.loads(path.read_text())
        if record.get("schema") == "dev2-gpu-time/1" or record.get("schema_version") == "decision2-27b-launch-receipt/1":
            items.append({"file": str(path), "gpu_hours": record["gpu_hours"],
                          "reused": root == readout and root != out})
total = sum(i["gpu_hours"] for i in items if not i["reused"])
(out / "GPU-HOURS.json").write_text(json.dumps(
    {"schema": "decision2-27b-m3-finalist-gpu-hours/1", "gpu_hours": total,
     "reused_readout_gpu_hours": sum(i["gpu_hours"] for i in items if i["reused"]), "items": items},
    indent=1, sort_keys=True) + "\n")
print(json.dumps({"finalist_gpu_hours": round(total, 4)}))
EOF
echo "finalist $NAME stages $STAGES complete"
