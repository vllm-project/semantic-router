#!/usr/bin/env bash
# M4b finalist package and formal post-key same-panel run of one full checkpoint (node B host side, one GPU
# of GPU0-2; preregistration "Finalists (at most three) and formal runs"). run_finalist.sh's cal698 / adopt /
# package stages and run_formal_kernel.sh's collection, seal, report and paired compares, with launch3.py and
# track 27b-m4b (m4b/common3.sh).
# Usage: run_formal.sh NAME CKPT GPU MIRROR_SHA
#   NAME        finalist slot, output directory /data/dev2/runs/27b/m4b/NAME (F-a, F-b, F-c; RULES.json)
#   CKPT        the finalist's full checkpoint; READOUT (default /data/dev2/runs/27b/m4b/readouts/NAME) is
#               its development readout, whose READOUT-M4B.json must name CKPT. CHECKPOINT_FORMAT=peft-lora/1
#               takes a LoRA checkpoint instead (M5's L128 soup); LOADED_PARAMETERS is then required
#   MIRROR_SHA  code commit; the mirror /data/dev2/src/<sha>[-src_training_decision2]
# Stages (STAGES, default verify,cal698,adopt,package,smoke,collect,score):
#   verify   mirror record, inputs, both frozen caches' tree hashes, sealed comparators, the GPU lease
#   cal698   v2.release.calibrate_frozen --require-kernels at 32,768 via launch3.py on a fresh copy of the
#            readout cache 583241fb -> cal698/calibration.json, cal698/cal.logits.jsonl
#   adopt    v2.release.dev_calibration (host): the readout's typed-DEV / CSS-pilot predictions re-tempered to
#            the CAL698 fit (23:15 rule) -> adopt/dev-calibration.json, ADOPTION.json
#   package  frozen before any formal collection -> package/PACKAGE.json, package/calibration.json (the CAL698
#            fit if adopted, else its T = 1 binding, kernel_readout t1-calibration); loaded parameters from the
#            safetensors headers must equal LOADED_PARAMETERS (25,629,863,936)
#   smoke    8 items per formal panel -> formal-smoke/
#   collect  typed FINAL, CSS15, public 231 -> formal/
#            both on their own fresh verified copy of F1's scored post-run cache 03b172f1 (FORMAL_FROZEN,
#            FORMAL_CACHE_SHA); the post-run check must add no autotune entry
#   score    seal, report (tier 27B, LOADED_PARAMETERS), paired compares (5,000 draws) vs F1 (M3-A-soup),
#            AutoJev-27B, Eikos-27B, Jebadiah and EXTRA_COMPARATOR ("NAME=RUN_DIR ...", e.g. another finalist)
# GPU-HOURS.json sums the launch receipts and runner GPU-TIME.json files under NAME. DRY_RUN=1: kernel_common.sh.
set -euo pipefail
echo "m4b formal $*: start $(date -u +%Y-%m-%dT%H:%M:%SZ)"

NAME=${1:?NAME} CKPT=${2:?CKPT} GPU=${3:?GPU} MIRROR_SHA=${4:?MIRROR_SHA}
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "NAME must be one directory name" >&2; exit 2; }
[[ "$MIRROR_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
SRC=$MIRROR_SHA
[ -d "/data/dev2/src/$SRC" ] || SRC=$MIRROR_SHA-src_training_decision2
S=/data/dev2/src/$SRC/src/training/decision2
R=/data/dev2/runs/27b
LIMIT=32768
FROZEN=${FROZEN:-$R/m3-warm-32768/triton-cache}
CACHE_SHA=${CACHE_SHA:-583241fbc3bc89e22be51a49722996eab742162356100400d64fb4208cb20daf}
FORMAL_FROZEN=${FORMAL_FROZEN:-$R/m3-f2/f1-scored-cache}
FORMAL_CACHE_SHA=${FORMAL_CACHE_SHA:-03b172f1a6adeef6c6a6c491d04389b355c9d8579480008023f408c8659b502b}
CHECKPOINT_FORMAT=${CHECKPOINT_FORMAT:-full}
case "$CHECKPOINT_FORMAT" in
  full) LOADED_PARAMETERS=${LOADED_PARAMETERS:-25629863936} ;;
  peft-lora/1) : "${LOADED_PARAMETERS:?a LoRA checkpoint needs LOADED_PARAMETERS}" ;;
  *) echo "CHECKPOINT_FORMAT is full or peft-lora/1" >&2; exit 2 ;;
esac
STAGES=${STAGES:-verify,cal698,adopt,package,smoke,collect,score}
LABEL=${LABEL:-DEV2.0-27B (M4b $NAME)}
COMPARATORS=(
  "M3-A-soup=$R/M3-A-soup/formal"
  "AutoJev-27B=$R/m2-peer-autojev27-nodeB-kernel"
  "Eikos-27B=$R/m3-peer-eikos27-nodeB-kernel"
  "Jebadiah=$R/m3-peer-jebadiah-nodeB-kernel"
)
read -r -a extra <<< "${EXTRA_COMPARATOR:-}"
for spec in "${extra[@]}"; do
  [[ "$spec" == ?*=/?* ]] || { echo "EXTRA_COMPARATOR entries are NAME=RUN_DIR" >&2; exit 2; }
  COMPARATORS+=("$spec")
done
cd "$S"
export PYTHONPATH=$S PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/dev2/tmp
source "$S/v2/27b/kernel_common.sh"
source "$S/v2/27b/m4b/common3.sh"
OUT=$M4B/$NAME
READOUT=${READOUT:-$M4B/readouts/$NAME}
JOB=d2-m4b-$NAME
has() { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
json() {  # FILE KEY -> value
  python3 -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$1" "$2"
}
params() {  # CHECKPOINT -> "LOADED_PARAMETERS<TAB>PARAMETER_SOURCE" from safetensors headers
  python3 -m v2.27b.m4b.ckpt_format params --checkpoint "$1" --format "$CHECKPOINT_FORMAT" --base "$BASE"
}
check_readout() {
  python3 - "$READOUT/READOUT-M4B.json" "$CKPT" <<'EOF'
import json, os, sys
summary = json.load(open(sys.argv[1]))
if os.path.realpath(summary["checkpoint"] or "") != os.path.realpath(sys.argv[2]):
    raise SystemExit(f"{sys.argv[1]} reads out {summary['checkpoint']}, not {sys.argv[2]}")
print(f"development readout {summary['readout_dir']}: P_dev {summary['P_dev']:.4f}")
EOF
}
collect() {  # RUN_DIR [--max-items N]
  local dir=$1 status=0
  shift
  mkdir -p "$dir"
  cache_copy "$FORMAL_FROZEN" "$FORMAL_CACHE_SHA" "$dir/triton-cache"
  runner "$dir" "$CKPT" "$dir/triton-cache" "27b-m4b $NAME kernel-path post-key same-panel collection" \
    --mount "$(dirname "$CAL")" -- --revision "$REVISION" --extra "source=$BASE" \
    --extra "calibration=$CAL" --extra "max_length=$LIMIT" "$@" || status=$?
  cache_finish "$dir/triton-cache"
  [ "$status" = 0 ] || return "$status"
  [ "$DRY_RUN" = 1 ] || no_autotune_added "$dir/triton-cache"
}

if has verify; then
  verify_mirror "$MIRROR_SHA"
  need "$CKPT" "$BASE" "$CAL_FILE" "$FROZEN" "$FORMAL_FROZEN" "$PANEL_ROOT" "$READOUT/READOUT.json"
  check_readout
  verify_cache "$FROZEN" "$CACHE_SHA"
  verify_cache "$FORMAL_FROZEN" "$FORMAL_CACHE_SHA"
  counted=$(params "$CKPT")
  echo "loaded parameters ${counted%%$'\t'*} (expected $LOADED_PARAMETERS)"
  for spec in "${COMPARATORS[@]}"; do
    [ -f "${spec#*=}/SEAL.json" ] || echo "comparator ${spec%%=*} is not sealed yet: ${spec#*=}" >&2
  done
  [ "$DRY_RUN" = 1 ] || verify_lease
fi
mkdir -p "$OUT/receipts"

if has cal698; then
  need "$CKPT"
  mkdir -p "$OUT/cal698"
  cache_copy "$FROZEN" "$CACHE_SHA" "$OUT/cal698/triton-cache"
  status=0
  launcher "$JOB-cal698" 0.5 "M4b finalist $NAME CAL698 fit at $LIMIT" "$OUT/receipts/cal698.json" \
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
    check_readout
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
    "schema": "decision2-27b-m4b-adoption/1",
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
    counted=$(params "$CKPT")
    IFS=$'\t' read -r loaded source <<< "$counted"
    [ "$loaded" = "$LOADED_PARAMETERS" ] || { echo "loaded parameters $loaded != $LOADED_PARAMETERS" >&2; exit 1; }
    mkdir -p "$OUT/package"
    if [ "$(json "$OUT/ADOPTION.json" adopt)" = True ]; then
      cp "$OUT/cal698/calibration.json" "$OUT/package/calibration.json"
    else
      python3 -m v2.27b.kernel_readout t1-calibration --rejected "$OUT/cal698/calibration.json" \
        --adoption "$OUT/adopt/dev-calibration.json" --output "$OUT/package/calibration.json"
    fi
    python3 - "$OUT" "$CKPT" "$LIMIT" "$FORMAL_FROZEN" "$FORMAL_CACHE_SHA" "$loaded" "$source" "$CHECKPOINT_FORMAT" <<'EOF'
import hashlib, json, pathlib, sys
from datetime import datetime, timezone
out, ckpt, limit, frozen, cache_sha, loaded, source, checkpoint_format = sys.argv[1:]
out = pathlib.Path(out)
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
adoption = json.loads((out / "ADOPTION.json").read_text())
calibration = json.loads((out / "package/calibration.json").read_text())
if calibration["model_sha256"] != adoption["model_sha256"]:
    raise SystemExit("package calibration binds another checkpoint")
package = {
    "schema": "decision2-27b-m4b-package/1",
    "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "checkpoint": ckpt,
    "checkpoint_format": checkpoint_format,
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
if has smoke || has collect || has score; then
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
    IFS=$'\t' read -r CAL loaded PARAMETER_SOURCE <<< "$frozen"
    [ "$loaded" = "$LOADED_PARAMETERS" ] || { echo "PACKAGE.json loaded parameters $loaded" >&2; exit 1; }
  elif [ "$DRY_RUN" = 1 ]; then
    CAL=$OUT/package/calibration.json PARAMETER_SOURCE="dry run"
  else
    echo "the formal stages need a frozen package (package stage)" >&2
    exit 2
  fi
  REVISION="checkpoint-sha256:$(model_sha "$CAL")"
fi
if has smoke; then
  collect "$OUT/formal-smoke" --max-items 8
fi
if has collect; then
  collect "$OUT/formal"
fi
if has score; then
  for spec in "${COMPARATORS[@]}"; do
    [ -f "${spec#*=}/SEAL.json" ] && continue
    [ "$DRY_RUN" = 1 ] || { echo "comparator ${spec%%=*} is not sealed: ${spec#*=}" >&2; exit 2; }
    echo "dry-run: comparator ${spec%%=*} is not sealed yet: ${spec#*=}" >&2
  done
  score() {  # same_panel ARGS...: run, or print and argcheck with DRY_RUN=1
    dry python3 -m v2.eval.same_panel "$@"
    [ "$DRY_RUN" != 1 ] || argcheck v2.eval.same_panel "$@"
  }
  [ -f "$OUT/formal/SEAL.json" ] || score seal --run-dir "$OUT/formal"
  score report --run-dir "$OUT/formal" --label "$LABEL" --tier 27B \
    --family decision2 --model-id llm-semantic-router/DEV2.0-27B --revision "$REVISION" \
    --loaded-parameters "$LOADED_PARAMETERS" --parameter-source "$PARAMETER_SOURCE"
  for spec in "${COMPARATORS[@]}"; do
    score compare --run-dir "$OUT/formal" --comparator-run-dir "${spec#*=}" \
      --left-name "$LABEL" --right-name "${spec%%=*}"
  done
fi

python3 - "$OUT" <<'EOF'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
items = []
for path in sorted(out.rglob("*.json")):
    if path.name != "GPU-TIME.json" and path.parent.name != "receipts":
        continue
    record = json.loads(path.read_text())
    if record.get("schema") == "dev2-gpu-time/1" or record.get("track") == "27b-m4b":
        items.append({"file": str(path), "gpu_hours": record["gpu_hours"]})
total = sum(i["gpu_hours"] for i in items)
(out / "GPU-HOURS.json").write_text(json.dumps(
    {"schema": "decision2-27b-m4b-finalist-gpu-hours/1", "gpu_hours": total, "items": items},
    indent=1, sort_keys=True) + "\n")
print(json.dumps({"finalist_gpu_hours": round(total, 4)}))
EOF
echo "m4b formal $NAME stages $STAGES complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
