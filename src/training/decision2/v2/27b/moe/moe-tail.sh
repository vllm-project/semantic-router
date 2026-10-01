#!/usr/bin/env bash
# 27B MoE milestone Stage B (host side): the preregistered steps after the screen, one stage per call.
# Usage: moe-tail.sh STAGE MIRROR ARGS...   (MIRROR: the mirror directory name under /data/dev2/src)
#   relay ARM...        node A: each finished arm-seed's BEST checkpoint (BEST.json; no trainer_state.pt) with a SHA-256
#                       list -> node B /data/dev2/xfer/27b-moe/relay/ARM-best/ over the temporary link; node A's receipt
#                       total -> relay/BUDGET-nodeA.json
#   soup NAME ARM...    node B: v2.27b.lora_soup over the relayed BEST checkpoints (CPU container; exact uniform rank
#                       concatenation, head averaged) -> R/NAME/checkpoint; members = the relayed files, rank / alpha
#                       = the members' sums, verification within tolerance
#   readout NAME        node B GPU7: moe-readout.sh at T = 1 (typed DEV, CSS pilot, HT-DEV v2 vs A20r's predictions)
#                       -> R/readouts/NAME
#   cal698 NAME         node B GPU7: v2.release.calibrate_frozen at 32,768 (the checkpoint's own prompt encoder)
#                       -> R/NAME/cal698/calibration.json
#   adopt NAME          node B host: v2.release.dev_calibration on the T = 1 readout (23:15 rule) -> R/NAME/ADOPTION.json
#   devgates NAME...    node B host: moe_devgates.py vs A20r's readout -> R/readouts/DEVGATES-<UTC>.json
#   package NAME        node B host: the frozen package, before any formal collection -> R/NAME/package/PACKAGE.json
#                       (adopted CAL698 fit or its T = 1 binding; loaded / active parameters; base and kernel identity)
#   formal NAME         node B GPU7: moe-formal.sh smoke,collect,score with the package calibration -> R/formal/NAME
#   gates NAME          node B host: v2.eval.gates paired / types / public231 and the a20 overlap exposure -> R/gates
#   mlx NAME            node B GPU7: moe-formal.sh mlx -> R/formal/NAME-mlx; staged in X/mlx/NAME (+ NAME.PUSHED)
#   mlx-score NAME      node A host: pull X/mlx/NAME, score mlx-diag-v1, mlx-paired vs A20r's node A collection;
#                       the pairing -> node B X/mlx/NAME-vs-A20r.json
#   mlx-pull NAME       node B host: X/mlx/NAME-vs-A20r.json -> R/gates/mlx/NAME-vs-A20r.json
#   latency NAME        node B GPU7: latency.py on the package checkpoint (M1's SELECT roster) -> R/NAME/latency.json
#   latency-ref         node B GPU7: the same measurement of A20r (dense, FLA kernel path) -> R/latency-ref/A20r
#   verdicts NAME...    node B host: moe_verdicts.py -> R/gates/VERDICTS-<UTC>.json
# Development readouts are never release scores; formal numbers are post-key same-panel.
# MOE_ROOT (default /data/dev2/runs/27b-moe) moves the cal698 / adopt / devgates / package / latency outputs for a path
# check; readout, formal and mlx always write under /data/dev2/runs/27b-moe.
set -euo pipefail
echo "moe tail $*: start $(date -u +%FT%TZ)"
STAGE=${1:?STAGE} MIR=${2:?MIRROR}
shift 2
S=/data/dev2/src/$MIR/src/training/decision2
[ -f "$S/v2/27b/moe/moe-tail.sh" ] || { echo "missing mirror $MIR" >&2; exit 2; }
R=${MOE_ROOT:-/data/dev2/runs/27b-moe}
X=/data/dev2/xfer/27b-moe
BASE_NAME=gemma-4-26B-A4B-it
BASE=/data/dev2/models/moe/$BASE_NAME
BASE_REPO=google/gemma-4-26B-A4B-it
BASE_REV=4d7ae4984b7db7de8f8457170b3f1a419ee76d52
BASE_TREE=b05e076d09618da6071bcda007e8702e9b4ab4badd999189a64a8bb589753a31
A20R_REF=/data/dev2/runs/27b/m5/readouts/A20r-ref
A20R_RUN=/data/dev2/runs/27b/M4-A20r-soup/formal
CAL_FILE=/data/dev2/hf-cache/datasets--llm-semantic-router--decision-2.0-training-data/snapshots/ed87a03ab80ca5b9560780bba51a83a77ff47d14/m2/cal/CAL698/cal.jsonl
CAL_SHA256=19cc1a8c4ebe6fd13031079f6b7d131046ce503f435f894e3ea672f91c2ed41f
SELECT_ROWS=/data/decision20-20260926/data/rights_clean_goemotions_v2/select.jsonl
MIX=/data/dev2/private/27b/m4-data/mixtures-m4-1/a20.train.jsonl
MIX_SHA=4aa0dc964505682b2840fa5167a7ec14983e5a0cea427480bed1996f1befc0d4
FLAGGED=/data/dev2/runs/eval/m5/overlap-effects/final/excluded-groups.json
PANEL_ROOT=/data/dev2/private/panels
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
LIMIT=32768
KEY=/data/dev2/tmp/27b-moe-xfer
cd "$S"
export PYTHONPATH=$S PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/dev2/tmp
name_ok() { [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "bad NAME $1" >&2; exit 2; }; }
json() { python3 -c "import functools,json,sys; print(functools.reduce(lambda v, k: v[k], sys.argv[2:], json.load(open(sys.argv[1]))))" "$@"; }
rs() { rsync -a --mkpath -e "ssh -i $KEY/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=$KEY/known_hosts -o StrictHostKeyChecking=yes -o BatchMode=yes" "$@"; }
launch() {  # NAME JOB CAP PURPOSE MOUNTS... -- ARGV...
  local name=$1 job=$2 cap=$3 purpose=$4
  shift 4
  mkdir -p "$R/$name/receipts"
  DEV2_NODE=b python3 -m v2.27b.moe.launch --name "d2-27b-moe-$name-$job" --gpu 7 --cap-hours "$cap" \
    --purpose "27b-moe $name $purpose" --receipt "$R/$name/receipts/$job.json" --mount "$S:/code" "$@"
  [ "$(json "$R/$name/receipts/$job.json" exit_code)" = 0 ] || { echo "$name $job failed" >&2; exit 1; }
}
package_field() { json "$R/$1/package/PACKAGE.json" "${@:2}"; }
frozen_check() {  # NAME: the package's calibration is unchanged since the freeze
  python3 - "$R/$1/package/PACKAGE.json" <<'EOF'
import hashlib, json, sys
package = json.load(open(sys.argv[1]))
path = package["calibration"]["path"]
if hashlib.sha256(open(path, "rb").read()).hexdigest() != package["calibration"]["sha256"]:
    raise SystemExit("package calibration changed after the freeze")
EOF
}

case "$STAGE" in
  relay)
    [ $# -ge 1 ] || { echo "relay ARM..." >&2; exit 2; }
    PEER=$(cat "$KEY/peer")
    for arm in "$@"; do
      name_ok "$arm"
      run=$R/$arm/full/run out=/data/dev2/tmp/27b-moe-relay/$arm-best
      [ -f "$run/COMPLETE.json" ] || { echo "$arm has not finished" >&2; exit 2; }
      best=$(json "$run/BEST.json" checkpoint)
      [ -f "$run/$best/decision_config.json" ] || { echo "$arm BEST $best is missing" >&2; exit 2; }
      rm -rf "$out" && mkdir -p "$out/$best"
      rsync -a --exclude trainer_state.pt "$run/$best/" "$out/$best/"
      cp -p "$run/BEST.json" "$run/COMPLETE.json" "$out/"
      (cd "$out/$best" && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$out/SHA256SUMS"
      rs "$out/" "root@$PEER:relay/$arm-best/"
      date -u +%FT%TZ > "$out/RELAYED"
      rs "$out/RELAYED" "root@$PEER:relay/$arm-best/RELAYED"
      echo "relayed $arm $best ($(wc -l < "$out/SHA256SUMS") files)"
    done
    python3 - "$R" > /data/dev2/tmp/27b-moe-relay/BUDGET-nodeA.json <<'EOF'
import glob, json, sys
from datetime import datetime, timezone
items = {p: json.load(open(p)).get("gpu_hours", 0) for p in sorted(glob.glob(f"{sys.argv[1]}/*/receipts/*.json"))}
print(json.dumps({"node": "a", "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "gpu_hours": round(sum(items.values()), 4), "receipts": len(items)}))
EOF
    rs /data/dev2/tmp/27b-moe-relay/BUDGET-nodeA.json "root@$PEER:relay/BUDGET-nodeA.json" ;;
  soup)
    NAME=${1:?NAME}; shift; name_ok "$NAME"
    [ $# -ge 2 ] || { echo "soup NAME ARM ARM..." >&2; exit 2; }
    OUT=$R/$NAME
    [ ! -e "$OUT/checkpoint" ] || { echo "$OUT/checkpoint exists: refusing to overwrite" >&2; exit 66; }
    members=() mounts=() relays=()
    for arm in "$@"; do
      d=$X/relay/$arm-best
      [ -f "$d/RELAYED" ] || { echo "$arm-best has not been relayed" >&2; exit 2; }
      best=$(json "$d/BEST.json" checkpoint)
      (cd "$d/$best" && sha256sum -c --quiet ../SHA256SUMS)
      python3 -m v2.27b.m4b.ckpt_format check --checkpoint "$d/$best" --format peft-lora/1
      members+=(--member "$d/$best")
      mounts+=(--mount "type=bind,src=$d/$best,dst=$d/$best,readonly")
      relays+=("$d/$best")
    done
    mkdir -p "$OUT/soup"
    docker run --rm --name "d2-27b-moe-$NAME-soup" --network none --cpus 16 -e OMP_NUM_THREADS=16 \
      -e HIP_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES= -e PYTHONPATH=/code -e PYTHONDONTWRITEBYTECODE=1 \
      --mount "type=bind,src=$S,dst=/code,readonly" --mount "type=bind,src=$BASE,dst=$BASE,readonly" "${mounts[@]}" \
      --mount "type=bind,src=$OUT,dst=$OUT" -w /code --entrypoint python3 "$IMAGE" \
      -m v2.27b.lora_soup "${members[@]}" --source-path "$BASE" --output "$OUT/checkpoint" 2>&1 | tee "$OUT/soup/soup.log"
    python3 - "$OUT/checkpoint/soup_manifest.json" "${relays[@]}" <<'EOF' | tee "$OUT/soup/check.json"
import json, pathlib, sys
manifest = json.load(open(sys.argv[1]))
members = [pathlib.Path(p) for p in sys.argv[2:]]
configs = [json.loads((m / "decision_config.json").read_text())["lora"] for m in members]
lora, check = manifest["lora"], manifest["verification"]
expect = (len(members), sum(c["rank"] for c in configs), sum(c["alpha"] for c in configs))
problems = []
if (lora["members"], lora["rank"], lora["alpha"]) != expect:
    problems.append(f"soup lora {lora}, expected members / rank / alpha {expect}")
if not check["verify_adapter_config"] or check["max_relative_diff"] > check["tolerance_relative"]:
    problems.append(f"soup verification {check['max_relative_diff']} / {check['verify_adapter_config']}")
if [pathlib.Path(m["path"]) for m in manifest["members"]] != members:
    problems.append("soup_manifest.json lists other members")
for member in manifest["members"]:
    sums = pathlib.Path(member["path"]).parent / "SHA256SUMS"
    listed = {line.split(None, 1)[1].strip().removeprefix("./"): line.split(None, 1)[0]
              for line in sums.read_text().splitlines() if line.strip()}
    for name, digest in member["files_sha256"].items():
        if listed.get(name) != digest:
            problems.append(f"{member['path']}/{name} is not the relayed file")
if problems:
    raise SystemExit("; ".join(problems))
print(json.dumps({"soup": sys.argv[1], "model_sha256": manifest["output"]["model_sha256"], "members": lora["members"],
                  "rank": lora["rank"], "alpha": lora["alpha"], "max_relative_diff": check["max_relative_diff"],
                  "projections": check["projections"]}))
EOF
    ;;
  readout)
    NAME=${1:?NAME}; name_ok "$NAME"
    bash "$S/v2/27b/moe/moe-readout.sh" b 7 "$NAME" "$R/$NAME/checkpoint" "$BASE" "$MIR" \
      "$A20R_REF/output/ht-dev2.predictions.jsonl" ;;
  cal698)
    NAME=${1:?NAME}; name_ok "$NAME"
    CKPT=$R/$NAME/checkpoint OUT=$R/$NAME/cal698
    [ ! -e "$OUT/calibration.json" ] || { echo "$OUT/calibration.json exists" >&2; exit 66; }
    [ "$(sha256sum < "$CAL_FILE" | cut -c1-64)" = "$CAL_SHA256" ] || { echo "CAL698 changed" >&2; exit 2; }
    mkdir -p "$OUT"
    launch "$NAME" cal698 0.5 "CAL698 fit at $LIMIT" --mount "$BASE:$BASE" --mount "$CKPT:$CKPT" \
      --mount "$CAL_FILE:/data/cal.jsonl" --mount "$OUT:$OUT:rw" --env PYTHONPATH=/code -- \
      python3 -m v2.release.calibrate_frozen --checkpoint "$CKPT" --source-path "$BASE" --cal /data/cal.jsonl \
      --cal-sha256 "$CAL_SHA256" --max-length "$LIMIT" --output "$OUT/calibration.json" --logits "$OUT/cal.logits.jsonl"
    json "$OUT/calibration.json" temperature_by_type ;;
  adopt)
    NAME=${1:?NAME}; name_ok "$NAME"
    OUT=$R/$NAME RD=$R/readouts/$NAME
    [ ! -e "$OUT/ADOPTION.json" ] || { echo "$OUT/ADOPTION.json exists" >&2; exit 66; }
    python3 - "$RD" "$OUT/cal698/calibration.json" "$LIMIT" <<'EOF'
import json, pathlib, sys
readout, candidate, limit = pathlib.Path(sys.argv[1]), json.load(open(sys.argv[2])), int(sys.argv[3])
if candidate["inference"]["max_length"] != limit:
    raise SystemExit("the CAL698 fit is not at the package limit")
for panel in ("typed-dev", "css-pilot"):
    for line in (readout / f"output/{panel}.predictions.jsonl").read_text().splitlines():
        row = json.loads(line) if line.strip() else None
        if row is None:
            continue
        if row.get("model_sha256") != candidate["model_sha256"]:
            raise SystemExit(f"{panel} predictions and the CAL698 fit bind different checkpoints")
        if row.get("calibration_sha256") is not None:
            raise SystemExit(f"{panel} predictions were not collected at T = 1")
EOF
    mkdir -p "$OUT/adopt"
    python3 -m v2.release.dev_calibration --label "$NAME" --panel-root "$PANEL_ROOT" \
      --typed-dev "$RD/output/typed-dev.predictions.jsonl" --css-pilot "$RD/output/css-pilot.predictions.jsonl" \
      --candidate "$OUT/cal698/calibration.json" --work "$OUT/adopt/work" --output "$OUT/adopt/dev-calibration.json"
    python3 - "$OUT" "$RD" "$OUT/checkpoint" <<'EOF'
import hashlib, json, pathlib, sys
out, readout, ckpt = sys.argv[1:]
out = pathlib.Path(out)
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
receipt = json.loads((out / "adopt/dev-calibration.json").read_text())
candidate = json.loads((out / "cal698/calibration.json").read_text())
adoption = {
    "schema": "decision2-27b-moe-adoption/1",
    "rule": receipt["rule"],
    "checkpoint": ckpt,
    "model_sha256": candidate["model_sha256"],
    "decision": "CAL698" if receipt["adopt"] else "T = 1",
    "adopt": receipt["adopt"],
    "worsened": receipt["worsened"],
    "criteria": receipt["criteria"],
    "candidate": {"path": str(out / "cal698/calibration.json"), "sha256": receipt["candidate_sha256"],
                  "temperature_by_type": candidate["temperature_by_type"]},
    "development_readout": readout,
    "receipt": {"path": str(out / "adopt/dev-calibration.json"), "sha256": sha(out / "adopt/dev-calibration.json")},
}
with (out / "ADOPTION.json").open("x") as stream:
    json.dump(adoption, stream, indent=1, sort_keys=True)
    stream.write("\n")
print(json.dumps({k: adoption[k] for k in ("decision", "worsened")}))
EOF
    ;;
  devgates)
    [ $# -ge 1 ] || { echo "devgates NAME..." >&2; exit 2; }
    cands=()
    for NAME in "$@"; do name_ok "$NAME"; cands+=(--candidate "$NAME=$R/readouts/$NAME"); done
    OUTPUT=$R/readouts/DEVGATES-$(date -u +%Y%m%dT%H%M%SZ).json
    python3 -m v2.27b.moe.moe_devgates --reference "A20r=$A20R_REF" "${cands[@]}" --output "$OUTPUT"
    ln -sfn "$(basename "$OUTPUT")" "$R/readouts/DEVGATES.json" ;;
  package)
    NAME=${1:?NAME}; name_ok "$NAME"
    OUT=$R/$NAME CKPT=$R/$NAME/checkpoint
    [ ! -e "$OUT/package/PACKAGE.json" ] || { echo "package is frozen: $OUT/package/PACKAGE.json" >&2; exit 66; }
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if sys.argv[2] in d['finalists'] else 3)" \
      "$R/readouts/DEVGATES.json" "$NAME" || { echo "$NAME did not pass the development gates" >&2; exit 3; }
    mkdir -p "$OUT/package"
    python3 -m v2.27b.moe.moe_params --checkpoint "$CKPT" --base "$BASE" --output "$OUT/package/params.json"
    if [ "$(json "$OUT/ADOPTION.json" adopt)" = True ]; then
      cp "$OUT/cal698/calibration.json" "$OUT/package/calibration.json"
    else
      python3 -m v2.27b.kernel_readout t1-calibration --rejected "$OUT/cal698/calibration.json" \
        --adoption "$OUT/adopt/dev-calibration.json" --output "$OUT/package/calibration.json"
    fi
    python3 - "$OUT" "$CKPT" "$LIMIT" "$BASE_REPO" "$BASE_REV" "$BASE_TREE" "$BASE" "$(readlink -f "$R/readouts/DEVGATES.json")" <<'EOF'
import hashlib, json, pathlib, sys
from datetime import datetime, timezone
out, ckpt, limit, repo, rev, tree, base, devgates = sys.argv[1:]
out = pathlib.Path(out)
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
adoption = json.loads((out / "ADOPTION.json").read_text())
calibration = json.loads((out / "package/calibration.json").read_text())
params = json.loads((out / "package/params.json").read_text())
meta = json.loads((pathlib.Path(ckpt) / "decision_config.json").read_text())
manifest = pathlib.Path(ckpt) / "soup_manifest.json"
if calibration["model_sha256"] != adoption["model_sha256"]:
    raise SystemExit("package calibration binds another checkpoint")
if meta["lora"]["base_revision"] != rev or meta.get("base_revision") != rev:
    raise SystemExit("the checkpoint pins another base revision")
package = {
    "schema": "decision2-27b-moe-package/1",
    "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "checkpoint": ckpt,
    "checkpoint_format": meta["checkpoint_format"],
    "model_sha256": calibration["model_sha256"],
    "checkpoint_sha256": calibration["checkpoint_sha256"],
    "base": {"repo": repo, "revision": rev, "path": base, "tree_sha256": tree, "source_kind": meta["lora"]["source_kind"]},
    "architecture": meta["architecture"],
    "experts_implementation": meta["experts_implementation"],
    "prompt_version": meta["prompt_version"],
    "lora": {k: meta["lora"][k] for k in ("rank", "alpha", "dropout")},
    "soup_manifest_sha256": sha(manifest),
    "decision": adoption["decision"],
    "calibration": {"path": str(out / "package/calibration.json"), "sha256": sha(out / "package/calibration.json"),
                    "temperature_by_type": calibration["temperature_by_type"]},
    "release_calibration": adoption["candidate"] if adoption["adopt"] else None,
    "rejected_temperature_by_type": None if adoption["adopt"] else adoption["candidate"]["temperature_by_type"],
    "max_input_tokens": int(limit),
    "adoption_sha256": sha(out / "ADOPTION.json"),
    "devgates": {"path": devgates, "sha256": sha(devgates)},
    "loaded_parameters": params["loaded_parameters"],
    "active_parameters": params["active_parameters"],
    "parameter_source": params["loaded_source"] + "; active: " + params["active_source"],
}
with (out / "package/PACKAGE.json").open("x") as stream:
    json.dump(package, stream, indent=1, sort_keys=True)
    stream.write("\n")
print(json.dumps({k: package[k] for k in ("decision", "model_sha256", "loaded_parameters", "active_parameters")}))
EOF
    ;;
  formal | mlx)
    NAME=${1:?NAME}; name_ok "$NAME"
    frozen_check "$NAME"
    stages=smoke,collect,score
    [ "$STAGE" = formal ] || stages=mlx
    CALIBRATION=$(package_field "$NAME" calibration path) bash "$S/v2/27b/moe/moe-formal.sh" b 7 "$NAME" \
      "$(package_field "$NAME" checkpoint)" "$BASE" "$MIR" "$(package_field "$NAME" loaded_parameters)" \
      "$(package_field "$NAME" active_parameters)" "$stages"
    if [ "$STAGE" = mlx ]; then
      D=$X/mlx/$NAME
      [ -f "$R/formal/$NAME-mlx/COLLECT.json" ] || { echo "no finished mlx-diag collection" >&2; exit 2; }
      mkdir -p "$D"
      rsync -a --exclude triton-cache/ "$R/formal/$NAME-mlx/" "$D/"
      (cd "$D" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum) > "$D.SHA256SUMS"
      mv "$D.SHA256SUMS" "$D/SHA256SUMS"
      date -u +%FT%TZ > "$X/mlx/$NAME.PUSHED"
    fi ;;
  gates)
    NAME=${1:?NAME}; name_ok "$NAME"
    RUN=$R/formal/$NAME G=$R/gates D=$R/gates/$NAME
    [ -f "$RUN/SEAL.json" ] || { echo "$NAME has no sealed formal run" >&2; exit 2; }
    mkdir -p "$D" "$G/overlap"
    declare -A RIGHT=([A20r]=$A20R_RUN [autojev27]=/data/dev2/runs/27b/m2-peer-autojev27-nodeB-kernel
      [eikos27b]=/data/dev2/runs/27b/m3-peer-eikos27-nodeB-kernel
      [jebadiah27b]=/data/dev2/runs/27b/m3-peer-jebadiah-nodeB-kernel [F1]=/data/dev2/runs/27b/M3-A-soup/formal)
    python3 -m v2.eval.panels verify --panel typed-final --panel css15 --panel public231 > "$G/panels-verify.json"
    for right in "${!RIGHT[@]}"; do
      python3 -m v2.eval.gates paired --left "$RUN" --left-name "$NAME" --right "${RIGHT[$right]}" \
        --right-name "$right" --output "$D/paired-vs-$right.json" > "$D/paired-vs-$right.log"
    done
    python3 -m v2.eval.gates paired --left "$A20R_RUN" --left-name A20r --right "$RUN" --right-name "$NAME" \
      --output "$D/paired-A20r-minus-cand.json" > "$D/paired-A20r-minus-cand.log"
    python3 -m v2.eval.gates types --run "$RUN" --label "$NAME" --output "$D/types.json" > "$D/types.log"
    python3 -m v2.eval.gates public231 --left "$RUN" --right "$A20R_RUN" --left-name "$NAME" --right-name A20r \
      --output "$D/public231-vs-A20r.json" > "$D/public231-vs-A20r.log"
    [ -f "$G/overlap/exposure-moe-a20.json" ] || python3 -m v2.eval.overlap_effects exposure --groups "$FLAGGED" \
      --train "$MIX" --expect-sha256 "$MIX_SHA" --label "DEV2.0-27B MoE a20.train.jsonl" \
      --output "$G/overlap/exposure-moe-a20.json" > "$G/overlap/exposure-moe-a20.log"
    python3 - "$D" "$G/overlap/exposure-moe-a20.json" <<'EOF'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
out = {}
for p in sorted(d.glob("paired-vs-*.json")):
    r = json.loads(p.read_text())
    out[p.stem] = {"delta": round(r["point"]["delta"]["score"], 3), "ci95": [round(r["ci95"]["low"], 3), round(r["ci95"]["high"], 3)]}
out["types"] = {t: v["verdict"] for t, v in json.loads((d / "types.json").read_text())["types"].items()}
out["public231"] = json.loads((d / "public231-vs-A20r.json").read_text())["verdict"]
out["overlap_groups"] = len(json.loads(open(sys.argv[2]).read())["groups"])
print(json.dumps(out, sort_keys=True))
EOF
    ;;
  mlx-score)
    NAME=${1:?NAME}; name_ok "$NAME"
    PEER=$(cat "$KEY/peer")
    IN=/data/dev2/tmp/27b-moe-mlx/$NAME OUT=$R/mlx-diag/$NAME PANEL=$PANEL_ROOT/mlx-diag-v1
    REF=/data/dev2/runs/27b/m4-mlx/M4-A20r-soup PAIR=$R/mlx-diag/mlx-paired-$NAME-vs-A20r.json
    [ ! -e "$OUT" ] || { echo "$OUT exists" >&2; exit 66; }
    rm -rf "$IN" && mkdir -p "$IN"
    rs "root@$PEER:mlx/$NAME/" "$IN/"
    (cd "$IN" && sha256sum -c --quiet SHA256SUMS)
    mkdir -p "$(dirname "$OUT")"
    cp -a "$IN" "$OUT"
    python3 -m v2.eval.multilingual_panel score --panel "$PANEL" --predictions "$OUT/output/mlx-diag.predictions.jsonl" \
      --output "$OUT/mlx-diag.score.json" > "$OUT/score.log"
    python3 -m v2.06b.m8_scorebias mlx-paired --candidate-run "$OUT" --released-run "$REF" --panel "$PANEL" \
      --output "$PAIR"
    rs "$PAIR" "root@$PEER:mlx/$NAME-vs-A20r.json"
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({'R4': d['R4']['pass'], 'card_macro_ci95': d['bootstrap']['card_macro_ci95'], 'delta': d['delta']}))" "$PAIR" ;;
  mlx-pull)
    NAME=${1:?NAME}; name_ok "$NAME"
    [ -f "$X/mlx/$NAME-vs-A20r.json" ] || { echo "node A has not returned the mlx-diag pairing" >&2; exit 2; }
    mkdir -p "$R/gates/mlx"
    cp -p "$X/mlx/$NAME-vs-A20r.json" "$R/gates/mlx/$NAME-vs-A20r.json"
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({'R4': d['R4']['pass'], 'card_macro_ci95': d['bootstrap']['card_macro_ci95'], 'delta': d['delta']}))" \
      "$R/gates/mlx/$NAME-vs-A20r.json" ;;
  latency)
    NAME=${1:?NAME}; name_ok "$NAME"
    CKPT=$(package_field "$NAME" checkpoint) OUT=$R/$NAME/latency
    [ ! -e "$OUT/latency.json" ] || { echo "$OUT/latency.json exists" >&2; exit 66; }
    mkdir -p "$OUT"
    launch "$NAME" latency 0.3 "BF16 decision latency (M1 SELECT roster)" --mount "$BASE:$BASE" --mount "$CKPT:$CKPT" \
      --mount "$SELECT_ROWS:/data/select.jsonl" --mount "$OUT:$OUT:rw" --env PYTHONPATH=/code -- \
      python3 -m v2.27b.moe.latency --checkpoint "$CKPT" --source-path "$BASE" --select /data/select.jsonl \
      --output "$OUT/latency.json" ;;
  latency-ref)
    OUT=$R/latency-ref/A20r DENSE=/data/decision20-20260926/models/Qwen3.8-27B
    CKPT=/data/dev2/runs/27b/M4-A20r-soup/soup/checkpoint
    [ ! -e "$OUT/latency.json" ] || { echo "$OUT/latency.json exists" >&2; exit 66; }
    mkdir -p "$OUT"
    python3 -m v2.27b.triton_cache copy --frozen /data/dev2/runs/27b/m3-warm-32768/triton-cache \
      --expect 583241fbc3bc89e22be51a49722996eab742162356100400d64fb4208cb20daf --dest "$OUT/triton-cache"
    launch latency-ref A20r 0.3 "A20r BF16 decision latency (M1 SELECT roster, kernel path)" \
      --mount "$DENSE:$DENSE" --mount "$CKPT:$CKPT" --mount "$SELECT_ROWS:/data/select.jsonl" --mount "$OUT:$OUT:rw" \
      --env PYTHONPATH=/code --env TRITON_CACHE_AUTOTUNING=1 --env "TRITON_CACHE_DIR=$OUT/triton-cache" \
      --env HIP_FORCE_DEV_KERNARG=1 -- python3 -m v2.27b.moe.latency --checkpoint "$CKPT" --source-path "$DENSE" \
      --select /data/select.jsonl --output "$OUT/latency.json"
    python3 -m v2.27b.triton_cache finish --dest "$OUT/triton-cache" > /dev/null || true ;;
  verdicts)
    [ $# -ge 1 ] || { echo "verdicts NAME..." >&2; exit 2; }
    args=()
    for NAME in "$@"; do name_ok "$NAME"; args+=(--finalist "$NAME=$R/formal/$NAME=$R/$NAME/package/PACKAGE.json"); done
    python3 -m v2.27b.moe.moe_verdicts --gates "$R/gates" "${args[@]}" \
      --output "$R/gates/VERDICTS-$(date -u +%Y%m%dT%H%M%SZ).json" ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
echo "moe tail $STAGE complete: $(date -u +%FT%TZ)"
