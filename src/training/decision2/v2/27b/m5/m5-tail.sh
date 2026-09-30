#!/usr/bin/env bash
# ~27B M5 post-training stages on node B (host side): M4b's verified drivers with Milestone 5's allocation and roots.
# Usage: m5-tail.sh STAGE MIRROR_SHA ARGS...
#   lease GPU                 take an auxiliary GPU (node B GPU0-2) for track 27b (launch3 lease; a foreign idle owner
#                             moves to owner.prev-<UTC>); refused while another track's job holds it
#   pull ARM-SEED             m5-pull.sh: a node A relay (BEST checkpoint + SHA-256 list) over the private link
#   soup NAME CKPT CKPT...    combine_full.sh soup -> /data/dev2/runs/27b/m5/NAME/checkpoint (+ verify.json)
#   lsoup NAME CKPT CKPT...   LoRA members (branch B1's L128): v2.27b.lora_soup (exact rank concatenation, as M4's
#                             run_finalist.sh soup) in a CPU-only container -> /data/dev2/runs/27b/m5/NAME/checkpoint
#                             (soup_manifest.json); rank / alpha = the members' sums and the verification are checked;
#                             a member with a relay list (RELAY_SUMS=CKPT=SHA256SUMS ...) must match it file by file
#   reference GPU             A20r's typed DEV + CSS pilot + HT-DEV v2 readout (m5-htdev2.sh, htdev2-refs-nodeB.json)
#                             -> /data/dev2/runs/27b/m5/readouts/A20r-ref
#   readout NAME CKPT GPU     run_readout.sh (CAL698 kernel fit, then typed DEV + CSS pilot + HT-DEV v2 with it, score,
#                             summary) -> /data/dev2/runs/27b/m5/readouts/NAME
#   devgates NAME...          m5_devgates.py: the four development gates vs A20r-ref -> readouts/DEVGATES-<UTC>.json
#   formal NAME CKPT GPU      run_formal.sh (CAL698 release fit, adoption, frozen package, smoke, typed FINAL + CSS15 +
#                             public 231, seal, report, paired compares) -> /data/dev2/runs/27b/m5/NAME
#   mlx NAME GPU              run_mlx.sh on NAME's frozen package -> /data/dev2/runs/27b/m5/mlx-diag/NAME
#   mlx-push NAME             NAME's mlx-diag collection (no cache) -> node A relay mlx/NAME with a SHA-256 list
#   mlx-pull NAME             node A's mlx-paired output (m5-mlx-nodeA.sh) -> gates/mlx/NAME-vs-A20r.json
# Every inference job: 32,768 tokens, kernel path, a fresh verified copy of DEV2.0-27B's scored cache 03b172f1
# (formal runs: no autotune entry may be added). Readouts, CAL698 fits and formal runs never run on a training GPU.
# readout / formal / mlx take CHECKPOINT_FORMAT from the environment (default full; peft-lora/1 for M5-L128, whose
# formal run also needs LOADED_PARAMETERS); m4b/run_readout.sh and run_formal.sh refuse any other format.
set -euo pipefail
echo "m5 tail $*: start $(date -u +%FT%TZ)"
STAGE=${1:?STAGE} SHA=${2:?MIRROR_SHA}
shift 2
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
S=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2
[ -f "$S/v2/27b/m5/m5-tail.sh" ] || { echo "missing mirror $SHA" >&2; exit 2; }
M4B=$S/v2/27b/m4b M5=$S/v2/27b/m5
R=/data/dev2/runs/27b/m5
F1_CACHE=/data/dev2/runs/27b/m3-f2/f1-scored-cache
F1_CACHE_SHA=03b172f1a6adeef6c6a6c491d04389b355c9d8579480008023f408c8659b502b
export DEV2_27B_LAUNCH_ALLOC=m5-b M4B_ROOT=$R PYTHONPATH=$S PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/dev2/tmp
aux() {  # GPU: only node B GPU0-2 serve single-GPU M5 inference jobs
  case "$1" in 0 | 1 | 2) ;; *) echo "GPU$1 is not an M5 auxiliary GPU (node B GPU0-2)" >&2; exit 2 ;; esac
}
case "$STAGE" in
  lease)
    GPU=${1:?GPU}; aux "$GPU"
    (cd "$S" && python3 -m v2.27b.m4b.launch3 lease --gpus "$GPU" --purpose "27b M5 auxiliary inference" \
      --status reserved-idle) ;;
  pull) bash "$M5/m5-pull.sh" "${1:?ARM-SEED}" ;;
  soup)
    NAME=${1:?NAME}; shift
    COMBINE_ROOT=$R COMBINE_PREFIX=d2-27b-m5 bash "$M4B/combine_full.sh" "$SHA" "$NAME" soup "$@" ;;
  lsoup)
    NAME=${1:?NAME}; shift
    [ $# -ge 2 ] || { echo "lsoup NAME CKPT CKPT..." >&2; exit 2; }
    [[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "NAME must be one directory name" >&2; exit 2; }
    OUT=$R/$NAME BASE=/data/decision20-20260926/models/Qwen3.8-27B
    IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
    [ ! -e "$OUT/checkpoint" ] || { echo "$OUT/checkpoint exists: refusing to overwrite" >&2; exit 66; }
    args=() mounts=()
    for c in "$@"; do
      (cd "$S" && python3 -m v2.27b.m4b.ckpt_format check --checkpoint "$c" --format peft-lora/1)
      args+=(--member "$c")
      mounts+=(--mount "type=bind,src=$c,dst=$c,readonly")
    done
    mkdir -p "$OUT/soup"
    docker run --rm --name "d2-27b-m5-$NAME-soup" --network none --cpus 16 -e OMP_NUM_THREADS=16 \
      -e HIP_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES= -e PYTHONPATH=/code -e PYTHONDONTWRITEBYTECODE=1 \
      --mount "type=bind,src=$S,dst=/code,readonly" --mount "type=bind,src=$BASE,dst=$BASE,readonly" "${mounts[@]}" \
      --mount "type=bind,src=$OUT,dst=$OUT" -w /code --entrypoint python3 "$IMAGE" \
      -m v2.27b.lora_soup "${args[@]}" --source-path "$BASE" --output "$OUT/checkpoint" 2>&1 | tee "$OUT/soup/soup.log"
    python3 - "$OUT/checkpoint/soup_manifest.json" "${RELAY_SUMS:-}" "$@" <<'EOF' | tee "$OUT/soup/check.json"
import json, pathlib, sys
manifest = json.load(open(sys.argv[1]))
relays = dict(spec.split("=", 1) for spec in sys.argv[2].split())
members = [pathlib.Path(p) for p in sys.argv[3:]]
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
    sums = relays.get(member["path"])
    if not sums:
        continue
    listed = {line.split(None, 1)[1].strip().removeprefix("./"): line.split(None, 1)[0]
              for line in open(sums) if line.strip()}
    for name, digest in member["files_sha256"].items():
        if listed.get(name) != digest:
            problems.append(f"{member['path']}/{name} is not the relayed file")
if problems:
    raise SystemExit("; ".join(problems))
print(json.dumps({"soup": sys.argv[1], "model_sha256": manifest["output"]["model_sha256"], "members": lora["members"],
                  "rank": lora["rank"], "alpha": lora["alpha"], "max_relative_diff": check["max_relative_diff"],
                  "relay_checked": sorted(relays)}))
EOF
    ;;
  reference)
    GPU=${1:?GPU}; aux "$GPU"
    ROOT=$R/readouts PANELS=typed-dev,css-pilot,ht-dev2 STAGES=collect \
      bash "$M5/m5-htdev2.sh" "$M5/htdev2-refs-nodeB.json" m4-a20r-soup "$GPU" "$SHA"
    mkdir -p "$R/readouts"
    [ -e "$R/readouts/A20r-ref" ] || ln -s m4-a20r-soup "$R/readouts/A20r-ref"
    (cd "$S" && python3 -m v2.eval.dev_readout --run-dir "$R/readouts/m4-a20r-soup" --label "27b M5 A20r reference" \
      --output "$R/readouts/m4-a20r-soup/READOUT.json") ;;
  readout)
    NAME=${1:?NAME} CKPT=${2:?CKPT} GPU=${3:?GPU}; aux "$GPU"
    FROZEN=$F1_CACHE CACHE_SHA=$F1_CACHE_SHA PANELS=typed-dev,css-pilot,ht-dev2 \
      STAGES=${STAGES:-verify,cal,collect,score,summary} LABEL="27b M5 $NAME" \
      bash "$M4B/run_readout.sh" "$NAME" "$CKPT" "$GPU" "$SHA" ;;
  devgates)
    [ $# -ge 1 ] || { echo "devgates NAME..." >&2; exit 2; }
    cands=()
    for NAME in "$@"; do cands+=(--candidate "$NAME=$R/readouts/$NAME"); done
    (cd "$S" && python3 -m v2.27b.m5.m5_devgates --reference "A20r=$R/readouts/A20r-ref" "${cands[@]}" \
      --output "$R/readouts/DEVGATES-$(date -u +%Y%m%dT%H%M%SZ).json") ;;
  formal)
    NAME=${1:?NAME} CKPT=${2:?CKPT} GPU=${3:?GPU}; aux "$GPU"
    FROZEN=$F1_CACHE CACHE_SHA=$F1_CACHE_SHA READOUT=$R/readouts/$NAME LABEL="DEV2.0-27B (M5 $NAME)" \
      EXTRA_COMPARATOR="M4-A20r-soup=/data/dev2/runs/27b/M4-A20r-soup/formal M4-A20-soup=/data/dev2/runs/27b/M4-A20-soup/formal F-b=/data/dev2/runs/27b/m4b/F-b/formal ${EXTRA_COMPARATOR:-}" \
      bash "$M4B/run_formal.sh" "$NAME" "$CKPT" "$GPU" "$SHA" ;;
  mlx)
    NAME=${1:?NAME} GPU=${2:?GPU}; aux "$GPU"
    bash "$M4B/run_mlx.sh" "$NAME" "$R/$NAME/package/PACKAGE.json" "$GPU" "$SHA" ;;
  mlx-push | mlx-pull)
    NAME=${1:?NAME} KEY=/data/dev2/tmp/27b-m5-xfer
    X="ssh -i $KEY/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=$KEY/known_hosts -o StrictHostKeyChecking=yes -o BatchMode=yes"
    PEER=$(cat "$KEY/peer") D=$R/mlx-diag/$NAME
    if [ "$STAGE" = mlx-push ]; then
      [ -f "$D/COLLECT.json" ] || { echo "no finished mlx-diag collection $D" >&2; exit 2; }
      (cd "$D" && find output -type f | sort | xargs sha256sum > SHA256SUMS)
      rsync -a --mkpath -e "$X" --exclude triton-cache/ "$D/" "root@$PEER:mlx/$NAME/"
    else
      mkdir -p "$R/gates/mlx"
      rsync -a -e "$X" "root@$PEER:mlx/$NAME-vs-A20r.json" "$R/gates/mlx/$NAME-vs-A20r.json"
      python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({'R4': d['R4']['pass'], 'card_macro_ci95': d['bootstrap']['card_macro_ci95'], 'delta': d['delta']}))" \
        "$R/gates/mlx/$NAME-vs-A20r.json"
    fi ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
echo "m5 tail $STAGE complete: $(date -u +%FT%TZ)"
