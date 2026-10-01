#!/usr/bin/env bash
# Forward-token-budget runtime revisions of DEV2.0-0.8B / 2B / 9B / 27B (user release round 2026-10-01 18:08 UTC+8;
# release worker 4c0a68cd), on node E.
#   --stage    release.sh without parity or upload on the final spec: build, native examples and card, AutoConfig /
#              AutoTokenizer / AutoModel / pipeline vs native and the card's Transformers block (the package that
#              --extra tests and --release must rebuild byte for byte)
#   --extra    on the newest staged package: the synthetic long-input regression (one prompt at the cap minus 300
#              tokens; 32 questions, 48 on 0.8B / 2B so the batch passes the budget there too; each answer vs the
#              question alone), runtime_bench old (the released auto_map
#              package) vs new (400 typed-final requests after 400 warm-up, a fresh frozen-cache copy each side) and,
#              for 27B, the two private recorded requests (counts only; request files stay in a private directory)
#   --release  refuses unless main is the superseded revision, --extra passed and a fresh build equals the staged
#              package; then hf_headroom.sh and release.sh --upload --collect --already-collected with full parity
#              before and after the real download, the AutoModel steps, Hub smoke under 5.17 and 5.18, readback,
#              gate and collection; then card HTTP, links and gate evaluate
# Usage: bash <mirror>/v2/release/records/dev2-budget-2026-10-01/ops/budget.sh <tier> <mode> --gpu 6|7 [--resume REV]
set -euo pipefail
tier="${1:-}" mode="${2:-}"
shift 2 || true
gpu="" resume=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu=$2; shift 2 ;;
    # --release after a run that uploaded this package and then failed: main may be that revision.
    --resume) resume=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ "$mode" =~ ^--(stage|extra|release)$ ]] || { echo "mode: --stage|--extra|--release" >&2; exit 2; }
[[ "$gpu" =~ ^[67]$ ]] || { echo "node E GPU6 or GPU7 only (--gpu)" >&2; exit 2; }
TS=$(date -u +%Y%m%dT%H%M%SZ)
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
SRC=$(basename "$(cd "$S/../../.." && pwd)")
R=$S/v2/release/records/dev2-budget-2026-10-01
G=/data/dev2/private/panels/goldfree
D=/data/dev2/runs/release/decisions
HFC=/data/dev2/hf-cache
TF518=/data/dev2/tools/tf518
HFPY=/data/dev2/tools/hf-cli/bin/python
REPRO=/data/dev2/private/eval/budget-repro
image=decision20-train-fast:host2 cache_tool=digest PM="" long_q=32 tie_args=() base_args=() base_mount=() base_run=()
case "$tier" in
  0.8B)
    key=0p8b P=/data/dev2/runs/release/inputs/dev2-0p8b-t1/derived IN=/data/dev2/runs/release/inputs/dev2-0p8b-bf16
    frozen=/data/dev2/runs/dec/formal/m2/m2-E8F-soup-nodeA-triton cap=16384 long_q=48
    frozen_digest=5e37a14373b70584bcc2e0056ad7ef5ebaf3d01b0a1820d216717a23232f09e2 ;;
  2B)
    key=2b P=/data/dev2/runs/release/inputs/dev2-2b-t1/derived IN=/data/dev2/runs/release/inputs/dev2-2b-bf16
    frozen=/data/dev2/runs/dec/formal/m3/m3-S2T-soup-nodeA-triton cap=16384 long_q=48
    frozen_digest=abdfd6872ccee3efc2b8e67358e9726e83eec882b1b0de3059d2ad408b2e26f0 ;;
  9B)
    key=9b P=/data/dev2/runs/release/inputs/dev2-8b-t1/derived IN=/data/dev2/runs/release/inputs/dev2-8b-bf16
    frozen=/data/dev2/runs/9b/formal-m4/triton-cache cap=16384
    frozen_digest=5604ffdc5f1916068c0b8df0526efc455f7bbec708081533020b834aba52586d ;;
  27B)
    key=27b P=/data/dev2/runs/27b/M4-A20r-soup/formal/output PM=/data/dev2/runs/27b/m4-mlx/M4-A20r-soup/output
    IN=/data/dev2/runs/27b/M4-A20r-soup/soup/checkpoint cap=32768
    frozen=/data/dev2/runs/27b/M4-A20r-soup/formal/triton-cache cache_tool=27b
    frozen_digest=f474e2e9dbdf3a2900465bce74256326ba2756250861b45359742685982cb5f7
    image=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
    base_repo=$HFC/models--Qwen--Qwen3.8-27B
    base_snapshot=$base_repo/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
    base_args=(--base-path "$base_snapshot" --env "HF_HUB_CACHE=$HFC" --mount "$base_repo")
    base_mount=(-v "$base_repo:$base_repo:ro")
    base_run=(--base-path "$base_snapshot")
    # A side flip of a question that asked alone sits within 0.02 of its decision boundary is a tie.
    tie_args=(--tie-margin 0.02) ;;
  *) echo "tier must be one of 0.8B 2B 9B 27B" >&2; exit 2 ;;
esac
name=DEV2.0-$tier REPO=llm-semantic-router/$name
SPEC=$S/v2/release/specs/dev2-$key-budget.json
export TMPDIR=/data/dev2/tmp PYTHONPATH=$S
mkdir -p "$TMPDIR" /data/dev2/runs/release/triton "$D"
digest() { (cd "$1" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64); }
copy_cache() {
  if [[ "$cache_tool" == 27b ]]; then
    (cd "$S" && python3 -m v2.27b.triton_cache copy --frozen "$frozen" --expect "$frozen_digest" --dest "$1")
  else
    [[ "$(digest "$frozen")" == "$frozen_digest" ]] || { echo "frozen cache $frozen changed" >&2; exit 1; }
    cp -a "$frozen" "$1"
  fi
}
finish_cache() {
  [[ "$cache_tool" != 27b || ! -d "$1" ]] || (cd "$S" && python3 -m v2.27b.triton_cache finish --dest "$1") || true
}
decision=$D/$name.decision.budget.json
if [[ -e "$decision" ]]; then cmp "$R/$name.decision.budget.json" "$decision"; else
  cp "$R/$name.decision.budget.json" "$decision"; chmod 444 "$decision"; fi
render=$(readlink -f "/dev/dri/by-path/pci-$(amd-smi list 2>/dev/null | awk -v g="GPU: $gpu" '$0 ~ "^"g"$" {getline; print tolower($2)}')-render")
[[ -c "$render" ]] || { echo "no render node for GPU$gpu" >&2; exit 1; }
newest() { compgen -G "$1" | sort | tail -1 || true; }
manifest_sha() { sha256sum "$1/MODEL_MANIFEST.json" | cut -c1-64; }

# One container on the leased GPU only: its render node + /dev/kfd, no network, offline hub, kernels required.
gpu_run() {  # gpu_run <triton-cache-dir> <log> <volume args...> -- <python args...>
  local tc="$1" logf="$2"; shift 2
  local vols=()
  while [[ "$1" != -- ]]; do vols+=("$1"); shift; done
  shift
  docker run --rm --network none --ipc host --device /dev/kfd --device "$render" --group-add video \
    --group-add render --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 -e TOKENIZERS_PARALLELISM=false -e HIP_FORCE_DEV_KERNARG=1 \
    -e TRITON_CACHE_AUTOTUNING=1 -e "TRITON_CACHE_DIR=$tc" -e PYTHONPATH="$S:/opt/decision-fla" \
    -v "$S:$S:ro" -v "$tc:$tc" "${base_mount[@]}" "${vols[@]}" -w "$S" --entrypoint python3 "$image" "$@" \
    > "$logf" 2>&1
}

case "$mode" in
  --stage)
    W=/data/dev2/runs/release/dev2-budget-$tier-stage-$TS
    TC=/data/dev2/runs/release/triton/dev2-budget-$tier-stage-$TS
    copy_cache "$TC"
    echo "mirror $SRC tier $tier gpu $gpu mode $mode work $W"
    status=0
    "$S/v2/release/release.sh" --spec "$SPEC" --src "$SRC" --work "$W" --image "$image" \
      --gpu "$gpu" --track release-automap --threads 4 --site /opt/decision-fla --require-kernels \
      --env TRITON_CACHE_AUTOTUNING=1 "${base_args[@]}" --env HIP_FORCE_DEV_KERNARG=1 \
      --env "TRITON_CACHE_DIR=$TC" --mount-rw "$TC" || status=$?
    finish_cache "$TC"
    echo "work=$W manifest=$(manifest_sha "$W/package/$name") status=$status"
    exit "$status" ;;
  --extra)
    V=$(newest "/data/dev2/runs/release/dev2-budget-$tier-stage-*")
    O=$(newest "/data/dev2/runs/release/dev2-automap-$tier-2026*")
    [[ -n "$V" && -n "$O" ]] || { echo "no staged package or no released auto_map package" >&2; exit 1; }
    released=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' \
      "$O/receipts/gate.json")
    [[ "$(manifest_sha "$O/package/$name")" == "$released" ]] \
      || { echo "$O package is not the released auto_map package" >&2; exit 1; }
    W=/data/dev2/runs/release/dev2-budget-$tier-extra-$TS
    mkdir -p "$W/receipts" "$W/logs" "$W/answers"
    new=$V/package/$name old=$O/package/$name
    pk=(-v "$new:$new:ro" -v "$old:$old:ro" -v "$W:$W" -v "$G:$G:ro")
    echo "tier $tier staged $(basename "$V") ($(manifest_sha "$new")) old $(basename "$O") ($released) work $W"
    status=0
    TCL=/data/dev2/runs/release/triton/dev2-budget-$tier-long-$TS; copy_cache "$TCL"
    gpu_run "$TCL" "$W/logs/long-request.log" "${pk[@]}" -- -B -m v2.release.tests.gpu_long_request \
      --package "$new" "${base_run[@]}" --tokens $((cap - 300)) --questions "$long_q" "${tie_args[@]}" \
      --out "$W/receipts/long-request.json" || status=1
    finish_cache "$TCL"
    for side in old new; do
      TCB=/data/dev2/runs/release/triton/dev2-budget-$tier-bench-$side-$TS; copy_cache "$TCB"
      pkg=$old; [[ $side == new ]] && pkg=$new
      gpu_run "$TCB" "$W/logs/bench-$side.log" "${pk[@]}" -- -I -B "$S/v2/release/runtime_bench.py" run \
        --package "$pkg" --prompts "$G/typed-final.prompts.jsonl" --count 400 --warmup 400 \
        --output "$W/answers/bench-$side.json" --threads 4 "${base_run[@]}" --site /opt/decision-fla \
        --require-kernels || status=1
      finish_cache "$TCB"
    done
    python3 "$S/v2/release/runtime_bench.py" compare "$W/answers/bench-old.json" "$W/answers/bench-new.json" \
      --output "$W/receipts/bench-compare.json" || status=1
    for side in old new; do
      python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); r.pop("answers", None); json.dump(r, open(sys.argv[2], "w"), indent=2)' \
        "$W/answers/bench-$side.json" "$W/receipts/bench-$side.json"
    done
    if [[ "$tier" == 27B ]]; then
      TCP=/data/dev2/runs/release/triton/dev2-budget-$tier-private-$TS; copy_cache "$TCP"
      gpu_run "$TCP" "$W/answers/private-requests.log" "${pk[@]}" -v "$REPRO:$REPRO:ro" -- -B \
        -m v2.release.tests.gpu_private_requests --package "$new" "${base_run[@]}" \
        --rows "$REPRO/invalid.jsonl.gz" --rows "$REPRO/fault.jsonl.gz" --repeat 2 \
        --out "$W/receipts/private-requests.json" || status=1
      finish_cache "$TCP"
    fi
    python3 - "$W/receipts" "$status" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
out = {"staged_manifest": None, "pass": sys.argv[2] == "0"}
for name in ("long-request", "bench-compare", "private-requests"):
    p = d / f"{name}.json"
    if p.is_file():
        r = json.loads(p.read_text())
        out[name] = r.get("pass", r.get("passed"))
        out["pass"] = out["pass"] and bool(out[name])
print(json.dumps(out))
json.dump(out, open(d / "extra.json", "w"), indent=2)
PY
    python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); r["staged_manifest"]=sys.argv[2]; r["staged_work"]=sys.argv[3]; json.dump(r, open(sys.argv[1], "w"), indent=2)' \
      "$W/receipts/extra.json" "$(manifest_sha "$new")" "$V"
    echo "work=$W status=$status"
    exit "$status" ;;
esac

# --release
expected=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["supersedes"]["released_as"].split("@")[1])' \
  "$R/$name.decision.budget.json")
main=$("$HFPY" -c 'import sys; from huggingface_hub import HfApi; print(HfApi().model_info(sys.argv[1]).sha)' "$REPO")
if [[ -n "$resume" && "$main" == "$resume" ]]; then
  echo "$REPO main $main = the revision an interrupted run of this package uploaded (resume)"
else
  [[ "$main" == "$expected" ]] || { echo "$REPO main is $main, not the superseded revision $expected: refusing" >&2; exit 1; }
  echo "$REPO main $main = superseded revision"
fi
E=$(newest "/data/dev2/runs/release/dev2-budget-$tier-extra-*")
if [[ -z "$E" ]] || ! python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["pass"] else 1)' \
  "$E/receipts/extra.json"; then
  echo "no passing --extra run for $tier" >&2; exit 1
fi
staged=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["staged_manifest"])' "$E/receipts/extra.json")
check=$TMPDIR/dev2-budget-$tier-check-$TS
python3 -m v2.release.build --spec "$SPEC" --output "$check/$name" > /dev/null
rebuilt=$(manifest_sha "$check/$name")
rm -rf "$check"
[[ "$rebuilt" == "$staged" ]] || { echo "fresh build $rebuilt differs from the tested package $staged" >&2; exit 1; }
echo "fresh build = tested package ($staged)"
bash "$S/v2/common/hf_headroom.sh" --min-free-gb 3
W=/data/dev2/runs/release/dev2-budget-$tier-$TS
TC=/data/dev2/runs/release/triton/dev2-budget-$tier-$TS
copy_cache "$TC"
parity_args=(
  --parity "typed-final:$G/typed-final.prompts.jsonl:$P/typed-final.predictions.jsonl:1600"
  --parity "css15:$G/css15.prompts.jsonl:$P/css15.predictions.jsonl:6547"
  --parity "public231:$G/public231.prompts.jsonl:$P/public231.predictions.jsonl:231"
  --parity "mlx-diag:$G/mlx-diag.prompts.jsonl:${PM:-$P}/mlx-diag.predictions.jsonl:2275"
)
mounts=(--mount "$G" --mount "$P" --mount "$IN")
[[ -z "$PM" ]] || mounts+=(--mount "$PM")
echo "mirror $SRC tier $tier gpu $gpu mode $mode work $W"
set -x
status=0
"$S/v2/release/release.sh" --spec "$SPEC" --src "$SRC" --work "$W" --image "$image" \
  --gpu "$gpu" --track release-automap --threads 4 --site /opt/decision-fla --require-kernels \
  --env TRITON_CACHE_AUTOTUNING=1 "${base_args[@]}" --env HIP_FORCE_DEV_KERNARG=1 \
  --env "TRITON_CACHE_DIR=$TC" --mount-rw "$TC" "${mounts[@]}" "${parity_args[@]}" \
  --upload --collect --already-collected --hub-site "tf518=$TF518" || status=$?
set +x
finish_cache "$TC"
python3 - "$W/receipts" <<'PY' || true
import json, sys
from pathlib import Path
receipts = Path(sys.argv[1])
for name in ("parity-pre", "automap-pre", "automap-card-pre", "automap-parity-pre", "automap-vs-native-pre",
             "parity-post", "automap-post", "automap-hub", "automap-hub-tf518", "gate"):
    path = receipts / f"{name}.json"
    if path.is_file():
        r = json.loads(path.read_text())
        extra = ""
        if "panels" in r:
            extra = " " + json.dumps({k: [v.get("prompts"), v.get("category_changes"), v.get("missing"),
                                         v.get("max_abs_drift")] for k, v in r["panels"].items()})
        print(f"{name}: passed={r.get('passed', 'n/a')}{extra}")
PY
[[ "$status" == 0 ]] || { echo "release.sh failed ($status); work=$W" >&2; exit "$status"; }
REV=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['revision'])" "$W/receipts/upload.json")
[[ "$(manifest_sha "$W/package/$name")" == "$staged" ]] || { echo "released package differs from the tested one" >&2; status=1; }
mkdir -p "$W/extra"
cd "$S"
"$HFPY" -m v2.release.tests.hub_card_http_check --repo "$REPO" --revision "$REV" \
  --package "$W/package/$name" --output "$W/extra/card-http.json" || status=1
"$HFPY" -m v2.release.hub_links --repo "$REPO" --revision "$REV" \
  --package "$W/package/$name" --output "$W/extra/hub-links.json" || status=1
python3 -m v2.release.gate evaluate --work "$W" > "$W/extra/gate-evaluate.json" || status=1
bash "$S/v2/common/hf_headroom.sh" --min-free-gb 0
echo "work=$W revision=$REV post_checks=$([[ $status == 0 ]] && echo ok || echo FAILED)"
exit "$status"
