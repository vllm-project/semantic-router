#!/usr/bin/env bash
# auto_map revisions of the DEV2.0 repositories (user request 2026-10-01 10:53 UTC+8; worker 4c0a68cd), on node E.
#   --verify   release.sh without upload on the draft spec: build, native examples and card, full native parity
#              (typed-final 1,600, css15 6,547, public231 231, mlx-diag 2,275 unless the tier's mlx-diag run had
#              its own cache), then AutoConfig / AutoTokenizer / AutoModel / pipeline vs native, the card's
#              Transformers block and AutoModel parity compared prompt by prompt with the native run
#   --mlx      the same, mlx-diag only, with the mlx-diag run's own cache (4B)
#   --release  release.sh --upload --collect --already-collected with the final spec and decision (same steps,
#              then the real download, native + AutoModel on it, the card's Transformers block from the Hub in
#              a fresh cache under Transformers 5.17 and 5.18, readback, gate, collection), then card HTTP,
#              links and storage
#   --tf518    AutoModel parity only, under Transformers 5.18 (site /data/dev2/tools/tf518), on the newest
#              verified package; compared prompt by prompt with that run's native answers
# Usage: bash <mirror>/v2/release/records/dev2-automap-2026-10-01/ops/automap.sh <tier> <mode> --gpu 6|7
set -euo pipefail
tier="${1:-}" mode="${2:-}"
shift 2 || true
gpu=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ "$mode" =~ ^--(verify|mlx|release|tf518)$ ]] || { echo "mode: --verify|--mlx|--release|--tf518" >&2; exit 2; }
[[ "$gpu" =~ ^[67]$ ]] || { echo "node E GPU6 or GPU7 only (--gpu)" >&2; exit 2; }
TS=$(date -u +%Y%m%dT%H%M%SZ)
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
SRC=$(basename "$(cd "$S/../../.." && pwd)")
R=$S/v2/release/records/dev2-automap-2026-10-01
G=/data/dev2/private/panels/goldfree
D=/data/dev2/runs/release/decisions
HFC=/data/dev2/hf-cache
TF518=/data/dev2/tools/tf518
HFPY=/data/dev2/tools/hf-cli/bin/python
image=decision20-train-fast:host2 kernels=1 tolerance_args=() frozen="" frozen_digest="" cache_tool=digest
mlx_cache="" mlx_digest="" PM="" base_args=()
case "$tier" in
  0.6B)
    key=0p6b P=/data/dev2/runs/06b/m8/formal/m8-s5-b05/output PM=/data/dev2/runs/06b/m8/formal/m8-s5-b05-mlx/output
    IN=/data/dev2/runs/release/inputs/dev2-0p6b-m8-bf16 kernels=0 tolerance_args=(--parity-tolerance 0) ;;
  0.8B)
    key=0p8b P=/data/dev2/runs/release/inputs/dev2-0p8b-t1/derived IN=/data/dev2/runs/release/inputs/dev2-0p8b-bf16
    frozen=/data/dev2/runs/dec/formal/m2/m2-E8F-soup-nodeA-triton
    frozen_digest=5e37a14373b70584bcc2e0056ad7ef5ebaf3d01b0a1820d216717a23232f09e2 ;;
  2B)
    key=2b P=/data/dev2/runs/release/inputs/dev2-2b-t1/derived IN=/data/dev2/runs/release/inputs/dev2-2b-bf16
    frozen=/data/dev2/runs/dec/formal/m3/m3-S2T-soup-nodeA-triton
    frozen_digest=abdfd6872ccee3efc2b8e67358e9726e83eec882b1b0de3059d2ad408b2e26f0 ;;
  4B)
    key=4b P=/data/dev2/runs/release/inputs/dev2-4b-t1/derived IN=/data/dev2/runs/release/inputs/dev2-4b-bf16
    frozen=/data/dev2/runs/dec/formal/m4/m4-N4XF-soup-nodeA-triton
    frozen_digest=438618a6e3beb39407dabb96e689b5740d021a9c93b9df9769453c3d94deb119
    mlx_cache=/data/dev2/runs/dec/formal/m4/m4-N4XF-soup-triton
    mlx_digest=ba8f21322724f57beafac7dbce5348af0a9c148368d89394e459287226c481fc ;;
  9B)
    key=9b P=/data/dev2/runs/release/inputs/dev2-8b-t1/derived IN=/data/dev2/runs/release/inputs/dev2-8b-bf16
    frozen=/data/dev2/runs/9b/formal-m4/triton-cache
    frozen_digest=5604ffdc5f1916068c0b8df0526efc455f7bbec708081533020b834aba52586d ;;
  27B)
    key=27b P=/data/dev2/runs/27b/M4-A20r-soup/formal/output PM=/data/dev2/runs/27b/m4-mlx/M4-A20r-soup/output
    IN=/data/dev2/runs/27b/M4-A20r-soup/soup/checkpoint
    frozen=/data/dev2/runs/27b/M4-A20r-soup/formal/triton-cache cache_tool=27b
    frozen_digest=f474e2e9dbdf3a2900465bce74256326ba2756250861b45359742685982cb5f7
    image=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
    base_repo=$HFC/models--Qwen--Qwen3.8-27B
    base_snapshot=$base_repo/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
    base_args=(--base-path "$base_snapshot" --env "HF_HUB_CACHE=$HFC" --mount "$base_repo") ;;
  *) echo "tier must be one of 0.6B 0.8B 2B 4B 9B 27B" >&2; exit 2 ;;
esac
name=DEV2.0-$tier REPO=llm-semantic-router/$name
export TMPDIR=/data/dev2/tmp PYTHONPATH=$S
mkdir -p "$TMPDIR" /data/dev2/runs/release/triton "$D"
digest() { (cd "$1" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64); }
copy_cache() {
  if [[ "$cache_tool" == 27b ]]; then
    (cd "$S" && python3 -m v2.27b.triton_cache copy --frozen "$1" --expect "$2" --dest "$3")
  else
    [[ "$(digest "$1")" == "$2" ]] || { echo "frozen cache $1 changed" >&2; exit 1; }
    cp -a "$1" "$3"
  fi
}
kernel_args=() env_kernel=()
if [[ "$kernels" == 1 ]]; then
  kernel_args=(--site /opt/decision-fla --require-kernels)
  env_kernel=(--env TRITON_CACHE_AUTOTUNING=1)
fi
mounts=(--mount "$G" --mount "$P" --mount "$IN")
[[ -z "$PM" ]] || mounts+=(--mount "$PM")
TC=/data/dev2/runs/release/triton/dev2-automap-$tier-$TS
case "$mode" in
  --verify|--tf518)
    suffix=.draft
    [[ "$mode" == --tf518 ]] && suffix=""
    [[ -z "$frozen" ]] || copy_cache "$frozen" "$frozen_digest" "$TC"
    parity_args=(
      --parity "typed-final:$G/typed-final.prompts.jsonl:$P/typed-final.predictions.jsonl:1600"
      --parity "css15:$G/css15.prompts.jsonl:$P/css15.predictions.jsonl:6547"
      --parity "public231:$G/public231.prompts.jsonl:$P/public231.predictions.jsonl:231"
    )
    [[ -n "$mlx_cache" ]] || parity_args+=(--parity "mlx-diag:$G/mlx-diag.prompts.jsonl:${PM:-$P}/mlx-diag.predictions.jsonl:2275")
    hub_args=() ;;
  --mlx)
    suffix=.draft
    [[ -n "$mlx_cache" ]] || { echo "--mlx is for a tier whose mlx-diag run had its own cache" >&2; exit 2; }
    copy_cache "$mlx_cache" "$mlx_digest" "$TC"
    parity_args=(--parity "mlx-diag:$G/mlx-diag.prompts.jsonl:$P/mlx-diag.predictions.jsonl:2275")
    hub_args=() ;;
  --release)
    suffix=""
    bash "$S/v2/common/hf_headroom.sh" --min-free-gb 3
    [[ -z "$frozen" ]] || copy_cache "$frozen" "$frozen_digest" "$TC"
    parity_args=(
      --parity "typed-final:$G/typed-final.prompts.jsonl:$P/typed-final.predictions.jsonl:1600"
      --parity "css15:$G/css15.prompts.jsonl:$P/css15.predictions.jsonl:6547"
      --parity "public231:$G/public231.prompts.jsonl:$P/public231.predictions.jsonl:231"
    )
    [[ -n "$mlx_cache" ]] || parity_args+=(--parity "mlx-diag:$G/mlx-diag.prompts.jsonl:${PM:-$P}/mlx-diag.predictions.jsonl:2275")
    hub_args=(--upload --collect --already-collected --hub-site "tf518=$TF518") ;;
esac
SPEC=$S/v2/release/specs/dev2-$key-automap$suffix.json
decision=$D/$name.decision.automap$suffix.json
if [[ -e "$decision" ]]; then cmp "$R/$name.decision.automap$suffix.json" "$decision"; else
  cp "$R/$name.decision.automap$suffix.json" "$decision"; chmod 444 "$decision"; fi
cache_args=()
[[ ! -d "$TC" ]] || cache_args=(--env "TRITON_CACHE_DIR=$TC" --mount-rw "$TC")
if [[ "$mode" == --tf518 ]]; then
  # AutoModel parity under Transformers 5.18 on the newest verified package, vs that run's native answers.
  V=$(ls -d "/data/dev2/runs/release/dev2-automap-$tier-verify-"* | tail -1)
  W=/data/dev2/runs/release/dev2-automap-$tier-tf518-$TS
  mkdir -p "$W/receipts" "$W/logs"
  render=$(readlink -f "/dev/dri/by-path/pci-$(amd-smi list 2>/dev/null | awk -v g="GPU: $gpu" '$0 ~ "^"g"$" {getline; print tolower($2)}')-render")
  volumes=(-v "$S:$S:ro" -v "$V/package:$V/package:ro" -v "$W:$W" -v "$G:$G:ro" -v "$P:$P:ro" -v "$TF518:$TF518:ro")
  [[ -z "$PM" ]] || volumes+=(-v "$PM:$PM:ro")
  envs=(-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HIP_FORCE_DEV_KERNARG=1)
  [[ ! -d "$TC" ]] || { volumes+=(-v "$TC:$TC"); envs+=(-e TRITON_CACHE_AUTOTUNING=1 -e "TRITON_CACHE_DIR=$TC"); }
  [[ ${#base_args[@]} -eq 0 ]] || { volumes+=(-v "$base_repo:$base_repo:ro"); envs+=(-e "HF_HUB_CACHE=$HFC"); }
  panels=()
  for p in "${parity_args[@]}"; do [[ "$p" == --parity ]] || panels+=(--panel "$p"); done
  docker run --rm --network none --ipc host --device /dev/kfd --device "$render" --group-add video --group-add render \
    --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=0 -e TOKENIZERS_PARALLELISM=false "${envs[@]}" \
    "${volumes[@]}" --entrypoint python3 "$image" -I -B "$S/v2/release/examples.py" automap-parity \
    --package "$V/package/$name" --output "$W/receipts/automap-parity-tf518.json" --answers "$W/automap-tf518.jsonl" \
    --threads 4 --device cuda:0 --site "$TF518" "${kernel_args[@]}" "${panels[@]}" > "$W/logs/automap-parity-tf518.log" 2>&1
  python3 "$S/v2/release/examples.py" compare-answers "$V/answers/native-pre.jsonl" "$W/automap-tf518.jsonl" \
    --output "$W/receipts/automap-tf518-vs-native.json"
  python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(json.dumps({"passed": r["passed"], "max_abs_drift": r["max_abs_drift"], "panels": {k: [v["prompts"], v["identical_prompts"], v["category_changes"], v["missing"]] for k, v in r["panels"].items()}}))' \
    "$W/receipts/automap-tf518-vs-native.json"
  echo "work=$W"
  exit 0
fi
case "$mode" in
  --verify) W=/data/dev2/runs/release/dev2-automap-$tier-verify-$TS ;;
  --mlx) W=/data/dev2/runs/release/dev2-automap-$tier-mlx-$TS ;;
  --release) W=/data/dev2/runs/release/dev2-automap-$tier-$TS ;;
esac
echo "mirror $SRC tier $tier gpu $gpu mode $mode work $W"
set -x
status=0
"$S/v2/release/release.sh" --spec "$SPEC" --src "$SRC" --work "$W" --image "$image" \
  --gpu "$gpu" --track release-automap --threads 4 \
  "${kernel_args[@]}" "${env_kernel[@]}" "${base_args[@]}" --env HIP_FORCE_DEV_KERNARG=1 "${cache_args[@]}" \
  "${mounts[@]}" "${tolerance_args[@]}" "${parity_args[@]}" "${hub_args[@]}" || status=$?
set +x
[[ "$cache_tool" != 27b || ! -d "$TC" ]] || (cd "$S" && python3 -m v2.27b.triton_cache finish --dest "$TC") || true
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
if [[ "$mode" == --release ]]; then
  REV=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['revision'])" "$W/receipts/upload.json")
  mkdir -p "$W/extra"
  cd "$S"
  "$HFPY" -m v2.release.tests.hub_card_http_check --repo "$REPO" --revision "$REV" \
    --package "$W/package/$name" --output "$W/extra/card-http.json" || status=1
  "$HFPY" -m v2.release.hub_links --repo "$REPO" --revision "$REV" \
    --package "$W/package/$name" --output "$W/extra/hub-links.json" || status=1
  python3 -m v2.release.gate evaluate --work "$W" > "$W/extra/gate-evaluate.json" || status=1
  bash "$S/v2/common/hf_headroom.sh" --min-free-gb 0
  echo "work=$W revision=$REV post_checks=$([[ $status == 0 ]] && echo ok || echo FAILED)"
fi
echo "work=$W"
exit "$status"
