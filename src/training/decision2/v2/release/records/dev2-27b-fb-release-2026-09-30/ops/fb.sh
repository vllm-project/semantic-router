#!/usr/bin/env bash
# DEV2.0-27B = 27B M4b F-b in bf16z storage (coordinator notes 2026-09-30 06:10 and 07:15). Node B unless noted.
#   fb.sh verify                     bf16z verify of the compressed checkpoint (decompress + hash every file; CPU, image)
#   fb.sh c1pkg                      the plain (uncompressed) package of the same weights: the C1 frozen package (CPU)
#   fb.sh parity --gpu N             release.sh without upload, draft spec: full parity on typed-final 1,600, css15
#                                    6,547 and public231 231 with a fresh copy of the formal run's autotune cache
#   fb.sh parity-mlx --gpu N         the same for mlx-diag 2,275 with the mlx-diag run's own cache
#   fb.sh release --gpu N            final spec: release.sh --upload --collect --already-collected (subset parity
#                                    200 / 300 / 100), then bf16z verify of the real download and the post checks
# GPU runs take a shared lease entry owner.release-27b on node B GPU0-2 and restore the weights into a rw cache.
set -euo pipefail
mode="${1:-}"
shift || true
gpu=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
TS=$(date -u +%Y%m%dT%H%M%SZ)
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
SRC=$(basename "$(cd "$S/../../.." && pwd)")
R=$S/v2/release/records/dev2-27b-fb-release-2026-09-30
RENAME_OPS=$S/v2/release/records/dev2-rename-9b-27b-2026-09-29/ops
IN=/data/dev2/runs/release/inputs/dev2-27b-fb
G=/data/dev2/private/panels/goldfree
D=/data/dev2/runs/release/decisions
F=/data/dev2/runs/27b/m4b/F-b/formal
X=/data/dev2/runs/27b/m4b/mlx-diag/F-b
IMG=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
HFPY=/data/dev2/tools/hf-cli/bin/python
COLL=llm-semantic-router/decision-20-6ab7cf7bdfb506bf8269cb00
CACHE=/data/dev2/runs/release/bf16z-cache
REPO=llm-semantic-router/DEV2.0-27B
export TMPDIR=/data/dev2/tmp PYTHONPATH=$S
mkdir -p "$TMPDIR" "$D" "$CACHE" /data/dev2/runs/release/triton
cpu=(docker run --rm --network none -e HIP_VISIBLE_DEVICES= -e PYTHONPATH="$S" -v "$S:$S:ro" -w "$S" --entrypoint python3)
decision() { # $1 records name, $2 node name
  if [[ -e "$D/$2" ]]; then cmp "$R/$1" "$D/$2"; else cp "$R/$1" "$D/$2" && chmod 444 "$D/$2"; fi
}
cache_copy() { # $1 frozen dir, $2 post json -> copy path on stdout
  local expect dest
  expect=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["post_sha256"])' "$2")
  dest=/data/dev2/runs/release/triton/dev2-27b-fb-$(basename "$(dirname "$1")")-$TS
  python3 "$S/v2/27b/triton_cache.py" copy --frozen "$1" --dest "$dest" --expect "$expect" >&2
  echo "$dest"
}
common=(--src "$SRC" --image "$IMG" --track release-27b --shared-lease release-27b --threads 4
  --site /opt/decision-fla --require-kernels --env TRITON_CACHE_AUTOTUNING=1 --env HIP_FORCE_DEV_KERNARG=1
  --env "DECISION2_CACHE=$CACHE" --mount-rw "$CACHE" --mount "$G" --mount "$IN" --mount "$F" --mount "$X")
case "$mode" in
  verify)
    "${cpu[@]}" -v "$IN:$IN" "$IMG" -B -m v2.release.runtime.bf16z verify --dir "$IN/bf16z" \
      --receipt "$IN/bf16z.json" --output "$IN/bf16z-verify.json"
    sha256sum "$IN/bf16z-verify.json" ;;
  c1pkg)
    decision DEV2.0-27B.decision.draft.json DEV2.0-27B.decision.fb-draft.json
    [[ ! -e "$IN/c1pkg" ]] || { echo "$IN/c1pkg exists" >&2; exit 1; }
    python3 -m v2.release.build --spec "$S/v2/release/specs/dev2-27b-fb-c1pkg.json" --output "$IN/c1pkg/DEV2.0-27B"
    sha256sum "$IN/c1pkg/DEV2.0-27B/MODEL_MANIFEST.json" ;;
  parity | parity-mlx | release)
    [[ "$gpu" =~ ^[012]$ ]] || { echo "node B GPU0-2 only (--gpu 0|1|2)" >&2; exit 2; }
    rocm-smi --showuse --showmeminfo vram --json | python3 "$RENAME_OPS/pick_gpu.py" 130 "$gpu" >/dev/null \
      || { echo "GPU$gpu is busy or lacks free VRAM" >&2; exit 1; }
    if [[ "$mode" == release ]]; then
      decision DEV2.0-27B.decision.json DEV2.0-27B.decision.fb.json
      spec=$S/v2/release/specs/dev2-27b-fb-release.json
    else
      decision DEV2.0-27B.decision.draft.json DEV2.0-27B.decision.fb-draft.json
      spec=$S/v2/release/specs/dev2-27b-fb-draft.json
    fi
    if [[ "$mode" == parity-mlx ]]; then
      TC=$(cache_copy "$X/triton-cache" "$X/triton-cache.post.json")
      parity=(--parity "mlx-diag:$G/mlx-diag.prompts.jsonl:$X/output/mlx-diag.predictions.jsonl:2275")
    else
      TC=$(cache_copy "$F/triton-cache" "$F/triton-cache.post.json")
      n=(1600 6547 231)
      [[ "$mode" == release ]] && n=(200 300 100)
      parity=(--parity "typed-final:$G/typed-final.prompts.jsonl:$F/output/typed-final.predictions.jsonl:${n[0]}"
        --parity "css15:$G/css15.prompts.jsonl:$F/output/css15.predictions.jsonl:${n[1]}"
        --parity "public231:$G/public231.prompts.jsonl:$F/output/public231.predictions.jsonl:${n[2]}")
    fi
    hub=()
    [[ "$mode" == release ]] && hub=(--upload --collect --already-collected)
    W=/data/dev2/runs/release/dev2-27b-fb-$mode-$TS
    trap 'rm -f "/data/dev2/leases/gpu$gpu.lock/owner.release-27b"' EXIT
    [[ "$mode" != release ]] || bash "$S/v2/common/hf_headroom.sh" --min-free-gb 40
    echo "mirror $SRC mode $mode gpu $gpu work $W cache $TC"
    status=0
    "$S/v2/release/release.sh" --spec "$spec" --work "$W" --gpu "$gpu" "${common[@]}" \
      --env "TRITON_CACHE_DIR=$TC" --mount-rw "$TC" "${parity[@]}" "${hub[@]}" || status=$?
    python3 "$S/v2/27b/triton_cache.py" finish --dest "$TC" || true
    [[ "$status" == 0 ]] || { echo "release.sh failed ($status); work=$W" >&2; exit "$status"; }
    [[ "$mode" == release ]] || { echo "work=$W $mode passed"; exit 0; }
    REV=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['revision'])" "$W/receipts/upload.json")
    mkdir -p "$W/extra"
    "${cpu[@]}" -v "$W:$W" "$IMG" -B -m v2.release.runtime.bf16z verify --dir "$W/download/DEV2.0-27B" \
      --output "$W/extra/bf16z-verify-download.json" || status=1
    cd "$S"
    "$HFPY" "$RENAME_OPS/collection_order.py" "$COLL" "$W/extra/collection-order.json" || status=1
    "$HFPY" -m v2.release.tests.hub_card_http_check --repo "$REPO" --revision "$REV" \
      --package "$W/package/DEV2.0-27B" --output "$W/extra/card-http.json" || status=1
    "$HFPY" -m v2.release.hub_links --repo "$REPO" --revision "$REV" \
      --package "$W/package/DEV2.0-27B" --output "$W/extra/hub-links.json" || status=1
    python3 -m v2.release.gate evaluate --work "$W" > "$W/extra/gate-evaluate.json" || status=1
    bash "$S/v2/common/hf_headroom.sh" --min-free-gb 0
    echo "work=$W revision=$REV post_checks=$([[ $status == 0 ]] && echo ok || echo FAILED)"
    exit "$status" ;;
  *) echo "mode: verify | c1pkg | parity | parity-mlx | release" >&2; exit 2 ;;
esac
