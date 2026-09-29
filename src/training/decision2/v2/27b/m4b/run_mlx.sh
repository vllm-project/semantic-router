#!/usr/bin/env bash
# M4b mlx-diag collection of one frozen finalist package (node B host side, one GPU of GPU0-2). Multilingual
# development diagnostic (2,275 prompts), never a release score: m3f2/f2-mlx.sh with launch3's lease and
# track 27b-m4b (m4b/common3.sh). The package's checkpoint, calibration and limit; one fresh copy of the
# package's own scored post-run cache (<package>/../formal/triton-cache, its tree hash from
# triton-cache.post.json) checked before each collection, and no autotune entry added after it. Node B has
# no mlx-diag gold: score on node A with score_mlx_nodeA.sh.
# Usage: run_mlx.sh NAME PKG GPU MIRROR_SHA
#   NAME        output /data/dev2/runs/27b/m4b/mlx-diag/NAME (and NAME-smoke); the finalist slot
#   PKG         the finalist's frozen package/PACKAGE.json (run_formal.sh)
#   MIRROR_SHA  code commit; the mirror /data/dev2/src/<sha>[-src_training_decision2]
# Stages (STAGES, default verify,smoke,collect): verify (mirror, package, cache hash, lease), smoke (8 prompts
# into NAME-smoke; must precede collect), collect (all prompts into NAME). DRY_RUN=1: see kernel_common.sh.
set -euo pipefail
echo "m4b mlx-diag $*: start $(date -u +%Y-%m-%dT%H:%M:%SZ)"

NAME=${1:?NAME} PKG=${2:?PKG} GPU=${3:?GPU} MIRROR_SHA=${4:?MIRROR_SHA}
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "NAME must be one directory name" >&2; exit 2; }
[[ "$MIRROR_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
SRC=$MIRROR_SHA
[ -d "/data/dev2/src/$SRC" ] || SRC=$MIRROR_SHA-src_training_decision2
S=/data/dev2/src/$SRC/src/training/decision2
STAGES=${STAGES:-verify,smoke,collect}
cd "$S"
export PYTHONPATH=$S PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/dev2/tmp
source "$S/v2/27b/kernel_common.sh"
source "$S/v2/27b/m4b/common3.sh"
OUT=$M4B/mlx-diag/$NAME
has() { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
need "$PKG" "$BASE" "$PANEL_ROOT/goldfree/mlx-diag.prompts.jsonl"
FROZEN=$(cd "$(dirname "$PKG")/.." && pwd -P)/formal/triton-cache
frozen=$(python3 - "$PKG" "$FROZEN.post.json" <<'EOF'
import hashlib, json, sys
package, post = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
path = package["calibration"]["path"]
if hashlib.sha256(open(path, "rb").read()).hexdigest() != package["calibration"]["sha256"]:
    raise SystemExit("package calibration changed after the freeze")
if not post["frozen_check"]["passed"] or post["classified"]["autotune"]["added"]:
    raise SystemExit(f"{sys.argv[2]}: the formal run's cache failed its frozen check")
print(package["checkpoint"], path, package["max_input_tokens"], post["post_sha256"], sep="\t")
EOF
)
IFS=$'\t' read -r CKPT CAL MAXLEN CACHE_SHA <<< "$frozen"
need "$CKPT" "$FROZEN"
REVISION="checkpoint-sha256:$(model_sha "$CAL")"

collect() {  # RUN_DIR [--max-items N]
  local dir=$1 status=0
  shift
  mkdir -p "$dir"
  cache_copy "$FROZEN" "$CACHE_SHA" "$dir/triton-cache"
  runner "$dir" "$CKPT" "$dir/triton-cache" "27b-m4b $NAME mlx-diag kernel-path diagnostic collection" \
    --mount "$(dirname "$CAL")" -- --revision "$REVISION" --extra "source=$BASE" \
    --extra "calibration=$CAL" --extra "max_length=$MAXLEN" --panels mlx-diag "$@" || status=$?
  cache_finish "$dir/triton-cache"
  [ "$status" = 0 ] || return "$status"
  [ "$DRY_RUN" = 1 ] || no_autotune_added "$dir/triton-cache"
}

if has verify; then
  verify_mirror "$MIRROR_SHA"
  verify_cache "$FROZEN" "$CACHE_SHA"
  [ "$DRY_RUN" = 1 ] || verify_lease
fi
if has smoke; then
  collect "$OUT-smoke" --max-items 8
fi
if has collect; then
  [ "$DRY_RUN" = 1 ] || [ -f "$OUT-smoke/SMOKE.json" ] || { echo "run the smoke stage first" >&2; exit 2; }
  collect "$OUT"
fi
echo "m4b mlx-diag $NAME stages $STAGES complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
