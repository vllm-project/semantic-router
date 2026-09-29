#!/usr/bin/env bash
# ~27B M3 mlx-diag collection of one frozen finalist package (node B host). Multilingual
# development diagnostic (2,275 prompts), never a release score. It is run_formal_kernel.sh's collect() with
# --panels mlx-diag: the 27B kernel adapter (KERNEL_SPEC), the package's checkpoint, calibration and limit, one
# fresh copy of FROZEN checked against CACHE_SHA before the run and cache_finish after it, and kernel_common.sh's
# runner (idle wait, 27b lease, eval runner). FROZEN defaults to the snapshot of F1's post-run scored cache, so
# every autotune key shared with the formal runs stays identical. Node B has no mlx-diag gold: score on node A
# (f2-mlx-score.sh).
# Usage: f2-mlx.sh NAME GPU [smoke]
#   NAME   finalist under /data/dev2/runs/27b with a frozen package/PACKAGE.json (M3-A-soup, M3-S-soup, ...)
#   smoke  8 prompts into MLX_ROOT/NAME-smoke; run it once first (first mlx-diag use of the kernel adapter)
# Env: SRC (mirror; default 35fa052d2), FROZEN, CACHE_SHA, MLX_ROOT (default /data/dev2/runs/27b/m3-f2/mlx-diag),
#      DRY_RUN=1 (kernel_common.sh; the cache copy still runs, so dry-run only into a scratch MLX_ROOT).
set -euo pipefail

NAME=$1 GPU=$2 MODE=${3:-full}
SRC=${SRC:-35fa052d2b7c3ad0f9b2ee9bd1529e6b28076029-src_training_decision2}
S=/data/dev2/src/$SRC/src/training/decision2
R=/data/dev2/runs/27b
FROZEN=${FROZEN:-$R/m3-f2/f1-scored-cache}
CACHE_SHA=${CACHE_SHA:-03b172f1a6adeef6c6a6c491d04389b355c9d8579480008023f408c8659b502b}
OUT=${MLX_ROOT:-$R/m3-f2/mlx-diag}/$NAME
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "NAME must be one directory name" >&2; exit 2; }
case "$GPU" in 5 | 6 | 7) ;; *) echo "GPU$GPU is outside the ~27B allocation" >&2; exit 2 ;; esac
case "$MODE" in
  full) smoke=() ;;
  smoke) OUT=$OUT-smoke smoke=(--max-items 8) ;;
  *) echo "mode is full or smoke" >&2; exit 2 ;;
esac
[[ "$CACHE_SHA" =~ ^[0-9a-f]{64}$ ]] || { echo "CACHE_SHA must be a full SHA-256" >&2; exit 2; }
export TMPDIR=/data/dev2/tmp PYTHONDONTWRITEBYTECODE=1
cd "$S"
export PYTHONPATH=$S
source "$S/v2/27b/kernel_common.sh"
need "$R/$NAME/package/PACKAGE.json" "$BASE" "$FROZEN" "$PANEL_ROOT/goldfree/mlx-diag.prompts.jsonl"
frozen=$(python3 - "$R/$NAME/package/PACKAGE.json" <<'EOF'
import hashlib, json, sys
package = json.load(open(sys.argv[1]))
path = package["calibration"]["path"]
if hashlib.sha256(open(path, "rb").read()).hexdigest() != package["calibration"]["sha256"]:
    raise SystemExit("package calibration changed after the freeze")
print(package["checkpoint"], path, package["max_input_tokens"], sep="\t")
EOF
)
IFS=$'\t' read -r CKPT CAL MAXLEN <<< "$frozen"
need "$CKPT"
REVISION="checkpoint-sha256:$(model_sha "$CAL")"

mkdir -p "$OUT"
cache_copy "$FROZEN" "$CACHE_SHA" "$OUT/triton-cache"
status=0
runner "$OUT" "$CKPT" "$OUT/triton-cache" "27b $NAME mlx-diag kernel-path diagnostic collection" \
  --mount "$(dirname "$CAL")" -- --revision "$REVISION" --extra "source=$BASE" \
  --extra "calibration=$CAL" --extra "max_length=$MAXLEN" --panels mlx-diag "${smoke[@]}" || status=$?
cache_finish "$OUT/triton-cache"
[ "$status" = 0 ] || exit "$status"
echo "mlx-diag $NAME ($MODE) collected into $OUT"
