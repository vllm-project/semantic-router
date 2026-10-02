#!/usr/bin/env bash
# ~27B M6 host-CPU gates of the sealed finalists (node B; no GPU): successor items 1-7 vs A20r's scored run, the
# beats-AutoJev check and the attribution contrasts (preregistration "Formal, successor rule and Index",
# "Attribution"). M5's m5-gates.sh with the M6 root, M5-L128 as a comparator and the M6 mixtures.
# Usage: m6-gates.sh MIRROR_SHA STAGE NAME...      (NAME: M6-IB, M6-IBX, M6-IB2, M6-IB2PN, a cross-arm soup
#                                                   M6-IBxIB2-mNN or an M7 arm soup, with a sealed formal run; the
#                                                   overlap stage reads M6 builds only)
#   gates     panels verify; per NAME v2.eval.gates paired vs A20r, AutoJev-27B, Eikos-27B, Jebadiah-27B, F1, M5-L128 and
#             A20r - NAME; types; public231 vs A20r (item 7); family contrasts (m4_contrast.py) -> /data/dev2/runs/27b/m6/gates/
#   overlap   v2.eval.overlap_effects exposure of each M6 TRAIN file listed in BUILD.json -> gates/overlap/exposure-m6-<mix>.json
#   verdicts  m5_verdicts.py with the M6 TRAIN mapping (items 1-7 and beats-AutoJev; item 4 from node A's mlx-paired
#             output relayed to gates/mlx/NAME-vs-A20r.json; item 8 PENDING, the eval custodian's) -> gates/VERDICTS-<UTC>.json
#   contrast  the family contrasts of `gates` only, over every NAME given (the attribution after all chains; each
#             chain's gates stage already wrote its per-arm files, which refuse to be rewritten)
#             -> gates/contrast-all-<UTC>.json
set -euo pipefail
echo "m6 gates $*: start $(date -u +%FT%TZ)"
SHA=${1:?MIRROR_SHA} STAGE=${2:?STAGE}
shift 2
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
S=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2
[ -f "$S/v2/27b/m6/m6-gates.sh" ] || { echo "missing mirror $SHA" >&2; exit 2; }
cd "$S"
export PYTHONPATH=$S PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/dev2/tmp
B=/data/dev2/runs/27b R=$B/m6 G=$B/m6/gates
A20R=$B/M4-A20r-soup/formal
BUILDS=${BUILDS:-/data/dev2/private/27b/m6-data}
declare -A PEER=([AutoJev-27B]=$B/m2-peer-autojev27-nodeB-kernel [Eikos-27B]=$B/m3-peer-eikos27-nodeB-kernel
  [Jebadiah-27B]=$B/m3-peer-jebadiah-nodeB-kernel)
declare -A RUN=([M4-A20r-soup]=$A20R [M5-L128]=$B/m5/M5-L128/formal ["DEV2.0-27B (F1)"]=$B/M3-A-soup/formal)
declare -A MIX=([M6-IB]=a20ib1 [M6-IBX]=a20ib1x [M6-IB2]=${IB2_MIX:-a20ib12} [M6-IB2PN]=a20ib12pn
  [M7-IB124ML]=a20ib124ml [M7-IB14ML]=a20ib14ml [M8-IB14]=a20ib14 [M8-IB124]=a20ib124 [X8-IBxIB2-8]=a20ib12)
for name in "$@"; do
  # a cross-arm soup of M6-IB and M6-IB2 seeds: every a20ib1 row is an a20ib12 row, so a20ib12 is its training rows
  [[ "$name" =~ ^M6-IBxIB2-m[0-9]{2}$ ]] && MIX[$name]=a20ib12
  # the M7 / M8 cross-arm soups train on several mixtures, none holding the others: no single exposure file
  [[ "$name" =~ ^X7-|^X8-ML$ ]] && MIX[$name]=several
  [ -n "${MIX[$name]:-}" ] || { echo "unknown M6 finalist $name" >&2; exit 2; }
  [ -f "$R/$name/formal/SEAL.json" ] || { echo "$name has no sealed formal run" >&2; exit 2; }
  RUN[$name]=$R/$name/formal
done
paired() {  # LEFT_NAME RIGHT_NAME OUTPUT
  python3 -m v2.eval.gates paired --left "${RUN[$1]}" --left-name "$1" --right "${RUN[$2]}" --right-name "$2" \
    --output "$3" > "${3%.json}.log"
}
mkdir -p "$G"
contrast() {  # OUTPUT NAME...: m4_contrast over every run, the M6 pairs present and the NAMEs as finalists
  local out=$1 name spec
  shift
  local -a args=() pairs=()
  for name in "${!RUN[@]}"; do args+=(--run "$name=${RUN[$name]}"); done
  for spec in "M6-IB:M4-A20r-soup" "M6-IBX:M4-A20r-soup" "M6-IB2:M4-A20r-soup" "M6-IB2PN:M4-A20r-soup" \
    "M6-IB:M5-L128" "M6-IBX:M5-L128" "M6-IB2:M5-L128" "M6-IB2PN:M5-L128" "M6-IB:M6-IBX" "M6-IB2:M6-IB" \
    "M6-IB2PN:M6-IB2" "M5-L128:M4-A20r-soup"; do
    [[ -n "${RUN[${spec%%:*}]:-}" && -n "${RUN[${spec#*:}]:-}" ]] && pairs+=(--pair "$spec")
  done
  for name in "$@"; do args+=(--finalist "$name"); done
  python3 -m v2.27b.m4_contrast "${args[@]}" "${pairs[@]}" --output "$out" > "${out%.json}.log"
}
case "$STAGE" in
  contrast)
    for p in "${!PEER[@]}"; do RUN[$p]=${PEER[$p]}; done
    contrast "$G/contrast-all-$(date -u +%Y%m%dT%H%M%SZ).json" "$@"
    ;;
  gates)
    for p in "${!PEER[@]}"; do RUN[$p]=${PEER[$p]}; done
    python3 -m v2.eval.panels verify --panel typed-final --panel css15 --panel public231 > "$G/panels-verify.json"
    for name in "$@"; do
      D=$G/$name
      mkdir -p "$D"
      paired "$name" M4-A20r-soup "$D/paired-vs-A20r.json"
      paired M4-A20r-soup "$name" "$D/paired-A20r-minus-cand.json"
      paired "$name" AutoJev-27B "$D/paired-vs-autojev27.json"
      paired "$name" Eikos-27B "$D/paired-vs-eikos27b.json"
      paired "$name" Jebadiah-27B "$D/paired-vs-jebadiah27b.json"
      paired "$name" "DEV2.0-27B (F1)" "$D/paired-vs-F1.json"
      paired "$name" M5-L128 "$D/paired-vs-L128.json"
      python3 -m v2.eval.gates types --run "${RUN[$name]}" --label "$name" --output "$D/types.json" > "$D/types.log"
      python3 -m v2.eval.gates public231 --left "${RUN[$name]}" --right "$A20R" --left-name "$name" \
        --right-name M4-A20r-soup --output "$D/public231-vs-A20r.json" > "$D/public231-vs-A20r.log"
    done
    contrast "$G/contrast.json" "$@"
    ;;
  overlap)
    V=$G/overlap
    mkdir -p "$V"
    FLAGGED=/data/dev2/runs/eval/m5/overlap-effects/final/excluded-groups.json
    for name in "$@"; do
      mix=${MIX[$name]}
      [ "$mix" != several ] || { echo "$name trains on several mixtures: run overlap for each M6 arm" >&2; exit 2; }
      file=$(find "$BUILDS" -path "*/mixtures-m6*-1/$mix.train.jsonl" | sort | head -n 1)
      [ -n "$file" ] || { echo "no frozen $mix.train.jsonl under $BUILDS" >&2; exit 2; }
      sha=$(python3 - "$(dirname "$(dirname "$file")")" "$(basename "$(dirname "$file")")" "$mix.train.jsonl" <<'EOF'
import glob, json, os, sys
root, mixdir, name = sys.argv[1:]
for path in sorted(glob.glob(os.path.join(root, "BUILD*.json"))):
    build = json.load(open(path))
    if build.get("mixtures_dir", "mixtures-m6-1") == mixdir and name in build["files_sha256"]:
        print(build["files_sha256"][name])
        break
else:
    raise SystemExit(f"no build record lists {mixdir}/{name}")
EOF
)
      python3 -m v2.eval.overlap_effects exposure --groups "$FLAGGED" --train "$file" --expect-sha256 "$sha" \
        --label "DEV2.0-27B M6 $mix.train.jsonl" --output "$V/exposure-m6-$mix.json" > "$V/exposure-$mix.log"
    done
    ;;
  verdicts)
    args=()
    for name in "$@"; do args+=(--finalist "$name=${RUN[$name]}" --train-of "$name=${MIX[$name]}"); done
    python3 -m v2.27b.m5.m5_verdicts --gates "$G" "${args[@]}" --exposure-prefix exposure-m6- \
      --output "$G/VERDICTS-$(date -u +%Y%m%dT%H%M%SZ).json"
    ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
sums=$(cd "$G" && find . -type f ! -name SHA256SUMS.txt -print0 | LC_ALL=C sort -z | xargs -0 sha256sum)
printf '%s\n' "$sums" > "$G/SHA256SUMS.txt"
echo "m6 gates $STAGE complete: $(date -u +%FT%TZ)"
