#!/usr/bin/env bash
# ~27B M5 host-CPU gates of the sealed finalists (node B; no GPU): the successor rule vs A20r's scored run, the
# beats-AutoJev check and the attribution contrasts (preregistration "Formal runs, rules and attribution").
# Usage: m5-gates.sh MIRROR_SHA STAGE NAME...
#   gates     panels verify; per NAME v2.eval.gates paired vs A20r, AutoJev-27B, Eikos-27B, Jebadiah-27B, F1 (M3-A) and
#             A20r - NAME; types; public231 vs A20r (item 7); family contrasts (m4_contrast.py, group-level CIs) for the
#             preregistered pairs whose runs are sealed -> /data/dev2/runs/27b/m5/gates/
#   overlap   v2.eval.overlap_effects exposure of each NAME's TRAIN (a20 for FF20 / L128, a20h for FF20H, both for SX)
#             -> /data/dev2/runs/27b/m5/gates/overlap/
#   verdicts  m5_verdicts.py: items 1-7 and the beats-AutoJev check per NAME (item 4 from node A's mlx-paired output
#             relayed to /data/dev2/runs/27b/m5/gates/mlx/NAME-vs-A20r.json) -> gates/VERDICTS.json
set -euo pipefail
echo "m5 gates $*: start $(date -u +%FT%TZ)"
SHA=${1:?MIRROR_SHA} STAGE=${2:?STAGE}
shift 2
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
S=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2
[ -f "$S/v2/27b/m5/m5_verdicts.py" ] || { echo "missing mirror $SHA" >&2; exit 2; }
cd "$S"
export PYTHONPATH=$S PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/dev2/tmp
B=/data/dev2/runs/27b R=$B/m5 G=$B/m5/gates
A20R=$B/M4-A20r-soup/formal
declare -A PEER=([AutoJev-27B]=$B/m2-peer-autojev27-nodeB-kernel [Eikos-27B]=$B/m3-peer-eikos27-nodeB-kernel
  [Jebadiah-27B]=$B/m3-peer-jebadiah-nodeB-kernel)
declare -A RUN=([M4-A20r-soup]=$A20R [M4-A20-soup]=$B/M4-A20-soup/formal [F-b]=$B/m4b/F-b/formal
  ["DEV2.0-27B (F1)"]=$B/M3-A-soup/formal)
for name in "$@"; do
  [ -f "$R/$name/formal/SEAL.json" ] || { echo "$name has no sealed formal run" >&2; exit 2; }
  RUN[$name]=$R/$name/formal
done
paired() {  # LEFT_NAME RIGHT_NAME OUTPUT
  python3 -m v2.eval.gates paired --left "${RUN[$1]}" --left-name "$1" --right "${RUN[$2]}" --right-name "$2" \
    --output "$3" > "${3%.json}.log"
}
mkdir -p "$G"
case "$STAGE" in
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
      python3 -m v2.eval.gates types --run "${RUN[$name]}" --label "$name" --output "$D/types.json" > "$D/types.log"
      python3 -m v2.eval.gates public231 --left "${RUN[$name]}" --right "$A20R" --left-name "$name" \
        --right-name M4-A20r-soup --output "$D/public231-vs-A20r.json" > "$D/public231-vs-A20r.log"
    done
    args=() pairs=()
    for name in "${!RUN[@]}"; do args+=(--run "$name=${RUN[$name]}"); done
    for spec in "M5-FF20:M4-A20-soup" "M5-FF20:M4-A20r-soup" "M5-FF20:F-b" "M5-FF20H:M5-FF20" "M5-SX:M5-FF20H" \
      "M5-SX:M5-FF20" "M5-L128:M4-A20r-soup" "M5-FF20H:M4-A20r-soup" "M5-SX:M4-A20r-soup"; do
      [[ -n "${RUN[${spec%%:*}]:-}" && -n "${RUN[${spec#*:}]:-}" ]] && pairs+=(--pair "$spec")
    done
    for name in "$@"; do args+=(--finalist "$name"); done
    python3 -m v2.27b.m4_contrast "${args[@]}" "${pairs[@]}" --output "$G/contrast.json" > "$G/contrast.log"
    ;;
  overlap)
    V=$G/overlap
    mkdir -p "$V"
    FLAGGED=/data/dev2/runs/eval/m5/overlap-effects/final/excluded-groups.json
    MX=/data/dev2/private/27b/m5-data/mixtures-m5-1
    for mix in a20:4aa0dc964505682b2840fa5167a7ec14983e5a0cea427480bed1996f1befc0d4 \
      a20h:4a9d93f56e5dc4f5715dfe7c310375199484546b51d452dc94b6ca0332306202; do
      python3 -m v2.eval.overlap_effects exposure --groups "$FLAGGED" --train "$MX/${mix%%:*}.train.jsonl" \
        --expect-sha256 "${mix#*:}" --label "DEV2.0-27B M5 ${mix%%:*}.train.jsonl" \
        --output "$V/exposure-m5-${mix%%:*}.json" > "$V/exposure-${mix%%:*}.log"
    done
    ;;
  verdicts)
    args=()
    for name in "$@"; do args+=(--finalist "$name=${RUN[$name]}"); done
    python3 -m v2.27b.m5.m5_verdicts --gates "$G" "${args[@]}" --output "$G/VERDICTS-$(date -u +%Y%m%dT%H%M%SZ).json"
    ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
sums=$(cd "$G" && find . -type f ! -name SHA256SUMS.txt -print0 | LC_ALL=C sort -z | xargs -0 sha256sum)
printf '%s\n' "$sums" > "$G/SHA256SUMS.txt"
echo "m5 gates $STAGE complete: $(date -u +%FT%TZ)"
