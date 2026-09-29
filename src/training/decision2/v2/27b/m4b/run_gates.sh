#!/usr/bin/env bash
# M4b host-CPU gates and rule verdicts of one sealed finalist (node B host; m3f2/f2-phase2.sh gates/overlap as
# run for F2). The gate tools run from the eval track's mirrors (gates c8a5504b5, overlap_effects 1bbebc2fe);
# m4b_rules.py from this script's own mirror.
# Usage: run_gates.sh NAME
#   NAME  finalist slot (F-a, F-b, F-c): formal run /data/dev2/runs/27b/m4b/NAME/formal, label from the
#         committed overlap-spec-27b-m4b.json
# Stages (STAGES, default gates,overlap,verdicts), outputs under /data/dev2/runs/27b/m4b/NAME/:
#   gates     panels verify; v2.eval.gates paired vs AutoJev-27B, Eikos-27B, Jebadiah-27B and candidate - F1;
#             types; the F1 record's item_cis.py driver; typed-FINAL families and Score levels -> gates/
#   overlap   overlap_effects exposure on F1's TRAIN a7.train.jsonl (de00df03), the per-finalist spec derived
#             from overlap-spec-27b-m4b.json (m4b_rules overlap-spec: sealed finalists only), run -> overlap/
#   verdicts  successor rule vs F1 and the "beats AutoJev" bar -> verdicts.json; item 4 (mlx-diag) is PENDING
#             unless MLX_OVERLAP names an overlap_effects run output with the mlx panel (node A), which writes
#             verdicts-mlx.json instead
set -euo pipefail
echo "m4b gates $*: start $(date -u +%Y-%m-%dT%H:%M:%SZ)"

NAME=${1:?NAME}
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "NAME must be one directory name" >&2; exit 2; }
STAGES=${STAGES:-gates,overlap,verdicts}
M=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)
TEMPLATE=$M/v2/27b/m4b/overlap-spec-27b-m4b.json
R=/data/dev2/runs/27b
OUT=$R/m4b/$NAME
CAND=$OUT/formal
F1=$R/M3-A-soup/formal
F1_LABEL="DEV2.0-27B (F1)"
TRAIN=/data/dev2/private/27b/m3-data/mixtures-m3-1/a7.train.jsonl
TRAIN_SHA=de00df035439346e0da2bf77b594ce0c3f7d058b61040f149f0c75c61ce7f397
GATES=/data/dev2/src/c8a5504b5b39049390fccfbe3bcac5ef65e1e710-src_training_decision2/src/training/decision2
OVERLAP=/data/dev2/src/1bbebc2fe3136858bcda63a12c2cde13c45d9238-src_training_decision2/src/training/decision2
ITEM_CIS=/data/dev2/src/5553298a1a7e1c8b192eab1bb7c2699689b7a3dd-src_training_decision2/src/training/decision2/v2/eval/records/m4-dev2-27b-f1-gates/item_cis.py
O=/data/dev2/runs/eval/m5/overlap-effects
G=$OUT/gates V=$OUT/overlap
PEERS=(
  "autojev27 AutoJev-27B $R/m2-peer-autojev27-nodeB-kernel"
  "eikos27b Eikos-27B $R/m3-peer-eikos27-nodeB-kernel"
  "jebadiah27b Jebadiah-27B $R/m3-peer-jebadiah-nodeB-kernel"
)
export TMPDIR=/data/dev2/tmp PYTHONDONTWRITEBYTECODE=1
has() { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
LABEL=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["m4b"][sys.argv[2]]["label"])' \
  "$TEMPLATE" "$NAME")
[ -f "$CAND/SEAL.json" ] && [ -f "$CAND/REPORT.json" ] || { echo "$CAND is not sealed and reported" >&2; exit 2; }
echo "candidate $LABEL: $CAND"

if has gates; then
  [ ! -e "$G" ] || { echo "$G exists" >&2; exit 66; }
  mkdir -p "$G"
  cd "$GATES"
  export PYTHONPATH=$GATES
  date -u +%FT%TZ > "$G/run.start"
  python3 -m v2.eval.panels verify --panel typed-final --panel css15 --panel public231 > "$G/panels-verify.json"
  for peer in "${PEERS[@]}"; do
    read -r key name run <<< "$peer"
    python3 -m v2.eval.gates paired --left "$CAND" --right "$run" --left-name "$LABEL" \
      --right-name "$name" --output "$G/paired-vs-$key.json" > "$G/paired-vs-$key.log"
  done
  python3 -m v2.eval.gates paired --left "$CAND" --right "$F1" --left-name "$LABEL" \
    --right-name "$F1_LABEL" --output "$G/paired-vs-f1.json" > "$G/paired-vs-f1.log"
  python3 -m v2.eval.gates types --run "$CAND" --label "$LABEL" --output "$G/types-cand.json" \
    > "$G/types-cand.log"
  cfg=$(python3 - "$R" "$CAND" "$F1" "$G" "$LABEL" "$F1_LABEL" <<'EOF'
import json, sys
r, cand, f1, g, label, f1_label = sys.argv[1:]
peers = {"AutoJev-27B": f"{r}/m2-peer-autojev27-nodeB-kernel", "Eikos-27B": f"{r}/m3-peer-eikos27-nodeB-kernel",
         "Jebadiah-27B": f"{r}/m3-peer-jebadiah-nodeB-kernel"}
print(json.dumps({"models": {label: cand, f1_label: f1, **peers},
                  "pairs": [[label, n] for n in [*peers, f1_label]], "output": f"{g}/item-cis.json", "jobs": 8}))
EOF
  )
  python3 - "$cfg" < "$ITEM_CIS" > "$G/item-cis.log"
  python3 - "$F1/REPORT.json" "$CAND/REPORT.json" "$G/types-cand.json" > "$G/families-score-levels.log" <<'EOF'
import json, sys
f1, cand, types = (json.load(open(p)) for p in sys.argv[1:])
fam1, fam2 = (r["panels"]["typed-final"]["by_family"] for r in (f1, cand))
for k in sorted(fam1):  # point values only: the tooling gives intervals for T, not per family
    print(f"{k}: F1 {fam1[k]:.4f}  candidate {fam2[k]:.4f}  candidate-F1 {fam2[k] - fam1[k]:+.4f}")
score = types["types"]["score"]
print("Score predicted", score["predicted_distribution"], "gold", score["gold_distribution"])
print("Score recall by level", score["recall_by_level"])
EOF
  date -u +%FT%TZ > "$G/run.end"
  (cd "$G" && sha256sum -- * > SHA256SUMS.txt)
fi
if has overlap; then
  [ ! -e "$V" ] || { echo "$V exists" >&2; exit 66; }
  mkdir -p "$V"
  (cd "$OVERLAP" && PYTHONPATH=$OVERLAP python3 -m v2.eval.overlap_effects exposure \
    --groups "$O/final/excluded-groups.json" --train "$TRAIN" --expect-sha256 "$TRAIN_SHA" \
    --label "$LABEL (M4b; F1's TRAIN a7.train.jsonl)" --output "$V/exposure.json") > "$V/exposure.log"
  (cd "$M" && PYTHONPATH=$M python3 -m v2.27b.m4b.m4b_rules overlap-spec --template "$TEMPLATE" --name "$NAME" \
    --exposure "$V/exposure.json" --output "$V/spec.json")
  (cd "$OVERLAP" && PYTHONPATH=$OVERLAP python3 -m v2.eval.overlap_effects run --spec "$V/spec.json" \
    --flagged "$O/final-27b/flagged.json" --output "$V/overlap-effects.json" --jobs 8) > "$V/run.log" 2>&1
fi
if has verdicts; then
  mlx=() output=$OUT/verdicts.json
  if [ -n "${MLX_OVERLAP:-}" ]; then
    mlx=(--mlx-overlap "$MLX_OVERLAP") output=$OUT/verdicts-mlx.json
  fi
  (cd "$M" && PYTHONPATH=$M python3 -m v2.27b.m4b.m4b_rules verdicts --label "$LABEL" --f1-label "$F1_LABEL" \
    --paired-f1 "$G/paired-vs-f1.json" --paired-peer "AutoJev-27B=$G/paired-vs-autojev27.json" \
    --paired-peer "Eikos-27B=$G/paired-vs-eikos27b.json" --paired-peer "Jebadiah-27B=$G/paired-vs-jebadiah27b.json" \
    --types "$G/types-cand.json" --exposure "$V/exposure.json" --overlap "$V/overlap-effects.json" \
    "${mlx[@]}" --report "$CAND/REPORT.json" --output "$output")
fi
echo "m4b gates $NAME stages $STAGES complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
