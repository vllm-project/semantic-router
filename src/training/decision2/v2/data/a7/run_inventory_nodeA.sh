#!/usr/bin/env bash
# Count-only census of the Decision 1.0 encoder-family (Kai/Lex) corpora on node A,
# run from an exact mirror with host python. Reads the 1.0 encoder data project
# read-only and writes /data/dev2/private/a7/runs/inventory-nodeA/<commit12>/census.json.
set -euo pipefail

S="$(cd "$(dirname "$0")/../../.." && pwd)"
mirror="$(cd "$S/../../.." && pwd)"
commit="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "$mirror/.dev2-mirror.json")"
W="/data/dev2/private/a7/runs/inventory-nodeA/${commit:0:12}"
E=/data/vela-jina-20260916/decision-1.0
umask 077
mkdir -p "$W"
export PYTHONPATH="$S"
cd "$S"
files=(
  KAI_SHARED=data/kai-shared-typed-v1/admitted-v2/train.jsonl
  KAI_SHARED_SEL=data/kai-shared-typed-v1/admitted-v2/select.jsonl
  KAI_SHARED_CAL=data/kai-shared-typed-v1/admitted-v2/cal.jsonl
  EXP2_NATURAL=data/expanded-stage2-v1/natural_train.jsonl
  SRC_BALANCED2=data/source-balanced-stage2-v1/train.jsonl
  SCHEMA_STAGE3=data/schema-transfer-stage3-v1/train.jsonl
  NATURAL_CHOICE=data/natural-choice-transfer-v1/train.jsonl
  NATURAL_NOUL=data/natural-noul-transfer-v1/prepared-v1/train.jsonl
  OASST_SCORE=data/oasst-score-transfer-v1/train.jsonl
  OASST1_SCORE=data/oasst1-score-v1/prepared-v1/train.jsonl
  OASST_ES_ZH=data/oasst-es-zh-schema-v1/prepared-v1/admitted-train.jsonl
  NI_MULTIFAMILY=data/natural-instructions-multifamily-v1/prepared-v1/train.jsonl
  HINGLISH_MIXED=data/hinglish-score-mixed-v1/prepared-v1/train.jsonl
  SENTIMIX=data/sentimix-hinglish-score-v1/prepared-v1/train.jsonl
  AFRISENTI=data/afrisenti-swahili-score-v1/prepared-v1/train.jsonl
  SCORE_WORLD=data/score-world-diversity-v1/prepared-v1/train.jsonl
  SCORE_MULTIRUBRIC=data/score-multirubric-v1/prepared-v2/TRAIN.jsonl.gz
  HELPSTEER2_SCORE=data/helpsteer2-score-increment-v1/prepared-v1/TRAIN.jsonl.gz
  CHOICE_CONFIDENCE=data/choice-confidence-v1/prepared-v1/train.jsonl
  BOOLQ_NOUL=data/boolq-noul-increment-v2/prepared-v1/train.jsonl
  NOUL_CATEGORY=data/noul-category-binding-v1/prepared-v1/train.jsonl
  COSMOS_CHOICE=data/cosmosqa-choice-v3/prepared-v1/train.jsonl
  QASC_CHOICE=data/qasc-choice-v1/prepared-v1/train.jsonl
  LONG_RULE=data/long-rule-coverage-v1/prepared-v1/short-train.jsonl
  KAI_NOUL_MIX=research/kai-noul-product-mix-v1/prepared-v1/train.jsonl
  TYPED_DECISIONS=data/typed-decisions-adaptation-v1/prepared-v1/train.jsonl
  TYPED_DECISIONS_DEV=data/typed-decisions-adaptation-v1/prepared-v1/dev.jsonl
)
args=()
for item in "${files[@]}"; do
  [[ -f "$E/${item#*=}" ]] && args+=(--file "${item%%=*}=$E/${item#*=}")
done
python3 -m v2.data.a7.inventory "${args[@]}" --out "$W/census.json"
echo "census written: $W/census.json commit=$commit"
