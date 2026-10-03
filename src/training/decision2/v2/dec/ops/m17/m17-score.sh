#!/usr/bin/env bash
# Decoder M17 node-A scoring (prereg dec-m17-prereg-2026-10-02.md, "Development readouts and gates"), CPU only, from an
# exact mirror. Gold stays on node A; node F readouts are pulled gold-free over the temporary transfer key. The
# retention probes use the 4B tier gold of M12's hit list (M15's file, f712fc53...: every M17 TRAIN row is a released
# or IB row); IB DEV uses M11's panel gold; MLX-DEV2 uses the eval track's panel gold (9d9398f4...).
#
#   m17-score.sh pull <user@node> <point> [<point> ...]   lines/<point>/ predictions, manifests, receipts, old MLX-DEV
#   m17-score.sh mlx2-pull <user@node> <name> [...]        private MLX-DEV2 predictions of <name> (node F mlx2/<name>/)
#   m17-score.sh probe-gold                                the M17 4B probe gold (M15's file, hash-checked)
#   m17-score.sh mlx2 <REF> <point> [<point> ...]          MLX-DEV2 compare (v2.eval.mlx_dev2 compare, 5,000
#                                                          replicates, seed 20260927) -> lines/<point>/mlx2cmp/
#   m17-score.sh mlx2-agree <REF>                          report only: REF's node-F read vs the eval track's LH read
#   m17-score.sh ibdev <REF> <point> [<point> ...]         IB DEV (12 families) and transfer (9 families) macros
#   m17-score.sh points <REF> <point> [<point> ...]        HT-DEV v2, Score5-typed-DEV, retention probes, hs1-dev
#   m17-score.sh contrast <A> <B>                          HT-DEV v2 and probes of A vs B (report only)
#   m17-score.sh readout <REF> <point> [...]               typed DEV + CSS pilot (v2.dec.dev_readout) -> readout/4b.json
#   m17-score.sh rules arms <X=REF> [...]                  m17_rules.py arms -> m17/select/4b-arms.json (runs once)
#   m17-score.sh rules final [<X=REF> ...]                 m17_rules.py final -> m17/select/4b-finalists.json (once)
set -u
S=$(cd "$(dirname "$0")/../../../.." && pwd)
MIRROR=${S%/src/training/decision2}
[ -f "$MIRROR/.dev2-mirror.json" ] || { echo "run from an exact mirror under /data/dev2/src" >&2; exit 2; }
M=/data/dev2/runs/dec/m17
L=$M/lines
OPS=$S/v2/dec/ops
GOLD=/data/dev2/private/panels/gold
HT_GOLD=$GOLD/ht-dev2.gold.jsonl HT_GOLD_SHA=659c92b45d2a5b37e250ccb728cdbf5580ba361b3fc4b2f2ab505a13e0a556cc
PG=/data/dev2/private/dec/m17
PROBE_GOLD_M15=/data/dev2/private/dec/m15/m10-probes.4b.gold.jsonl
PROBE_GOLD_SHA=f712fc5354e2cd3d25b7cf6127d2677b61bf0ffc73f3833d284134d91b1de2df
HITS_M15=/data/dev2/runs/dec/m15/probes/hits-4b.json HITS_SHA=0083e8b91f8a997402be43e6c569939d62493c845f1a6f6213fcdd1a3bf4f328
IB_GOLD=/data/dev2/private/dec/m11/ib-dev.gold.jsonl
IB_GOLD_SHA=a69cf36d9fa6f85b1c1139dd3ecc1d613479ff6ed589675e9bc4ef3af0adb1d3
MLX2_PANEL=/data/dev2/private/panels/mlx-dev2-v1
MLX2_GOLD_SHA=9d9398f4a5753cebf750429c6089ef833e7bf1f9d8f67bb5c330f9cdc01ee426
MLX2=$PG/mlx2
MLX2_EVAL_LH=/data/dev2/private/eval/mlx-dev2/runs/LH/mlx-dev2.predictions.jsonl
KEY='ssh -i /root/.ssh/d2_temp_cd -o BatchMode=yes'
log() { echo "$(date -u +%FT%TZ) score $*" | tee -a "$L/OPERATIONS-nodeA.log"; }
die() { log "$*"; exit 1; }
py() { (cd "$S" && PYTHONPATH=$S python3 -B "$@"); }
pred() { echo "$L/$1/$2/$2.predictions.jsonl"; }
mkdir -p "$L/diag" "$L/readout" "$M/select" "$PG"
chmod 700 "$PG"
exec 9> "$M/select/.m17-score.lock"
flock -n 9 || die "another m17-score.sh is running"

ht() {  # <left> <right> <out name>
  local out=$L/diag/$3.htdev2.json
  [ -f "$out" ] && return 0
  py "$OPS/m7/m7_htdev2.py" --gold "$HT_GOLD" --left "$(pred "$1" ht-dev2)" --left-name "$1" \
    --right "$(pred "$2" ht-dev2)" --right-name "$2" --output "$out" > "$L/diag/$3.htdev2.log" 2>&1 \
    || die "$3 HT-DEV v2 FAILED (see $L/diag/$3.htdev2.log)"
  log "$3 ht-dev2: $(tail -1 "$L/diag/$3.htdev2.log" | cut -c1-300)"
}
probes() {  # <left> <right> <out name>
  local out=$L/diag/$3.probes.json gold=$PG/m10-probes.4b.gold.jsonl
  [ -f "$out" ] && return 0
  [ -f "$gold" ] || die "no probe gold $gold (m17-score.sh probe-gold)"
  local lp=$L/diag/$3.left.m10-probes.jsonl rp=$L/diag/$3.right.m10-probes.jsonl
  python3 - "$gold" "$(pred "$1" m10-probes)" "$lp" "$(pred "$2" m10-probes)" "$rp" << 'EOF2' || die "$3 probe filter FAILED"
import json, sys
ids = {json.loads(line)["id"] for line in open(sys.argv[1])}
for src, dst in ((sys.argv[2], sys.argv[3]), (sys.argv[4], sys.argv[5])):
    kept = 0
    with open(src) as f, open(dst, "w") as out:
        for line in f:
            if json.loads(line)["id"] in ids:
                out.write(line)
                kept += 1
    if kept != len(ids):
        raise SystemExit(f"{src}: {kept} predictions for {len(ids)} gold items")
EOF2
  py "$OPS/m10/m10_probes.py" score --gold "$gold" --predictions "$lp" \
    --reference "$rp" --output "$out" > "$L/diag/$3.probes.log" 2>&1 \
    || die "$3 probes FAILED (see $L/diag/$3.probes.log)"
  log "$3 probes: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print({k: d.get(k) for k in ("macro_mmlu_arc_gsm8k","macro_delta","macro_delta_ci95")})' "$out")"
}
s5() {  # <point>
  local out=$L/diag/$1.score5t.json
  [ -f "$out" ] && return 0
  py "$OPS/m8/m8_rules.py" score5t --panel-root /data/dev2/private/panels --predictions "$(pred "$1" score5t-dev)" \
    --output "$out" > "$L/diag/$1.score5t.log" 2>&1 || die "$1 Score5-typed-DEV FAILED (see $L/diag/$1.score5t.log)"
  log "$1 score5t: $(tail -1 "$L/diag/$1.score5t.log" | cut -c1-300)"
}
hs1() {  # <left> <right>
  local out=$L/diag/$1.hs1.json
  [ -f "$out" ] || [ ! -f "$(pred "$1" hs1-dev)" ] && return 0
  py -m v2.data.hs1.validity --gold "$GOLD/hs1-dev.gold.jsonl" --left "$(pred "$1" hs1-dev)" --left-name "$1" \
    --right "$(pred "$2" hs1-dev)" --right-name "$2" --output "$out" > "$L/diag/$1.hs1.log" 2>&1 \
    || log "$1 hs1-dev score failed (see $L/diag/$1.hs1.log)"
}
mlx2_gold() {
  [ "$(sha256sum "$MLX2_PANEL/gold.jsonl" | cut -d' ' -f1)" = "$MLX2_GOLD_SHA" ] || die "MLX-DEV2 gold is not $MLX2_GOLD_SHA"
}

case ${1:-} in
  pull)
    from=$2
    shift 2
    for p in "$@"; do
      rsync -a -e "$KEY" --include='*/' --include='*.predictions.jsonl' --include='*.manifest.json' \
        --include='*.launch.json' --include='mlxdev-predictions.jsonl' --include='*.mlxdev.json' \
        --include='*.mlxscore.json' --exclude='*' "$from:$L/$p/" "$L/$p/" || die "pull of $p failed"
      log "pulled $p: $(find "$L/$p" -name '*.predictions.jsonl' | wc -l) prediction files"
    done
    ;;
  mlx2-pull)
    from=$2
    shift 2
    for n in "$@"; do
      mkdir -p "$MLX2/$n"
      rsync -a -e "$KEY" "$from:$MLX2/$n/mlx-dev2.predictions.jsonl" "$MLX2/$n/" || die "MLX-DEV2 pull of $n failed"
      log "pulled MLX-DEV2 $n: $(wc -l < "$MLX2/$n/mlx-dev2.predictions.jsonl") predictions ($(sha256sum "$MLX2/$n/mlx-dev2.predictions.jsonl" | cut -c1-16))"
    done
    ;;
  probe-gold)
    [ "$(sha256sum "$HITS_M15" | cut -d' ' -f1)" = "$HITS_SHA" ] || die "M15's 4B hits are not M12's $HITS_SHA"
    [ "$(sha256sum "$PROBE_GOLD_M15" | cut -d' ' -f1)" = "$PROBE_GOLD_SHA" ] || die "M15's 4B probe gold is not $PROBE_GOLD_SHA"
    [ -f "$PG/m10-probes.4b.gold.jsonl" ] || cp "$PROBE_GOLD_M15" "$PG/m10-probes.4b.gold.jsonl"
    log "4B probe gold: $(sha256sum "$PG/m10-probes.4b.gold.jsonl" | cut -c1-16) ($(wc -l < "$PG/m10-probes.4b.gold.jsonl") items)"
    ;;
  mlx2)
    ref=$2
    shift 2
    mlx2_gold
    for p in "$@"; do
      out=$L/$p/mlx2cmp/$p.mlx2.json
      [ -f "$out" ] && continue
      mkdir -p "$L/$p/mlx2cmp"
      py -m v2.eval.mlx_dev2 compare --panel "$MLX2_PANEL" --candidate "$MLX2/$p/mlx-dev2.predictions.jsonl" \
        --reference "$MLX2/$ref/mlx-dev2.predictions.jsonl" --candidate-name "$p" --reference-name "$ref" \
        --replicates 5000 --seed 20260927 --output "$out" \
        > "$L/$p/mlx2cmp/$p.mlx2.log" 2>&1 || die "$p MLX-DEV2 compare FAILED (see $L/$p/mlx2cmp/$p.mlx2.log)"
      log "$p MLX-DEV2 vs $ref: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); c=d["card_eligible"]; print(round(c["delta"],4), [round(c["ci95"]["low"],4), round(c["ci95"]["high"],4)], "guard_pass", d["guard_pass"])' "$out")"
    done
    ;;
  mlx2-agree)
    ref=$2
    mlx2_gold
    out=$L/diag/$ref.mlx2-vs-eval-LH.json
    [ -f "$out" ] && exit 0
    py -m v2.eval.mlx_dev2 compare --panel "$MLX2_PANEL" --candidate "$MLX2/$ref/mlx-dev2.predictions.jsonl" \
      --reference "$MLX2_EVAL_LH" --candidate-name "$ref" --reference-name eval-LH --replicates 5000 --seed 20260927 \
      --output "$out" > "$L/diag/$ref.mlx2-vs-eval-LH.log" 2>&1 \
      || die "MLX-DEV2 agreement compare FAILED"
    python3 - "$MLX2/$ref/mlx-dev2.predictions.jsonl" "$MLX2_EVAL_LH" "$out" << 'EOF2'
import json, sys
a = {json.loads(l)["id"]: json.loads(l)["answers"] for l in open(sys.argv[1])}
b = {json.loads(l)["id"]: json.loads(l)["answers"] for l in open(sys.argv[2])}
same = sum(1 for k in a if b.get(k) == a[k])
d = json.load(open(sys.argv[3]))
d["answer_agreement"] = {"same": same, "items": len(a), "missing_in_eval": len(set(a) - set(b))}
json.dump(d, open(sys.argv[3], "w"), indent=2)
print(json.dumps(d["answer_agreement"]))
EOF2
    log "$ref vs the eval track's LH read: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["answer_agreement"], round(d["card_eligible"]["delta"],4))' "$out")"
    ;;
  points)
    ref=$2
    shift 2
    [ "$(sha256sum "$HT_GOLD" | cut -d' ' -f1)" = "$HT_GOLD_SHA" ] || die "HT-DEV v2 gold is not $HT_GOLD_SHA"
    s5 "$ref"
    for p in "$@"; do
      [ "$p" = "$ref" ] && continue
      ht "$p" "$ref" "$p"
      s5 "$p"
      probes "$p" "$ref" "$p"
      hs1 "$p" "$ref"
    done
    ;;
  ibdev)
    ref=$2
    shift 2
    [ "$(sha256sum "$IB_GOLD" | cut -d' ' -f1)" = "$IB_GOLD_SHA" ] || die "IB DEV gold is not $IB_GOLD_SHA"
    for p in "$@"; do
      out=$L/diag/$p.ibdev.json
      [ -f "$out" ] && continue
      py "$OPS/m11/m11_ibdev.py" score --gold "$IB_GOLD" --predictions "$(pred "$p" ib-dev)" --name "$p" \
        --reference "$(pred "$ref" ib-dev)" --reference-name "$ref" --exclude isarc,hover,gsm2 --output "$out" \
        > "$L/diag/$p.ibdev.log" 2>&1 || die "$p IB DEV FAILED (see $L/diag/$p.ibdev.log)"
      log "$p ib-dev: $(tail -1 "$L/diag/$p.ibdev.log" | cut -c1-300)"
    done
    ;;
  contrast)
    ht "$2" "$3" "$2-vs-$3"
    probes "$2" "$3" "$2-vs-$3"
    ;;
  readout)
    ref=$2
    shift 2
    args=(--arm "$ref=$(pred "$ref" dev),$(pred "$ref" css-pilot)")
    for p in "$@"; do
      [ "$p" = "$ref" ] && continue
      args+=(--arm "$p=$(pred "$p" dev),$(pred "$p" css-pilot)" --compare "$ref:$p")
    done
    out=$L/readout/4b.json
    [ -f "$out" ] && mv "$out" "$L/readout/4b.$(date -u +%Y%m%dT%H%M%SZ).json"
    py -m v2.dec.dev_readout --typed-gold "$GOLD/typed-dev.gold.jsonl" --css-gold "$GOLD/css-pilot.gold.jsonl" \
      "${args[@]}" --output "$out" > "$L/readout/4b.log" 2>&1 || die "dev_readout FAILED (see $L/readout/4b.log)"
    log "readout $(sha256sum "$out" | cut -c1-16) ($ref $*)"
    ;;
  rules)
    mode=$2
    shift 2
    pts=()
    for x in "$@"; do pts+=(--point "$x"); done
    case $mode in
      arms) out=$M/select/4b-arms.json extra=() ;;
      final) out=$M/select/4b-finalists.json extra=(--arms-result "$M/select/4b-arms.json") ;;
      *) die "rules mode must be arms or final" ;;
    esac
    py "$OPS/m17/m17_rules.py" "$mode" --lines-root "$L" --mlx2-root "$MLX2" --readout "$L/readout/4b.json" \
      "${pts[@]}" "${extra[@]}" --output "$out" > "$M/select/rules-$mode.log" 2>&1 \
      || die "rules $mode FAILED (see $M/select/rules-$mode.log)"
    log "rules $mode: $(tail -1 "$M/select/rules-$mode.log" | cut -c1-900)"
    ;;
  *) sed -n '2,22p' "$0"; exit 2 ;;
esac
