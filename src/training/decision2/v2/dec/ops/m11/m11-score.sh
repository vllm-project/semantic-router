#!/usr/bin/env bash
# Decoder M11 node-A scoring (prereg "Development readouts" / "Development gates"), CPU only, from an exact mirror.
# Gold stays on node A; node E / F readouts are pulled gold-free over the temporary transfer key. Points are named
# <tier>-<name> (tier 2b or 08b); the retention probes are scored on the tier's probe gold (M10's gold minus the probe
# items that hit the tier's TRAIN, prereg "Probe TRAIN-overlap per tier").
#
#   m11-score.sh pull <user@node> <point> [<point> ...]   lines/<point>/ predictions and manifests from node E / F
#   m11-score.sh probe-gold <user@node> [<tier> ...]      probes/hits-<tier>.json from node E (ids only), then the
#                                                         tier probe gold files (private/dec/m11/); tiers 2b 08b, or
#                                                         4b (stage 2: hits against both stage-2 TRAIN files)
#   m11-score.sh ibdev-panel <user@node> [...]            stage 2: the IB DEV panel from node A's IB1-r3 / IB2 read-back
#                                                         DEV files (m11_ibdev.py prompts; gold stays in private/dec/m11),
#                                                         prompts relayed to each node's panels/ib-dev.prompts.jsonl
#   m11-score.sh ibdev <REF> <point> [<point> ...]        stage 2: IB DEV / transfer macros vs REF -> diag/<point>.ibdev.json
#   m11-score.sh points <REF> <point> [<point> ...]       per point vs REF: HT-DEV v2 (m7_htdev2.py), Score5-typed-DEV
#                                                         (m8_rules.py score5t; also REF), retention probes
#                                                         (m10_probes.py score, tier gold); hs1-dev (v2.data.hs1.validity)
#   m11-score.sh contrast <A> <B>                         HT-DEV v2 and probes of A vs B (A-vs-B.*.json; report only)
#   m11-score.sh readout <tier> <REF> <point> [...]       typed DEV + CSS pilot (v2.dec.dev_readout) -> readout/<tier>.json
#   m11-score.sh rules <tier> <X=REF> [...] [-- <A:B> ...] m11_rules.py -> m11/select/<tier>-finalists.json (runs once)
set -u
S=$(cd "$(dirname "$0")/../../../.." && pwd)
MIRROR=${S%/src/training/decision2}
[ -f "$MIRROR/.dev2-mirror.json" ] || { echo "run from an exact mirror under /data/dev2/src" >&2; exit 2; }
M=/data/dev2/runs/dec/m11
L=$M/lines
OPS=$S/v2/dec/ops
GOLD=/data/dev2/private/panels/gold
HT_GOLD=$GOLD/ht-dev2.gold.jsonl HT_GOLD_SHA=659c92b45d2a5b37e250ccb728cdbf5580ba361b3fc4b2f2ab505a13e0a556cc
PROBE_GOLD_M10=/data/dev2/private/dec/m10/m10-probes.gold.jsonl
PROBE_GOLD_SHA=5c674e35142bc1aae2526d38b6930a0e570d99b5ca5185b0a7ac428dc8b7bae2
PG=/data/dev2/private/dec/m11
log() { echo "$(date -u +%FT%TZ) score $*" | tee -a "$L/OPERATIONS-nodeA.log"; }
die() { log "$*"; exit 1; }
py() { (cd "$S" && PYTHONPATH=$S python3 -B "$@"); }
pred() { echo "$L/$1/$2/$2.predictions.jsonl"; }
mkdir -p "$L/diag" "$L/readout" "$M/select"

ht() {  # <left> <right> <out name>
  local out=$L/diag/$3.htdev2.json
  [ -f "$out" ] && return 0
  py "$OPS/m7/m7_htdev2.py" --gold "$HT_GOLD" --left "$(pred "$1" ht-dev2)" --left-name "$1" \
    --right "$(pred "$2" ht-dev2)" --right-name "$2" --output "$out" > "$L/diag/$3.htdev2.log" 2>&1 \
    || die "$3 HT-DEV v2 FAILED (see $L/diag/$3.htdev2.log)"
  log "$3 ht-dev2: $(tail -1 "$L/diag/$3.htdev2.log" | cut -c1-300)"
}
probes() {  # <left> <right> <out name>
  local out=$L/diag/$3.probes.json
  [ -f "$out" ] && return 0
  local gold=$PG/m10-probes.${1%%-*}.gold.jsonl
  [ -f "$gold" ] || die "no tier probe gold $gold (m11-score.sh probe-gold)"
  py "$OPS/m10/m10_probes.py" score --gold "$gold" --predictions "$(pred "$1" m10-probes)" \
    --reference "$(pred "$2" m10-probes)" --output "$out" > "$L/diag/$3.probes.log" 2>&1 \
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
    || log "$1 hs1-dev score failed (report only; see $L/diag/$1.hs1.log)"
}

case ${1:-} in
  pull)
    from=$2
    shift 2
    for p in "$@"; do
      rsync -a -e 'ssh -i /root/.ssh/d2_temp_cd -o BatchMode=yes' --include='*/' --include='*.predictions.jsonl' \
        --include='*.manifest.json' --include='*.launch.json' --include='parity-*.json' --exclude='*' \
        "$from:$L/$p/" "$L/$p/" || die "pull of $p failed"
      log "pulled $p: $(find "$L/$p" -name '*.predictions.jsonl' | wc -l) prediction files"
    done
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
  probe-gold)
    from=$2
    shift 2
    [ "$(sha256sum "$PROBE_GOLD_M10" | cut -d' ' -f1)" = "$PROBE_GOLD_SHA" ] || die "M10 probe gold is not $PROBE_GOLD_SHA"
    mkdir -p "$M/probes" "$PG"
    [ $# -gt 0 ] || set -- 2b 08b
    for t in "$@"; do
      rsync -a -e 'ssh -i /root/.ssh/d2_temp_cd -o BatchMode=yes' "$from:$M/probes/hits-$t.json" "$M/probes/" \
        || die "pull of hits-$t.json failed"
      [ -f "$PG/m10-probes.$t.gold.jsonl" ] && continue
      python3 - "$M/probes/hits-$t.json" "$PROBE_GOLD_M10" "$PG/m10-probes.$t.gold.jsonl" << 'EOF2' || die "tier gold $t FAILED"
import json, sys
hits = set(json.load(open(sys.argv[1]))["panel_hits"])
kept = dropped = 0
with open(sys.argv[2]) as src, open(sys.argv[3], "x") as out:
    for line in src:
        if json.loads(line)["id"] in hits:
            dropped += 1
            continue
        out.write(line)
        kept += 1
print(kept, dropped)
EOF2
      log "tier probe gold $t: $(sha256sum "$PG/m10-probes.$t.gold.jsonl" | cut -c1-16) ($(wc -l < "$PG/m10-probes.$t.gold.jsonl") items)"
    done
    ;;
  ibdev-panel)
    shift
    IB1D=/data/dev2/private/data/ib1/r3/hf/readback/m6/ib1/ib1.dev.jsonl
    IB2D=/data/dev2/private/data/ib2/c3/hf/readback/m6/ib2/ib2.dev.jsonl
    B=$PG/ibdev-build
    if [ ! -f "$B/manifest.json" ]; then
      mkdir -p "$PG" && chmod 700 "$PG"
      py "$OPS/m11/m11_ibdev.py" prompts --ib1 "$IB1D" \
        --ib1-sha 3f56aa418e90e58f3fbaa50bf2f9f4f0405eeb9dd2f0c51ffca6f602711c693f --ib2 "$IB2D" \
        --ib2-sha ab009fb12f9563c3ef4a846f5455dd5233f2a2c5fafe6ac8a9c74349532eb923 --output "$B" > "$PG/ibdev-build.log" 2>&1 \
        || die "IB DEV panel build FAILED (see $PG/ibdev-build.log)"
      chmod 600 "$B"/*.jsonl
      cp "$B/ib-dev.gold.jsonl" "$PG/ib-dev.gold.jsonl" && chmod 600 "$PG/ib-dev.gold.jsonl"
      log "IB DEV panel: $(tail -1 "$PG/ibdev-build.log")"
    fi
    for to in "$@"; do
      rsync -a --chmod=F444 -e 'ssh -i /root/.ssh/d2_temp_cd -o BatchMode=yes' "$B/ib-dev.prompts.jsonl" \
        "$to:/data/dev2/runs/dec/panels/ib-dev.prompts.jsonl" || die "relay of ib-dev prompts failed"
      log "ib-dev prompts relayed ($(sha256sum "$B/ib-dev.prompts.jsonl" | cut -c1-16))"
    done
    ;;
  ibdev)
    ref=$2
    shift 2
    for p in "$@"; do
      out=$L/diag/$p.ibdev.json
      [ -f "$out" ] && continue
      py "$OPS/m11/m11_ibdev.py" score --gold "$PG/ib-dev.gold.jsonl" --predictions "$(pred "$p" ib-dev)" --name "$p" \
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
    tier=$2 ref=$3
    shift 3
    args=(--arm "$ref=$(pred "$ref" dev),$(pred "$ref" css-pilot)")
    for p in "$@"; do
      [ "$p" = "$ref" ] && continue
      args+=(--arm "$p=$(pred "$p" dev),$(pred "$p" css-pilot)" --compare "$ref:$p")
    done
    out=$L/readout/$tier.json
    [ -f "$out" ] && mv "$out" "$L/readout/$tier.$(date -u +%Y%m%dT%H%M%SZ).json"
    py -m v2.dec.dev_readout --typed-gold "$GOLD/typed-dev.gold.jsonl" --css-gold "$GOLD/css-pilot.gold.jsonl" \
      "${args[@]}" --output "$out" > "$L/readout/$tier.log" 2>&1 || die "dev_readout FAILED (see $L/readout/$tier.log)"
    log "readout $(sha256sum "$out" | cut -c1-16) ($ref $*)"
    ;;
  rules)
    tier=$2
    shift 2
    pts=() cons=()
    while [ $# -gt 0 ]; do
      [ "$1" = -- ] && { shift; break; }
      pts+=(--point "$1")
      shift
    done
    for c in "$@"; do cons+=(--contrast "$c"); done
    py "$OPS/m11/m11_rules.py" --tier "$tier" --lines-root "$L" --readout "$L/readout/$tier.json" "${pts[@]}" \
      "${cons[@]}" --output "$M/select/$tier-finalists.json" > "$M/select/rules-$tier.log" 2>&1 \
      || die "rules FAILED (see $M/select/rules-$tier.log)"
    log "finalists $tier: $(tail -1 "$M/select/rules-$tier.log" | cut -c1-600)"
    ;;
  *) sed -n '2,22p' "$0"; exit 2 ;;
esac
