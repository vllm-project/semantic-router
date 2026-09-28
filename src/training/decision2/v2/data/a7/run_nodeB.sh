#!/usr/bin/env bash
# A7 pipeline on node B, run from an exact mirror (CPU only); version from the spec.
#
# Usage: run_nodeB.sh <stage> [workers]
#   build     pre-admission sub-arms, views and build manifest (host python)
#   build-enc pre-admission encoder-family sub-arms (a7-enc10) in the runtime image,
#             after running their unit tests there (pyarrow)
#   recover   pre-admission A7r (rule 7e) from the spec sources, deduplicated and
#             isolated against the frozen files in A7_FROZEN_FINAL (host python)
#   lengths   per-row token lengths in the pinned runtime image (no GPU devices)
#   screens   overlap vs PI-v2 and shortcut receipts per sub-arm (host python)
#   rescreen  overlap of the admitted files vs A7_RESCREEN_PI into A7_RESCREEN_DIR
#   requarantine  new version A7_VERSION = A7_FROM_RUN minus rescreen/embedding hits
#   admit     apply overlap/budget/shortcut rules, resolve views
#   post      post-admission shortcut and TRAIN<->AHO self-scan diagnostics
#   freeze    content hash + manifest (+tokens) per final file (runtime image)
#   isolation cross-partition isolation against SELECT/CAL/A0 and published arms
#   assemble  HF upload folder (a7/ only) with registry, path-stripped JSON
#   upload    upload a7/ to v2/a7 of the private dataset, verify private + readback
#   inventory token-annotated census of the 1.0 decoder corpora
# Work dir: /data/dev2/private/a7/runs/<version>/<commit12>; A7_SPEC overrides the spec and
# A7_RUN_DIR the work dir (assemble/upload of a run built by an earlier commit).
set -euo pipefail

stage="${1:?stage}"
workers="${2:-64}"
S="$(cd "$(dirname "$0")/../../.." && pwd)"
mirror="$(cd "$S/../../.." && pwd)"
commit="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "$mirror/.dev2-mirror.json")"
A7="$S/v2/data/a7"
SPEC="${A7_SPEC:-$A7/specs/a7-dec10-v2.nodeB.json}"
version="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$SPEC")"
W="${A7_RUN_DIR:-/data/dev2/private/a7/runs/$version/${commit:0:12}}"
PI="${A7_PI:-/data/dev2/private/data/pi-v2/manifest.json}"
IMAGE="${A7_IMAGE:-decision20-lux-runtime:latest}"
LICENSES="${A7_LICENSE_REGISTRY:-$A7/license-registry-a7-v1.json}"
read -r -a SUBS <<< "${A7_SUBS:-A7h A7m A7g A7i A7p A7o}"
export PYTHONPATH="$S"
cd "$S"
umask 077
mkdir -p "$W/logs"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$W/logs/run.log"; }

in_image() {
  docker run --rm --network none --entrypoint python3 \
    -e PYTHONPATH="$S" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -v /data/dev2:/data/dev2 -w "$S" "$IMAGE" "$@"
}

case "$stage" in
  build)
    log "build commit=$commit"
    python3 -m v2.data.a7.build_a7 --spec "$SPEC" \
      --out-dir "$W/build" --commit "$commit" | tee -a "$W/logs/build.out"
    ;;
  build-enc)
    log "build-enc commit=$commit"
    in_image -m unittest v2.data.a7.tests.test_build_enc 2>&1 | tail -3 | tee -a "$W/logs/build.out"
    args=()
    while read -r file; do args+=(--data-track-file "$file"); done < <(
      python3 -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1])).get("data_track_files", [])))' "$SPEC")
    in_image -m v2.data.a7.build_enc --spec "$SPEC" "${args[@]}" --out-dir "$W/build" \
      --commit "$commit" | tee -a "$W/logs/build.out"
    ;;
  recover)
    log "recover commit=$commit frozen=${A7_FROZEN_FINAL:?A7_FROZEN_FINAL}"
    python3 -m v2.data.a7.recover --spec "$SPEC" --frozen-final "$A7_FROZEN_FINAL" \
      --out-dir "$W/build" --commit "$commit" | tee -a "$W/logs/recover.out"
    ;;
  lengths)
    log "lengths image=$(docker image inspect --format '{{.Id}}' "$IMAGE")"
    args=()
    for file in "$W"/build/prelim/*.jsonl; do args+=(--rows "$file"); done
    in_image -m v2.data.a7.lengths "${args[@]}" --tokenizers "$A7/specs/tokenizers.nodeB.json" \
      --out "$W/lengths.jsonl" --workers "$workers" | tee -a "$W/logs/lengths.out"
    ;;
  screens)
    mkdir -p "$W/screens"
    for sub in "${SUBS[@]}"; do
      files=()
      for part in train aho; do
        [[ -f "$W/build/prelim/$sub.$part.jsonl" ]] && files+=(--candidates "$W/build/prelim/$sub.$part.jsonl")
      done
      [[ ${#files[@]} -gt 0 ]] || continue
      set +e
      start=$(date +%s)
      python3 -m v2.data.overlap "${files[@]}" --protected-inventory "$PI" \
        --private-receipt "$W/screens/$sub.overlap.private.json" \
        --public-receipt "$W/screens/$sub.overlap.public.json" --workers "$workers" \
        > "$W/screens/$sub.overlap.stdout" 2> "$W/screens/$sub.overlap.stderr"
      log "$sub overlap rc=$? wall=$(( $(date +%s) - start ))"
      start=$(date +%s)
      python3 -m v2.data.shortcut --rows "$W/build/prelim/$sub.train.jsonl" \
        --receipt "$W/screens/$sub.shortcut.json" --workers "$workers" \
        > "$W/screens/$sub.shortcut.stdout" 2> "$W/screens/$sub.shortcut.stderr"
      log "$sub shortcut rc=$? wall=$(( $(date +%s) - start ))"
      set -e
    done
    ;;
  rescreen)
    # Lexical overlap of the admitted files against another protected manifest
    # (A7_RESCREEN_PI, e.g. PI-v3 without TRAIN roles) into $W/$A7_RESCREEN_DIR.
    pi="${A7_RESCREEN_PI:?A7_RESCREEN_PI}"
    dir="$W/${A7_RESCREEN_DIR:?A7_RESCREEN_DIR}"
    mkdir -p "$dir"
    log "rescreen manifest=$(sha256sum "$pi" | cut -c1-64) dir=$dir"
    for sub in "${SUBS[@]}"; do
      files=()
      for part in train aho; do
        [[ -f "$W/final/$sub.$part.jsonl" ]] && files+=(--candidates "$W/final/$sub.$part.jsonl")
      done
      [[ ${#files[@]} -gt 0 ]] || continue
      set +e
      start=$(date +%s)
      python3 -m v2.data.overlap "${files[@]}" --protected-inventory "$pi" \
        --private-receipt "$dir/$sub.overlap.private.json" \
        --public-receipt "$dir/$sub.overlap.public.json" --workers "$workers" \
        > "$dir/$sub.overlap.stdout" 2> "$dir/$sub.overlap.stderr"
      log "$sub rescreen overlap rc=$? wall=$(( $(date +%s) - start ))"
      set -e
    done
    ;;
  requarantine)
    # New version (A7_VERSION) in $W from the admitted run A7_FROM_RUN minus the groups
    # flagged by its A7_RESCREEN_DIR lexical and A7_EMBED_DIR embedding receipts.
    from="${A7_FROM_RUN:?A7_FROM_RUN}"
    args=()
    for file in "$from/${A7_RESCREEN_DIR:?A7_RESCREEN_DIR}"/*.overlap.private.json; do
      args+=(--overlap-receipt "$file")
    done
    args+=(--embed-receipt "$from/${A7_EMBED_DIR:?A7_EMBED_DIR}/embed.private.json")
    python3 -m v2.data.a7.requarantine --from-run "$from" "${args[@]}" \
      --version "${A7_VERSION:?A7_VERSION}" --out-run "$W" | tee -a "$W/logs/requarantine.out"
    mkdir -p "$W/rescreen" "$W/embed"
    cp "$from/$A7_RESCREEN_DIR"/*.overlap.public.json "$W/rescreen/"
    cp "$from/$A7_EMBED_DIR/embed.public.json" "$from/$A7_EMBED_DIR/run.json" "$W/embed/"
    ;;
  admit)
    args=()
    for sub in "${SUBS[@]}"; do
      [[ -f "$W/screens/$sub.overlap.private.json" ]] && args+=(--overlap-receipt "$sub=$W/screens/$sub.overlap.private.json")
      [[ -f "$W/screens/$sub.shortcut.json" ]] && args+=(--shortcut-receipt "$sub=$W/screens/$sub.shortcut.json")
    done
    python3 -m v2.data.a7.admit --build-dir "$W/build" "${args[@]}" --lengths "$W/lengths.jsonl" \
      --report-only-role rights_clean_train --out-dir "$W/final" | tee -a "$W/logs/admit.out"
    ;;
  post)
    mkdir -p "$W/post"
    for sub in "${SUBS[@]}"; do
      [[ -f "$W/final/$sub.train.jsonl" ]] || continue
      set +e
      python3 -m v2.data.shortcut --rows "$W/final/$sub.train.jsonl" \
        --receipt "$W/post/$sub.shortcut.json" --workers "$workers" > "$W/post/$sub.shortcut.stdout"
      log "$sub post-shortcut rc=$?"
      if [[ -f "$W/final/$sub.aho.jsonl" ]]; then
        python3 -m v2.data.overlap --self-scan --candidates "$W/final/$sub.train.jsonl" \
          --candidates "$W/final/$sub.aho.jsonl" --private-receipt "$W/post/$sub.selfscan.private.json" \
          --public-receipt "$W/post/$sub.selfscan.public.json" --workers "$workers" > "$W/post/$sub.selfscan.stdout"
        log "$sub self-scan rc=$?"
      fi
      set -e
    done
    ;;
  freeze)
    mkdir -p "$W/manifests"
    for file in "$W"/final/*.jsonl; do
      name="$(basename "$file" .jsonl)"
      role="${name##*.}"
      [[ "$role" == "train" || "$role" == "aho" ]] || continue
      in_image -m v2.data.freeze freeze --rows "$file" --arm-id "${name/./-}" --role "$role" \
        --license-registry "$LICENSES" --tokenizers "$A7/specs/tokenizers.nodeB.json" \
        --out-manifest "$W/manifests/$name.freeze.json" | tee -a "$W/logs/freeze.out"
    done
    ;;
  isolation)
    B=/data/dev2/private/a7/base/5c0255ed
    parts=(--partition "train/A0=$B/rights_clean.train.jsonl" --partition "select/SELECT700=$B/select.jsonl"
           --partition "cal/CAL700=$B/cal.jsonl" --partition "train/RP-v1q=$B/v2/replay/RP-v1q/train.jsonl")
    for arm in A0p A0s A1 A2 A3 A4v2h A4v2r A5 A6g A6h; do
      parts+=(--partition "train/$arm=$B/v2/arms/$arm/train.jsonl")
      [[ -f "$B/v2/arms/$arm/aho.jsonl" ]] && parts+=(--partition "aho/$arm=$B/v2/arms/$arm/aho.jsonl")
    done
    for tier in eos kai lex lux nox sol; do parts+=(--partition "train/R2-$tier=$B/v2/replay/R2/$tier/replay.jsonl"); done
    for file in "$W"/final/*.jsonl; do
      name="$(basename "$file" .jsonl)"
      parts+=(--partition "${name##*.}/${name%%.*}=$file")
    done
    for extra in ${A7_ISOLATION_WITH:-}; do
      for file in "$extra"/*.jsonl; do
        name="$(basename "$file" .jsonl)"
        parts+=(--partition "${name##*.}/frozen-${name%%.*}=$file")
      done
    done
    python3 -m v2.data.freeze isolation "${parts[@]}" --report "$W/isolation.json" | tee -a "$W/logs/isolation.out"
    ;;
  assemble)
    # A7_EXTRA_LICENSES: more licence registries; A7_EXTRA_RUNS: "NAME=DIR ..." runs whose
    # sub-arms join this upload (records under versions/NAME/).
    lic=(--license-registry "$LICENSES")
    for extra in ${A7_EXTRA_LICENSES:-}; do lic+=(--license-registry "$extra"); done
    ext=()
    for item in ${A7_EXTRA_RUNS:-}; do ext+=(--extra-run "$item"); done
    python3 -m v2.data.a7.hf_spec --run-dir "$W" --readme "$A7/records/hf-dataset-a7-readme.md" \
      "${lic[@]}" "${ext[@]}" --out "$W/hf-spec.json"
    python3 -m v2.data.assemble_hf_upload --spec "$W/hf-spec.json" --out-dir "$W/hf-upload"
    ;;
  upload)
    export HF_HUB_CACHE=/data/dev2/hf-cache
    repo=llm-semantic-router/decision-2.0-training-data
    private() {
      hf datasets info "$repo" --format json | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("private"), d.get("sha"))'
    }
    read -r before parent < <(private)
    [[ "$before" == "True" ]] || { echo "dataset is not private; refusing to upload" >&2; exit 1; }
    log "upload parent=$parent"
    hf upload "$repo" "$W/hf-upload/a7" v2/a7 --repo-type dataset \
      --commit-message "A7 ${A7_VERSION:-$version}: own Decision 1.0 corpora sub-arms (${commit:0:12})" | tee -a "$W/logs/upload.out"
    read -r after revision < <(private)
    [[ "$after" == "True" ]] || { echo "dataset private flag changed" >&2; exit 1; }
    mkdir -p "$W/readback"
    hf download "$repo" v2/a7/registry.json --repo-type dataset --revision "$revision" --local-dir "$W/readback" >/dev/null
    cmp "$W/readback/v2/a7/registry.json" "$W/hf-upload/a7/registry.json"
    log "upload revision=$revision private=$after registry readback identical"
    ;;
  inventory)
    D=/data/dev2/private/a7/sources/dec10
    mkdir -p "$W/inventory"
    files=(S1=train.jsonl S1val=validation.jsonl S2=stage2_pointer-train.jsonl S3=stage3_train.jsonl
           S3val=stage3_validation.jsonl S4v2=stage4-prepared_combined-v2_train.jsonl
           S4v2sel=stage4-prepared_combined-v2_select.jsonl S4v2cal=stage4-prepared_combined-v2_cal.jsonl
           NAT8K=natural-reading-ab-v1_natural-train.jsonl NT24K=natural-reading-ab-v1_natural-treatment.jsonl
           NRC24K=natural-reading-ab-v1_replay-control.jsonl SF24K=nox-semantic-format-v1_treatment.jsonl
           NATSEL=natural-reading-ab-v1_natural-select.jsonl NATCAL=natural-reading-ab-v1_natural-cal.jsonl)
    rows=(); census=()
    for item in "${files[@]}"; do rows+=(--rows "${item%%=*}=$D/${item#*=}"); census+=(--file "${item%%=*}=$D/${item#*=}"); done
    [[ -f "$W/inventory/lengths.jsonl" ]] || in_image -m v2.data.a7.lengths "${rows[@]}" \
      --tokenizers "$A7/specs/tokenizer-qwen35.nodeB.json" --out "$W/inventory/lengths.jsonl" --workers "$workers"
    python3 -m v2.data.a7.inventory "${census[@]}" --lengths "$W/inventory/lengths.jsonl" \
      --out "$W/inventory/census.json"
    ;;
  *) echo "unknown stage $stage" >&2; exit 2 ;;
esac
log "stage $stage done"
