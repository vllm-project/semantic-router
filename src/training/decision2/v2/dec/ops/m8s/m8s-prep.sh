#!/usr/bin/env bash
# Decoder M8-small data preparation on node B, CPU only (prereg dec-m8s-prereg-2026-09-30.md, "Top-up rows").
#   0. the released starts: hf download of DEV2.0-2B@a53cf66a / DEV2.0-0.8B@bede7938 into m8s/start/<name> (real
#      files); every MODEL_MANIFEST file re-hashed and the checkpoint identity equal to the BF16 copy's;
#   1. the M8s Triton cache: one cp -a copy of the node's decoder cache (isolated from other decoder jobs);
#   2. m8s_compose.py per tier -> m8s/data/<tier>/topup/{train.jsonl,train.parts.jsonl,compose.json};
#   3. 2B control teacher: v2.dec.compose_teacher (S2T's own-Sol targets 947bc65b) -> m8s/teacher/2b-C/teacher.jsonl;
#   4. an overlap exposure receipt per tier against the r2 payload (2194716a) -> m8s/exposure/<tier>/.
# A finished step is skipped on a rerun; a failed step is not rerun. READY files are written by hand only after the
# lock record is committed.
# usage: m8s-prep.sh [2b 08b]
set -uo pipefail
. "$(dirname "$0")/m8s-lib.sh"
TIERS=${*:-2b 08b}
mkdir -p "$M/start" "$M/data" "$M/teacher" "$M/exposure"
export DEC_IMAGE=$IMAGE DEC_DATA=$SELCAL
PAYLOAD_DIR=/data/dev2/runs/eval/m5/overlap-effects/final
PAYLOAD_SHA=2194716a179b2e6c3ba529dee3952c5dd62c0f3281638b7236029d5d542f4914
TOK=/hf/models--llm-semantic-router--Decision-1.0-Eos-0.8B/snapshots/363c4a5e56afc115b1c78c837633956d0bbb63ab
declare -A RECIPE=([2b]=/runs/m4/data/m4-v2m-ret-r2/train.jsonl [08b]=/runs/m6/data/m6-e8f-r2clean/train.jsonl)
declare -A RECIPE_SHA=([2b]=1527b38b1ba888695fe48dd43e92827d1719d57674009cfc29d5ab759b08dd2c
  [08b]=f9f3c0229551357ad1cb8e6923c64c636ee697e71d162f059548ae223e78d8ae)
S2T_T=/runs/m3/teacher/sol/sol-teacher.jsonl
QUAR=/code/v2/dec/ops/m7/specs/m7-quarantine-groups.json

cpu() {  # <job> <out dir under m8s> <python args...>
  local job=$1 out=$M/$2
  shift 2
  if [ -f "$out.launch.json" ]; then
    grep -q '"exit_status": 0' "$out.launch.json" && return 0
    log "prep $job failed earlier (receipt $out.launch.json); not rerun"
    return 1
  fi
  if bash "$LAUNCH" "m8s-$job" "$SRC" "$out" --cpu -- "$@"; then
    log "prep $job done: $(tail -c 400 "$out.stdout.log" | tr '\n' ' ')"
  else
    log "prep $job FAILED: $(tail -c 600 "$out.stderr.log" | tr '\n' ' ')"
    return 1
  fi
}

start() {  # <tier>
  local t=$1 dir=${START[$1]}
  if [ ! -f "$dir/START-VERIFIED.json" ]; then
    HF_HUB_CACHE=/data/dev2/hf-cache hf download "${START_REPO[$t]}" --revision "${START_REV[$t]}" --local-dir "$dir" \
      > "$M/logs/download-$t.log" 2>&1 || { log "download ${START_REPO[$t]}@${START_REV[$t]:0:8} FAILED"; return 1; }
    (cd "$S" && PYTHONPATH=$S python3 -B - "$dir" "${START_ID[$t]}" "${START_REPO[$t]}" "${START_REV[$t]}") <<'EOF' || return 1
import hashlib, json, sys
from pathlib import Path
from training.model.infer import checkpoint_fingerprint
d, want, repo, rev = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
man = json.loads((d / "MODEL_MANIFEST.json").read_text())
bad, n = [], 0
for rel, digest in man["files_sha256"].items():
    h = hashlib.sha256()
    with open(d / rel, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    n += 1
    if h.hexdigest() != digest:
        bad.append(rel)
ident = checkpoint_fingerprint(d)["model_sha256"]
if bad or not n or ident != want or man["identity"].get("model_sha256", want) != want:
    raise SystemExit(f"start {d} FAILED: {len(bad)} of {n} files differ; identity {ident} (want {want})")
(d / "START-VERIFIED.json").write_text(json.dumps({"repo": repo, "revision": rev, "files_checked": n,
    "model_sha256": ident}, indent=1) + "\n")
print(f"start {d.name}: {n} files verified, identity {ident[:12]}")
EOF
  fi
  log "start $t verified: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["repo"], d["revision"][:8], d["model_sha256"][:12])' "$dir/START-VERIFIED.json")"
}

[ "$(sha256sum "$PAYLOAD_DIR/excluded-groups.json" | cut -d' ' -f1)" = "$PAYLOAD_SHA" ] || { log "payload hash MISMATCH"; exit 1; }
for t in $TIERS; do start "$t" || exit 1; done
if [ ! -d "$TCACHE" ]; then
  mkdir -p "$(dirname "$TCACHE")"
  if ! cp -a "$R/triton-cache/dbe5f32b2263" "$TCACHE.pending" || ! mv -T "$TCACHE.pending" "$TCACHE"; then
    log "Triton cache copy FAILED"
    exit 1
  fi
  log "Triton cache copied to $TCACHE ($(find "$TCACHE" -type f | wc -l) files)"
fi
for t in $TIERS; do
  cpu "compose-$t" "data/$t" v2/dec/ops/m8s/m8s_compose.py --tier "$t" --recipe "${RECIPE[$t]}" \
    --recipe-sha "${RECIPE_SHA[$t]}" --quarantine "$QUAR" --tokenizer "$TOK" --workers 40 --output /out/topup || exit 1
  if [ "$t" = 2b ]; then
    cpu "teacher-2b-C" "teacher/2b-C" -m v2.dec.compose_teacher compose --train /runs/m8s/data/2b/topup/train.jsonl \
      --source "$S2T_T" --output /out/teacher.jsonl || exit 1
  fi
  tsha=$(sha "$M/data/$t/topup/train.jsonl")
  DEC_DATA=$PAYLOAD_DIR cpu "exposure-$t" "exposure/$t" -m v2.eval.overlap_effects exposure \
    --groups /data/excluded-groups.json --train "/runs/m8s/data/$t/topup/train.jsonl" --expect-sha256 "$tsha" \
    --label "decoder M8-small $t top-up" --output "/out/exposure-$t.json" || exit 1
done
log "prep finished ($TIERS)"
