#!/usr/bin/env bash
# JevArena-C1 content recheck of new training files against the C1 v1.2 items: the custodian step
# before successor-rule item 8 for a model trained on rows that C1 was not rechecked against. Node A,
# CPU only.
# Usage (on node A from an exact mirror; the key arrives once on stdin, never in argv or a file):
#   c1-recheck.sh --src MIRROR --spec SPEC [--verify-only]
# SPEC (dev2-c1-recheck-spec/1, relative to the mirror's src/training/decision2, so it is committed)
# pins every training file by path, SHA-256 and row count. Before the key: the mirror, the pinned
# image, the spec's files (recheck verify), the encrypted bundle, the build-5 manifest and candidate
# files, the v1.2 retired list and the protected rows. --verify-only stops there. Then, still without
# the key, the registered scanner (v2.eval.sealed.overlap, the rescan parameters) screens the
# protected rows (all splits of the C1 sources; disclosed only). Then the key is read, only the
# prompts are decrypted into a private temp dir under the sealed directory, the key is dropped, and
# inside the pinned image (--network none, no GPU device, the rest of the sealed directory hidden):
#   recheck items   maps every prompt to its build-5 candidate (no gold, no salt), marks the v1.2 items;
#   recheck scan    the G0 rules (E, N1, N2, G), the declared boilerplate rule, the planted controls;
#   overlap scan    the registered scanner over the selected items, as the build screened them;
#   recheck judge   the verdict per training file and the item-8 exposure per arm.
# The temp dir is removed on any exit. Public-safe receipts (counts and hashes) go to
# /data/dev2/runs/eval/c1-recheck/<name>-<UTC>/ (SHA256SUMS lists them; RECHECK.log is the log);
# hits and quarantine lists stay in $C1/recheck/<name>-<UTC>/. Each step is logged in C1's ACCESS.log
# (RECHECK lines). One custodial C1 process at a time (the post-key lock).
set -uo pipefail
umask 077

SRC="" SPEC="" PHASE=run
while [ $# -gt 0 ]; do
  case $1 in
  --src) SRC=$2; shift 2 ;;
  --spec) SPEC=$2; shift 2 ;;
  --verify-only) PHASE=verify; shift ;;
  *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
# No child process may read the key: stdin becomes /dev/null; only a key-reading run keeps it on fd 3.
if [ "$PHASE" = run ]; then
  exec 3<&0 0</dev/null
else
  exec 0</dev/null
fi
[ -n "$SRC" ] && [ -n "$SPEC" ] || { echo "usage: c1-recheck.sh --src MIRROR --spec SPEC [--verify-only]" >&2; exit 2; }
case $SPEC in
/* | *..*) echo "--spec must be a path inside the mirror's src/training/decision2" >&2; exit 2 ;;
esac
[ -f "/data/dev2/src/$SRC/.dev2-mirror.json" ] || { echo "no verified mirror $SRC" >&2; exit 1; }

S="/data/dev2/src/$SRC/src/training/decision2"
C1=/data/dev2/private/sealed/c1
PK=$C1/postkey
BUNDLE_SHA=d924389a7ffc3d852d9204c3dfca1c12c32f82685a7e5c65277ba9980110534f
PROMPTS_SHA=0b29686f60c980f3fbc8a03b88537fc0bf90ee967afa67d4fe0c958b1bfde16a
MANIFEST=$C1/receipts/build-5-manifest.json
MANIFEST_SHA=71297ecb5458a182c68e88f88b52eb37f8f45770d02e82aa93f39d04dcbedd86
POOLS=$C1/v1/candidates-2
RETIRED=$C1/v1_2/RETIRED-v1_2.json
RETIRED_SHA=bbf095c70917f028d691fce570b114725e4f22fd6a1987456f6ae2c30f22990a
PROT=$C1/v1_1/protected-all-splits.jsonl
PROT_SHA=36797f509bd96c3cb703df37cc48114c9bbdf0d2241802e56262139f4bef0a1a
COVERAGE="$S/v2/eval/sealed/c1-rescan-coverage.json"
UTC=$(date -u +%Y%m%dT%H%M%SZ)
export PYTHONPATH="$S" PYTHONDONTWRITEBYTECODE=1
cd "$S" || exit 1
[ -f "$SPEC" ] || { echo "no spec $SPEC in the mirror" >&2; exit 2; }
SPEC="$S/$SPEC"
spec() { python3 -c 'import json, sys
v = json.load(open(sys.argv[1]))
for k in sys.argv[2].split("/"): v = v[k]
print("\n".join(v) if isinstance(v, list) else v)' "$1" "$2" 3<&-; }
NAME=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["name"])' "$SPEC" 3<&-)
[[ "$NAME" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { echo "bad spec name" >&2; exit 2; }
mapfile -t KEYS < <(python3 -c 'import json, sys; [print(d["key"]) for d in json.load(open(sys.argv[1]))["datasets"]]' "$SPEC" 3<&-)
mapfile -t PATHS < <(python3 -c 'import json, sys; [print(d["path"]) for d in json.load(open(sys.argv[1]))["datasets"]]' "$SPEC" 3<&-)
IMG=$(spec "$COVERAGE" image)
mapfile -t SCAN_ARGS < <(spec "$COVERAGE" scan)
J="/data/dev2/runs/eval/c1-recheck/$NAME-$UTC"
if [ "$PHASE" != run ]; then J="$J.verify"; fi
P="$C1/recheck/$NAME-$UTC"
mkdir -p "$PK" "$J" || exit 1
exec 9>>"$PK/.lock"
flock -n 9 || { echo "another custodial C1 process holds $PK/.lock" >&2; exit 1; }

KEY="" T="" LOG="$J/RECHECK.log" PLAINTEXT=""
log() {
  local t
  t=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "$t $*" >>"$LOG"
  echo "$t RECHECK $NAME $*" >>"$C1/ACCESS.log"
  echo "$t $*"
}
abort() {
  log "ABORT: $*"
  exit 1
}
cleanup() {
  local code=$?
  if [ -n "$T" ]; then rm -rf "$T"; fi
  KEY=""
  unset KEY
  log "cleanup (exit $code): ${PLAINTEXT:-nothing was decrypted}"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
sha() { sha256sum <"$1" | cut -c1-64; }
DATA_MOUNTS=()
for path in "${PATHS[@]}"; do
  real=$(readlink -f "$path") && [ -f "$real" ] || { echo "no training file $path" >&2; exit 1; }
  DATA_MOUNTS+=(-v "$real:$path:ro")
done
corpora() {
  local i
  for i in "${!KEYS[@]}"; do printf -- '--corpus\0%s=%s\0' "${KEYS[$i]}" "${PATHS[$i]}"; done
}
# dock MOUNT... -- MODULE ARGS...: the pinned image, offline, no GPU device, the sealed dir hidden
dock() {
  local -a mounts=()
  while [ "$1" != -- ]; do mounts+=("$1"); shift; done
  shift
  docker run --rm --network none --mount "type=tmpfs,destination=/data/dev2/private/sealed" \
    -v "$S:$S:ro" -v "$J:$J" "${mounts[@]}" -e PYTHONPATH="$S" -e PYTHONDONTWRITEBYTECODE=1 \
    -w "$S" --entrypoint python3 "$IMG" -m "$@" 3<&-
}

log "start: C1 v1.2 content recheck $PHASE, spec $NAME ($(sha "$SPEC")), mirror $SRC, job $J"
[ "$(docker image inspect --format '{{.Id}}' "$IMG" 3<&-)" = "$IMG" ] || abort "image $IMG missing"
[ "$(sha "$C1/c1-v1-bundle.tar.enc")" = "$BUNDLE_SHA" ] || abort "encrypted bundle hash mismatch"
[ "$(sha "$MANIFEST")" = "$MANIFEST_SHA" ] || abort "build-5 manifest hash mismatch"
[ "$(sha "$RETIRED")" = "$RETIRED_SHA" ] || abort "the v1.2 retired list differs from ${RETIRED_SHA:0:12}"
[ "$(sha "$PROT")" = "$PROT_SHA" ] || abort "protected rows differ from ${PROT_SHA:0:12}"
while read -r source digest; do
  [ "$(sha "$POOLS/$source.jsonl")" = "$digest" ] || abort "candidate file $source differs from the build-5 manifest"
done < <(python3 -c 'import json, sys
for k, v in sorted(json.load(open(sys.argv[1]))["candidates_sha256"].items()): print(k, v)' "$MANIFEST" 3<&-)
dock "${DATA_MOUNTS[@]}" -- v2.eval.sealed.recheck verify --spec "$SPEC" --output "$J/VERIFY.json" >>"$LOG" 2>&1 ||
  abort "training files differ from the spec (see $J/VERIFY.json); key not read"
log "verified the image, bundle, build-5 manifest and candidate files, v1.2 retired list, protected rows and ${#KEYS[@]} training files"
if [ "$PHASE" = verify ]; then
  log "verify-only: done"
  exit 0
fi

mkdir -p "$P" || abort "cannot create $P"
mapfile -d '' -t CORPORA < <(corpora)
log "protected rows $PROT_SHA read in place (read-only) for the disclosure scan"
dock "${DATA_MOUNTS[@]}" -v "$PROT:$PROT:ro" -v "$P:$P" -- v2.eval.sealed.overlap scan --protected "$PROT" \
  "${CORPORA[@]}" "${SCAN_ARGS[@]}" --output "$J/protected-rows/overlap-receipt.json" \
  --hits "$P/protected-rows-hits.jsonl" >"$J/protected-rows.log" 2>&1 || abort "protected-row scan failed (see $J/protected-rows.log); key not read"
log "protected-row scan done: receipt $(sha "$J/protected-rows/overlap-receipt.json")"

[ -s "$C1/c1-v1-bundle.tar.enc" ] || abort "no encrypted bundle"
IFS= read -r KEY <&3 || [ -n "$KEY" ] || abort "no key on stdin"
exec 3<&-
[ -n "$KEY" ] || abort "empty key"
T=$(mktemp -d "$C1/.recheck.XXXXXX") || abort "no temp dir"
PLAINTEXT="prompts decrypted to a private temp dir and removed"
openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -pass fd:4 -in "$C1/c1-v1-bundle.tar.enc" 4<<<"$KEY" |
  tar -xOf - v1/build-5/prompts.jsonl >"$T/prompts.jsonl"
KEY=""
unset KEY
[ "$(sha "$T/prompts.jsonl")" = "$PROMPTS_SHA" ] || abort "prompts hash mismatch"
log "prompts decrypted to a private temp dir (the gold and the salt stay encrypted); key dropped"

dock -v "$T:$T" -v "$POOLS:$POOLS:ro" -v "$MANIFEST:$MANIFEST:ro" -v "$RETIRED:$RETIRED:ro" -- \
  v2.eval.sealed.recheck items --prompts "$T/prompts.jsonl" --candidates-dir "$POOLS" --build-manifest "$MANIFEST" \
  --retired "$RETIRED" --output "$T/items.jsonl" --protected-output "$T/items.protected.jsonl" \
  --receipt "$J/ITEMS.json" >>"$LOG" 2>&1 || abort "item mapping failed (see $LOG)"
rm -f "$T/prompts.jsonl"
log "items mapped to their build-5 candidates ($(sha "$J/ITEMS.json")); prompts file removed"
dock "${DATA_MOUNTS[@]}" -v "$T:$T:ro" -v "$P:$P" -- v2.eval.sealed.recheck scan --spec "$SPEC" \
  --items "$T/items.jsonl" --output "$J/SCAN.json" --private "$P/SCAN-PRIVATE.json" >>"$LOG" 2>&1 ||
  abort "G0 scan failed or a planted control was missed (see $LOG)"
log "G0 scan done: $(sha "$J/SCAN.json")"
dock "${DATA_MOUNTS[@]}" -v "$T:$T:ro" -v "$P:$P" -- v2.eval.sealed.overlap scan --protected "$T/items.protected.jsonl" \
  "${CORPORA[@]}" "${SCAN_ARGS[@]}" --output "$J/items-scan/overlap-receipt.json" --hits "$P/items-hits.jsonl" \
  >"$J/items-scan.log" 2>&1 || abort "registered scan of the items failed (see $J/items-scan.log)"
log "registered scan of the items done: receipt $(sha "$J/items-scan/overlap-receipt.json")"
dock "${DATA_MOUNTS[@]}" -v "$T:$T:ro" -v "$P:$P" -- v2.eval.sealed.recheck judge --spec "$SPEC" \
  --items "$T/items.jsonl" --protected-items "$T/items.protected.jsonl" --scan "$J/SCAN.json" \
  --scan-private "$P/SCAN-PRIVATE.json" --registered-hits "$P/items-hits.jsonl" \
  --registered-receipt "$J/items-scan/overlap-receipt.json" --protected-hits "$P/protected-rows-hits.jsonl" \
  --protected-receipt "$J/protected-rows/overlap-receipt.json" --output "$J/VERDICT.json" \
  --quarantine-dir "$P/quarantine" | tee -a "$LOG"
rc=${PIPESTATUS[0]}
[ "$rc" = 0 ] || abort "judge failed (exit $rc)"
rm -rf "$T"
T=""
log "temp dir removed"
sums=$(cd "$J" && find . -type f ! -name SHA256SUMS ! -name RECHECK.log -print0 | LC_ALL=C sort -z | xargs -0 sha256sum)
printf '%s\n' "$sums" >"$J/SHA256SUMS"
log "end: verdict $(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["verdict"])' "$J/VERDICT.json") ($(sha "$J/VERDICT.json"))"
