#!/usr/bin/env bash
# Publish sanitized snapshots of the coordinator's local program docs on the
# review branch xunzhuo/decision-2-training.
#
# Usage: sync_program_docs.sh [--dry-run]
#
# Regenerates BRIEF.md, COORDINATION.md and STATUS.md next to this script from
# ${DEV2_PROGRAM_DIR:-$HOME/code/decision2-program}/{BRIEF.md,COORDINATION.md,00-decision-2-program-status.md},
# replacing every node address and hostname listed in the private nodes.env and
# node-names.env (see v2/common/check_no_private.sh) with the node's label
# (alias node-a -> "node A"). Lines from one containing <!-- private --> through
# one containing <!-- /private --> stay local: the snapshot shows a one-line
# omission note instead (an unterminated section is an error).
#
# Then, in the worktree holding this script, which must be on
# xunzhuo/decision-2-training with nothing staged: merges
# origin/xunzhuo/decision-2-training, commits the snapshots with a DCO sign-off
# only if they changed, checks every unpushed commit with the leak guard, pushes
# as a fast-forward (on rejection: fetch, merge, retry) and prints the resulting
# commit SHA on stdout. Nothing is copied, committed or pushed when the guard
# reports anything. Re-running without source changes commits nothing.
# --dry-run only regenerates and checks the snapshots in a scratch directory
# and prints their diff against the copies in the worktree.
set -euo pipefail

branch="xunzhuo/decision-2-training"
remote="origin"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(git -C "$here" rev-parse --show-toplevel)"
rel="${here#"$repo"/}"
guard="$repo/src/training/decision2/v2/common/check_no_private.sh"
source_dir="${DEV2_PROGRAM_DIR:-$HOME/code/decision2-program}"
snapshots=(BRIEF.md COORDINATION.md STATUS.md)

dry_run=0
case "${1:-}" in
  "") ;;
  --dry-run) dry_run=1 ;;
  *) sed -n '2,/^set -euo pipefail$/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//' >&2; exit 2 ;;
esac

say() { echo "sync_program_docs: $*" >&2; }
die() { say "$*"; exit 1; }
g() { git -C "$repo" "$@"; }

scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

generate() {
  python3 - "$source_dir" "$1" <<'PY'
import os
import re
import sys

source_dir, out_dir = sys.argv[1], sys.argv[2]
SOURCES = [
    ("BRIEF.md", "BRIEF.md"),
    ("COORDINATION.md", "COORDINATION.md"),
    ("00-decision-2-program-status.md", "STATUS.md"),
]


def pairs(env_var, default, required):
    path = os.path.expanduser(os.environ.get(env_var) or default)
    if not os.path.isfile(path):
        if required:
            sys.exit(f"sync_program_docs: {path} not found")
        return []
    found = []
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = (part.strip() for part in line.split("=", 1))
            if key and value:
                found.append((key, value))
    return found


def label(alias):
    match = re.fullmatch(r"node[-_ ]?([A-Za-z0-9]+)", alias)
    return f"node {match.group(1).upper()}" if match else alias


def drop_private(text, source):
    kept, hidden, omitted = [], False, 0
    for line in text.splitlines(keepends=True):
        if "<!-- private -->" in line:
            if hidden:
                sys.exit(f"sync_program_docs: nested <!-- private --> in {source}")
            hidden, omitted = True, omitted + 1
            indent = line[: len(line) - len(line.lstrip())]
            kept.append(f"{indent}*(A private section is omitted from this snapshot.)*\n")
        elif "<!-- /private -->" in line:
            if not hidden:
                sys.exit(f"sync_program_docs: unmatched <!-- /private --> in {source}")
            hidden = False
        elif not hidden:
            kept.append(line)
    if hidden:
        sys.exit(f"sync_program_docs: unterminated <!-- private --> in {source}")
    return "".join(kept), omitted


def terms(value):
    host = re.sub(r"^[A-Za-z][A-Za-z0-9+.-]*://", "", value)
    host = host.rsplit("@", 1)[-1].split("/", 1)[0]
    if host.count(":") == 1:
        host = host.split(":", 1)[0]
    found = {value, host}
    if re.fullmatch(r"[0-9]{1,3}(?:\.[0-9]{1,3}){3}", host):
        found |= {host.replace(".", "-"), host.replace(".", "_")}
    elif "." in host:
        found.add(host.split(".", 1)[0])
    return {term for term in found if len(term) >= 4}


replacements = {}
for env_var, default, required in (
    ("DEV2_NODES_FILE", "~/.config/decision2/nodes.env", True),
    ("DEV2_NODE_NAMES_FILE", "~/.config/decision2/node-names.env", False),
):
    for alias, value in pairs(env_var, default, required):
        for term in terms(value):
            replacements[term.lower()] = label(alias)
if not replacements:
    sys.exit("sync_program_docs: no node addresses found in nodes.env")
labels = sorted(set(replacements.values()))
pattern = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    + "|".join(re.escape(t) for t in sorted(replacements, key=len, reverse=True))
    + r")(?![A-Za-z0-9])",
    re.IGNORECASE | re.ASCII,
)
shown = ", ".join(f'"{name}"' for name in labels)
for source, target in SOURCES:
    with open(os.path.join(source_dir, source), encoding="utf-8") as handle:
        text, omitted = drop_private(handle.read(), source)
    text = pattern.sub(lambda m: replacements[m.group(0).lower()], text)
    for name in labels:
        q = re.escape(name)
        text = re.sub(rf"{q} = `{q}`(?: \({q}\))?", name, text)
        text = re.sub(rf"`{q}` \({q}\)", name, text)
        text = re.sub(rf"{q} \({q}\)", name, text)
    unchanged = (
        f"{omitted} section(s) marked private are omitted; nothing else is changed"
        if omitted
        else "nothing else is changed"
    )
    header = (
        f"> Sanitized snapshot of the coordinator's local `{source}`, regenerated by "
        f"[`sync_program_docs.sh`](sync_program_docs.sh): node addresses and hostnames "
        f"are replaced by {shown}; {unchanged}. Do not edit this copy; "
        f"see [README.md](README.md).\n\n"
    )
    with open(os.path.join(out_dir, target), "w", encoding="utf-8", newline="\n") as handle:
        handle.write(header + text.rstrip("\n") + "\n")
PY
}

check_snapshots() {
  local out="$scratch/guard.out"
  if ! "$guard" --strict -- "${snapshots[@]/#/$scratch/}" >"$out" 2>&1; then
    sed "s|$scratch/|$rel/|g" "$out" >&2
    die "the leak guard rejected the regenerated snapshots; fix the local source (or add the node to nodes.env / node-names.env); nothing committed"
  fi
}

generate "$scratch"
check_snapshots

if (( dry_run )); then
  for f in "${snapshots[@]}"; do
    current="$here/$f"
    [[ -f "$current" ]] || current=/dev/null
    diff -u --label "a/$rel/$f" --label "b/$rel/$f" "$current" "$scratch/$f" || true
  done
  exit 0
fi

[[ "$(g symbolic-ref --short -q HEAD || true)" == "$branch" ]] || die "$repo is not on $branch"
g diff --cached --quiet || die "$repo has staged changes; commit or unstage them first"

merge_remote() {
  g fetch --quiet "$remote" "$branch"
  if ! g merge --quiet --no-edit --signoff -m "merge(decision2): integrate $remote/$branch into the program docs sync" "$remote/$branch"; then
    g merge --abort >/dev/null 2>&1 || true
    die "merging $remote/$branch failed; resolve it by hand"
  fi
}

merge_remote
for f in "${snapshots[@]}"; do cp "$scratch/$f" "$here/$f"; done
g add -- "${snapshots[@]/#/$rel/}"
if g diff --cached --quiet; then
  say "snapshots unchanged; nothing to commit"
else
  changed="$(g diff --cached --name-only | sed 's|.*/||; s|\.md$||' | paste -sd, - | sed 's/,/, /g')"
  if ! "$guard" --strict; then
    g reset --quiet -- "${snapshots[@]/#/$rel/}"
    die "the leak guard rejected the staged snapshots; nothing committed"
  fi
  g commit --quiet --signoff -m "docs(decision2): sync sanitized program docs ($changed)"
  say "committed $(g rev-parse --short HEAD) ($changed)"
fi

attempt=1
while :; do
  "$guard" --strict --log "$remote/$branch..HEAD" || die "the leak guard rejected an unpushed commit; nothing pushed"
  if g push --quiet "$remote" "HEAD:refs/heads/$branch" 2>"$scratch/push.err"; then
    break
  fi
  if ! grep -qE 'rejected|fetch first|non-fast-forward' "$scratch/push.err" || (( attempt >= 5 )); then
    cat "$scratch/push.err" >&2
    die "push failed"
  fi
  attempt=$((attempt + 1))
  say "push rejected; fetching, merging and retrying"
  merge_remote
done

head="$(g rev-parse HEAD)"
[[ "$(g rev-parse "$remote/$branch")" == "$head" ]] || die "$remote/$branch does not point at $head after the push"
say "$remote/$branch is at $head"
echo "$head"
