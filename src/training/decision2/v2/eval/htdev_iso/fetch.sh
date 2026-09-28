#!/usr/bin/env bash
# Download every HT-DEV candidate source at its pinned revision (node A host; network).
#
# Usage: fetch.sh <sources.json> <iso-dir> [<key> ...]
#
# Raw artefacts land in <iso-dir>/raw/<key>/ (mode 700): HF snapshots via the hf CLI
# at the full revision SHA, GitHub commit tarballs from codeload, Zenodo files with
# the record's md5 checked, plain URLs as given. A key whose raw directory exists is
# skipped. Conversion into row files is prepare.py's job (no network).
set -euo pipefail
umask 077
spec="$1"; iso="$2"; shift 2
HF=${HF_CLI:-/usr/local/bin/hf}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/data/dev2/hf-cache}
mkdir -p "$iso/raw"; chmod 700 "$iso" "$iso/raw"
keys=("$@")
if [[ ${#keys[@]} -eq 0 ]]; then
  mapfile -t keys < <(python3 -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["sources"]))' "$spec")
fi
field() { python3 -c 'import json,sys; v=json.load(open(sys.argv[1]))["sources"][sys.argv[2]].get(sys.argv[3]); print("\n".join(v) if isinstance(v,list) else ("" if v is None else v))' "$spec" "$1" "$2"; }
for key in "${keys[@]}"; do
  out="$iso/raw/$key"
  if [[ -e "$out" ]]; then echo "skip $key (exists)"; continue; fi
  origin=$(field "$key" origin); rev=$(field "$key" revision)
  tmp="$out.partial"; rm -rf "$tmp"; mkdir -p "$tmp"
  case "$origin" in
    hf)
      repo=$(field "$key" repo); args=()
      while read -r pattern; do [[ -n "$pattern" ]] && args+=(--include "$pattern"); done < <(field "$key" include)
      "$HF" download "$repo" --repo-type dataset --revision "$rev" --local-dir "$tmp" "${args[@]}" --quiet >/dev/null
      rm -rf "$tmp/.cache"
      ;;
    github)
      repo=$(field "$key" repo)
      curl -sSfL --retry 3 "https://codeload.github.com/$repo/tar.gz/$rev" -o "$tmp/${repo##*/}-$rev.tar.gz"
      ;;
    zenodo)
      record=$(field "$key" record)
      curl -sSfL --retry 3 "https://zenodo.org/api/records/$record" -o "$tmp/zenodo-record.json"
      while read -r name; do
        curl -sSfL --retry 3 "https://zenodo.org/records/$record/files/$name?download=1" -o "$tmp/$name"
        want=$(python3 -c 'import json,sys; print([f["checksum"] for f in json.load(open(sys.argv[1]))["files"] if f["key"]==sys.argv[2]][0].split(":")[1])' "$tmp/zenodo-record.json" "$name")
        got=$(md5sum "$tmp/$name" | cut -d' ' -f1)
        [[ "$want" == "$got" ]] || { echo "md5 mismatch $key/$name" >&2; exit 1; }
      done < <(field "$key" files)
      ;;
    url)
      while read -r url; do curl -sSfL --retry 3 "$url" -o "$tmp/${url##*/}"; done < <(field "$key" urls)
      ;;
    unavailable)
      echo "unavailable $key"; rmdir "$tmp"; continue
      ;;
    *) echo "unknown origin $origin" >&2; exit 1 ;;
  esac
  mv "$tmp" "$out"; chmod -R go-rwx "$out"
  echo "fetched $key files=$(find "$out" -type f | wc -l) bytes=$(du -sb "$out" | cut -f1)"
done
