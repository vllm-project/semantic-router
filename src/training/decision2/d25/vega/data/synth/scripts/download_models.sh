#!/usr/bin/env bash
# Download pinned permissive generator/teacher models into /data/d25/shared/models/<name>.
# Resumable: snapshot_download skips complete files; a .complete marker records the revision.
# Usage (inside a pod with HF_TOKEN): download_models.sh NAME=REPO@REV [NAME=REPO@REV ...]
set -euo pipefail
ROOT=/data/d25/shared/models
mkdir -p "$ROOT" /data/d25/vega/synth/logs
pids=()
for spec in "$@"; do
  name=${spec%%=*}; rest=${spec#*=}; repo=${rest%@*}; rev=${rest#*@}
  dest="$ROOT/$name"
  if [[ -f "$dest/.complete" ]] && grep -q "$rev" "$dest/.complete"; then
    echo "skip $name (complete at $rev)"; continue
  fi
  (
    python3 - "$repo" "$rev" "$dest" <<'PY'
import sys, time
from huggingface_hub import snapshot_download
repo, rev, dest = sys.argv[1:4]
t = time.time()
snapshot_download(repo_id=repo, revision=rev, local_dir=dest, max_workers=24)
print(f"downloaded {repo}@{rev} to {dest} in {time.time() - t:.0f}s", flush=True)
PY
    echo "$repo $rev $(date -u +%FT%TZ)" > "$dest/.complete"
  ) > "/data/d25/vega/synth/logs/download-$name.log" 2>&1 &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
du -sh "$ROOT"/* || true
exit $status
