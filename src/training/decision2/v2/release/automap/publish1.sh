#!/usr/bin/env bash
# Publish the Transformers remote code to one Decision 1.0 repository, step by step, on a node.
#
# Usage: publish1.sh <step> <Repo-Name> <work-root> [PR number]
#   stage       download config.json / README.md / MANIFEST.json at the live head, stage the update,
#               save the head's file listing and check that no open PR touches the same files
#   pr          open one PR with exactly the staged files (parent commit = staged head)
#   smoke-pr N  fresh venv (stock Transformers, CPU): load refs/pr/N with trust_remote_code
#               (the node's cache supplies the unchanged weights)
#   merge N     merge our PR N if the head is unchanged and no open PR touches the files
#   readback    every staged file and every unchanged file at the new main, by hash
#   smoke-main  fresh venv and an empty Hugging Face cache: load main, run the card's Python block
set -euo pipefail
step="$1"; repo="$2"; root="$3"; number="${4:-}"
here="$(cd "$(dirname "$0")" && pwd)"
hf_python=/data/dev2/tools/hf-cli/bin/python3
image=decision20-train-fast:host2
fresh=/data/dev2/tools/envs/fresh-cpu/bin/python
id="llm-semantic-router/$repo"
work="$root/$repo"
hub() { "$hf_python" "$here/hub1.py" "$@" --repo "$id" --stage "$work/stage"; }
fresh_run() {
  local home="$1" cache="$2"; shift 2
  docker run --rm --network host -v /data/dev2:/data/dev2 -e HF_HOME="$home" -e HF_HUB_CACHE="$cache" \
    -e HF_HUB_DISABLE_TELEMETRY=1 --entrypoint "$fresh" "$image" -B "$here/smoke1.py" "$@"
}
case "$step" in
  stage)
    [[ -e "$work" ]] && { echo "work dir exists: $work" >&2; exit 1; }
    mkdir -p "$work/head"
    head=$("$hf_python" -c "from huggingface_hub import HfApi; print(HfApi().model_info('$id').sha)")
    "$hf_python" - "$id" "$head" "$work/head" <<'PY'
import sys
from huggingface_hub import HfApi, hf_hub_download
repo, head, target = sys.argv[1:]
names = {s.rfilename for s in HfApi().model_info(repo, revision=head).siblings}
for name in ("config.json", "README.md", "MANIFEST.json"):
    if name in names:
        hf_hub_download(repo, name, revision=head, local_dir=target)
PY
    for path in "$here"/decision1/*.py; do
      name="$(basename "$path")"
      [[ -e "$work/head/$name" ]] && { echo "$name already in the repository" >&2; exit 1; }
    done
    python3 "$here/stage1.py" --repo "$id" --head "$head" --snapshot "$work/head" --output "$work/stage"
    "$hf_python" "$here/hub1.py" listing --repo "$id" --stage "$work/stage" --revision "$head" > "$work/before.json"
    hub check | tee "$work/check.json"
    ;;
  pr) hub pr | tee "$work/pr.json" ;;
  smoke-pr)
    mkdir -p "$work/home-pr"
    # The PR check reuses the node's cache for the unchanged weights; smoke-main starts empty.
    fresh_run "$work/home-pr" /data/dev2/hf-cache --repo "$id" --revision "refs/pr/$number" --device cpu --output "$work/smoke-pr.json" \
      2>&1 | tail -n 30
    ;;
  merge) hub merge --pr "$number" | tee "$work/merge.json" ;;
  readback)
    main=$("$hf_python" -c "from huggingface_hub import HfApi; print(HfApi().model_info('$id').sha)")
    hub readback --revision "$main" --before "$work/before.json" | tee "$work/readback.json"
    ;;
  smoke-main)
    mkdir -p "$work/home-main"
    fresh_run "$work/home-main" "$work/home-main/hub" --repo "$id" --device cpu --card --output "$work/smoke-main.json" 2>&1 | tail -n 30
    ;;
  *) echo "unknown step $step" >&2; exit 2 ;;
esac
