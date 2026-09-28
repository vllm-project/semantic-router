#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-}"
if [[ -z "${repo_root}" ]]; then
  if ! repo_root=$(git rev-parse --show-toplevel 2>/dev/null); then
    printf '%s\n' unavailable
    exit 0
  fi
fi

repo_root=$(cd "${repo_root}" && pwd -P)

if git -C "${repo_root}" diff --quiet HEAD -- &&
  git -C "${repo_root}" diff --cached --quiet HEAD -- &&
  [[ -z "$(git -C "${repo_root}" ls-files --others --exclude-standard)" ]]; then
  git -C "${repo_root}" rev-parse HEAD
  exit 0
fi

digest_root=$(mktemp -d "${TMPDIR:-/tmp}/vllm-sr-source-revision.XXXXXX")
trap 'rm -rf "${digest_root}"' EXIT

git init --quiet --bare --object-format=sha256 "${digest_root}/repo.git"
GIT_DIR="${digest_root}/repo.git" GIT_WORK_TREE="${repo_root}" \
  git add -A -- .
tree=$(GIT_DIR="${digest_root}/repo.git" git write-tree)

if [[ ! "${tree}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Unable to derive an immutable SHA-256 source-tree revision" >&2
  exit 1
fi

printf 'sha256:%s\n' "${tree}"
