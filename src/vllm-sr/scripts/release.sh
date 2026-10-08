#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
PYPROJECT_PATH="$PROJECT_DIR/pyproject.toml"
HELM_CHART_PATH="$REPO_ROOT/deploy/helm/semantic-router/Chart.yaml"
CONTRACT_CHECK="$REPO_ROOT/tools/release/check_version_contract.py"
# The source chart's images during a development cycle: the tag main publishes.
DEVELOPMENT_IMAGE_TAG="latest"

RELEASE_VERSION="${1:-}"
NEXT_VERSION="${2:-}"

usage() {
  cat <<'EOF'
Usage: scripts/release.sh <release-version> [next-version]

Examples:
  scripts/release.sh 0.3.0
  scripts/release.sh 0.3.0 0.4.0

When next-version is omitted, the script defaults to the next minor base
version (for example 0.3.0 -> 0.4.0).

The release commit sets the vllm-sr Python package version and pins the
source Helm chart's images to the release tag; the repo-level release
contract check runs on it before the stable tag is created. The next commit
starts the next development cycle: the package moves to next-version and the
chart goes back to the development images (latest).
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

require_clean_worktree() {
  if [ -n "$(git status --porcelain)" ]; then
    die "working tree must be clean before running a release"
  fi
}

validate_semver() {
  local value
  value="$1"
  [[ "$value" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "version must use semantic versioning (x.y.z): $value"
}

current_version() {
  grep '^version = ' "$PYPROJECT_PATH" | sed 's/version = "\(.*\)"/\1/'
}

write_pyproject_version() {
  local value
  value="$1"
  sed -i.bak 's/^version = .*/version = "'"$value"'"/' "$PYPROJECT_PATH"
  rm -f "$PYPROJECT_PATH.bak"
}

write_chart_app_version() {
  local value
  value="$1"
  grep -q '^appVersion: ' "$HELM_CHART_PATH" || die "no appVersion in $HELM_CHART_PATH"
  sed -i.bak 's/^appVersion: .*/appVersion: "'"$value"'"/' "$HELM_CHART_PATH"
  rm -f "$HELM_CHART_PATH.bak"
}

default_next_version() {
  local major minor
  IFS='.' read -r major minor _ <<EOF
$RELEASE_VERSION
EOF
  printf '%s\n' "$((major + 0)).$((minor + 1)).0"
}

sorts_after() {
  [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | tail -n 1)" = "$1" ]
}

commit_if_changed() {
  local message
  message="$1"
  if git diff --quiet -- "$PYPROJECT_PATH" "$HELM_CHART_PATH"; then
    return 1
  fi

  git add "$PYPROJECT_PATH" "$HELM_CHART_PATH"
  git commit -s -m "$message"
  return 0
}

main() {
  [ -n "$RELEASE_VERSION" ] || {
    usage
    exit 1
  }

  validate_semver "$RELEASE_VERSION"
  if [ -z "$NEXT_VERSION" ]; then
    NEXT_VERSION="$(default_next_version)"
  fi
  validate_semver "$NEXT_VERSION"
  [ "$NEXT_VERSION" != "$RELEASE_VERSION" ] || die "next version must differ from release version"
  sorts_after "$NEXT_VERSION" "$RELEASE_VERSION" || die "next version $NEXT_VERSION must sort after release version $RELEASE_VERSION"

  cd "$REPO_ROOT"
  require_clean_worktree

  local start_ref active_branch current tag_name reset
  start_ref="$(git rev-parse --verify HEAD)"
  active_branch="$(git rev-parse --abbrev-ref HEAD)"
  current="$(current_version)"
  tag_name="v$RELEASE_VERSION"
  reset="git reset --hard $(git rev-parse --short "$start_ref")"

  git rev-parse --verify "$tag_name" >/dev/null 2>&1 && die "tag already exists: $tag_name"

  echo "Current release version:"
  echo "  vllm-sr $current"

  write_pyproject_version "$RELEASE_VERSION"
  write_chart_app_version "$tag_name"
  commit_if_changed "chore(vllm-sr): release v$RELEASE_VERSION" || true

  python3 "$CONTRACT_CHECK" --version "$RELEASE_VERSION" \
    || die "release v$RELEASE_VERSION fails the version contract; nothing is tagged. Undo with: $reset"

  git tag -a "$tag_name" -m "vLLM Semantic Router v$RELEASE_VERSION"

  write_pyproject_version "$NEXT_VERSION"
  write_chart_app_version "$DEVELOPMENT_IMAGE_TAG"
  commit_if_changed "chore(vllm-sr): start $NEXT_VERSION dev cycle" || die "the next development cycle changes nothing"

  python3 "$CONTRACT_CHECK" \
    || die "the $NEXT_VERSION development cycle fails the version contract; nothing is pushed. Undo with: git tag -d $tag_name && $reset"

  cat <<EOF
Prepared local release state:
  branch        $active_branch
  release tag   $tag_name
  release ref   $(git rev-parse --short "$tag_name^{commit}")
  next version  $NEXT_VERSION
  start ref     $(git rev-parse --short "$start_ref")
  head ref      $(git rev-parse --short HEAD)

Next step:
  git push origin HEAD --follow-tags
EOF
}

main "$@"
