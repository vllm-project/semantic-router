#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
PYPROJECT_PATH="$PROJECT_DIR/pyproject.toml"
CANDLE_CARGO_PATH="$REPO_ROOT/candle-binding/Cargo.toml"
CANDLE_LOCK_PATH="$REPO_ROOT/candle-binding/Cargo.lock"

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

The script updates every versioned surface validated by the release workflow:
the vllm-sr Python package and the candle-semantic-router Rust crate. It also
runs the repo-level release contract check before creating the stable tag.
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

current_candle_version() {
  python3 - "$CANDLE_CARGO_PATH" <<'PY'
import re
import sys
from pathlib import Path

content = Path(sys.argv[1]).read_text()
package_section = re.split(r"^\[(?!package\])", content, maxsplit=1, flags=re.MULTILINE)[0]
match = re.search(r'^version\s*=\s*"([^"]+)"', package_section, re.MULTILINE)
print(match.group(1) if match else "")
PY
}

write_pyproject_version() {
  local value
  value="$1"
  sed -i.bak 's/^version = .*/version = "'"$value"'"/' "$PYPROJECT_PATH"
  rm -f "$PYPROJECT_PATH.bak"
}

write_candle_version() {
  local value
  value="$1"
  python3 - "$CANDLE_CARGO_PATH" "$CANDLE_LOCK_PATH" "$value" <<'PY'
import re
import sys
from pathlib import Path

cargo_path = Path(sys.argv[1])
lock_path = Path(sys.argv[2])
version = sys.argv[3]

def replace_one(path: Path, pattern: re.Pattern[str], label: str) -> None:
    content = path.read_text()
    updated, count = pattern.subn(rf'\g<prefix>{version}\g<suffix>', content, count=1)
    if count != 1:
        raise SystemExit(f"could not update {label} in {path}")
    path.write_text(updated)

replace_one(
    cargo_path,
    re.compile(
        r'(?ms)(?P<prefix>^\[package\]\n(?:(?!^\[).)*?^version\s*=\s*")[^"]+(?P<suffix>")'
    ),
    "candle-binding package version",
)

replace_one(
    lock_path,
    re.compile(
        r'(?ms)(?P<prefix>^\[\[package\]\]\nname = "candle-semantic-router"\nversion = ")[^"]+(?P<suffix>")'
    ),
    "candle-binding lockfile package version",
)
PY
}

write_version() {
  local value
  value="$1"
  write_pyproject_version "$value"
  write_candle_version "$value"
}

release_files_match() {
  local expected
  expected="$1"
  [ "$(current_version)" = "$expected" ] && [ "$(current_candle_version)" = "$expected" ]
}

default_next_version() {
  local major minor patch
  IFS='.' read -r major minor patch <<EOF
$RELEASE_VERSION
EOF
  printf '%s\n' "$((major + 0)).$((minor + 1)).0"
}

commit_if_changed() {
  local message
  message="$1"
  if git diff --quiet -- "$PYPROJECT_PATH" "$CANDLE_CARGO_PATH" "$CANDLE_LOCK_PATH"; then
    return 1
  fi

  git add "$PYPROJECT_PATH" "$CANDLE_CARGO_PATH" "$CANDLE_LOCK_PATH"
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

  cd "$REPO_ROOT"
  require_clean_worktree

  local start_ref active_branch current candle_current tag_name
  start_ref="$(git rev-parse --verify HEAD)"
  active_branch="$(git rev-parse --abbrev-ref HEAD)"
  current="$(current_version)"
  candle_current="$(current_candle_version)"
  tag_name="v$RELEASE_VERSION"

  git rev-parse --verify "$tag_name" >/dev/null 2>&1 && die "tag already exists: $tag_name"

  echo "Current release versions:"
  echo "  vllm-sr                $current"
  echo "  candle-semantic-router $candle_current"

  if ! release_files_match "$RELEASE_VERSION"; then
    write_version "$RELEASE_VERSION"
    commit_if_changed "chore(vllm-sr): release v$RELEASE_VERSION" || true
  fi

  python3 "$REPO_ROOT/tools/release/check_version_contract.py" --version "$RELEASE_VERSION"

  git tag -a "$tag_name" -m "vLLM Semantic Router v$RELEASE_VERSION"

  write_version "$NEXT_VERSION"
  commit_if_changed "chore(vllm-sr): start $NEXT_VERSION dev cycle" || die "next version bump did not change pyproject.toml"

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
