#!/usr/bin/env bash
set -euo pipefail

: "${GITHUB_OUTPUT:?GITHUB_OUTPUT is required}"

image_name="${MATRIX_IMAGE:?MATRIX_IMAGE is required}"
cargo_build_jobs="${CARGO_BUILD_JOBS:?CARGO_BUILD_JOBS is required}"
dashboard_version_mode="${DASHBOARD_VERSION_MODE:-main}"
project_version_file="${PROJECT_VERSION_FILE:-src/vllm-sr/pyproject.toml}"

dashboard_version=""

if [[ "${image_name}" == "dashboard" ]]; then
  project_version=$(sed -n 's/^version = "\(.*\)"/\1/p' "${project_version_file}" | head -n1)
  if [[ -z "${project_version}" ]]; then
    echo "::error file=${project_version_file}::Unable to resolve vllm-sr project version" >&2
    exit 1
  fi

  short_sha=$(git rev-parse --short=7 HEAD)

  case "${dashboard_version_mode}" in
    pr)
      pr_number="${PR_NUMBER:?PR_NUMBER is required for dashboard PR version}"
      dashboard_version="v${project_version}-dev.pr-${pr_number}.${short_sha}"
      ;;
    publish)
      if [[ "${IS_NIGHTLY:-}" == "true" ]]; then
        nightly_date="${NIGHTLY_DATE:?NIGHTLY_DATE is required for nightly dashboard version}"
        dashboard_version="v${project_version}-nightly.${nightly_date}.${short_sha}"
      else
        dashboard_version="v${project_version}-dev.${short_sha}"
      fi
      ;;
    *)
      echo "::error::Unsupported DASHBOARD_VERSION_MODE '${dashboard_version_mode}'" >&2
      exit 1
      ;;
  esac
fi

{
  echo 'args<<EOF'
  echo 'BUILDKIT_INLINE_CACHE=1'
  echo "CARGO_BUILD_JOBS=${cargo_build_jobs}"
  echo 'CARGO_INCREMENTAL=1'
  echo 'RUSTC_WRAPPER=""'
  echo 'CARGO_NET_GIT_FETCH_WITH_CLI=true'
  if [[ -n "${dashboard_version}" ]]; then
    echo "DASHBOARD_VERSION=${dashboard_version}"
  fi
  echo 'EOF'
} >> "${GITHUB_OUTPUT}"
