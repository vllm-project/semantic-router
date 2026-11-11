#!/usr/bin/env bash
# Download the dependency charts Chart.lock pins into the chart's charts/,
# retrying a failed download with exponential backoff.
#
#   deploy/helm/fetch-dependencies.sh [CHART_PATH]
#
# HELM_DEPENDENCIES_CACHED=true skips the download: a workflow sets it when
# its cache, keyed on Chart.lock, restored the archives. HELM_REPO_UPDATE=false
# uses the repositories already configured instead of adding them.
# HELM_DEPENDENCY_ATTEMPTS (4) and HELM_DEPENDENCY_BACKOFF_SECONDS (10,
# doubled after each failed attempt) tune the retries.
set -euo pipefail

chart="${1:-deploy/helm/semantic-router}"
attempts="${HELM_DEPENDENCY_ATTEMPTS:-4}"
delay="${HELM_DEPENDENCY_BACKOFF_SECONDS:-10}"

report() {
  local level=$1 message=$2
  if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
    echo "::${level}::${message}" >&2
  else
    echo "${level}: ${message}" >&2
  fi
}

if [[ "${HELM_DEPENDENCIES_CACHED:-}" == "true" ]]; then
  echo "Using the dependency charts restored from the cache for $chart/Chart.lock"
  exit 0
fi

add_repositories() {
  helm repo add bitnami https://charts.bitnami.com/bitnami --force-update &&
    helm repo add milvus https://milvus-io.github.io/milvus-helm/ --force-update &&
    helm repo add jaegertracing https://jaegertracing.github.io/helm-charts --force-update &&
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts --force-update &&
    helm repo add grafana https://grafana.github.io/helm-charts --force-update
}

fetch() {
  if [[ "${HELM_REPO_UPDATE:-true}" == "true" ]]; then
    add_repositories || return 1
  fi
  # Adding a repository fetched its index, so the build needs no refresh.
  helm dependency build "$chart" --skip-refresh
}

for ((attempt = 1; attempt <= attempts; attempt++)); do
  if fetch; then
    exit 0
  fi
  if ((attempt < attempts)); then
    report warning "Downloading the dependency charts of $chart failed (attempt $attempt of $attempts); retrying in ${delay}s"
    sleep "$delay"
    delay=$((delay * 2))
  fi
done

report error "Could not download the dependency charts of $chart after $attempts attempts. Chart.lock pins them; check that their chart repositories and release downloads are reachable, then re-run."
exit 1
