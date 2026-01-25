#!/bin/bash
# Install KServe and dependencies for OpenShift clusters without a preinstalled KServe stack.
# Mirrors the MaaS installer flow while using oc for OpenShift clusters.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

KSERVE_VERSION="v0.15.2"
CERT_MANAGER_VERSION="v1.14.5"
OCP=false

usage() {
    cat <<EOF
Usage: $0 [--ocp]

Options:
  --ocp    Validate OpenShift Serverless instead of installing vanilla KServe
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ocp)
            OCP=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

log() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

if ! command -v oc &>/dev/null; then
    error "oc CLI not found. Install OpenShift CLI first."
    exit 1
fi

if [[ "$OCP" == true ]]; then
    log "Validating OpenShift Serverless operator is installed..."
    if ! oc get subscription serverless-operator -n openshift-serverless >/dev/null 2>&1; then
        error "OpenShift Serverless operator not found. Please install it first."
        exit 1
    fi

    log "Validating OpenShift Serverless controller is running..."
    if ! oc wait --for=condition=ready pod --all -n openshift-serverless --timeout=60s >/dev/null 2>&1; then
        error "OpenShift Serverless controller is not ready."
        exit 1
    fi

    success "OpenShift Serverless operator is installed and running."
    exit 0
fi

if oc get crd inferenceservices.serving.kserve.io &>/dev/null; then
    success "KServe CRDs already installed."
    exit 0
fi

if ! oc get crd certificates.cert-manager.io &>/dev/null; then
    log "Installing cert-manager ($CERT_MANAGER_VERSION)..."
    oc apply -f "https://github.com/cert-manager/cert-manager/releases/download/${CERT_MANAGER_VERSION}/cert-manager.yaml"
    if oc get namespace cert-manager &>/dev/null; then
        oc wait --for=condition=Available deployment/cert-manager -n cert-manager --timeout=5m || true
        oc wait --for=condition=Available deployment/cert-manager-webhook -n cert-manager --timeout=5m || true
        oc wait --for=condition=Available deployment/cert-manager-cainjector -n cert-manager --timeout=5m || true
    else
        warn "cert-manager namespace not found after install; continuing."
    fi
else
    log "cert-manager CRDs already present."
fi

log "Installing KServe ($KSERVE_VERSION)..."
oc apply -f "https://github.com/kserve/kserve/releases/download/${KSERVE_VERSION}/kserve.yaml"
oc apply -f "https://github.com/kserve/kserve/releases/download/${KSERVE_VERSION}/kserve-cluster-resources.yaml"

if oc get namespace kserve &>/dev/null; then
    oc wait --for=condition=Available deployment/kserve-controller-manager -n kserve --timeout=5m || true
else
    warn "KServe namespace not found after install; verify installation."
fi

if oc get crd inferenceservices.serving.kserve.io &>/dev/null; then
    success "KServe CRDs installed."
else
    error "KServe CRDs still missing after install."
    exit 1
fi
