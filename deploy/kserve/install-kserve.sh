#!/bin/bash
# Install KServe and dependencies for OpenShift clusters without a preinstalled KServe stack.
# Mirrors the MaaS installer flow while using oc for OpenShift clusters.
# Supports both InferenceService and LLMInferenceService CRDs.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

KSERVE_VERSION="v0.15.2"
LLMISVC_VERSION="v0.15.2"
CERT_MANAGER_VERSION="v1.14.5"
OCP=false
INSTALL_LLMISVC=true

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

cleanup_cert_manager_webhooks() {
    # Stale webhook configs block installs if cert-manager namespace is gone.
    if ! oc get namespace cert-manager &>/dev/null; then
        local has_validating=false
        local has_mutating=false
        if oc get validatingwebhookconfiguration cert-manager-webhook &>/dev/null; then
            has_validating=true
        fi
        if oc get mutatingwebhookconfiguration cert-manager-webhook &>/dev/null; then
            has_mutating=true
        fi
        if [[ "$has_validating" == true || "$has_mutating" == true ]]; then
            warn "cert-manager namespace not found but webhook configs exist; removing stale webhooks"
            oc delete validatingwebhookconfiguration cert-manager-webhook --ignore-not-found
            oc delete mutatingwebhookconfiguration cert-manager-webhook --ignore-not-found
        fi
    fi
}

ensure_cert_manager() {
    cleanup_cert_manager_webhooks
    if ! oc get namespace cert-manager &>/dev/null; then
        log "Installing cert-manager ($CERT_MANAGER_VERSION)..."
        oc apply -f "https://github.com/cert-manager/cert-manager/releases/download/${CERT_MANAGER_VERSION}/cert-manager.yaml"
    else
        log "cert-manager namespace already present."
    fi

    if oc get namespace cert-manager &>/dev/null; then
        oc wait --for=condition=Available deployment/cert-manager -n cert-manager --timeout=5m || true
        oc wait --for=condition=Available deployment/cert-manager-webhook -n cert-manager --timeout=5m || true
        oc wait --for=condition=Available deployment/cert-manager-cainjector -n cert-manager --timeout=5m || true
    else
        warn "cert-manager namespace not found after install; continuing."
    fi
}

wait_for_endpoints() {
    local ns="$1"
    local svc="$2"
    local label="$3"
    for i in {1..60}; do
        local endpoints=""
        endpoints=$(oc get endpoints "$svc" -n "$ns" -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null || echo "")
        if [[ -n "$endpoints" ]]; then
            success "$label has ready endpoints"
            return 0
        fi
        if [[ $i -eq 60 ]]; then
            warn "Timeout waiting for $label endpoints"
        fi
        sleep 2
    done
    return 1
}

apply_llmisvc_core() {
    local llmisvc_url="https://raw.githubusercontent.com/kserve/kserve/master/hack/setup/quick-install/llmisvc-full-install-with-manifests.sh"
    local temp_script temp_core
    temp_script=$(mktemp)
    temp_core=$(mktemp)

    # ClusterRoleBinding roleRef is immutable; remove any stale binding so apply doesn't fail.
    if oc get clusterrolebinding llmisvc-manager-rolebinding &>/dev/null; then
        log "Removing stale llmisvc-manager-rolebinding to avoid immutable roleRef errors..."
        oc delete clusterrolebinding llmisvc-manager-rolebinding --ignore-not-found
    fi

    log "Downloading LLMInferenceService install manifests..."
    if ! curl -sL "$llmisvc_url" -o "$temp_script"; then
        warn "Failed to download LLMInferenceService install script."
        rm -f "$temp_script" "$temp_core"
        return 1
    fi

    local core_start core_end
    core_start=$(grep -n "KSERVE_CORE_MANIFEST_EOF" "$temp_script" | head -1 | cut -d: -f1)
    core_end=$(grep -n "KSERVE_CORE_MANIFEST_EOF" "$temp_script" | tail -1 | cut -d: -f1)
    if [[ -n "$core_start" ]] && [[ -n "$core_end" ]] && [[ "$core_start" -lt "$core_end" ]]; then
        sed -n "$((core_start + 1)),$((core_end - 1))p" "$temp_script" > "$temp_core"
        if [[ -s "$temp_core" ]]; then
            log "Applying LLMInferenceService core manifests (controller, webhook, etc.)..."
            oc apply --server-side --force-conflicts -f "$temp_core" || warn "Failed to apply LLMInferenceService core manifests"
        fi
    else
        warn "Could not find LLMInferenceService core manifest in install script."
    fi

    rm -f "$temp_script" "$temp_core"
    return 0
}

apply_llmisvc_crds() {
    local llmisvc_url="https://raw.githubusercontent.com/kserve/kserve/master/hack/setup/quick-install/llmisvc-full-install-with-manifests.sh"
    local temp_script temp_crd_config temp_crd_main
    temp_script=$(mktemp)
    temp_crd_config=$(mktemp)
    temp_crd_main=$(mktemp)

    log "Downloading LLMInferenceService install manifests..."
    if ! curl -sL "$llmisvc_url" -o "$temp_script"; then
        warn "Failed to download LLMInferenceService install script."
        rm -f "$temp_script" "$temp_crd_config" "$temp_crd_main"
        return 1
    fi

    local config_start main_start
    config_start=$(grep -n "name: llminferenceserviceconfigs.serving.kserve.io" "$temp_script" | head -1 | cut -d: -f1)
    main_start=$(grep -n "name: llminferenceservices.serving.kserve.io" "$temp_script" | head -1 | cut -d: -f1)

    if [[ -n "$config_start" ]] && [[ -n "$main_start" ]]; then
        local config_real_start main_real_start main_end
        config_real_start=$((config_start - 6))
        sed -n "${config_real_start},$((main_start - 8))p" "$temp_script" > "$temp_crd_config"

        main_real_start=$((main_start - 6))
        main_end=$(grep -n "KSERVE_CRD_MANIFEST_EOF" "$temp_script" | tail -1 | cut -d: -f1)
        if [[ -n "$main_end" ]]; then
            sed -n "${main_real_start},$((main_end - 1))p" "$temp_script" > "$temp_crd_main"
        fi

        if [[ -s "$temp_crd_config" ]]; then
            log "Applying LLMInferenceServiceConfig CRD..."
            oc apply --server-side --force-conflicts -f "$temp_crd_config" || warn "Failed to apply LLMInferenceServiceConfig CRD"
        fi
        if [[ -s "$temp_crd_main" ]]; then
            log "Applying LLMInferenceServices CRD..."
            oc apply --server-side --force-conflicts -f "$temp_crd_main" || warn "Failed to apply LLMInferenceServices CRD"
        fi
    else
        warn "Could not find LLMInferenceService CRD definitions in install script."
    fi

    rm -f "$temp_script" "$temp_crd_config" "$temp_crd_main"
    return 0
}

apply_llmisvc_full() {
    apply_llmisvc_crds
    apply_llmisvc_core
    apply_llmisvc_configs
}

apply_llmisvc_configs() {
    local local_dir="/home/ubuntu/tmp/kserve/config/llmisvcconfig"
    local base_url="https://raw.githubusercontent.com/kserve/kserve/${KSERVE_VERSION}/config/llmisvcconfig"
    local files=(
        "config-llm-decode-template.yaml"
        "config-llm-decode-worker-data-parallel.yaml"
        "config-llm-prefill-template.yaml"
        "config-llm-prefill-worker-data-parallel.yaml"
        "config-llm-router-route.yaml"
        "config-llm-scheduler.yaml"
        "config-llm-template.yaml"
        "config-llm-worker-data-parallel.yaml"
    )

    log "Ensuring LLMInferenceServiceConfig templates..."
    if [[ -d "$local_dir" ]]; then
        for f in "${files[@]}"; do
            if [[ -f "$local_dir/$f" ]]; then
                oc apply -n kserve -f "$local_dir/$f" 2>/dev/null || warn "Failed to apply $f from local kserve repo"
            fi
        done
    else
        for f in "${files[@]}"; do
            oc apply -n kserve -f "${base_url}/${f}" 2>/dev/null || warn "Failed to apply $f from ${base_url}"
        done
    fi
}

ensure_kserve_controllers() {
    if ! oc get deployment/kserve-controller-manager -n kserve &>/dev/null; then
        log "KServe controller not found; reapplying KServe manifests..."
        oc apply --server-side --force-conflicts -f "https://github.com/kserve/kserve/releases/download/${KSERVE_VERSION}/kserve.yaml"
    fi

    if oc get deployment/kserve-controller-manager -n kserve &>/dev/null; then
        if ! oc get secret kserve-webhook-server-cert -n kserve &>/dev/null; then
            log "KServe webhook cert secret missing; reapplying KServe manifests..."
            oc apply --server-side --force-conflicts -f "https://github.com/kserve/kserve/releases/download/${KSERVE_VERSION}/kserve.yaml"
        fi
        oc wait --for=condition=Available deployment/kserve-controller-manager -n kserve --timeout=5m || true
        wait_for_endpoints "kserve" "kserve-webhook-server-service" "KServe webhook service"
    fi
}

ensure_llmisvc_controllers() {
    local needs_reinstall=false
    if ! oc get deployment/llmisvc-controller-manager -n kserve &>/dev/null; then
        needs_reinstall=true
    fi
    if ! oc get service llmisvc-webhook-server-service -n kserve &>/dev/null; then
        needs_reinstall=true
    fi
    if ! oc get crd llminferenceserviceconfigs.serving.kserve.io &>/dev/null; then
        needs_reinstall=true
    fi
    if [[ "$needs_reinstall" == true ]]; then
        log "LLMInferenceService components incomplete; reinstalling CRDs and core manifests..."
        apply_llmisvc_full
    fi

    if oc get deployment/llmisvc-controller-manager -n kserve &>/dev/null; then
        if ! oc get secret llmisvc-webhook-server-cert -n kserve &>/dev/null; then
            log "LLMInferenceService webhook cert secret missing; reapplying core manifests..."
            apply_llmisvc_full
        fi
        oc adm policy add-scc-to-user anyuid -z llmisvc-controller-manager -n kserve 2>/dev/null || true
        oc rollout restart deployment/llmisvc-controller-manager -n kserve 2>/dev/null || true
        oc wait --for=condition=Available deployment/llmisvc-controller-manager -n kserve --timeout=3m || true
        wait_for_endpoints "kserve" "llmisvc-webhook-server-service" "LLMInferenceService webhook"
        if ! oc get llminferenceserviceconfig kserve-config-llm-template -n kserve &>/dev/null; then
            warn "LLMInferenceServiceConfig templates not found; applying templates..."
            apply_llmisvc_configs
        fi
    fi
}

tune_storage_initializer() {
    if ! oc get configmap inferenceservice-config -n kserve &>/dev/null; then
        warn "inferenceservice-config not found in kserve namespace; skipping storage initializer tuning"
        return 0
    fi

    python3 - <<'PY'
import json
import subprocess

cm = json.loads(subprocess.check_output([
    "oc","get","configmap","inferenceservice-config","-n","kserve","-o","json"
]).decode())

data = cm.get("data", {})
raw = data.get("storageInitializer")
if not raw:
    raise SystemExit(0)

cfg = json.loads(raw)
cfg["memoryRequest"] = "1Gi"
cfg["memoryLimit"] = "4Gi"

data["storageInitializer"] = json.dumps(cfg, indent=2)
cm["data"] = data

patch = json.dumps({"data": {"storageInitializer": data["storageInitializer"]}})
subprocess.check_call([
    "oc","patch","configmap","inferenceservice-config","-n","kserve","--type=merge","-p",patch
])
PY
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

# Check if both KServe CRDs are already installed
INFERENCESERVICE_INSTALLED=false
LLMISVC_INSTALLED=false

if oc get crd inferenceservices.serving.kserve.io &>/dev/null; then
    INFERENCESERVICE_INSTALLED=true
    log "InferenceService CRD already installed."
fi

if oc get crd llminferenceservices.serving.kserve.io &>/dev/null; then
    LLMISVC_INSTALLED=true
    log "LLMInferenceService CRD already installed."
fi

if [[ "$INFERENCESERVICE_INSTALLED" == true ]] && [[ "$LLMISVC_INSTALLED" == true ]]; then
    ensure_cert_manager
    ensure_kserve_controllers
    ensure_llmisvc_controllers
    apply_llmisvc_configs
    tune_storage_initializer
    success "All KServe CRDs already installed."
    exit 0
fi

# Install InferenceService CRD if not already installed
if [[ "$INFERENCESERVICE_INSTALLED" == false ]]; then
    ensure_cert_manager

    log "Installing KServe ($KSERVE_VERSION)..."
    # Use server-side apply to avoid annotation size limits on CRDs
    oc apply --server-side --force-conflicts -f "https://github.com/kserve/kserve/releases/download/${KSERVE_VERSION}/kserve.yaml"

    # Wait for KServe controller and webhook to be ready before applying cluster resources
    if oc get namespace kserve &>/dev/null; then
        log "Waiting for KServe controller manager to be ready..."
        oc wait --for=condition=Available deployment/kserve-controller-manager -n kserve --timeout=5m || true

        log "Waiting for KServe webhook service endpoints to be ready..."
        for i in {1..60}; do
            ENDPOINTS=$(oc get endpoints kserve-webhook-server-service -n kserve -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null || echo "")
            if [[ -n "$ENDPOINTS" ]]; then
                success "KServe webhook service has ready endpoints"
                break
            fi
            if [[ $i -eq 60 ]]; then
                warn "Timeout waiting for webhook endpoints, continuing anyway..."
            fi
            sleep 2
        done
    else
        warn "KServe namespace not found after install; verify installation."
    fi

    # Apply cluster resources after webhook is ready
    log "Applying KServe cluster resources..."
    oc apply --server-side --force-conflicts -f "https://github.com/kserve/kserve/releases/download/${KSERVE_VERSION}/kserve-cluster-resources.yaml"

    if oc get crd inferenceservices.serving.kserve.io &>/dev/null; then
        success "KServe CRDs installed."
        tune_storage_initializer
        if [[ "$LLMISVC_INSTALLED" == true ]]; then
            ensure_llmisvc_controllers
        fi
    else
        error "KServe CRDs still missing after install."
        exit 1
    fi
fi

# Install LLMInferenceService CRD if requested
if [[ "$INSTALL_LLMISVC" == true ]]; then
    if oc get crd llminferenceservices.serving.kserve.io &>/dev/null; then
        success "LLMInferenceService CRD already installed."
        ensure_llmisvc_controllers
    else
        log "Installing LLMInferenceService CRD..."

        # Download the kserve quick install script which contains embedded CRDs
        LLMISVC_INSTALL_URL="https://raw.githubusercontent.com/kserve/kserve/master/hack/setup/quick-install/llmisvc-full-install-with-manifests.sh"
        TEMP_SCRIPT=$(mktemp)
        TEMP_CRD_CONFIG=$(mktemp)
        TEMP_CRD_MAIN=$(mktemp)

        log "Downloading LLMInferenceService install manifests..."
        if curl -sL "$LLMISVC_INSTALL_URL" -o "$TEMP_SCRIPT"; then
            # Extract CRD definitions from the embedded heredocs in the script
            # LLMInferenceServiceConfig CRD starts at line containing 'apiVersion: apiextensions' for llminferenceserviceconfigs
            # LLMInferenceServices CRD starts at the second 'apiVersion: apiextensions' block

            # Find line numbers for the CRD blocks
            CONFIG_CRD_START=$(grep -n "name: llminferenceserviceconfigs.serving.kserve.io" "$TEMP_SCRIPT" | head -1 | cut -d: -f1)
            MAIN_CRD_START=$(grep -n "name: llminferenceservices.serving.kserve.io" "$TEMP_SCRIPT" | head -1 | cut -d: -f1)

            if [[ -n "$CONFIG_CRD_START" ]] && [[ -n "$MAIN_CRD_START" ]]; then
                # Extract from 6 lines before the name (to get apiVersion line) to the line before the next CRD
                CONFIG_CRD_REAL_START=$((CONFIG_CRD_START - 6))
                sed -n "${CONFIG_CRD_REAL_START},$((MAIN_CRD_START - 8))p" "$TEMP_SCRIPT" > "$TEMP_CRD_CONFIG"

                # For main CRD, extract from 6 lines before name to EOF marker
                MAIN_CRD_REAL_START=$((MAIN_CRD_START - 6))
                MAIN_CRD_END=$(grep -n "KSERVE_CRD_MANIFEST_EOF" "$TEMP_SCRIPT" | tail -1 | cut -d: -f1)
                if [[ -n "$MAIN_CRD_END" ]]; then
                    sed -n "${MAIN_CRD_REAL_START},$((MAIN_CRD_END - 1))p" "$TEMP_SCRIPT" > "$TEMP_CRD_MAIN"
                fi

                # Apply the CRDs using server-side apply to avoid annotation size limits
                if [[ -s "$TEMP_CRD_CONFIG" ]]; then
                    log "Applying LLMInferenceServiceConfig CRD..."
                    oc apply --server-side --force-conflicts -f "$TEMP_CRD_CONFIG" || warn "Failed to apply LLMInferenceServiceConfig CRD"
                fi

                if [[ -s "$TEMP_CRD_MAIN" ]]; then
                    log "Applying LLMInferenceServices CRD..."
                    oc apply --server-side --force-conflicts -f "$TEMP_CRD_MAIN" || warn "Failed to apply LLMInferenceServices CRD"
                fi

                # Extract and apply core manifests (controller, webhook service, etc.)
                TEMP_CORE=$(mktemp)
                CORE_START=$(grep -n "KSERVE_CORE_MANIFEST_EOF" "$TEMP_SCRIPT" | head -1 | cut -d: -f1)
                CORE_END=$(grep -n "KSERVE_CORE_MANIFEST_EOF" "$TEMP_SCRIPT" | tail -1 | cut -d: -f1)
                if [[ -n "$CORE_START" ]] && [[ -n "$CORE_END" ]] && [[ "$CORE_START" -lt "$CORE_END" ]]; then
                    sed -n "$((CORE_START + 1)),$((CORE_END - 1))p" "$TEMP_SCRIPT" > "$TEMP_CORE"
                    if [[ -s "$TEMP_CORE" ]]; then
                        log "Applying LLMInferenceService core manifests (controller, webhook, etc.)..."
                        oc apply --server-side --force-conflicts -f "$TEMP_CORE" || warn "Failed to apply LLMInferenceService core manifests"

                        # Grant privileged SCC to the controller service account (required for OpenShift)
                        log "Granting OpenShift SCC to LLMInferenceService controller..."
                        oc adm policy add-scc-to-user privileged -z llmisvc-controller-manager -n kserve 2>/dev/null || true

                        # Wait for controller to be ready
                        log "Waiting for LLMInferenceService controller to be ready..."
                        oc rollout restart deployment/llmisvc-controller-manager -n kserve 2>/dev/null || true
                        oc wait --for=condition=Available deployment/llmisvc-controller-manager -n kserve --timeout=3m || warn "Controller may not be ready yet"

                        # Wait for webhook endpoints to be ready
                        log "Waiting for LLMInferenceService webhook endpoints..."
                        for i in {1..30}; do
                            ENDPOINTS=$(oc get endpoints llmisvc-webhook-server-service -n kserve -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null || echo "")
                            if [[ -n "$ENDPOINTS" ]]; then
                                success "LLMInferenceService webhook has ready endpoints"
                                break
                            fi
                            sleep 2
                        done
                    fi
                fi
                rm -f "$TEMP_CORE"
            else
                warn "Could not find LLMInferenceService CRD definitions in install script."
            fi
        else
            warn "Failed to download LLMInferenceService install script."
        fi

        rm -f "$TEMP_SCRIPT" "$TEMP_CRD_CONFIG" "$TEMP_CRD_MAIN"

        # Wait for CRD to be established
        log "Waiting for LLMInferenceService CRD to be established..."
        for i in {1..30}; do
            if oc get crd llminferenceservices.serving.kserve.io -o jsonpath='{.status.conditions[?(@.type=="Established")].status}' 2>/dev/null | grep -q "True"; then
                success "LLMInferenceService CRD installed and established."
                break
            fi
            if [[ $i -eq 30 ]]; then
                warn "Timeout waiting for CRD to be established."
            fi
            sleep 2
        done
    fi
fi
