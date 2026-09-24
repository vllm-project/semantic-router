#!/bin/bash

# Helm Chart Validation Script
# This script validates the Helm chart for semantic-router

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CHART_PATH="deploy/helm/semantic-router"
TEMP_BASE="${TMPDIR:-/tmp}"
mkdir -p "$TEMP_BASE"
TEMP_DIR=$(mktemp -d "$TEMP_BASE/helm-test.XXXXXX")

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

cleanup() {
    log_info "Cleaning up..."
    rm -rf "$TEMP_DIR"
}

trap cleanup EXIT

# Create temp directory
mkdir -p "$TEMP_DIR"

echo "=================================================="
echo "Semantic Router Helm Chart Validation"
echo "=================================================="
echo ""

# Test 1: Helm lint
log_info "Running Helm lint..."
if helm lint "$CHART_PATH"; then
    log_success "Helm lint passed"
else
    log_error "Helm lint failed"
    exit 1
fi
echo ""

# Test 2: Helm template with default values
log_info "Testing Helm template with default values..."
if helm template test-release "$CHART_PATH" > "$TEMP_DIR/default-template.yaml"; then
    log_success "Helm template with default values succeeded"
    log_info "Output saved to $TEMP_DIR/default-template.yaml"
else
    log_error "Helm template with default values failed"
    exit 1
fi
echo ""

# A Dashboard config edit on Kubernetes is saved to a ConfigMap and activated
# after rollout. The backup history needed for rollback must outlive that pod.
log_info "Testing default Dashboard backup persistence..."
helm template dashboard-release "$CHART_PATH" --set dashboard.enabled=true \
    > "$TEMP_DIR/dashboard-default-template.yaml"
python3 - "$TEMP_DIR/dashboard-default-template.yaml" <<'PY'
import sys
import yaml

documents = [doc for doc in yaml.safe_load_all(open(sys.argv[1], encoding="utf-8")) if isinstance(doc, dict)]
dashboard = next(
    doc for doc in documents
    if doc.get("kind") == "Deployment"
    and doc.get("metadata", {}).get("labels", {}).get("app.kubernetes.io/component") == "dashboard"
)
claims = [
    doc for doc in documents
    if doc.get("kind") == "PersistentVolumeClaim"
    and doc.get("metadata", {}).get("labels", {}).get("app.kubernetes.io/component") == "dashboard"
]
assert len(claims) == 1, "Dashboard backup PVC must render when Dashboard is enabled"
pod = dashboard["spec"]["template"]["spec"]
mounts = pod["containers"][0]["volumeMounts"]
assert any(mount.get("name") == "dashboard-data" and mount.get("mountPath") == "/app/data" for mount in mounts)
assert any(
    volume.get("name") == "dashboard-data"
    and volume.get("persistentVolumeClaim", {}).get("claimName") == claims[0]["metadata"]["name"]
    for volume in pod["volumes"]
)
PY
log_success "Dashboard config backups survive a rollout by default"
echo ""

# Test 3: Canonical config override must be atomic and preserve explicit gates
log_info "Testing atomic canonical Router config rendering..."
cp deploy/helm/testdata/backend-target-values.yaml "$TEMP_DIR/canonical-config.yaml"

helm template canonical-release "$CHART_PATH" \
    -f "$TEMP_DIR/canonical-config.yaml" \
    > "$TEMP_DIR/canonical-template.yaml"

if grep -q "replace-with-your-model" "$TEMP_DIR/canonical-template.yaml"; then
    log_error "Chart defaults leaked into the atomic canonical Router config"
    exit 1
fi
if ! grep -A1 "skip_processing:" "$TEMP_DIR/canonical-template.yaml" | grep -q "enabled: true"; then
    log_error "Canonical skip_processing=true was not preserved"
    exit 1
fi

log_info "Testing backend target compatibility rendering..."
backend_fields=(
    "base_url: https://provider.example/v1"
    "provider_model_id: provider/model-id"
    "api_key_env: PROVIDER_API_KEY"
    "X-Tenant: production"
    "chat_path: /chat/completions"
    "weight: 75"
)
for field in "${backend_fields[@]}"; do
    if ! grep -q "$field" "$TEMP_DIR/canonical-template.yaml"; then
        log_error "Canonical backend target fields were not preserved: $field"
        exit 1
    fi
done

python3 tools/ci/check_backend_target_compatibility.py \
    --rendered-helm "$TEMP_DIR/canonical-template.yaml"

helm template canonical-false-release "$CHART_PATH" \
    -f "$TEMP_DIR/canonical-config.yaml" \
    --set configOverride.global.router.skip_processing.enabled=false \
    > "$TEMP_DIR/canonical-false-template.yaml"
if ! grep -A1 "skip_processing:" "$TEMP_DIR/canonical-false-template.yaml" | grep -q "enabled: false"; then
    log_error "Canonical skip_processing=false was not preserved"
    exit 1
fi
log_success "Canonical Router config rendering is atomic and preserves feature gates"
echo ""

if helm template canonical-multi-release "$CHART_PATH" \
    -f "$TEMP_DIR/canonical-config.yaml" \
    --set replicaCount=2 \
    > "$TEMP_DIR/canonical-multi-template.yaml" 2>&1; then
    log_error "Atomic canonical Router Learning bypassed the multi-replica guard"
    exit 1
fi
if ! grep -q "multi-replica router deployments cannot use Router Learning" \
    "$TEMP_DIR/canonical-multi-template.yaml"; then
    log_error "Multi-replica Router Learning failed without the safety-guard error"
    exit 1
fi
helm template canonical-multi-opt-out-release "$CHART_PATH" \
    -f "$TEMP_DIR/canonical-config.yaml" \
    --set replicaCount=2 \
    --set safetyGuards.rejectMultiReplicaLocalLearningState=false \
    > "$TEMP_DIR/canonical-multi-opt-out-template.yaml"
log_success "Every template consumer enforces atomic canonical config safety guards"
echo ""

if helm template canonical-empty-release "$CHART_PATH" \
    --set-json 'configOverride={}' \
    > "$TEMP_DIR/canonical-empty-template.yaml" 2>&1; then
    log_error "An empty canonical config silently fell back to chart defaults"
    exit 1
fi
if ! grep -q "configOverride must be a non-empty mapping" \
    "$TEMP_DIR/canonical-empty-template.yaml"; then
    log_error "Empty canonical config failed without the expected safety error"
    exit 1
fi
log_success "Empty canonical config fails closed instead of using chart samples"
echo ""

log_info "Testing model deployment and recipe binding preservation..."
helm template runtime-release "$CHART_PATH" \
    -f deploy/helm/testdata/model-runtime-values.yaml \
    > "$TEMP_DIR/model-runtime-template.yaml"
python3 deploy/helm/check-model-runtime.py "$TEMP_DIR/model-runtime-template.yaml"
log_success "Model deployments, input/admission budgets and isolated bindings are preserved"
echo ""

log_info "Testing custom workspace models mount rendering..."
cat > "$TEMP_DIR/workspace-models-values.yaml" <<'YAML'
extraVolumes:
  - name: workspace-models
    hostPath:
      path: /opt/semantic-router/workspace-models
      type: DirectoryOrCreate
extraVolumeMounts:
  - name: workspace-models
    mountPath: /app/models
YAML
helm template workspace-release "$CHART_PATH" \
    -f "$TEMP_DIR/workspace-models-values.yaml" \
    > "$TEMP_DIR/workspace-models-template.yaml"
python3 - "$TEMP_DIR/default-template.yaml" "$TEMP_DIR/workspace-models-template.yaml" <<'PY'
import sys
from pathlib import Path

import yaml


def router_pod(path):
    for document in yaml.safe_load_all(Path(path).read_text(encoding="utf-8")):
        if (
            document
            and document.get("kind") == "Deployment"
            and document["metadata"]["labels"].get("app.kubernetes.io/component") == "router"
        ):
            return document["spec"]["template"]["spec"]
    raise AssertionError(f"Router Deployment missing from {path}")


default_pod, workspace_pod = map(router_pod, sys.argv[1:])
for pod, expected_name in ((default_pod, "models-volume"), (workspace_pod, "workspace-models")):
    mounts = [mount for mount in pod["containers"][0]["volumeMounts"] if mount["mountPath"] == "/app/models"]
    assert len(mounts) == 1 and mounts[0]["name"] == expected_name, mounts
    volumes = [volume for volume in pod["volumes"] if volume["name"] == expected_name]
    assert len(volumes) == 1, volumes
    backup_env = [
        entry for entry in pod["containers"][0]["env"]
        if entry["name"] == "VLLM_SR_CONFIG_BACKUP_DIR"
    ]
    assert len(backup_env) == 1, backup_env
    assert backup_env[0]["value"] == "/app/models/.vllm-sr/config-backups", backup_env
default_models = next(volume for volume in default_pod["volumes"] if volume["name"] == "models-volume")
assert "persistentVolumeClaim" in default_models, default_models
assert all(volume["name"] != "models-volume" for volume in workspace_pod["volumes"])
PY
log_success "Custom /app/models mount replaces the default model volume"
echo ""



# Test 4: Validate YAML syntax
log_info "Validating YAML syntax..."
if command -v yamllint &> /dev/null; then
    if yamllint "$CHART_PATH/values.yaml" 2>&1 | grep -v "too many spaces inside braces"; then
        log_warning "YAML lint found some issues (Helm templates cause expected warnings)"
    else
        log_success "YAML validation passed"
    fi
else
    log_warning "yamllint not installed, skipping YAML validation"
fi
echo ""

# Test 5: Check required files exist
log_info "Checking required files..."
required_files=(
    "Chart.yaml"
    "values.yaml"
    "README.md"
    ".helmignore"
    "templates/_helpers.tpl"
    "templates/deployment.yaml"
    "templates/service.yaml"
    "templates/configmap.yaml"
    "templates/pvc.yaml"
    "templates/serviceaccount.yaml"
    "templates/ingress.yaml"
    "templates/hpa.yaml"
    "templates/NOTES.txt"
)

all_files_exist=true
for file in "${required_files[@]}"; do
    if [ -f "$CHART_PATH/$file" ]; then
        log_success "Found: $file"
    else
        log_error "Missing: $file"
        all_files_exist=false
    fi
done

if [ "$all_files_exist" = false ]; then
    log_error "Some required files are missing"
    exit 1
fi
echo ""

# Test 6: Validate generated resources
log_info "Validating generated Kubernetes resources..."
resource_types=(
    "ServiceAccount"
    "PersistentVolumeClaim"
    "ConfigMap"
    "Deployment"
    "Service"
)

for resource in "${resource_types[@]}"; do
    if grep -q "kind: $resource" "$TEMP_DIR/default-template.yaml"; then
        log_success "Found resource: $resource"
    else
        log_error "Missing resource: $resource"
        exit 1
    fi
done
log_info "Note: Namespace is managed by Helm's --create-namespace flag"
echo ""

# Test 7: Validate config file mount contract
log_info "Validating config file mount contract..."
if grep -qE 'mountPath: /app/config$' "$TEMP_DIR/default-template.yaml"; then
    log_error "Rendered templates still mount the full /app/config directory and would hide bundled KB assets"
    exit 1
fi

for expected in 'subPath: config.yaml' 'subPath: tools_db.json'; do
    if grep -q "$expected" "$TEMP_DIR/default-template.yaml"; then
        log_success "Found expected file mount: $expected"
    else
        log_error "Missing expected file mount: $expected"
        exit 1
    fi
done
echo ""

# Test 8: Validate the default router pod security contract
log_info "Validating hardened router pod security defaults..."
for expected in \
    'runAsNonRoot: true' \
    'runAsUser: 65532' \
    'runAsGroup: 65532' \
    'type: RuntimeDefault' \
    'allowPrivilegeEscalation: false' \
    'readOnlyRootFilesystem: true' \
    'mountPath: /tmp' \
    'mountPath: /app/models'; do
    if ! grep -q "$expected" "$TEMP_DIR/default-template.yaml"; then
        log_error "Missing hardened router pod setting: $expected"
        exit 1
    fi
done
if ! grep -A2 'capabilities:' "$TEMP_DIR/default-template.yaml" | grep -q 'ALL'; then
    log_error "Router container does not drop all Linux capabilities"
    exit 1
fi
log_success "Router pod defaults enforce the restricted security contract"
echo ""

# Test 9: Validate Chart.yaml
log_info "Validating Chart.yaml..."
if [ -f "$CHART_PATH/Chart.yaml" ]; then
    chart_name=$(grep "^name:" "$CHART_PATH/Chart.yaml" | awk '{print $2}')
    chart_version=$(grep "^version:" "$CHART_PATH/Chart.yaml" | awk '{print $2}')
    app_version=$(grep "^appVersion:" "$CHART_PATH/Chart.yaml" | awk '{print $2}')

    log_success "Chart name: $chart_name"
    log_success "Chart version: $chart_version"
    log_success "App version: $app_version"
else
    log_error "Chart.yaml not found"
    exit 1
fi
echo ""

# Test 10: Check for common Helm best practices
log_info "Checking Helm best practices..."
best_practices_passed=true

# Check if labels helper exists
if grep -q "semantic-router.labels" "$CHART_PATH/templates/_helpers.tpl"; then
    log_success "Labels helper template exists"
else
    log_error "Labels helper template missing"
    best_practices_passed=false
fi

# Check if selector labels helper exists
if grep -q "semantic-router.selectorLabels" "$CHART_PATH/templates/_helpers.tpl"; then
    log_success "Selector labels helper template exists"
else
    log_error "Selector labels helper template missing"
    best_practices_passed=false
fi

# Check if NOTES.txt exists
if [ -f "$CHART_PATH/templates/NOTES.txt" ]; then
    log_success "NOTES.txt exists"
else
    log_error "NOTES.txt missing"
    best_practices_passed=false
fi

if [ "$best_practices_passed" = false ]; then
    log_error "Some best practices checks failed"
    exit 1
fi
echo ""

# Test 11: Dry-run install (requires cluster)
if kubectl cluster-info &> /dev/null; then
    log_info "Testing dry-run install..."
    if helm install test-release "$CHART_PATH" --dry-run --debug > "$TEMP_DIR/dry-run.log" 2>&1; then
        log_success "Dry-run install succeeded"
    else
        log_error "Dry-run install failed"
        cat "$TEMP_DIR/dry-run.log"
        exit 1
    fi
else
    log_warning "No Kubernetes cluster available, skipping dry-run install test"
fi
echo ""

# Test 11: Package the chart
log_info "Testing chart packaging..."
if helm package "$CHART_PATH" --destination "$TEMP_DIR" > /dev/null 2>&1; then
    log_success "Chart packaged successfully"
    ls -lh "$TEMP_DIR"/*.tgz
else
    log_error "Chart packaging failed"
    exit 1
fi
echo ""

# Summary
echo "=================================================="
echo "Validation Summary"
echo "=================================================="
log_success "All validation tests passed!"
echo ""
echo "Generated files are available in: $TEMP_DIR"
echo ""
echo "Next steps:"
echo "1. Review the generated templates in $TEMP_DIR"
echo "2. Test installation: make helm-install"
echo ""
