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
if ! grep -A4 'startupProbe:' "$TEMP_DIR/default-template.yaml" | grep -q 'path: /ready'; then
    log_error "Default startup probe does not use Router readiness"
    exit 1
fi
if ! grep -A5 'livenessProbe:' "$TEMP_DIR/default-template.yaml" | grep -q 'path: /health'; then
    log_error "Default liveness probe does not use Router health"
    exit 1
fi
if ! grep -A5 'readinessProbe:' "$TEMP_DIR/default-template.yaml" | grep -q 'scheme: HTTP'; then
    log_error "Standalone readiness probe does not use HTTP"
    exit 1
fi
echo ""

# Test 3: Canonical config override must be atomic and preserve explicit gates
log_info "Testing atomic canonical Router config rendering..."
cat > "$TEMP_DIR/canonical-config.yaml" <<'EOF'
configOverride:
  version: v0.4
  listeners:
    - name: http
      address: 0.0.0.0
      port: 8899
  models:
    - name: local/custom
      connections:
        - provider: vllm
          endpoint: http://custom-model.default.svc.cluster.local:8000/v1
          model: local/custom
  recipes:
    - name: custom
      document:
        decisions:
          - name: custom-route
            priority: 100
            rules: {}
  entrypoints:
    - name: local/custom
      recipe: custom
      assignments:
        custom-route:
          models:
            - model: local/custom
  global:
    router:
      learning:
        enabled: true
        adaptation:
          enabled: true
EOF

helm template canonical-release "$CHART_PATH" \
    -f "$TEMP_DIR/canonical-config.yaml" \
    > "$TEMP_DIR/canonical-template.yaml"

if grep -q "replace-with-your-model" "$TEMP_DIR/canonical-template.yaml"; then
    log_error "Chart defaults leaked into the atomic canonical Router config"
    exit 1
fi
log_success "Canonical Router config rendering is atomic"
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
if ! grep -Eq "configOverride must be a non-empty mapping|configOverride: Must have at least 1 properties" \
    "$TEMP_DIR/canonical-empty-template.yaml"; then
    log_error "Empty canonical config failed without the expected safety error"
    exit 1
fi
log_success "Empty canonical config fails closed instead of using chart samples"
echo ""

# Test 4: Managed mode must isolate Management and render availability controls.
log_info "Testing managed listener isolation and availability resources..."
cat > "$TEMP_DIR/managed-config.yaml" <<'EOF'
configOverride:
  version: v0.4
  global:
    control_plane:
      mode: managed
    stores:
      access:
        postgres:
          dsn_env: TEST_DATABASE_URL
    services:
      management_api:
        port: 9443
      backend_dispatch:
        port: 8181
podDisruptionBudget:
  enabled: true
topologySpread:
  enabled: true
networkPolicy:
  enabled: true
  ingress:
    extProcPeers:
      - podSelector: {}
    backendDispatchPeers:
      - podSelector: {}
    metricsPeers:
      - podSelector: {}
dashboard:
  enabled: true
  routerTLS:
    ca:
      existingSecret: router-management-client-ca
      existingSecretKey: trust.pem
EOF

helm template managed-release "$CHART_PATH" \
    -f "$TEMP_DIR/managed-config.yaml" \
    > "$TEMP_DIR/managed-template.yaml"

awk '
    $0 == "kind: ConfigMap" { in_router_configmap = 1; next }
    in_router_configmap && $0 ~ /^  config\.yaml: \|/ { in_config = 1; next }
    in_config && $0 ~ /^  [^ ]/ { exit }
    in_config { sub(/^    /, ""); print }
' "$TEMP_DIR/managed-template.yaml" > "$TEMP_DIR/managed-rendered-config.yaml"

if ! grep -Eq '^[[:space:]]+mode: managed$' "$TEMP_DIR/managed-rendered-config.yaml"; then
    log_error "Managed mode was not preserved in the rendered Router config"
    exit 1
fi
if grep -Eq '^(models|recipes|entrypoints):' "$TEMP_DIR/managed-rendered-config.yaml"; then
    log_error "Chart sample routes leaked into the managed Router bootstrap"
    exit 1
fi
if ! grep -q 'name: managed-release-semantic-router-management' "$TEMP_DIR/managed-template.yaml"; then
    log_error "Managed mode did not render a dedicated Management Service"
    exit 1
fi
if ! awk '
    /^---$/ { in_service = 0; is_management = 0; publishes_not_ready = 0 }
    /^kind: Service$/ { in_service = 1 }
    in_service && $1 == "name:" && $2 == "managed-release-semantic-router-management" { is_management = 1 }
    in_service && $1 == "publishNotReadyAddresses:" && $2 == "true" { publishes_not_ready = 1 }
    is_management && publishes_not_ready { found = 1 }
    END { exit(found ? 0 : 1) }
' "$TEMP_DIR/managed-template.yaml"; then
    log_error "Private Management Service does not publish bootstrap endpoints before inference readiness"
    exit 1
fi
publish_not_ready_count=$(grep -c '^  publishNotReadyAddresses: true$' "$TEMP_DIR/managed-template.yaml" || true)
if [ "$publish_not_ready_count" -ne 1 ]; then
    log_error "Only the private Management Service may publish not-ready Router addresses"
    exit 1
fi
if ! grep -A5 'readinessProbe:' "$TEMP_DIR/managed-template.yaml" | grep -q 'scheme: HTTPS'; then
    log_error "Managed readiness probe does not use HTTPS"
    exit 1
fi
if ! grep -A1 'name: TARGET_ROUTER_API_URL' "$TEMP_DIR/managed-template.yaml" | \
    grep -q 'https://managed-release-semantic-router-management:8080'; then
    log_error "Dashboard does not use the private managed HTTPS Service"
    exit 1
fi
if ! grep -A1 'name: SSL_CERT_FILE' "$TEMP_DIR/managed-template.yaml" | \
    grep -q '/var/run/secrets/vllm-sr/router-management-ca/ca.crt'; then
    log_error "Dashboard does not use the configured Router Management trust bundle"
    exit 1
fi
if ! grep -A5 'name: router-management-ca' "$TEMP_DIR/managed-template.yaml" | \
    grep -q 'secretName: router-management-client-ca'; then
    log_error "Dashboard does not mount the configured Router Management CA Secret"
    exit 1
fi
if ! grep -A6 'name: router-management-ca' "$TEMP_DIR/managed-template.yaml" | \
    grep -q 'key: trust.pem'; then
    log_error "Dashboard Router Management CA Secret key was not projected"
    exit 1
fi
if ! grep -q 'kind: PodDisruptionBudget' "$TEMP_DIR/managed-template.yaml"; then
    log_error "Enabled Router PodDisruptionBudget was not rendered"
    exit 1
fi
if ! grep -q 'kind: NetworkPolicy' "$TEMP_DIR/managed-template.yaml"; then
    log_error "Enabled Router NetworkPolicy was not rendered"
    exit 1
fi
if ! grep -q 'topologySpreadConstraints:' "$TEMP_DIR/managed-template.yaml"; then
    log_error "Enabled Router topology spread was not rendered"
    exit 1
fi
if helm template unsafe-managed-ingress "$CHART_PATH" \
    -f "$TEMP_DIR/managed-config.yaml" \
    --set ingress.enabled=true \
    > "$TEMP_DIR/unsafe-managed-ingress.yaml" 2>&1; then
    log_error "Managed mode allowed the private Management listener through Ingress"
    exit 1
fi
if ! grep -q 'ingress cannot expose the managed Management listener' \
    "$TEMP_DIR/unsafe-managed-ingress.yaml"; then
    log_error "Managed Ingress rejection did not explain the isolation contract"
    exit 1
fi
log_success "Managed listener isolation and availability resources are explicit"
echo ""



# Test 5: Validate YAML syntax
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

# Test 6: Check required files exist
log_info "Checking required files..."
required_files=(
    "Chart.yaml"
    "values.yaml"
    "README.md"
    ".helmignore"
    "templates/_helpers.tpl"
    "templates/access-migrate-job.yaml"
    "templates/deployment.yaml"
    "templates/service.yaml"
    "templates/configmap.yaml"
    "templates/pvc.yaml"
    "templates/serviceaccount.yaml"
    "templates/ingress.yaml"
    "templates/hpa.yaml"
    "templates/networkpolicy.yaml"
    "templates/poddisruptionbudget.yaml"
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

# Test 7: Validate generated resources
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

# Test 8: Validate config file mount contract
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

# Test 8: Validate Chart.yaml
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

# Test 9: Check for common Helm best practices
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

# Test 10: Dry-run install (requires cluster)
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
