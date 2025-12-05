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
TEMP_DIR="/tmp/helm-test-$$"

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



# Test 3: Validate YAML syntax
log_info "Validating YAML syntax..."
yamllint_available=false
if command -v yamllint &> /dev/null; then
    yamllint_available=true
    if yamllint "$CHART_PATH/values.yaml" 2>&1 | grep -v "too many spaces inside braces"; then
        log_warning "YAML lint found some issues (Helm templates cause expected warnings)"
    else
        log_success "YAML validation passed"
    fi
else
    log_warning "yamllint not installed, skipping YAML validation"
fi
echo ""

# Test 4: Check required files exist
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

# Test 5: Validate generated resources
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

# Test 6: Validate Chart.yaml
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

# Test 7: Check for common Helm best practices
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

# Test 8: Dry-run install (requires cluster)
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

# Test 9: Package the chart
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
