#!/bin/bash
# Verification script for LLM Katan Kubernetes deployment
# Usage: ./verify-deployment.sh [namespace] [service-name]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
NAMESPACE="${1:-llm-katan-system}"
SERVICE="${2:-llm-katan}"
PORT=8000

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

# Track overall status
FAILED=0

echo "=================================================="
echo "LLM Katan Deployment Verification"
echo "=================================================="
log_info "Namespace: $NAMESPACE"
log_info "Service: $SERVICE"
echo ""

# Check 1: Namespace exists
log_info "Checking namespace..."
if kubectl get namespace "$NAMESPACE" &> /dev/null; then
    log_success "Namespace $NAMESPACE exists"
else
    log_error "Namespace $NAMESPACE not found"
    FAILED=1
fi
echo ""

# Check 2: Deployment exists
log_info "Checking deployments..."
if kubectl get deployment -n "$NAMESPACE" &> /dev/null; then
    DEPLOYMENT_COUNT=$(kubectl get deployment -n "$NAMESPACE" -o name | wc -l)
    log_success "Found $DEPLOYMENT_COUNT deployment(s)"
    kubectl get deployment -n "$NAMESPACE"
else
    log_error "No deployments found"
    FAILED=1
fi
echo ""

# Check 3: Pods are running
log_info "Checking pods..."
POD_STATUS=$(kubectl get pods -n "$NAMESPACE" -o jsonpath='{.items[*].status.phase}' 2>/dev/null || echo "")
if [ -z "$POD_STATUS" ]; then
    log_error "No pods found"
    FAILED=1
else
    RUNNING_PODS=$(echo "$POD_STATUS" | tr ' ' '\n' | grep -c "Running" || echo "0")
    TOTAL_PODS=$(echo "$POD_STATUS" | wc -w)
    
    if [ "$RUNNING_PODS" -eq "$TOTAL_PODS" ] && [ "$RUNNING_PODS" -gt 0 ]; then
        log_success "All $RUNNING_PODS/$TOTAL_PODS pods are running"
        kubectl get pods -n "$NAMESPACE"
    else
        log_error "Only $RUNNING_PODS/$TOTAL_PODS pods are running"
        kubectl get pods -n "$NAMESPACE"
        FAILED=1
    fi
fi
echo ""

# Check 4: Services exist
log_info "Checking services..."
if kubectl get svc -n "$NAMESPACE" -o name | grep -q "$SERVICE"; then
    log_success "Service $SERVICE exists"
    kubectl get svc -n "$NAMESPACE" | grep "$SERVICE" || true
else
    log_error "Service $SERVICE not found"
    FAILED=1
fi
echo ""

# Check 5: PVC bound
log_info "Checking PersistentVolumeClaims..."
PVC_COUNT=$(kubectl get pvc -n "$NAMESPACE" -o name 2>/dev/null | wc -l)
if [ "$PVC_COUNT" -gt 0 ]; then
    BOUND_PVCS=$(kubectl get pvc -n "$NAMESPACE" -o jsonpath='{.items[*].status.phase}' 2>/dev/null | tr ' ' '\n' | grep -c "Bound" || echo "0")
    if [ "$BOUND_PVCS" -eq "$PVC_COUNT" ]; then
        log_success "All $PVC_COUNT PVC(s) are bound"
        kubectl get pvc -n "$NAMESPACE"
    else
        log_error "Only $BOUND_PVCS/$PVC_COUNT PVC(s) are bound"
        kubectl get pvc -n "$NAMESPACE"
        FAILED=1
    fi
else
    log_warning "No PVCs found (optional)"
fi
echo ""

# Check 6: ConfigMaps exist
log_info "Checking ConfigMaps..."
CM_COUNT=$(kubectl get configmap -n "$NAMESPACE" -o name 2>/dev/null | wc -l)
if [ "$CM_COUNT" -gt 0 ]; then
    log_success "Found $CM_COUNT ConfigMap(s)"
    kubectl get configmap -n "$NAMESPACE" -o name
else
    log_warning "No ConfigMaps found (may use default config)"
fi
echo ""

# Check 7: Pod readiness
log_info "Checking pod readiness..."
READY_PODS=$(kubectl get pods -n "$NAMESPACE" -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' 2>/dev/null | tr ' ' '\n' | grep -c "True" || echo "0")
TOTAL_PODS=$(kubectl get pods -n "$NAMESPACE" -o name 2>/dev/null | wc -l)
if [ "$READY_PODS" -eq "$TOTAL_PODS" ] && [ "$READY_PODS" -gt 0 ]; then
    log_success "All $READY_PODS/$TOTAL_PODS pods are ready"
else
    log_error "Only $READY_PODS/$TOTAL_PODS pods are ready"
    FAILED=1
fi
echo ""

# Check 8: Recent pod restarts
log_info "Checking for pod restarts..."
MAX_RESTARTS=$(kubectl get pods -n "$NAMESPACE" -o jsonpath='{.items[*].status.containerStatuses[*].restartCount}' 2>/dev/null | tr ' ' '\n' | sort -rn | head -1 || echo "0")
if [ "$MAX_RESTARTS" -eq 0 ]; then
    log_success "No pod restarts detected"
elif [ "$MAX_RESTARTS" -lt 3 ]; then
    log_warning "Some pods have restarted (max: $MAX_RESTARTS times)"
else
    log_error "High restart count detected (max: $MAX_RESTARTS times)"
    FAILED=1
fi
echo ""

# Check 9: Endpoint connectivity - test all services
log_info "Testing endpoint connectivity for all services..."
SERVICES=$(kubectl get svc -n "$NAMESPACE" -o name 2>/dev/null | grep "$SERVICE" | sed 's|service/||' || echo "")

if [ -z "$SERVICES" ]; then
    log_warning "No services found matching '$SERVICE' - skipping connectivity tests"
else
    for SVC in $SERVICES; do
        log_info "Testing service: $SVC"
        
        # Start port-forward in background
        kubectl port-forward -n "$NAMESPACE" "svc/$SVC" "$PORT:$PORT" &> /dev/null &
        PF_PID=$!
        sleep 2
        
        # Test health endpoint
        if curl -f -s -m 3 "http://localhost:$PORT/health" &> /dev/null; then
            HEALTH_RESPONSE=$(curl -s -m 3 "http://localhost:$PORT/health" 2>/dev/null || echo "{}")
            log_success "$SVC - Health OK: $HEALTH_RESPONSE"
        else
            log_warning "$SVC - Health endpoint not responding"
        fi
        
        # Test models endpoint
        if curl -f -s -m 3 "http://localhost:$PORT/v1/models" &> /dev/null; then
            MODELS=$(curl -s -m 3 "http://localhost:$PORT/v1/models" 2>/dev/null | grep -o '"id":"[^"]*"' || echo "")
            log_success "$SVC - Models OK: $MODELS"
        else
            log_warning "$SVC - Models endpoint not responding"
        fi
        
        # Cleanup port-forward
        kill $PF_PID 2>/dev/null || true
        wait $PF_PID 2>/dev/null || true
        sleep 1
    done
fi
echo ""

# Check 10: Resource usage (if metrics-server available)
log_info "Checking resource usage..."
if kubectl top pod -n "$NAMESPACE" &> /dev/null; then
    log_success "Resource metrics available"
    kubectl top pod -n "$NAMESPACE"
else
    log_warning "metrics-server not available (optional)"
fi
echo ""

# Check 11: Recent logs for errors
log_info "Checking recent logs for errors..."
# Use app.kubernetes.io/name label or check all pods
ERROR_COUNT=$(kubectl logs -n "$NAMESPACE" --all-containers --tail=100 2>/dev/null | grep -ic "error\|exception\|failed" 2>/dev/null || echo "0")
ERROR_COUNT=$(echo "$ERROR_COUNT" | tr -d '\n' | tr -d ' ' | head -c 10)  # Clean the output and limit length

# Ensure it's a valid number
if ! [[ "$ERROR_COUNT" =~ ^[0-9]+$ ]]; then
    ERROR_COUNT=0
fi

if [ "$ERROR_COUNT" -eq 0 ]; then
    log_success "No errors in recent logs"
else
    log_warning "Found $ERROR_COUNT error messages in recent logs"
    log_info "Recent errors:"
    kubectl logs -n "$NAMESPACE" --all-containers --tail=100 2>/dev/null | grep -i "error\|exception\|failed" | tail -5 || true
fi
echo ""

# Final summary
echo "=================================================="
echo "Verification Summary"
echo "=================================================="

if [ $FAILED -eq 0 ]; then
    log_success "All critical checks passed!"
    echo ""
    log_info "Deployment is healthy and ready to use."
    echo ""
    log_info "Access the service:"
    echo "  kubectl port-forward -n $NAMESPACE svc/$SERVICE $PORT:$PORT"
    echo "  curl http://localhost:$PORT/health"
    echo ""
    exit 0
else
    log_error "Some checks failed!"
    echo ""
    log_info "Troubleshooting steps:"
    echo "  1. Check pod logs: kubectl logs -n $NAMESPACE --all-containers --tail=100"
    echo "  2. Describe pods: kubectl describe pod -n $NAMESPACE"
    echo "  3. Check events: kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp'"
    echo "  4. Check specific pod: kubectl logs -n $NAMESPACE <pod-name> -c llm-katan"
    echo ""
    exit 1
fi



