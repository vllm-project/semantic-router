#!/bin/bash

# validate-deployment.sh
# Validates that the OpenShift deployment is working correctly

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

NAMESPACE="${1:-vllm-semantic-router-system}"
FAILURES=0

log() {
    local level=$1
    shift
    local message="$@"
    case $level in
        "INFO")  echo -e "${BLUE}[INFO]${NC}  $message" ;;
        "PASS")  echo -e "${GREEN}[PASS]${NC}  $message" ;;
        "FAIL")  echo -e "${RED}[FAIL]${NC}  $message"; ((FAILURES++)) ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC}  $message" ;;
    esac
}

# Test 1: Check namespace exists
log "INFO" "Test 1: Checking namespace $NAMESPACE exists..."
if oc get namespace "$NAMESPACE" &> /dev/null; then
    log "PASS" "Namespace $NAMESPACE exists"
else
    log "FAIL" "Namespace $NAMESPACE does not exist"
fi

# Test 2: Check deployment is ready
log "INFO" "Test 2: Checking deployment is ready..."
if oc get deployment semantic-router -n "$NAMESPACE" &> /dev/null; then
    READY=$(oc get deployment semantic-router -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')
    DESIRED=$(oc get deployment semantic-router -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
    if [[ "$READY" == "$DESIRED" && "$READY" != "0" ]]; then
        log "PASS" "Deployment is ready ($READY/$DESIRED replicas)"
    else
        log "FAIL" "Deployment not ready ($READY/$DESIRED replicas)"
    fi
else
    log "FAIL" "Deployment semantic-router does not exist"
fi

# Test 3: Check all 4 containers are running
log "INFO" "Test 3: Checking all 4 containers are running..."
POD_NAME=$(oc get pods -n "$NAMESPACE" -l app=semantic-router -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
if [[ -n "$POD_NAME" ]]; then
    CONTAINER_COUNT=$(oc get pod "$POD_NAME" -n "$NAMESPACE" -o json | jq '.status.containerStatuses | length')
    READY_COUNT=$(oc get pod "$POD_NAME" -n "$NAMESPACE" -o json | jq '[.status.containerStatuses[] | select(.ready==true)] | length')
    if [[ "$CONTAINER_COUNT" == "4" && "$READY_COUNT" == "4" ]]; then
        log "PASS" "All 4 containers are running (semantic-router, model-a, model-b, envoy-proxy)"
    else
        log "FAIL" "Not all containers ready ($READY_COUNT/4 containers)"
    fi
else
    log "FAIL" "Could not find pod"
fi

# Test 4: Check GPU detection in model-a
log "INFO" "Test 4: Checking GPU detection in model-a container..."
if [[ -n "$POD_NAME" ]]; then
    GPU_CHECK=$(oc exec -n "$NAMESPACE" "$POD_NAME" -c model-a -- python3 -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')" 2>/dev/null || echo "ERROR")
    if [[ "$GPU_CHECK" == "CUDA" ]]; then
        GPU_NAME=$(oc exec -n "$NAMESPACE" "$POD_NAME" -c model-a -- python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
        log "PASS" "GPU detected in model-a: $GPU_NAME"
    elif [[ "$GPU_CHECK" == "CPU" ]]; then
        log "WARN" "GPU not detected, running on CPU (acceptable with --device auto)"
    else
        log "FAIL" "Could not check GPU: $GPU_CHECK"
    fi
fi

# Test 5: Check GPU detection in model-b
log "INFO" "Test 5: Checking GPU detection in model-b container..."
if [[ -n "$POD_NAME" ]]; then
    GPU_CHECK=$(oc exec -n "$NAMESPACE" "$POD_NAME" -c model-b -- python3 -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')" 2>/dev/null || echo "ERROR")
    if [[ "$GPU_CHECK" == "CUDA" ]]; then
        GPU_NAME=$(oc exec -n "$NAMESPACE" "$POD_NAME" -c model-b -- python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
        log "PASS" "GPU detected in model-b: $GPU_NAME"
    elif [[ "$GPU_CHECK" == "CPU" ]]; then
        log "WARN" "GPU not detected, running on CPU (acceptable with --device auto)"
    else
        log "FAIL" "Could not check GPU: $GPU_CHECK"
    fi
fi

# Test 6: Check model-a loaded successfully
log "INFO" "Test 6: Checking model-a loaded successfully..."
if [[ -n "$POD_NAME" ]]; then
    MODEL_STATUS=$(oc logs -n "$NAMESPACE" "$POD_NAME" -c model-a --tail=200 | grep -i "model loaded" | tail -1)
    if [[ -n "$MODEL_STATUS" ]]; then
        DEVICE=$(echo "$MODEL_STATUS" | grep -oE "(cuda|cpu)" || echo "unknown")
        log "PASS" "Model-A loaded successfully on $DEVICE"
    else
        log "FAIL" "Could not verify model-a loaded"
    fi
fi

# Test 7: Check model-b loaded successfully
log "INFO" "Test 7: Checking model-b loaded successfully..."
if [[ -n "$POD_NAME" ]]; then
    MODEL_STATUS=$(oc logs -n "$NAMESPACE" "$POD_NAME" -c model-b --tail=200 | grep -i "model loaded" | tail -1)
    if [[ -n "$MODEL_STATUS" ]]; then
        DEVICE=$(echo "$MODEL_STATUS" | grep -oE "(cuda|cpu)" || echo "unknown")
        log "PASS" "Model-B loaded successfully on $DEVICE"
    else
        log "FAIL" "Could not verify model-b loaded"
    fi
fi

# Test 8: Check semantic-router is running
log "INFO" "Test 8: Checking semantic-router container is running..."
if [[ -n "$POD_NAME" ]]; then
    SR_READY=$(oc get pod "$POD_NAME" -n "$NAMESPACE" -o json | jq -r '.status.containerStatuses[] | select(.name=="semantic-router") | .ready')
    if [[ "$SR_READY" == "true" ]]; then
        log "PASS" "Semantic-router container is ready and running"
    else
        log "FAIL" "Semantic-router container is not ready"
    fi
fi

# Test 9: Check PVCs are bound
log "INFO" "Test 9: Checking PVCs are bound..."
MODELS_PVC=$(oc get pvc semantic-router-models -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || echo "Missing")
CACHE_PVC=$(oc get pvc semantic-router-cache -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || echo "Missing")
if [[ "$MODELS_PVC" == "Bound" ]]; then
    log "PASS" "Models PVC is bound"
else
    log "FAIL" "Models PVC is $MODELS_PVC"
fi
if [[ "$CACHE_PVC" == "Bound" ]]; then
    log "PASS" "Cache PVC is bound"
else
    log "FAIL" "Cache PVC is $CACHE_PVC"
fi

# Test 10: Check services exist
log "INFO" "Test 10: Checking services exist..."
if oc get service semantic-router -n "$NAMESPACE" &> /dev/null; then
    log "PASS" "Service semantic-router exists"
else
    log "FAIL" "Service semantic-router missing"
fi

# Test 11: Check routes exist
log "INFO" "Test 11: Checking routes exist..."
API_ROUTE=$(oc get route semantic-router-api -n "$NAMESPACE" -o jsonpath='{.spec.host}' 2>/dev/null || echo "")
if [[ -n "$API_ROUTE" ]]; then
    log "PASS" "Route semantic-router-api exists: $API_ROUTE"
else
    log "FAIL" "Route semantic-router-api missing"
fi

# Test 12: Check GPU node scheduling
log "INFO" "Test 12: Checking pod scheduled on GPU node..."
if [[ -n "$POD_NAME" ]]; then
    NODE_NAME=$(oc get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.nodeName}')
    NODE_HAS_GPU=$(oc get node "$NODE_NAME" -o jsonpath='{.metadata.labels.nvidia\.com/gpu\.present}' 2>/dev/null || echo "false")
    if [[ "$NODE_HAS_GPU" == "true" ]]; then
        log "PASS" "Pod scheduled on GPU node: $NODE_NAME"
    else
        log "WARN" "Pod scheduled on non-GPU node: $NODE_NAME (acceptable if no GPU nodes available)"
    fi
fi

# Summary
echo ""
echo "================================"
if [[ $FAILURES -eq 0 ]]; then
    log "PASS" "All validation tests passed!"
    exit 0
else
    log "FAIL" "Validation completed with $FAILURES failure(s)"
    exit 1
fi
