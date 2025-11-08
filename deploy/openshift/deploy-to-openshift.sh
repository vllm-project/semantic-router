#!/bin/bash
# Deploy split-pod architecture to OpenShift
# Each vLLM model in separate pod, semantic-router + envoy together

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

NAMESPACE="vllm-semantic-router-system"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

# Check if logged in to OpenShift
if ! oc whoami &>/dev/null; then
    error "Not logged in to OpenShift. Please login first:"
    echo "  oc login <your-openshift-server-url>"
    exit 1
fi

success "Logged in as $(oc whoami)"

# Create namespace
log "Creating namespace: $NAMESPACE"
oc create namespace "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -
success "Namespace ready"

# Build llm-katan image if needed
log "Checking for llm-katan image..."
if ! oc get imagestream llm-katan -n "$NAMESPACE" &> /dev/null; then
    log "Building llm-katan image..."

    if [[ -f "$SCRIPT_DIR/Dockerfile.llm-katan" ]]; then
        oc new-build --dockerfile - --name llm-katan -n "$NAMESPACE" < "$SCRIPT_DIR/Dockerfile.llm-katan"
    else
        error "Dockerfile.llm-katan not found at: $SCRIPT_DIR/Dockerfile.llm-katan"
        exit 1
    fi

    log "Waiting for build to start..."
    sleep 5

    log "Starting build..."
    oc start-build llm-katan -n "$NAMESPACE" --follow || true

    log "Waiting for build to complete..."
    if ! oc wait --for=condition=Complete build/llm-katan-1 -n "$NAMESPACE" --timeout=600s 2>/dev/null; then
        warn "Build may still be in progress. Checking status..."
        oc get builds -n "$NAMESPACE"
        oc logs build/llm-katan-1 -n "$NAMESPACE" --tail=50 || true
    fi

    success "llm-katan image built"
else
    log "llm-katan image already exists"
fi

# Create PVCs
log "Creating PersistentVolumeClaims..."
cat <<EOF | oc apply -n "$NAMESPACE" -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: semantic-router-models
  labels:
    app: semantic-router
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: gp3-csi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: semantic-router-cache
  labels:
    app: semantic-router
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
  storageClassName: gp3-csi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: vllm-model-a-cache
  labels:
    app: vllm-model
    model: model-a
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: gp3-csi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: vllm-model-b-cache
  labels:
    app: vllm-model
    model: model-b
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: gp3-csi
EOF
success "PVCs created"

# Deploy vLLM models FIRST to get their ClusterIPs
log "Deploying vLLM model services and deployments..."
oc apply -f "$SCRIPT_DIR/deployment.yaml" -n "$NAMESPACE"

# Wait for services to be created and get ClusterIPs
log "Waiting for vLLM services to get ClusterIPs..."
for i in {1..30}; do
    MODEL_A_IP=$(oc get svc vllm-model-a -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "")
    MODEL_B_IP=$(oc get svc vllm-model-b -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "")

    if [[ -n "$MODEL_A_IP" ]] && [[ -n "$MODEL_B_IP" ]]; then
        success "Got ClusterIPs: vllm-model-a=$MODEL_A_IP, vllm-model-b=$MODEL_B_IP"
        break
    fi

    if [[ $i -eq 30 ]]; then
        error "Timeout waiting for service ClusterIPs"
        exit 1
    fi

    sleep 2
done

# Generate dynamic config with actual ClusterIPs
log "Generating dynamic configuration with ClusterIPs..."
TEMP_CONFIG="/tmp/config-openshift-dynamic.yaml"
sed -e "s|address: \".*\" # model-a-ip|address: \"$MODEL_A_IP\"|g" \
    -e "s|address: \".*\" # model-b-ip|address: \"$MODEL_B_IP\"|g" \
    "$SCRIPT_DIR/config-openshift.yaml" > "$TEMP_CONFIG"

# Verify the IPs were substituted
if ! grep -q "$MODEL_A_IP" "$TEMP_CONFIG" || ! grep -q "$MODEL_B_IP" "$TEMP_CONFIG"; then
    warn "IP substitution may have failed. Using template config instead..."
    # Fallback: create config with sed on known patterns
    sed -e "s/172\.30\.64\.134/$MODEL_A_IP/g" \
        -e "s/172\.30\.116\.177/$MODEL_B_IP/g" \
        "$SCRIPT_DIR/config-openshift.yaml" > "$TEMP_CONFIG"
fi

success "Dynamic config generated with IPs: Model-A=$MODEL_A_IP, Model-B=$MODEL_B_IP"

# Create ConfigMaps with dynamic config
log "Creating ConfigMaps with dynamic IPs..."
oc create configmap semantic-router-config \
    --from-file=config.yaml="$TEMP_CONFIG" \
    -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -

oc create configmap envoy-config \
    --from-file=envoy.yaml="$SCRIPT_DIR/envoy-openshift.yaml" \
    -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -

success "ConfigMaps created with dynamic IPs"

# Clean up temp file
rm -f "$TEMP_CONFIG"

success "Deployment manifests applied"

# Create routes
log "Creating OpenShift routes..."
cat <<EOF | oc apply -n "$NAMESPACE" -f -
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: semantic-router-api
  labels:
    app: semantic-router
spec:
  to:
    kind: Service
    name: semantic-router
  port:
    targetPort: api
---
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: semantic-router-grpc
  labels:
    app: semantic-router
spec:
  to:
    kind: Service
    name: semantic-router
  port:
    targetPort: grpc
---
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: semantic-router-metrics
  labels:
    app: semantic-router
spec:
  to:
    kind: Service
    name: semantic-router-metrics
  port:
    targetPort: metrics
---
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: envoy-http
  labels:
    app: semantic-router
spec:
  to:
    kind: Service
    name: semantic-router
  port:
    targetPort: envoy-http
---
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: envoy-admin
  labels:
    app: semantic-router
spec:
  to:
    kind: Service
    name: semantic-router
  port:
    targetPort: envoy-admin
EOF
success "Routes created"

log "Waiting for deployments to be ready..."
log "This may take several minutes as models are downloaded..."

success "Deployment initiated! Check status with the following commands:"
echo ""
echo "  # View all pods"
echo "  oc get pods -n $NAMESPACE"
echo ""
echo "  # View deployment status"
echo "  oc get deployments -n $NAMESPACE"
echo ""
echo "  # View services"
echo "  oc get services -n $NAMESPACE"
echo ""
echo "  # View routes"
echo "  oc get routes -n $NAMESPACE"
echo ""
echo "  # Check logs for vLLM Model-A"
echo "  oc logs -f deployment/vllm-model-a -n $NAMESPACE"
echo ""
echo "  # Check logs for vLLM Model-B"
echo "  oc logs -f deployment/vllm-model-b -n $NAMESPACE"
echo ""
echo "  # Check logs for Semantic Router"
echo "  oc logs -f deployment/semantic-router -c semantic-router -n $NAMESPACE"
echo ""
echo "  # Check logs for Envoy"
echo "  oc logs -f deployment/semantic-router -c envoy-proxy -n $NAMESPACE"
