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
DEPLOY_OBSERVABILITY=true
USE_SIMULATOR=false
USE_KSERVE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-observability)
            DEPLOY_OBSERVABILITY=false
            shift
            ;;
        --simulator)
            USE_SIMULATOR=true
            shift
            ;;
        --kserve)
            USE_KSERVE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --simulator           Use mock-vllm simulator instead of llm-katan (no GPU required)"
            echo "  --kserve              Deploy semantic-router with a KServe backend (use --simulator for KServe sim)"
            echo "  --no-observability    Skip deploying dashboard, OpenWebUI, Grafana, and Prometheus"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "By default, deploys the full stack with llm-katan (requires GPU)."
            echo "Use --simulator for CPU-only clusters without GPUs."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage information"
            exit 1
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

# Check if logged in to OpenShift
if ! oc whoami &>/dev/null; then
    error "Not logged in to OpenShift. Please login first:"
    echo "  oc login <your-openshift-server-url>"
    exit 1
fi

success "Logged in as $(oc whoami)"

# GPU mode (--kserve without --simulator) requires GPU resources
# Simulator mode (--kserve --simulator) works without GPU

# Create namespace
log "Creating namespace: $NAMESPACE"
oc create namespace "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -
for i in {1..30}; do
    if oc get namespace "$NAMESPACE" -o jsonpath='{.metadata.deletionTimestamp}' 2>/dev/null | grep -q .; then
        warn "Namespace $NAMESPACE is terminating, waiting..."
        sleep 2
        continue
    fi
    break
done
success "Namespace ready"

# KServe mode: deploy LLMInferenceService and semantic-router
if [[ "$USE_KSERVE" == "true" ]]; then
    KSERVE_SCRIPT="$SCRIPT_DIR/../kserve/deploy.sh"
    KSERVE_INSTALL_SCRIPT="$SCRIPT_DIR/../kserve/install-kserve.sh"

    if [[ ! -x "$KSERVE_SCRIPT" ]]; then
        error "KServe deploy script not found: $KSERVE_SCRIPT"
        exit 1
    fi

    if [[ ! -x "$KSERVE_INSTALL_SCRIPT" ]]; then
        error "KServe install script not found: $KSERVE_INSTALL_SCRIPT"
        exit 1
    fi

    log "Installing KServe and LLMInferenceService CRDs..."
    "$KSERVE_INSTALL_SCRIPT"

    # Ensure LLMInferenceServiceConfig templates exist before creating LLMInferenceServices
    if oc get crd llminferenceserviceconfigs.serving.kserve.io &>/dev/null; then
        if ! oc get llminferenceserviceconfig kserve-config-llm-template -n kserve &>/dev/null; then
            log "Applying LLMInferenceServiceConfig templates..."
            if [[ -d /home/ubuntu/tmp/kserve/config/llmisvcconfig ]]; then
                for f in /home/ubuntu/tmp/kserve/config/llmisvcconfig/config-llm-*.yaml; do
                    oc apply -n kserve -f "$f" 2>/dev/null || true
                done
            else
                warn "Local kserve repo not found at /home/ubuntu/tmp/kserve; LLMInferenceServiceConfig templates may be missing."
            fi
        fi
    fi

    # Wait briefly for llmisvc controller and webhook to settle before creating LLMInferenceServices
    oc wait --for=condition=Available deployment/llmisvc-controller-manager -n kserve --timeout=3m 2>/dev/null || true
    for i in {1..30}; do
        ENDPOINTS=$(oc get endpoints llmisvc-webhook-server-service -n kserve -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null || echo "")
        if [[ -n "$ENDPOINTS" ]]; then
            break
        fi
        sleep 2
    done

    if [[ "$USE_SIMULATOR" == "true" ]]; then
        # Simulator mode: deploy LLMInferenceServices for Model-A and Model-B
        SIM_ISVC_A="$SCRIPT_DIR/../kserve/inference-examples/inferenceservice-llm-d-sim-model-a.yaml"
        SIM_ISVC_B="$SCRIPT_DIR/../kserve/inference-examples/inferenceservice-llm-d-sim-model-b.yaml"

        if [[ ! -f "$SIM_ISVC_A" || ! -f "$SIM_ISVC_B" ]]; then
            error "Simulator LLMInferenceService manifests not found."
            exit 1
        fi

        log "Ensuring simulator service account and SCC..."
        oc create serviceaccount llmisvc-workload -n "$NAMESPACE" 2>/dev/null || true
        oc adm policy add-scc-to-user anyuid -z llmisvc-workload -n "$NAMESPACE" 2>/dev/null || true
        oc adm policy add-scc-to-user privileged -z llmisvc-workload -n "$NAMESPACE" 2>/dev/null || true
        oc adm policy add-scc-to-user privileged system:serviceaccount:kserve:llmisvc-controller-manager -n "$NAMESPACE" 2>/dev/null || true

        log "Deploying simulator LLMInferenceServices..."
        oc apply -n "$NAMESPACE" -f "$SIM_ISVC_A"
        oc apply -n "$NAMESPACE" -f "$SIM_ISVC_B"

        log "Waiting for simulator LLMInferenceServices to be ready..."
        oc wait --for=condition=Ready llminferenceservice/model-a -n "$NAMESPACE" --timeout=10m
        oc wait --for=condition=Ready llminferenceservice/model-b -n "$NAMESPACE" --timeout=10m

        KSERVE_ARGS=(--simulator -n "$NAMESPACE")
    else
        # GPU mode: deploy real Qwen model on GPU
        QWEN_LLMISVC="$SCRIPT_DIR/../kserve/inference-examples/inferenceservice-qwen-0.6b-gpu.yaml"

        if [[ ! -f "$QWEN_LLMISVC" ]]; then
            error "Qwen LLMInferenceService manifest not found: $QWEN_LLMISVC"
            exit 1
        fi

        # Check for GPU resources
        GPU_NODES=$(oc get nodes -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}' 2>/dev/null | tr ' ' '\n' | grep -v '^$' | grep -v '<none>' -c)
        if [[ "$GPU_NODES" -eq 0 ]]; then
            warn "No GPU resources detected. Install GPU operator first:"
            warn "  ./deploy/kserve/install-gpu-operator.sh"
            error "GPU required for non-simulator KServe deployment"
            exit 1
        fi
        log "Found $GPU_NODES node(s) with GPU resources"

        log "Granting privileged SCC to default service account for model download..."
        oc adm policy add-scc-to-user privileged -z default -n "$NAMESPACE" 2>/dev/null || true

        log "Deploying Qwen 0.6B LLMInferenceService on GPU..."
        for i in {1..3}; do
            if oc apply -n "$NAMESPACE" -f "$QWEN_LLMISVC"; then
                break
            fi
            warn "Failed to apply Qwen LLMInferenceService (attempt $i); retrying..."
            sleep 5
        done

        log "Waiting for Qwen LLMInferenceService to be ready (this may take several minutes for model download)..."
        oc wait --for=condition=Ready llminferenceservice/qwen-0-6b -n "$NAMESPACE" --timeout=15m

        KSERVE_ARGS=(-n "$NAMESPACE" --inferenceservice qwen-0-6b --model "Qwen/Qwen3-0.6B")
    fi

    log "KServe mode: Deploying semantic-router with KServe backend..."
    "$KSERVE_SCRIPT" "${KSERVE_ARGS[@]}"
    success "KServe deployment complete"
    exit 0
fi

# Build model backend image based on mode
if [[ "$USE_SIMULATOR" == "true" ]]; then
    # Use mock-vllm simulator (no GPU required)
    log "Simulator mode: Building mock-vllm image..."
    BACKEND_IMAGE_NAME="mock-vllm"
    MOCK_VLLM_DIR="$SCRIPT_DIR/../../tools/mock-vllm"

    if ! oc get imagestream mock-vllm -n "$NAMESPACE" &> /dev/null; then
        if [[ -f "$MOCK_VLLM_DIR/Dockerfile" ]]; then
            oc new-build --name mock-vllm --binary --strategy=docker -n "$NAMESPACE"
            log "Uploading mock-vllm source and building..."
            oc start-build mock-vllm --from-dir="$MOCK_VLLM_DIR" --follow -n "$NAMESPACE" || true

            log "Waiting for build to complete..."
            # Get the latest build to handle reruns (mock-vllm-1, mock-vllm-2, etc.)
            LATEST_BUILD=$(oc get builds -l buildconfig=mock-vllm -n "$NAMESPACE" -o name --sort-by=.metadata.creationTimestamp | tail -1)
            if [[ -n "$LATEST_BUILD" ]]; then
                if ! oc wait --for=condition=Complete "$LATEST_BUILD" -n "$NAMESPACE" --timeout=60s 2>/dev/null; then
                    warn "Build may still be in progress. Checking status..."
                    oc get builds -n "$NAMESPACE"
                fi
            else
                warn "No mock-vllm build found to wait for"
            fi
            success "mock-vllm image built"
        else
            error "mock-vllm Dockerfile not found at: $MOCK_VLLM_DIR/Dockerfile"
            exit 1
        fi
    else
        log "mock-vllm image already exists"
    fi
else
    # Use llm-katan (requires GPU)
    log "Standard mode: Checking for llm-katan image..."
    BACKEND_IMAGE_NAME="llm-katan"

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

if [[ "$USE_SIMULATOR" == "true" ]]; then
    log "Simulator mode: Using deployment-simulator.yaml (mock-vllm, no GPU required)..."
    oc apply -f "$SCRIPT_DIR/deployment-simulator.yaml" -n "$NAMESPACE"
else
    oc apply -f "$SCRIPT_DIR/deployment.yaml" -n "$NAMESPACE"
fi

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
sed -e "s/DYNAMIC_MODEL_A_IP/$MODEL_A_IP/g" \
    -e "s/DYNAMIC_MODEL_B_IP/$MODEL_B_IP/g" \
    "$SCRIPT_DIR/config-openshift.yaml" > "$TEMP_CONFIG"

# Verify the IPs were substituted
if ! grep -q "$MODEL_A_IP" "$TEMP_CONFIG" || ! grep -q "$MODEL_B_IP" "$TEMP_CONFIG"; then
    error "IP substitution failed! Check config-openshift.yaml for DYNAMIC_MODEL_A_IP and DYNAMIC_MODEL_B_IP placeholders"
    exit 1
fi

success "Dynamic config generated with IPs: Model-A=$MODEL_A_IP, Model-B=$MODEL_B_IP"

# Create ConfigMaps with dynamic config
log "Creating ConfigMaps with dynamic IPs..."
oc create configmap semantic-router-config \
    --from-file=config.yaml="$TEMP_CONFIG" \
    --from-file=tools_db.json="$SCRIPT_DIR/tools_db.json" \
    -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -

oc create configmap envoy-config \
    --from-file=envoy.yaml="$SCRIPT_DIR/envoy-openshift.yaml" \
    -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -

success "ConfigMaps created with dynamic IPs"

# Clean up temp file
rm -f "$TEMP_CONFIG"

success "Deployment manifests applied"

# Deploy Dashboard
log "Deploying Dashboard..."
oc apply -f "$SCRIPT_DIR/dashboard/dashboard-deployment.yaml" -n "$NAMESPACE"
success "Dashboard deployment applied"

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

# Deploy Jaeger for tracing
log "Deploying Jaeger for distributed tracing..."
if [[ -d "$SCRIPT_DIR/observability/jaeger" ]]; then
    oc apply -f "$SCRIPT_DIR/observability/jaeger/deployment.yaml" -n "$NAMESPACE"
    oc apply -f "$SCRIPT_DIR/observability/jaeger/service.yaml" -n "$NAMESPACE"
    oc apply -f "$SCRIPT_DIR/observability/jaeger/route.yaml" -n "$NAMESPACE"
    success "Jaeger deployed"
else
    warn "Jaeger deployment files not found at $SCRIPT_DIR/observability/jaeger, skipping..."
fi

log "Waiting for deployments to be ready..."
log "This may take several minutes as models are downloaded..."

# Deploy observability components if enabled
if [[ "$DEPLOY_OBSERVABILITY" == "true" ]]; then
    log "Deploying observability components..."

    # Deploy Grafana with dynamic route URL
    log "Deploying Grafana..."

    # First apply Grafana resources to create the route
    oc apply -f "$SCRIPT_DIR/observability/grafana/pvc.yaml" -n "$NAMESPACE" 2>/dev/null || true
    oc apply -f "$SCRIPT_DIR/observability/grafana/service.yaml" -n "$NAMESPACE" 2>/dev/null || true
    oc apply -f "$SCRIPT_DIR/observability/grafana/route.yaml" -n "$NAMESPACE" 2>/dev/null || true
    oc apply -f "$SCRIPT_DIR/observability/grafana/configmap-dashboard.yaml" -n "$NAMESPACE" 2>/dev/null || true
    oc apply -f "$SCRIPT_DIR/observability/grafana/configmap-datasource.yaml" -n "$NAMESPACE" 2>/dev/null || true
    oc apply -f "$SCRIPT_DIR/observability/grafana/configmap-provisioning.yaml" -n "$NAMESPACE" 2>/dev/null || true

    # Wait for route to be created and get its URL
    log "Waiting for Grafana route to be created..."
    for i in {1..30}; do
        GRAFANA_ROUTE_URL=$(oc get route grafana -n "$NAMESPACE" -o jsonpath='{.spec.host}' 2>/dev/null || echo "")

        if [[ -n "$GRAFANA_ROUTE_URL" ]]; then
            GRAFANA_ROUTE_URL="https://$GRAFANA_ROUTE_URL"
            success "Grafana route URL: $GRAFANA_ROUTE_URL"
            break
        fi

        if [[ $i -eq 30 ]]; then
            error "Timeout waiting for Grafana route"
            exit 1
        fi

        sleep 2
    done

    # Generate Grafana deployment with dynamic route URL
    log "Generating Grafana deployment with dynamic route URL..."
    TEMP_GRAFANA_DEPLOYMENT="/tmp/grafana-deployment-dynamic.yaml"
    sed "s|DYNAMIC_GRAFANA_ROUTE_URL|$GRAFANA_ROUTE_URL|g" \
        "$SCRIPT_DIR/observability/grafana/deployment.yaml" > "$TEMP_GRAFANA_DEPLOYMENT"

    # Verify the URL was substituted
    if ! grep -q "$GRAFANA_ROUTE_URL" "$TEMP_GRAFANA_DEPLOYMENT"; then
        error "Grafana route URL substitution failed!"
        exit 1
    fi

    # Apply Grafana deployment with dynamic URL
    oc apply -f "$TEMP_GRAFANA_DEPLOYMENT" -n "$NAMESPACE"
    rm -f "$TEMP_GRAFANA_DEPLOYMENT"

    success "Grafana deployed with dynamic route URL: $GRAFANA_ROUTE_URL"

    # Deploy Prometheus
    log "Deploying Prometheus..."
    oc apply -f "$SCRIPT_DIR/observability/prometheus/" -n "$NAMESPACE"
    success "Prometheus deployed"

    # Build and deploy Dashboard using binary build
    log "Building and deploying Dashboard..."

    # Check if dashboard imagestream exists
    if ! oc get imagestream dashboard-custom -n "$NAMESPACE" &>/dev/null; then
        log "Creating dashboard imagestream..."
        oc create imagestream dashboard-custom -n "$NAMESPACE"
    fi

    # Check if dashboard buildconfig exists
    if ! oc get buildconfig dashboard-custom -n "$NAMESPACE" &>/dev/null; then
        log "Creating dashboard build configuration..."
        cd "$SCRIPT_DIR/../.."
        oc new-build --name=dashboard-custom --binary --strategy=docker --to=dashboard-custom:latest -n "$NAMESPACE"
    fi
    # Build context is repo root, so we must point the BuildConfig at dashboard/Dockerfile.
    # Ensure dockerfilePath is set whether it already exists (replace) or not (add).
    oc patch buildconfig/dashboard-custom -n "$NAMESPACE" --type='json' \
        -p='[{"op":"test","path":"/spec/strategy/dockerStrategy/dockerfilePath","value":"dashboard/Dockerfile"},{"op":"replace","path":"/spec/strategy/dockerStrategy/dockerfilePath","value":"dashboard/Dockerfile"}]' \
        2>/dev/null || oc patch buildconfig/dashboard-custom -n "$NAMESPACE" --type='json' \
        -p='[{"op":"add","path":"/spec/strategy/dockerStrategy/dockerfilePath","value":"dashboard/Dockerfile"}]'

    # Start the build from repo root so the dashboard build can access src/semantic-router
    log "Building dashboard image from source..."
    cd "$SCRIPT_DIR/../.."
    oc start-build dashboard-custom --from-dir=. --follow -n "$NAMESPACE" || warn "Dashboard build may still be in progress"

    # Deploy dashboard
    log "Deploying dashboard..."
    oc apply -f "$SCRIPT_DIR/dashboard/dashboard-deployment.yaml" -n "$NAMESPACE"
    success "Dashboard deployed"

    # Wait for observability deployments
    log "Waiting for observability components to be ready..."
    oc rollout status deployment/dashboard -n "$NAMESPACE" --timeout=5m || warn "Dashboard may still be starting"

    success "Observability components deployed!"
    echo ""
    echo "  Dashboard: https://$(oc get route dashboard -n $NAMESPACE -o jsonpath='{.spec.host}' 2>/dev/null || echo 'route-not-ready')"
    echo "  Grafana:   https://$(oc get route grafana -n $NAMESPACE -o jsonpath='{.spec.host}' 2>/dev/null || echo 'route-not-ready')"
    echo "  Prometheus: https://$(oc get route prometheus -n $NAMESPACE -o jsonpath='{.spec.host}' 2>/dev/null || echo 'route-not-ready')"
    echo ""
else
    log "Skipping observability components (--no-observability flag provided)"
fi

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
echo ""
echo "  # Check logs for Dashboard"
echo "  oc logs -f deployment/dashboard -n $NAMESPACE"
echo ""
