#!/bin/bash
# Deploy semantic-router to a kind cluster

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

NAMESPACE="vllm-semantic-router-system"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_OBSERVABILITY=true
USE_SIMULATOR=true  # Kind mode always uses simulator (no GPU)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-observability)
            DEPLOY_OBSERVABILITY=false
            shift
            ;;
        --namespace|-n)
            NAMESPACE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Deploy semantic-router to a kind cluster with KServe simulator"
            echo ""
            echo "Options:"
            echo "  --namespace, -n NAME  Namespace to deploy to (default: vllm-semantic-router-system)"
            echo "  --no-observability    Skip deploying Grafana and Prometheus"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "This script deploys the semantic-router stack to a kind cluster using"
            echo "the llm-d-inference-sim simulator (no GPU required)."
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

# Check if kubectl is available
if ! command -v kubectl &>/dev/null; then
    error "kubectl not found. Please install kubectl first."
    exit 1
fi

# Check if connected to a cluster
if ! kubectl cluster-info &>/dev/null; then
    error "Cannot connect to Kubernetes cluster. Is kind running?"
    echo "  To create a kind cluster: kind create cluster --name semantic-router"
    exit 1
fi

CLUSTER_NAME=$(kubectl config current-context)
success "Connected to cluster: $CLUSTER_NAME"

# Create namespace
log "Creating namespace: $NAMESPACE"
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
success "Namespace ready"

# Check if KServe is installed
if kubectl get crd inferenceservices.serving.kserve.io &>/dev/null; then
    log "KServe CRD detected - using KServe deployment mode"
    USE_KSERVE=true
else
    log "KServe CRD not found - using standalone deployment mode"
    USE_KSERVE=false
fi

# Deploy simulator InferenceServices if KServe is available
if [[ "$USE_KSERVE" == "true" ]]; then
    log "Deploying KServe InferenceServices for simulator..."

    KSERVE_ISVC_MANIFEST_A="$SCRIPT_DIR/../../kserve/inference-examples/inferenceservice-llm-d-sim-model-a.yaml"
    KSERVE_ISVC_MANIFEST_B="$SCRIPT_DIR/../../kserve/inference-examples/inferenceservice-llm-d-sim-model-b.yaml"

    if [[ ! -f "$KSERVE_ISVC_MANIFEST_A" || ! -f "$KSERVE_ISVC_MANIFEST_B" ]]; then
        error "KServe simulator InferenceService manifests not found"
        exit 1
    fi

    kubectl apply -n "$NAMESPACE" -f "$KSERVE_ISVC_MANIFEST_A"
    kubectl apply -n "$NAMESPACE" -f "$KSERVE_ISVC_MANIFEST_B"

    log "Waiting for KServe InferenceServices to be ready..."
    kubectl wait --for=condition=Ready "inferenceservice/model-a" -n "$NAMESPACE" --timeout=5m || warn "model-a may still be starting"
    kubectl wait --for=condition=Ready "inferenceservice/model-b" -n "$NAMESPACE" --timeout=5m || warn "model-b may still be starting"

    # Get predictor service IPs
    MODEL_A_IP=$(kubectl get svc -n "$NAMESPACE" -l serving.kserve.io/inferenceservice=model-a -o jsonpath='{.items[0].spec.clusterIP}' 2>/dev/null || echo "")
    MODEL_B_IP=$(kubectl get svc -n "$NAMESPACE" -l serving.kserve.io/inferenceservice=model-b -o jsonpath='{.items[0].spec.clusterIP}' 2>/dev/null || echo "")

    if [[ -z "$MODEL_A_IP" || -z "$MODEL_B_IP" ]]; then
        warn "Could not get KServe predictor IPs, using service names instead"
        MODEL_A_IP="model-a-predictor-default.$NAMESPACE.svc.cluster.local"
        MODEL_B_IP="model-b-predictor-default.$NAMESPACE.svc.cluster.local"
    fi

    success "KServe InferenceServices deployed"
else
    # Deploy standalone simulator pods (no KServe)
    log "Deploying standalone simulator pods..."

    kubectl apply -n "$NAMESPACE" -f "$SCRIPT_DIR/deployment-kind-simulator.yaml"

    # Wait for services to be created and get ClusterIPs
    log "Waiting for simulator services to get ClusterIPs..."
    for i in {1..30}; do
        MODEL_A_IP=$(kubectl get svc vllm-model-a -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "")
        MODEL_B_IP=$(kubectl get svc vllm-model-b -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "")

        if [[ -n "$MODEL_A_IP" ]] && [[ -n "$MODEL_B_IP" ]]; then
            success "Got ClusterIPs: model-a=$MODEL_A_IP, model-b=$MODEL_B_IP"
            break
        fi

        if [[ $i -eq 30 ]]; then
            error "Timeout waiting for service ClusterIPs"
            exit 1
        fi

        sleep 2
    done
fi

# Create PVCs for semantic-router (using kind's standard storage class)
log "Creating PersistentVolumeClaims..."
cat <<EOF | kubectl apply -n "$NAMESPACE" -f -
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
      storage: 5Gi
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
      storage: 5Gi
EOF
success "PVCs created"

# Generate config with model IPs
log "Generating configuration..."
TEMP_CONFIG="/tmp/config-kind-dynamic.yaml"

# Use KServe config or standalone config based on mode
if [[ "$USE_KSERVE" == "true" ]]; then
    # For KServe, use port 8080 (KServe predictor port)
    MODEL_A_PORT=8080
    MODEL_B_PORT=8080
else
    # For standalone, use ports from the service definitions
    MODEL_A_PORT=8000
    MODEL_B_PORT=8001
fi

cat > "$TEMP_CONFIG" <<EOF
# Semantic Router configuration for kind cluster
# Auto-routing model trigger (client should send model: "auto")
auto_model_name: "auto"
default_model: "Model-A"
strategy: "priority"

# vLLM Endpoints Configuration
vllm_endpoints:
  - name: "model-a-endpoint"
    address: "${MODEL_A_IP}"
    port: ${MODEL_A_PORT}
    weight: 1
  - name: "model-b-endpoint"
    address: "${MODEL_B_IP}"
    port: ${MODEL_B_PORT}
    weight: 1

model_config:
  "Model-A":
    preferred_endpoints: ["model-a-endpoint"]
  "Model-B":
    preferred_endpoints: ["model-b-endpoint"]

# Classifier configuration for domain-based routing
classifier:
  category_model:
    model_id: "models/category_classifier_modernbert-base_model"
    use_modernbert: true
    threshold: 0.35
    use_cpu: true
    category_mapping_path: "models/category_classifier_modernbert-base_model/category_mapping.json"

# Categories for domain classification
categories:
  - name: math
  - name: physics
  - name: chemistry
  - name: biology
  - name: computer science
  - name: engineering
  - name: history
  - name: philosophy
  - name: psychology
  - name: economics
  - name: business
  - name: law
  - name: health
  - name: general
  - name: other

# Decisions - route STEM topics to Model-A, humanities to Model-B
decisions:
  - name: stem_topics
    description: "Route STEM queries to Model-A"
    priority: 10
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "math"
        - type: "domain"
          name: "physics"
        - type: "domain"
          name: "chemistry"
        - type: "domain"
          name: "biology"
        - type: "domain"
          name: "computer science"
        - type: "domain"
          name: "engineering"
    modelRefs:
      - model: Model-A
        use_reasoning: false
  - name: humanities_topics
    description: "Route humanities queries to Model-B"
    priority: 10
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "history"
        - type: "domain"
          name: "philosophy"
        - type: "domain"
          name: "psychology"
        - type: "domain"
          name: "business"
        - type: "domain"
          name: "law"
        - type: "domain"
          name: "economics"
        - type: "domain"
          name: "health"
    modelRefs:
      - model: Model-B
        use_reasoning: false
  - name: default
    description: "Default routing for general queries"
    priority: 1
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "general"
        - type: "domain"
          name: "other"
    modelRefs:
      - model: Model-A
        use_reasoning: false

# Observability
observability:
  metrics:
    enabled: true
    port: 9190
EOF

success "Configuration generated"

# Create ConfigMaps
log "Creating ConfigMaps..."
kubectl create configmap semantic-router-config \
    --from-file=config.yaml="$TEMP_CONFIG" \
    -n "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

# Use the openshift envoy config as base (it's generic)
kubectl create configmap envoy-config \
    --from-file=envoy.yaml="$SCRIPT_DIR/../envoy-openshift.yaml" \
    -n "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

success "ConfigMaps created"
rm -f "$TEMP_CONFIG"

# Deploy semantic-router (use kind-specific manifest)
log "Deploying semantic-router..."
kubectl apply -n "$NAMESPACE" -f "$SCRIPT_DIR/deployment-kind-semantic-router.yaml"
success "Semantic-router deployment applied"

# Create services
log "Creating services..."
cat <<EOF | kubectl apply -n "$NAMESPACE" -f -
apiVersion: v1
kind: Service
metadata:
  name: semantic-router
  labels:
    app: semantic-router
spec:
  type: ClusterIP
  ports:
  - name: grpc
    port: 50051
    targetPort: 50051
  - name: api
    port: 8080
    targetPort: 8080
  - name: envoy-http
    port: 8801
    targetPort: 8801
  - name: envoy-admin
    port: 19000
    targetPort: 19000
  selector:
    app: semantic-router
---
apiVersion: v1
kind: Service
metadata:
  name: semantic-router-metrics
  labels:
    app: semantic-router
spec:
  ports:
  - name: metrics
    port: 9190
    targetPort: 9190
  selector:
    app: semantic-router
EOF
success "Services created"

# Wait for deployments
log "Waiting for deployments to be ready..."
log "This may take several minutes as models are downloaded..."

if [[ "$USE_KSERVE" != "true" ]]; then
    kubectl rollout status deployment/vllm-model-a -n "$NAMESPACE" --timeout=5m || warn "model-a may still be starting"
    kubectl rollout status deployment/vllm-model-b -n "$NAMESPACE" --timeout=5m || warn "model-b may still be starting"
fi
kubectl rollout status deployment/semantic-router -n "$NAMESPACE" --timeout=10m || warn "semantic-router may still be starting"

# Deploy observability if enabled
if [[ "$DEPLOY_OBSERVABILITY" == "true" ]]; then
    log "Deploying observability stack..."

    # Check if kubernetes observability manifests exist
    K8S_OBS_DIR="$SCRIPT_DIR/../../kubernetes/observability"
    if [[ -d "$K8S_OBS_DIR" ]]; then
        kubectl apply -f "$K8S_OBS_DIR/prometheus/" -n "$NAMESPACE" 2>/dev/null || warn "Prometheus deployment skipped"
        kubectl apply -f "$K8S_OBS_DIR/grafana/" -n "$NAMESPACE" 2>/dev/null || warn "Grafana deployment skipped"
        success "Observability stack deployed"
    else
        warn "Observability manifests not found at $K8S_OBS_DIR, skipping..."
    fi
fi

success "Deployment complete!"
echo ""
echo "=================================================="
echo "  Kind Deployment Summary"
echo "=================================================="
echo ""
echo "Namespace: $NAMESPACE"
echo ""
echo "Access the services (run in a separate terminal):"
echo ""
echo "  kubectl port-forward -n $NAMESPACE svc/semantic-router 8080:8080 8801:8801"
echo ""
echo "Then test:"
echo ""
echo "  # Auto-routing (classifier picks the model)"
echo "  curl http://localhost:8801/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"auto\", \"messages\": [{\"role\": \"user\", \"content\": \"What is 2+2?\"}]}'"
echo ""
echo "  # STEM query -> routes to Model-A"
echo "  curl http://localhost:8801/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"auto\", \"messages\": [{\"role\": \"user\", \"content\": \"Explain quantum physics\"}]}'"
echo ""
echo "  # Humanities query -> routes to Model-B"
echo "  curl http://localhost:8801/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"auto\", \"messages\": [{\"role\": \"user\", \"content\": \"Explain the elements of a contract under common law and give a simple example.\"}]}'"
echo ""
echo "View logs:"
echo "  kubectl logs -f deployment/semantic-router -c semantic-router -n $NAMESPACE"
echo "  kubectl logs -f deployment/semantic-router -c envoy-proxy -n $NAMESPACE"
echo ""
echo "View status:"
echo "  kubectl get pods -n $NAMESPACE"
echo "  kubectl get svc -n $NAMESPACE"
echo ""
