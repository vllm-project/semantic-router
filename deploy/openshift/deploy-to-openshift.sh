#!/bin/bash

# deploy-to-openshift.sh
# Automated deployment script for vLLM Semantic Router on OpenShift
#
# Usage: ./deploy-to-openshift.sh [OPTIONS]
#
# This script provides a complete automation solution for deploying
# the semantic router to OpenShift with support for different environments
# and configuration options.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
OPENSHIFT_SERVER=""
OPENSHIFT_USER="admin"
OPENSHIFT_PASSWORD=""
NAMESPACE="vllm-semantic-router-system"
DEPLOYMENT_METHOD="enhanced"  # Use enhanced deployment with llm-katan specialists
CONTAINER_IMAGE="ghcr.io/vllm-project/semantic-router/extproc"
CONTAINER_TAG="latest"
STORAGE_SIZE="10Gi"
MEMORY_REQUEST="3Gi"
MEMORY_LIMIT="6Gi"
CPU_REQUEST="1"
CPU_LIMIT="2"
LOG_LEVEL="info"
SKIP_MODEL_DOWNLOAD="false"
WAIT_FOR_READY="true"
CLEANUP_FIRST="false"
DRY_RUN="false"
PORT_FORWARD="false"
PORT_FORWARD_PORTS="8080:8080 8000:8000 8001:8001 50051:50051 8801:8801 19000:19000"
WITH_OBSERVABILITY="true"
OBSERVABILITY_ONLY="false"
CLEANUP_OBSERVABILITY="false"

# Function to print colored output
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case $level in
        "INFO")  echo -e "${timestamp} ${BLUE}[INFO]${NC}  $message" ;;
        "WARN")  echo -e "${timestamp} ${YELLOW}[WARN]${NC}  $message" ;;
        "ERROR") echo -e "${timestamp} ${RED}[ERROR]${NC} $message" ;;
        "SUCCESS") echo -e "${timestamp} ${GREEN}[SUCCESS]${NC} $message" ;;
    esac
}

# Function to show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Automated deployment script for vLLM Semantic Router on OpenShift

OPTIONS:
    -s, --server URL         OpenShift API server URL
    -u, --user USER          OpenShift username (default: admin)
    -p, --password PASS      OpenShift password
    -n, --namespace NS       Deployment namespace (default: vllm-semantic-router-system)
    -m, --method METHOD      Deployment method: kustomize|template|enhanced (default: enhanced)
    -i, --image IMAGE        Container image (default: ghcr.io/vllm-project/semantic-router/extproc)
    -t, --tag TAG            Container tag (default: latest)
    --storage SIZE           Storage size (default: 10Gi)
    --memory-request SIZE    Memory request (default: 3Gi)
    --memory-limit SIZE      Memory limit (default: 6Gi)
    --cpu-request SIZE       CPU request (default: 1)
    --cpu-limit SIZE         CPU limit (default: 2)
    --log-level LEVEL        Log level: debug|info|warn|error (default: info)
    --skip-models            Skip model download (for demo/testing)
    --no-wait                Don't wait for deployment to be ready
    --cleanup                Clean up existing deployment first
    --dry-run                Show what would be deployed without executing
    --port-forward           Set up port forwarding after successful deployment (default: enabled)
    --no-port-forward        Disable automatic port forwarding
    --port-forward-ports PORTS   Custom port mappings (default: "8080:8080 8000:8000 8001:8001")
    --no-observability       Skip observability stack deployment (observability enabled by default)
    --observability-only     Deploy ONLY observability stack (requires existing semantic-router deployment)
    --cleanup-observability  Remove ONLY observability components (keeps semantic-router intact)
    -h, --help               Show this help message

EXAMPLES:
    # Simple deployment (if already logged in with 'oc login')
    $0

    # Deploy with manual server specification
    $0 -s https://api.cluster.example.com:6443 -p mypassword

    # Deploy with custom namespace and resources
    $0 -n my-semantic-router --memory-limit 8Gi --cpu-limit 4

    # Deploy using basic method instead of enhanced
    $0 --method kustomize

    # Dry run to see what would be deployed
    $0 --dry-run

    # Deploy without automatic port forwarding
    $0 --no-port-forward

    # Deploy without observability stack
    $0 --no-observability

    # Deploy only observability (if semantic-router already exists)
    $0 --observability-only

    # Remove only observability stack
    $0 --cleanup-observability

ENVIRONMENT VARIABLES:
    OPENSHIFT_SERVER         OpenShift API server URL
    OPENSHIFT_USER           OpenShift username
    OPENSHIFT_PASSWORD       OpenShift password
    SEMANTIC_ROUTER_NAMESPACE    Deployment namespace

EOF
}

# Function to parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -s|--server)
                OPENSHIFT_SERVER="$2"
                shift 2
                ;;
            -u|--user)
                OPENSHIFT_USER="$2"
                shift 2
                ;;
            -p|--password)
                OPENSHIFT_PASSWORD="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -m|--method)
                DEPLOYMENT_METHOD="$2"
                shift 2
                ;;
            -i|--image)
                CONTAINER_IMAGE="$2"
                shift 2
                ;;
            -t|--tag)
                CONTAINER_TAG="$2"
                shift 2
                ;;
            --storage)
                STORAGE_SIZE="$2"
                shift 2
                ;;
            --memory-request)
                MEMORY_REQUEST="$2"
                shift 2
                ;;
            --memory-limit)
                MEMORY_LIMIT="$2"
                shift 2
                ;;
            --cpu-request)
                CPU_REQUEST="$2"
                shift 2
                ;;
            --cpu-limit)
                CPU_LIMIT="$2"
                shift 2
                ;;
            --log-level)
                LOG_LEVEL="$2"
                shift 2
                ;;
            --skip-models)
                SKIP_MODEL_DOWNLOAD="true"
                shift
                ;;
            --no-wait)
                WAIT_FOR_READY="false"
                shift
                ;;
            --cleanup)
                CLEANUP_FIRST="true"
                shift
                ;;
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            --port-forward)
                PORT_FORWARD="true"
                shift
                ;;
            --no-port-forward)
                PORT_FORWARD="false"
                shift
                ;;
            --port-forward-ports)
                PORT_FORWARD_PORTS="$2"
                shift 2
                ;;
            --with-observability)
                WITH_OBSERVABILITY="true"
                shift
                ;;
            --observability-only)
                OBSERVABILITY_ONLY="true"
                shift
                ;;
            --cleanup-observability)
                CLEANUP_OBSERVABILITY="true"
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log "ERROR" "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    # Override with environment variables if set
    OPENSHIFT_SERVER="${OPENSHIFT_SERVER:-$OPENSHIFT_SERVER}"
    OPENSHIFT_USER="${OPENSHIFT_USER:-$OPENSHIFT_USER}"
    OPENSHIFT_PASSWORD="${OPENSHIFT_PASSWORD:-$OPENSHIFT_PASSWORD}"
    NAMESPACE="${SEMANTIC_ROUTER_NAMESPACE:-$NAMESPACE}"
}

# Function to validate prerequisites
validate_prerequisites() {
    log "INFO" "Validating prerequisites..."

    # Check if oc is installed
    if ! command -v oc &> /dev/null; then
        log "ERROR" "OpenShift CLI (oc) is not installed or not in PATH"
        log "INFO" "Install from: https://docs.openshift.com/container-platform/latest/cli_reference/openshift_cli/getting-started-cli.html"
        exit 1
    fi

    # Check required parameters - server is only required if not already logged in
    if [[ -z "$OPENSHIFT_SERVER" ]]; then
        if oc whoami >/dev/null 2>&1; then
            OPENSHIFT_SERVER=$(oc whoami --show-server 2>/dev/null || echo "")
            log "INFO" "Auto-detected OpenShift server: $OPENSHIFT_SERVER"
        else
            log "ERROR" "Not logged in to OpenShift. Please login first using:"
            log "INFO" "  oc login <your-openshift-server-url>"
            log "INFO" ""
            log "INFO" "Example:"
            log "INFO" "  oc login https://api.cluster.example.com:6443"
            log "INFO" ""
            log "INFO" "After logging in, simply run this script again without any arguments:"
            log "INFO" "  $0"
            exit 1
        fi
    fi

    # Password is only required if we need to login (not already logged in)
    if [[ -z "$OPENSHIFT_PASSWORD" ]]; then
        if oc whoami >/dev/null 2>&1; then
            log "INFO" "No password specified, but already logged in as $(oc whoami)"
        else
            log "ERROR" "OpenShift password is required when not logged in. Use -p option or OPENSHIFT_PASSWORD env var"
            log "ERROR" "Or login manually first with: oc login"
            exit 1
        fi
    fi

    # Validate deployment method
    if [[ "$DEPLOYMENT_METHOD" != "kustomize" && "$DEPLOYMENT_METHOD" != "template" && "$DEPLOYMENT_METHOD" != "enhanced" ]]; then
        log "ERROR" "Invalid deployment method: $DEPLOYMENT_METHOD. Must be 'kustomize', 'template', or 'enhanced'"
        exit 1
    fi

    log "SUCCESS" "Prerequisites validated"
}

# Function to login to OpenShift
login_openshift() {
    log "INFO" "Checking OpenShift login status..."

    # Check if already logged in
    if oc whoami >/dev/null 2>&1; then
        local current_user=$(oc whoami)
        local current_server=$(oc whoami --show-server 2>/dev/null || echo "unknown")
        log "SUCCESS" "Already logged in as '$current_user' to '$current_server'"

        # If server matches what we want, we're good
        if [[ -n "$OPENSHIFT_SERVER" && "$current_server" == "$OPENSHIFT_SERVER" ]]; then
            log "INFO" "Current session matches target server, continuing..."
            return 0
        elif [[ -z "$OPENSHIFT_SERVER" ]]; then
            log "INFO" "No server specified, using current session..."
            return 0
        else
            log "WARN" "Current server '$current_server' differs from target '$OPENSHIFT_SERVER'"
            log "INFO" "Will login to target server..."
        fi
    else
        log "INFO" "Not currently logged in to OpenShift"
    fi

    # Need to login
    if [[ -z "$OPENSHIFT_SERVER" ]]; then
        log "ERROR" "No OpenShift server specified and not currently logged in"
        log "ERROR" "Please specify server with -s option or login manually with:"
        log "ERROR" "  oc login https://your-openshift-server:6443"
        exit 1
    fi

    if [[ -z "$OPENSHIFT_PASSWORD" ]]; then
        log "ERROR" "No OpenShift password specified"
        log "ERROR" "Please specify password with -p option or login manually with:"
        log "ERROR" "  oc login -u $OPENSHIFT_USER $OPENSHIFT_SERVER"
        exit 1
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would login with: oc login -u $OPENSHIFT_USER -p [REDACTED] $OPENSHIFT_SERVER --insecure-skip-tls-verify"
        return 0
    fi

    log "INFO" "Logging into OpenShift at $OPENSHIFT_SERVER as $OPENSHIFT_USER"
    if ! oc login -u "$OPENSHIFT_USER" -p "$OPENSHIFT_PASSWORD" "$OPENSHIFT_SERVER" --insecure-skip-tls-verify; then
        log "ERROR" "Failed to login to OpenShift"
        log "ERROR" "Please check your credentials and try again, or login manually with:"
        log "ERROR" "  oc login -u $OPENSHIFT_USER $OPENSHIFT_SERVER"
        exit 1
    fi

    log "SUCCESS" "Successfully logged into OpenShift as $(oc whoami)"
}

# Function to cleanup existing deployment
cleanup_deployment() {
    if [[ "$CLEANUP_FIRST" == "true" ]]; then
        log "INFO" "Cleaning up existing deployment in namespace $NAMESPACE"

        if [[ "$DRY_RUN" == "true" ]]; then
            log "INFO" "[DRY RUN] Would delete namespace: $NAMESPACE"
            return 0
        fi

        if oc get namespace "$NAMESPACE" &> /dev/null; then
            oc delete namespace "$NAMESPACE" --ignore-not-found=true
            log "INFO" "Waiting for namespace deletion to complete..."
            while oc get namespace "$NAMESPACE" &> /dev/null; do
                sleep 2
            done
            log "SUCCESS" "Namespace $NAMESPACE deleted"
        else
            log "INFO" "Namespace $NAMESPACE does not exist, skipping cleanup"
        fi
    fi
}

# Function to deploy using Enhanced OpenShift deployment (with llm-katan specialists)
deploy_enhanced() {
    log "INFO" "Deploying using Enhanced OpenShift method (with llm-katan specialists)"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would deploy enhanced deployment with 4-container pod:"
        log "INFO" "[DRY RUN]   - semantic-router (main ExtProc service on port 50051)"
        log "INFO" "[DRY RUN]   - math-specialist (llm-katan on port 8000)"
        log "INFO" "[DRY RUN]   - coding-specialist (llm-katan on port 8001)"
        log "INFO" "[DRY RUN]   - envoy-proxy (gateway on port 8801)"
        return 0
    fi

    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    # Create namespace first
    oc create namespace "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -
    log "SUCCESS" "Namespace $NAMESPACE created/verified"

    # Build llm-katan image if it doesn't exist
    log "INFO" "Checking for llm-katan image..."
    if ! oc get imagestream llm-katan -n "$NAMESPACE" &> /dev/null; then
        log "INFO" "Building llm-katan image from Dockerfile..."

        # Create build config and start build
        if [[ -f "$script_dir/Dockerfile.llm-katan" ]]; then
            oc new-build --dockerfile - --name llm-katan -n "$NAMESPACE" < "$script_dir/Dockerfile.llm-katan"
        else
            log "ERROR" "Dockerfile.llm-katan not found. Expected at: $script_dir/Dockerfile.llm-katan"
            exit 1
        fi

        # Wait for python imagestream to be ready
        log "INFO" "Waiting for python imagestream to be ready..."
        sleep 5
        while ! oc get istag python:3.10-slim -n "$NAMESPACE" &> /dev/null; do
            sleep 2
        done
        log "SUCCESS" "Python imagestream ready"

        # Start the build and wait for completion
        log "INFO" "Starting llm-katan build..."
        oc start-build llm-katan -n "$NAMESPACE"

        # Wait for build to complete
        log "INFO" "Waiting for llm-katan build to complete..."
        if ! oc wait --for=condition=Complete build/llm-katan-1 -n "$NAMESPACE" --timeout=600s; then
            log "ERROR" "llm-katan build failed or timed out"
            oc logs build/llm-katan-1 -n "$NAMESPACE" --tail=50
            exit 1
        fi

        log "SUCCESS" "llm-katan image built successfully"
    else
        log "INFO" "llm-katan image already exists, skipping build"
    fi

    # Create PVC for models
    log "INFO" "Creating PVC for models..."
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
      storage: $STORAGE_SIZE
  storageClassName: gp3-csi
EOF

    # Create PVC for cache (model weights from HuggingFace)
    log "INFO" "Creating PVC for cache..."
    cat <<EOF | oc apply -n "$NAMESPACE" -f -
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
EOF

    # Create ConfigMap using OpenShift-specific config based on e2e config
    log "INFO" "Creating ConfigMap with OpenShift-specific config..."
    oc create configmap semantic-router-config \
        --from-file=config.yaml="$script_dir/config-openshift.yaml" \
        -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -

    # Create Envoy ConfigMap using OpenShift-specific configuration
    log "INFO" "Creating Envoy ConfigMap with OpenShift-specific configuration..."
    oc create configmap envoy-config \
        --from-file=envoy.yaml="$script_dir/envoy-openshift.yaml" \
        -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -

    # Apply the enhanced deployment
    log "INFO" "Applying enhanced deployment with llm-katan specialists..."
    oc apply -f "$script_dir/deployment.yaml" -n "$NAMESPACE"

    # Create services and routes
    log "INFO" "Creating services..."
    cat <<EOF | oc apply -n "$NAMESPACE" -f -
apiVersion: v1
kind: Service
metadata:
  name: semantic-router
  labels:
    app: semantic-router
spec:
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
  - name: math-specialist
    port: 8000
    targetPort: 8000
  - name: coding-specialist
    port: 8001
    targetPort: 8001
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

    # Create routes
    log "INFO" "Creating OpenShift routes..."
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

    log "SUCCESS" "Enhanced deployment applied successfully"
}

# Function to deploy using kustomize
deploy_kustomize() {
    log "INFO" "Deploying using Kustomize method"

    # Create temporary directory for manifests
    local temp_dir=$(mktemp -d)
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    # Copy manifests to temp directory
    cp -r "$script_dir"/* "$temp_dir/"
    cd "$temp_dir"

    # Modify deployment if skipping models
    if [[ "$SKIP_MODEL_DOWNLOAD" == "true" ]]; then
        log "INFO" "Configuring deployment to skip model download"
        # Comment out init container (already done in our version)
    fi

    # Update image and tag in kustomization
    if [[ "$CONTAINER_IMAGE:$CONTAINER_TAG" != "ghcr.io/vllm-project/semantic-router/extproc:latest" ]]; then
        log "INFO" "Updating container image to $CONTAINER_IMAGE:$CONTAINER_TAG"
        yq eval ".images[0].name = \"$CONTAINER_IMAGE\"" -i kustomization.yaml
        yq eval ".images[0].newTag = \"$CONTAINER_TAG\"" -i kustomization.yaml
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would deploy with: oc apply -k ."
        log "INFO" "[DRY RUN] Manifest preview:"
        oc kustomize . | head -50
        return 0
    fi

    # Apply the manifests
    if ! oc apply -k .; then
        log "ERROR" "Failed to apply Kustomize manifests"
        exit 1
    fi

    # Cleanup temp directory
    cd - &> /dev/null
    rm -rf "$temp_dir"

    log "SUCCESS" "Kustomize deployment applied successfully"
}

# Function to deploy using template
deploy_template() {
    log "INFO" "Deploying using OpenShift Template method"

    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local template_file="$script_dir/template.yaml"

    if [[ ! -f "$template_file" ]]; then
        log "ERROR" "Template file not found: $template_file"
        exit 1
    fi

    local template_params=(
        "NAMESPACE=$NAMESPACE"
        "CONTAINER_IMAGE=$CONTAINER_IMAGE"
        "CONTAINER_TAG=$CONTAINER_TAG"
        "STORAGE_SIZE=$STORAGE_SIZE"
        "MEMORY_REQUEST=$MEMORY_REQUEST"
        "MEMORY_LIMIT=$MEMORY_LIMIT"
        "CPU_REQUEST=$CPU_REQUEST"
        "CPU_LIMIT=$CPU_LIMIT"
        "LOG_LEVEL=$LOG_LEVEL"
    )

    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would process template with parameters:"
        for param in "${template_params[@]}"; do
            log "INFO" "  $param"
        done
        return 0
    fi

    # Process and apply template
    local param_args=""
    for param in "${template_params[@]}"; do
        param_args="$param_args -p $param"
    done

    if ! oc process -f "$template_file" $param_args | oc apply -f -; then
        log "ERROR" "Failed to process and apply template"
        exit 1
    fi

    log "SUCCESS" "Template deployment applied successfully"
}

# Function to wait for deployment to be ready
wait_for_ready() {
    if [[ "$WAIT_FOR_READY" == "false" || "$DRY_RUN" == "true" ]]; then
        return 0
    fi

    log "INFO" "Waiting for deployment to be ready..."

    # Wait for namespace to be active
    log "INFO" "Waiting for namespace $NAMESPACE to be active..."
    if ! oc wait --for=condition=Active namespace/"$NAMESPACE" --timeout=60s; then
        log "WARN" "Namespace did not become active within timeout, but continuing..."
    fi

    # Wait for deployment to be available
    log "INFO" "Waiting for deployment to be available..."
    if ! oc wait --for=condition=Available deployment/semantic-router -n "$NAMESPACE" --timeout=300s; then
        log "WARN" "Deployment did not become available within timeout"
        log "INFO" "Checking deployment status..."
        oc get pods -n "$NAMESPACE"
        oc describe deployment/semantic-router -n "$NAMESPACE" | tail -20
        return 1
    fi

    log "SUCCESS" "Deployment is ready!"
}

# Function to setup port forwarding
setup_port_forwarding() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would setup port forwarding with ports: $PORT_FORWARD_PORTS"
        return 0
    fi

    if [[ "$PORT_FORWARD" != "true" ]]; then
        return 0
    fi

    log "INFO" "Setting up port forwarding..."

    # Get the pod name
    local pod_name=$(oc get pods -n "$NAMESPACE" -l app=semantic-router -o jsonpath='{.items[0].metadata.name}')
    if [[ -z "$pod_name" ]]; then
        log "ERROR" "Could not find semantic-router pod for port forwarding"
        return 1
    fi

    log "INFO" "Setting up port forwarding to pod: $pod_name"

    # Kill any existing port-forward processes for this namespace
    pkill -f "oc port-forward.*$NAMESPACE" || true
    sleep 2

    # Set up port forwarding in background
    log "INFO" "Port forwarding: $PORT_FORWARD_PORTS"
    oc port-forward "$pod_name" $PORT_FORWARD_PORTS -n "$NAMESPACE" &
    local pf_pid=$!

    # Give it a moment to establish
    sleep 3

    if kill -0 $pf_pid 2>/dev/null; then
        log "SUCCESS" "Port forwarding established (PID: $pf_pid)"
        log "INFO" "Access endpoints at:"
        for port_mapping in $PORT_FORWARD_PORTS; do
            local local_port=$(echo $port_mapping | cut -d: -f1)
            log "INFO" "  - localhost:$local_port"
        done
        log "INFO" "To stop port forwarding: kill $pf_pid"
        echo "PID $pf_pid" > /tmp/semantic-router-port-forward.pid
    else
        log "WARN" "Port forwarding may have failed to establish"
    fi
}

# Function to show deployment info
show_deployment_info() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would show deployment information"
        return 0
    fi

    log "INFO" "Deployment information:"

    echo ""
    echo "=== Pods ==="
    oc get pods -n "$NAMESPACE" -o wide

    echo ""
    echo "=== Services ==="
    oc get services -n "$NAMESPACE"

    echo ""
    echo "=== Routes ==="
    oc get routes -n "$NAMESPACE"

    echo ""
    echo "=== External URLs ==="
    local api_route=$(oc get route semantic-router-api -n "$NAMESPACE" -o jsonpath='{.spec.host}' 2>/dev/null)
    local grpc_route=$(oc get route semantic-router-grpc -n "$NAMESPACE" -o jsonpath='{.spec.host}' 2>/dev/null)
    local metrics_route=$(oc get route semantic-router-metrics -n "$NAMESPACE" -o jsonpath='{.spec.host}' 2>/dev/null)
    local envoy_route=$(oc get route envoy-http -n "$NAMESPACE" -o jsonpath='{.spec.host}' 2>/dev/null)

    if [[ -n "$envoy_route" ]]; then
        echo ""
        log "SUCCESS" "OpenWebUI Endpoint (use this in OpenWebUI settings):"
        echo "  http://$envoy_route/v1"
        echo ""
    fi

    if [[ -n "$api_route" ]]; then
        echo "Classification API: http://$api_route"
        echo "Health Check: http://$api_route/health"
    fi
    if [[ -n "$grpc_route" ]]; then
        echo "gRPC API: http://$grpc_route"
    fi
    if [[ -n "$metrics_route" ]]; then
        echo "Metrics: http://$metrics_route/metrics"
    fi

    echo ""
    echo "=== Quick Test Commands ==="
    if [[ -n "$api_route" ]]; then
        echo "curl http://$api_route/health"
        echo "curl -X POST http://$api_route/api/v1/classify/intent -H 'Content-Type: application/json' -d '{\"text\": \"What is 2+2?\"}'"
    fi
}

# Function to display observability stack information
show_observability_info() {
    log "INFO" "Observability deployment information:"

    echo ""
    echo "=== Observability Pods ==="
    oc get pods -n "$NAMESPACE" -l app.kubernetes.io/component=observability

    echo ""
    echo "=== Observability Routes ==="
    oc get routes -n "$NAMESPACE" -l app.kubernetes.io/component=observability

    local grafana_route=$(oc get route grafana -n "$NAMESPACE" -o jsonpath='{.spec.host}' 2>/dev/null)
    local prometheus_route=$(oc get route prometheus -n "$NAMESPACE" -o jsonpath='{.spec.host}' 2>/dev/null)

    echo ""
    log "SUCCESS" "Access URLs:"
    if [[ -n "$grafana_route" ]]; then
        echo "  Grafana:    https://$grafana_route (Login: admin/admin)"
        echo "  Dashboard:  https://$grafana_route/d/llm-router-metrics"
    fi
    if [[ -n "$prometheus_route" ]]; then
        echo "  Prometheus: https://$prometheus_route"
        echo "  Targets:    https://$prometheus_route/targets"
    fi

    echo ""
    log "INFO" "Verify Prometheus is scraping semantic-router:"
    echo "  oc logs deployment/prometheus -n $NAMESPACE | grep semantic-router"
    echo ""
    log "WARN" "Default Grafana password is 'admin'. Please change it after first login!"
}

# Function to deploy observability stack
deploy_observability() {
    log "INFO" "Deploying observability stack (Prometheus + Grafana)..."

    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would deploy: oc apply -k $script_dir/observability/"
        return 0
    fi

    # Verify semantic-router is deployed
    if ! oc get deployment semantic-router -n "$NAMESPACE" &> /dev/null; then
        log "ERROR" "Semantic router deployment not found in namespace $NAMESPACE"
        log "ERROR" "Deploy semantic-router first or use --with-observability flag"
        exit 1
    fi

    log "INFO" "Semantic router deployment found, proceeding with observability..."

    # Apply observability stack
    log "INFO" "Applying observability manifests from $script_dir/observability/"
    if ! oc apply -k "$script_dir/observability/" -n "$NAMESPACE"; then
        log "ERROR" "Failed to apply observability manifests"
        exit 1
    fi

    # Wait for deployments
    log "INFO" "Waiting for Prometheus to be ready..."
    if ! oc wait --for=condition=Available deployment/prometheus -n "$NAMESPACE" --timeout=180s 2>/dev/null; then
        log "WARN" "Prometheus may not be ready yet. Check status with: oc get pods -n $NAMESPACE"
    else
        log "SUCCESS" "Prometheus is ready"
    fi

    log "INFO" "Waiting for Grafana to be ready..."
    if ! oc wait --for=condition=Available deployment/grafana -n "$NAMESPACE" --timeout=180s 2>/dev/null; then
        log "WARN" "Grafana may not be ready yet. Check status with: oc get pods -n $NAMESPACE"
    else
        log "SUCCESS" "Grafana is ready"
    fi

    # Show access info
    echo ""
    show_observability_info

    log "SUCCESS" "Observability stack deployed!"
}

# Function to cleanup observability stack
cleanup_observability() {
    log "INFO" "Cleaning up observability stack (keeping semantic-router)..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would delete observability resources"
        return 0
    fi

    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    # Delete using kustomize (preserves semantic-router)
    log "INFO" "Deleting observability resources..."
    if ! oc delete -k "$script_dir/observability/" -n "$NAMESPACE" --ignore-not-found=true; then
        log "WARN" "Some errors occurred during cleanup, but continuing..."
    fi

    # Wait for cleanup
    log "INFO" "Waiting for cleanup to complete..."
    sleep 5

    # Verify cleanup
    local observability_pods=$(oc get pods -n "$NAMESPACE" -l app.kubernetes.io/component=observability --no-headers 2>/dev/null | wc -l)

    if [[ "$observability_pods" -eq 0 ]]; then
        log "SUCCESS" "Observability stack cleaned up successfully"
    else
        log "WARN" "Some observability resources may still exist:"
        oc get all -n "$NAMESPACE" -l app.kubernetes.io/component=observability
    fi
}

# Main function
main() {
    log "INFO" "Starting vLLM Semantic Router OpenShift deployment"

    parse_args "$@"
    validate_prerequisites
    login_openshift

    # Handle observability-only mode
    if [[ "$OBSERVABILITY_ONLY" == "true" ]]; then
        deploy_observability
        exit 0
    fi

    # Handle cleanup-observability mode
    if [[ "$CLEANUP_OBSERVABILITY" == "true" ]]; then
        cleanup_observability
        exit 0
    fi

    cleanup_deployment

    case "$DEPLOYMENT_METHOD" in
        "kustomize")
            deploy_kustomize
            ;;
        "template")
            deploy_template
            ;;
        "enhanced")
            deploy_enhanced
            ;;
    esac

    if wait_for_ready; then
        show_deployment_info

        # Deploy observability if requested
        if [[ "$WITH_OBSERVABILITY" == "true" ]]; then
            echo ""
            log "INFO" "Deploying observability stack as requested..."
            deploy_observability
        fi

        setup_port_forwarding
        log "SUCCESS" "Deployment completed successfully!"
    else
        log "ERROR" "Deployment may have issues. Check the logs and status above."
        exit 1
    fi
}

# Run main function with all arguments
main "$@"