#!/usr/bin/env bash

# This script validates all commands in website/docs/installation/milvus.md
#
# Usage:
#   # Non-interactive standalone deployment
#   MILVUS_MODE=standalone RECREATE_CLUSTER=false CLEANUP=false ./tools/milvus/test-milvus-deployment.sh
#
#   # Non-interactive cluster deployment with cleanup
#   MILVUS_MODE=cluster RECREATE_CLUSTER=true CLEANUP=true ./tools/milvus/test-milvus-deployment.sh

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="vllm-semantic-router-system"
CLUSTER_NAME="semantic-router-cluster"
RELEASE_NAME="milvus-semantic-cache"

# Environment variable defaults (empty means interactive)
MILVUS_MODE="${MILVUS_MODE:-}"              # standalone or cluster
RECREATE_CLUSTER="${RECREATE_CLUSTER:-}"    # true or false
CLEANUP="${CLEANUP:-}"                      # true or false

# Helper functions
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

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 is not installed"
        return 1
    fi
}

# Section: Prerequisites Check
section_prerequisites() {
    log_info "=== Checking Prerequisites ==="
    
    check_command kubectl || exit 1
    check_command kind || exit 1
    check_command helm || exit 1
    
    log_success "All prerequisites are installed"
}

# Section: Create Kind Cluster
section_create_cluster() {
    log_info "=== Creating Kind Cluster ==="
    
    if kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
        log_warning "Cluster ${CLUSTER_NAME} already exists"
        
        local recreate="$RECREATE_CLUSTER"
        if [ -z "$recreate" ]; then
            read -r -p "Delete and recreate? (y/N): " confirm
            [[ "$confirm" =~ ^[Yy]$ ]] && recreate="true" || recreate="false"
        fi
        
        if [ "$recreate" = "true" ]; then
            log_info "Recreating cluster..."
            make delete-cluster
        else
            log_info "Using existing cluster"
            return 0
        fi
    fi
    
    make create-cluster
    log_success "Cluster created successfully"
}

# Section: Deploy Milvus with Helm
section_deploy_milvus_helm() {
    local mode=$1
    log_info "=== Deploying Milvus ${mode^} Mode with Helm ==="
    
    # Add Milvus Helm repo
    log_info "Adding Milvus Helm repository..."
    helm repo add milvus https://zilliztech.github.io/milvus-helm/
    helm repo update
    
    # Check if release exists
    if helm list -n "$NAMESPACE" | grep -q "$RELEASE_NAME"; then
        log_warning "Helm release '$RELEASE_NAME' already exists"
        log_info "Uninstalling existing release..."
        helm uninstall "$RELEASE_NAME" -n "$NAMESPACE" || true
        sleep 10
    fi
    
    # Create namespace if not exists
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Install Milvus based on mode
    log_info "Installing Milvus ${mode} mode..."
    
    if [ "$mode" = "cluster" ]; then
        log_info "Cluster mode: Using Pulsar v3 (disabling old Pulsar)"
        helm install "$RELEASE_NAME" milvus/milvus \
            --set cluster.enabled=true \
            --set etcd.replicaCount=3 \
            --set minio.mode=distributed \
            --set pulsar.enabled=false \
            --set pulsarv3.enabled=true \
            --set metrics.serviceMonitor.enabled=false \
            --namespace "$NAMESPACE" \
            --wait --timeout=15m
    else
        helm install "$RELEASE_NAME" milvus/milvus \
            --set cluster.enabled=false \
            --set etcd.replicaCount=1 \
            --set minio.mode=standalone \
            --set pulsar.enabled=false \
            --set metrics.serviceMonitor.enabled=false \
            --namespace "$NAMESPACE" \
            --wait --timeout=15m
    fi
    
    log_info "Note: ServiceMonitor is disabled. To enable, install Prometheus Operator first."
    
    log_success "Milvus ${mode} mode deployed successfully"
}

# Section: Verify Milvus Deployment
section_verify_milvus() {
    log_info "=== Verifying Milvus Deployment ==="
    
    # Wait for Milvus pods
    log_info "Waiting for Milvus pods to be ready..."
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=milvus \
        -n "$NAMESPACE" --timeout=600s || {
        log_error "Milvus pods not ready"
        kubectl get pods -n "$NAMESPACE"
        return 1
    }
    
    # Check pods
    log_info "Milvus pods:"
    kubectl get pods -l app.kubernetes.io/name=milvus -n "$NAMESPACE"
    
    # Check services
    log_info "Milvus services:"
    kubectl get svc -l app.kubernetes.io/name=milvus -n "$NAMESPACE"
    
    log_success "Milvus deployment verified"
}

# Section: Apply Milvus Client Config
section_apply_client_config() {
    log_info "=== Applying Milvus Client Config ==="
    
    kubectl apply -n "$NAMESPACE" -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: milvus-client-config
data:
  milvus.yaml: |
    connection:
      host: "milvus-semantic-cache.${NAMESPACE}.svc.cluster.local"
      port: 19530
      timeout: 60
      auth:
        enabled: false
      tls:
        enabled: false
    collection:
      name: "semantic_cache"
      description: "Semantic cache"
      vector_field:
        name: "embedding"
        dimension: 384
        metric_type: "IP"
      index:
        type: "HNSW"
        params:
          M: 16
          efConstruction: 64
    search:
      params:
        ef: 64
      topk: 10
      consistency_level: "Session"
    development:
      auto_create_collection: true
      verbose_errors: true
EOF
    
    log_success "Milvus client config applied"
}

# Section: Networking and Security
section_networking_security() {
    log_info "=== Configuring Networking and Security ==="
    
    # Apply NetworkPolicy
    log_info "Applying NetworkPolicy..."
    kubectl apply -n "$NAMESPACE" -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-router-to-milvus
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: milvus
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: ${NAMESPACE}
          podSelector:
            matchLabels:
              app.kubernetes.io/name: semantic-router
      ports:
        - protocol: TCP
          port: 19530
EOF
    
    # Create auth secret (with example credentials)
    log_info "Creating auth secret..."
    kubectl create secret generic milvus-auth -n "$NAMESPACE" \
        --from-literal=username="admin" \
        --from-literal=password="Milvus123!" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log_success "Networking and security configured"
}

# Section: Monitoring
section_monitoring() {
    log_info "=== Configuring Monitoring ==="
    
    # Check if Prometheus Operator is installed
    if kubectl get crd servicemonitors.monitoring.coreos.com &> /dev/null; then
        log_info "Prometheus Operator detected"
        log_info "ServiceMonitor is enabled by default in Helm values"
        if kubectl get servicemonitor -n "$NAMESPACE" 2>/dev/null; then
            log_success "ServiceMonitor found"
        else
            log_warning "ServiceMonitor not yet created"
        fi
    else
        log_warning "Prometheus Operator not installed, ServiceMonitor will not be created"
    fi
}

# Section: Connection Tests
section_connection_tests() {
    log_info "=== Testing Milvus Connection ==="
    
    # Get Milvus service
    local milvus_svc
    milvus_svc=$(kubectl get svc -n "$NAMESPACE" -l app.kubernetes.io/name=milvus -o jsonpath='{.items[0].metadata.name}')
    
    if [ -z "$milvus_svc" ]; then
        log_error "Milvus service not found"
        return 1
    fi
    
    log_info "Milvus service: $milvus_svc"
    
    # Port forward for testing
    log_info "Setting up port-forward for testing..."
    kubectl port-forward -n "$NAMESPACE" "svc/$milvus_svc" 19530:19530 &
    local pf_pid=$!
    sleep 5
    
    # Test connection with nc
    log_info "Testing connection with netcat..."
    if command -v nc &> /dev/null; then
        if nc -zv localhost 19530 2>&1 | grep -q "succeeded"; then
            log_success "Connection test passed"
        else
            log_warning "Connection test failed (this is expected if Milvus is still starting)"
        fi
    else
        log_warning "netcat not installed, skipping connection test"
    fi
    
    # Cleanup port-forward
    kill $pf_pid 2>/dev/null || true
}

# Section: Troubleshooting Commands
section_troubleshooting() {
    log_info "=== Running Troubleshooting Commands ==="
    
    # Overall health check
    log_info "Overall health check:"
    kubectl get all -l app.kubernetes.io/name=milvus -n "$NAMESPACE"
    
    # Check PVC status
    log_info "PVC status:"
    kubectl get pvc -n "$NAMESPACE"
    
    # Check StorageClass
    log_info "StorageClass:"
    kubectl get sc
    
    # Check NetworkPolicy
    log_info "NetworkPolicy:"
    kubectl get networkpolicy -n "$NAMESPACE"
    
    # Component logs (last 20 lines)
    log_info "Milvus logs (last 20 lines):"
    kubectl logs -l app.kubernetes.io/name=milvus -n "$NAMESPACE" --tail=20 || {
        log_warning "Could not retrieve logs"
    }
    
    log_success "Troubleshooting commands completed"
}

# Section: Cleanup
section_cleanup() {
    log_info "=== Cleanup ==="
    
    local do_cleanup="$CLEANUP"
    if [ -z "$do_cleanup" ]; then
        read -r -p "Do you want to cleanup Milvus deployment? (y/N): " confirm
        [[ "$confirm" =~ ^[Yy]$ ]] && do_cleanup="true" || do_cleanup="false"
    fi
    
    if [ "$do_cleanup" = "true" ]; then
        log_info "Uninstalling Milvus..."
        helm uninstall "$RELEASE_NAME" -n "$NAMESPACE" || true
        
        log_info "Deleting namespace resources..."
        kubectl delete configmap milvus-client-config -n "$NAMESPACE" --ignore-not-found=true || true
        kubectl delete networkpolicy allow-router-to-milvus -n "$NAMESPACE" --ignore-not-found=true || true
        kubectl delete secret milvus-auth -n "$NAMESPACE" --ignore-not-found=true || true
        
        log_success "Cleanup completed"
    else
        log_info "Skipping cleanup"
    fi
}

# Section: Select Deployment Mode
section_select_mode() {
    log_info "=== Select Milvus Deployment Mode ==="
    
    # Validate if MILVUS_MODE is set
    if [ -n "${MILVUS_MODE}" ]; then
        if [ "${MILVUS_MODE}" != "standalone" ] && [ "${MILVUS_MODE}" != "cluster" ]; then
            log_error "Invalid MILVUS_MODE: ${MILVUS_MODE}. Must be 'standalone' or 'cluster'"
            exit 1
        fi
        log_info "Using MILVUS_MODE from environment: ${MILVUS_MODE}"
    else
        # Interactive mode selection
        echo "Available deployment modes:"
        echo "  1) Standalone - Single instance (development/testing)"
        echo "  2) Cluster - High availability (production)"
        echo ""
        read -r -p "Select mode (1/2) [default: 1]: " mode_choice
        
        case "$mode_choice" in
            2)
                MILVUS_MODE="cluster"
                ;;
            1|"")
                MILVUS_MODE="standalone"
                ;;
            *)
                log_error "Invalid choice"
                exit 1
                ;;
        esac
    fi
    
    log_info "Selected mode: ${MILVUS_MODE}"
    
    if [ "${MILVUS_MODE}" = "cluster" ]; then
        log_warning "Cluster mode requires more resources (etcd, minio, pulsar)"
    fi
}

# Main execution
main() {
    log_info "Starting Milvus Installation Validation"
    log_info "This script validates commands from website/docs/installation/milvus.md"
    echo ""
    
    # Run sections
    section_prerequisites
    echo ""
    
    section_create_cluster
    echo ""
    
    section_select_mode
    echo ""
    
    section_deploy_milvus_helm "$MILVUS_MODE"
    echo ""
    
    section_verify_milvus
    echo ""
    
    section_apply_client_config
    echo ""
    
    section_networking_security
    echo ""
    
    section_monitoring
    echo ""
    
    section_connection_tests
    echo ""
    
    section_troubleshooting
    echo ""
    
    log_success "All validation steps completed!"
    echo ""
    log_info "Summary:"
    echo "  - Cluster: $CLUSTER_NAME"
    echo "  - Namespace: $NAMESPACE"
    echo "  - Milvus Mode: $MILVUS_MODE"
    echo "  - Deployment: Helm release '$RELEASE_NAME'"
    echo ""
    log_info "Next steps:"
    echo "  - Test Milvus connection: kubectl port-forward -n $NAMESPACE svc/$RELEASE_NAME 19530:19530"
    echo "  - View logs: kubectl logs -l app.kubernetes.io/name=milvus -n $NAMESPACE -f"
    
    if [ "$MILVUS_MODE" = "cluster" ]; then
        echo "  - Check etcd: kubectl get pods -l app.kubernetes.io/name=etcd -n $NAMESPACE"
        echo "  - Check minio: kubectl get pods -l app.kubernetes.io/name=minio -n $NAMESPACE"
        echo "  - Check pulsar: kubectl get pods -l app.kubernetes.io/name=pulsar -n $NAMESPACE"
    fi
    
    echo "  - Cleanup: Run this script with cleanup option or manually uninstall"
    echo ""
    
    section_cleanup
}

# Run main function
main "$@"
