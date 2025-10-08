#!/bin/bash

# cleanup-openshift.sh
# Comprehensive cleanup script for vLLM Semantic Router OpenShift deployment
#
# Usage: ./cleanup-openshift.sh [OPTIONS]
#
# This script provides complete cleanup capabilities for semantic router
# deployments on OpenShift, with support for different cleanup levels.

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
CLEANUP_LEVEL="namespace"  # namespace, deployment, or all
DRY_RUN="false"
FORCE="false"
WAIT_FOR_COMPLETION="true"

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

Cleanup script for vLLM Semantic Router OpenShift deployment

OPTIONS:
    -s, --server URL         OpenShift API server URL
    -u, --user USER          OpenShift username (default: admin)
    -p, --password PASS      OpenShift password
    -n, --namespace NS       Deployment namespace (default: vllm-semantic-router-system)
    -l, --level LEVEL        Cleanup level: deployment|namespace|all (default: namespace)
    -f, --force              Force cleanup without confirmation
    --no-wait                Don't wait for cleanup completion
    --dry-run                Show what would be cleaned up without executing
    -h, --help               Show this help message

CLEANUP LEVELS:
    deployment    - Remove deployment, services, routes, configmap (keep namespace and PVC)
    namespace     - Remove entire namespace and all resources (default)
    all           - Remove namespace and any cluster-wide resources

EXAMPLES:
    # Clean up entire namespace (default)
    $0 -s https://api.cluster.example.com:6443 -p mypassword

    # Clean up only deployment resources, keep namespace
    $0 -s https://api.cluster.example.com:6443 -p mypassword --level deployment

    # Dry run to see what would be cleaned up
    $0 -s https://api.cluster.example.com:6443 -p mypassword --dry-run

    # Force cleanup without confirmation
    $0 -s https://api.cluster.example.com:6443 -p mypassword --force

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
            -l|--level)
                CLEANUP_LEVEL="$2"
                shift 2
                ;;
            -f|--force)
                FORCE="true"
                shift
                ;;
            --no-wait)
                WAIT_FOR_COMPLETION="false"
                shift
                ;;
            --dry-run)
                DRY_RUN="true"
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

    # Validate cleanup level
    if [[ "$CLEANUP_LEVEL" != "deployment" && "$CLEANUP_LEVEL" != "namespace" && "$CLEANUP_LEVEL" != "all" ]]; then
        log "ERROR" "Invalid cleanup level: $CLEANUP_LEVEL. Must be 'deployment', 'namespace', or 'all'"
        exit 1
    fi

    log "SUCCESS" "Prerequisites validated"
}

# Function to login to OpenShift
login_openshift() {
    log "INFO" "Logging into OpenShift at $OPENSHIFT_SERVER"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would login with: oc login -u $OPENSHIFT_USER -p [REDACTED] $OPENSHIFT_SERVER --insecure-skip-tls-verify"
        return 0
    fi

    if ! oc login -u "$OPENSHIFT_USER" -p "$OPENSHIFT_PASSWORD" "$OPENSHIFT_SERVER" --insecure-skip-tls-verify; then
        log "ERROR" "Failed to login to OpenShift"
        exit 1
    fi

    log "SUCCESS" "Successfully logged into OpenShift"
}

# Function to check if namespace exists
check_namespace_exists() {
    if [[ "$DRY_RUN" == "true" ]]; then
        return 0
    fi

    if ! oc get namespace "$NAMESPACE" &> /dev/null; then
        log "WARN" "Namespace $NAMESPACE does not exist"
        return 1
    fi
    return 0
}

# Function to show current resources
show_current_resources() {
    log "INFO" "Current resources in namespace $NAMESPACE:"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would show current resources"
        return 0
    fi

    if ! check_namespace_exists; then
        log "INFO" "No resources to show (namespace doesn't exist)"
        return 0
    fi

    echo ""
    echo "=== Pods ==="
    oc get pods -n "$NAMESPACE" 2>/dev/null || echo "No pods found"

    echo ""
    echo "=== Services ==="
    oc get services -n "$NAMESPACE" 2>/dev/null || echo "No services found"

    echo ""
    echo "=== Routes ==="
    oc get routes -n "$NAMESPACE" 2>/dev/null || echo "No routes found"

    echo ""
    echo "=== PVCs ==="
    oc get pvc -n "$NAMESPACE" 2>/dev/null || echo "No PVCs found"

    echo ""
    echo "=== ConfigMaps ==="
    oc get configmaps -n "$NAMESPACE" 2>/dev/null || echo "No configmaps found"

    echo ""
}

# Function to confirm cleanup
confirm_cleanup() {
    if [[ "$FORCE" == "true" || "$DRY_RUN" == "true" ]]; then
        return 0
    fi

    echo ""
    log "WARN" "This will permanently delete resources!"
    log "WARN" "Cleanup level: $CLEANUP_LEVEL"
    log "WARN" "Namespace: $NAMESPACE"

    case "$CLEANUP_LEVEL" in
        "deployment")
            log "WARN" "Will delete: deployment, services, routes, configmaps (keeping namespace and PVCs)"
            ;;
        "namespace")
            log "WARN" "Will delete: entire namespace and all resources including PVCs"
            ;;
        "all")
            log "WARN" "Will delete: namespace and any cluster-wide resources"
            ;;
    esac

    echo ""
    read -p "Are you sure you want to proceed? (yes/no): " confirm
    if [[ "$confirm" != "yes" && "$confirm" != "y" ]]; then
        log "INFO" "Cleanup cancelled by user"
        exit 0
    fi
}

# Function to cleanup deployment level resources
cleanup_deployment() {
    log "INFO" "Cleaning up deployment-level resources..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would delete deployment resources in namespace $NAMESPACE"
        return 0
    fi

    if ! check_namespace_exists; then
        log "INFO" "Nothing to clean up (namespace doesn't exist)"
        return 0
    fi

    # Delete specific resources but keep namespace and PVCs
    local resources=(
        "deployment/semantic-router"
        "service/semantic-router"
        "service/semantic-router-metrics"
        "route/semantic-router-api"
        "route/semantic-router-grpc"
        "route/semantic-router-metrics"
        "configmap/semantic-router-config"
    )

    for resource in "${resources[@]}"; do
        if oc get "$resource" -n "$NAMESPACE" &> /dev/null; then
            log "INFO" "Deleting $resource..."
            oc delete "$resource" -n "$NAMESPACE" --ignore-not-found=true
        else
            log "INFO" "Resource $resource not found, skipping..."
        fi
    done

    log "SUCCESS" "Deployment-level cleanup completed"
}

# Function to cleanup namespace
cleanup_namespace() {
    log "INFO" "Cleaning up namespace: $NAMESPACE"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would delete namespace: $NAMESPACE"
        return 0
    fi

    if ! check_namespace_exists; then
        log "INFO" "Nothing to clean up (namespace doesn't exist)"
        return 0
    fi

    oc delete namespace "$NAMESPACE" --ignore-not-found=true

    if [[ "$WAIT_FOR_COMPLETION" == "true" ]]; then
        log "INFO" "Waiting for namespace deletion to complete..."
        local timeout=300  # 5 minutes
        local count=0
        while oc get namespace "$NAMESPACE" &> /dev/null && [ $count -lt $timeout ]; do
            sleep 2
            count=$((count + 2))
            if [ $((count % 30)) -eq 0 ]; then
                log "INFO" "Still waiting for namespace deletion... (${count}s elapsed)"
            fi
        done

        if oc get namespace "$NAMESPACE" &> /dev/null; then
            log "WARN" "Namespace deletion is taking longer than expected"
            log "INFO" "You can check the status manually with: oc get namespace $NAMESPACE"
        else
            log "SUCCESS" "Namespace deleted successfully"
        fi
    fi
}

# Function to cleanup cluster-wide resources (if any)
cleanup_cluster_wide() {
    log "INFO" "Checking for cluster-wide resources to clean up..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would check for cluster-wide resources"
        return 0
    fi

    # For semantic router, there typically aren't cluster-wide resources
    # But this is where you would clean up CRDs, ClusterRoles, etc. if they existed

    log "INFO" "No cluster-wide resources to clean up for semantic router"
}

# Function to cleanup port forwarding
cleanup_port_forwarding() {
    log "INFO" "Cleaning up port forwarding processes..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would kill port forwarding processes for namespace: $NAMESPACE"
        return 0
    fi

    # Kill any port forwarding processes for this namespace
    local pf_pids=$(pgrep -f "oc port-forward.*$NAMESPACE" 2>/dev/null || true)
    if [[ -n "$pf_pids" ]]; then
        log "INFO" "Found port forwarding processes: $pf_pids"
        pkill -f "oc port-forward.*$NAMESPACE" || true
        sleep 2

        # Verify they're gone
        local remaining_pids=$(pgrep -f "oc port-forward.*$NAMESPACE" 2>/dev/null || true)
        if [[ -z "$remaining_pids" ]]; then
            log "SUCCESS" "Port forwarding processes terminated"
        else
            log "WARN" "Some port forwarding processes may still be running: $remaining_pids"
        fi
    else
        log "INFO" "No port forwarding processes found for namespace $NAMESPACE"
    fi

    # Clean up PID file if it exists
    if [[ -f "/tmp/semantic-router-port-forward.pid" ]]; then
        local saved_pid=$(cat /tmp/semantic-router-port-forward.pid 2>/dev/null | grep -o '^[0-9]*' || true)
        if [[ -n "$saved_pid" ]]; then
            log "INFO" "Cleaning up saved PID file (PID: $saved_pid)"
            kill "$saved_pid" 2>/dev/null || true
        fi
        rm -f /tmp/semantic-router-port-forward.pid
        log "INFO" "Removed PID file"
    fi

    log "SUCCESS" "Port forwarding cleanup completed"
}

# Function to verify cleanup completion
verify_cleanup() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would verify cleanup completion"
        return 0
    fi

    log "INFO" "Verifying cleanup completion..."

    case "$CLEANUP_LEVEL" in
        "deployment")
            if check_namespace_exists; then
                log "INFO" "Namespace $NAMESPACE still exists (as expected for deployment-level cleanup)"
                local remaining_resources=$(oc get all -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
                if [[ "$remaining_resources" -eq 0 ]]; then
                    log "SUCCESS" "All deployment resources have been removed"
                else
                    log "INFO" "Some resources remain in namespace (may include PVCs or other preserved resources)"
                fi
            else
                log "WARN" "Namespace was also deleted (unexpected for deployment-level cleanup)"
            fi
            ;;
        "namespace"|"all")
            if check_namespace_exists; then
                log "WARN" "Namespace $NAMESPACE still exists (deletion may still be in progress)"
            else
                log "SUCCESS" "Namespace $NAMESPACE has been completely removed"
            fi
            ;;
    esac
}

# Main function
main() {
    log "INFO" "Starting vLLM Semantic Router OpenShift cleanup"

    parse_args "$@"
    validate_prerequisites
    login_openshift

    show_current_resources
    confirm_cleanup

    # Clean up port forwarding first (before deleting resources)
    cleanup_port_forwarding

    case "$CLEANUP_LEVEL" in
        "deployment")
            cleanup_deployment
            ;;
        "namespace")
            cleanup_namespace
            ;;
        "all")
            cleanup_namespace
            cleanup_cluster_wide
            ;;
    esac

    verify_cleanup
    log "SUCCESS" "Cleanup completed successfully!"

    if [[ "$CLEANUP_LEVEL" == "namespace" || "$CLEANUP_LEVEL" == "all" ]]; then
        echo ""
        log "INFO" "To redeploy the semantic router, simply run:"
        log "INFO" "  ./deploy-to-openshift.sh"
        log "INFO" ""
        log "INFO" "The deploy script will auto-detect your OpenShift server and use your existing login."
    fi
}

# Run main function with all arguments
main "$@"