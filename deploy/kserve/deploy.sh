#!/bin/bash
# Semantic Router KServe Deployment Helper Script
# This script simplifies deploying the semantic router to work with OpenShift AI KServe LLMInferenceServices
# It handles variable substitution, validation, and deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
NAMESPACE=""
INFERENCESERVICE_NAME=""
MODEL_NAME=""
SIMULATOR=false
SIM_INFERENCESERVICE_A="model-a"
SIM_INFERENCESERVICE_B="model-b"
MODEL_NAME_A="Model-A"
MODEL_NAME_B="Model-B"
STORAGE_CLASS=""
MODELS_PVC_SIZE="10Gi"
CACHE_PVC_SIZE="5Gi"
# Embedding model for semantic caching and tools similarity
# Common options from sentence-transformers:
#   - all-MiniLM-L12-v2 (default, balanced speed/quality)
#   - all-mpnet-base-v2 (higher quality, slower)
#   - all-MiniLM-L6-v2 (faster, lower quality)
#   - paraphrase-multilingual-MiniLM-L12-v2 (multilingual)
EMBEDDING_MODEL="all-MiniLM-L12-v2"
DRY_RUN=false
SKIP_VALIDATION=false

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy vLLM Semantic Router for OpenShift AI KServe LLMInferenceServices

Required Options:
  -n, --namespace NAMESPACE          OpenShift namespace to deploy to

Required Options (non-simulator):
  -i, --inferenceservice NAME        Name of the KServe LLMInferenceService
  -m, --model MODEL_NAME             Model name as reported by the LLMInferenceService

Optional:
  --simulator                        Use KServe simulator with Model-A and Model-B
  --sim-inferenceservice-a NAME      Simulator LLMInferenceService A name (default: model-a)
  --sim-inferenceservice-b NAME      Simulator LLMInferenceService B name (default: model-b)
  --sim-model-a NAME                 Simulator model name for A (default: Model-A)
  --sim-model-b NAME                 Simulator model name for B (default: Model-B)
  -s, --storage-class CLASS          StorageClass for PVCs (default: cluster default)
  --models-pvc-size SIZE             Size for models PVC (default: 10Gi)
  --cache-pvc-size SIZE              Size for cache PVC (default: 5Gi)
  --embedding-model MODEL            BERT embedding model (default: all-MiniLM-L12-v2)
  --dry-run                          Generate manifests without applying
  --skip-validation                  Skip pre-deployment validation
  -h, --help                         Show this help message

Examples:
  # Deploy to namespace 'semantic' with granite32-8b model
  $0 -n semantic -i granite32-8b -m granite32-8b

  # Deploy with KServe simulator (Model-A / Model-B)
  $0 -n semantic --simulator

  # Deploy with custom storage class and embedding model
  $0 -n myproject -i llama3-70b -m llama3-70b -s gp3-csi --embedding-model all-mpnet-base-v2

  # Dry run to see what will be deployed
  $0 -n semantic -i granite32-8b -m granite32-8b --dry-run

Prerequisites:
  - OpenShift CLI (oc) installed and logged in
  - KServe installed
  - LLMInferenceService already deployed
  - Cluster admin or namespace admin permissions

For more information, see README.md
EOF
    exit 1
}

# Function to substitute variables in a file
substitute_vars() {
    local input_file="$1"
    local output_file="$2"

    sed -e "s|{{NAMESPACE}}|$NAMESPACE|g" \
        -e "s|{{INFERENCESERVICE_NAME}}|$INFERENCESERVICE_NAME|g" \
        -e "s|{{INFERENCESERVICE_NAME_A}}|$SIM_INFERENCESERVICE_A|g" \
        -e "s|{{INFERENCESERVICE_NAME_B}}|$SIM_INFERENCESERVICE_B|g" \
        -e "s|{{MODEL_NAME}}|$MODEL_NAME|g" \
        -e "s|{{MODEL_NAME_A}}|$MODEL_NAME_A|g" \
        -e "s|{{MODEL_NAME_B}}|$MODEL_NAME_B|g" \
        -e "s|{{EMBEDDING_MODEL}}|$EMBEDDING_MODEL|g" \
        -e "s|{{PREDICTOR_SERVICE_IP}}|${PREDICTOR_SERVICE_IP:-10.0.0.1}|g" \
        -e "s|{{PREDICTOR_SERVICE_IP_A}}|${PREDICTOR_SERVICE_IP_A:-10.0.0.1}|g" \
        -e "s|{{PREDICTOR_SERVICE_IP_B}}|${PREDICTOR_SERVICE_IP_B:-10.0.0.1}|g" \
        -e "s|{{MODELS_PVC_SIZE}}|$MODELS_PVC_SIZE|g" \
        -e "s|{{CACHE_PVC_SIZE}}|$CACHE_PVC_SIZE|g" \
        "$input_file" > "$output_file"

    # Handle storage class (optional)
    if [ -n "$STORAGE_CLASS" ]; then
        sed -i.bak "s/# storageClassName:.*/storageClassName: $STORAGE_CLASS/g" "$output_file"
        rm -f "${output_file}.bak"
    fi
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -i|--inferenceservice)
            INFERENCESERVICE_NAME="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --simulator)
            SIMULATOR=true
            shift
            ;;
        --sim-inferenceservice-a)
            SIM_INFERENCESERVICE_A="$2"
            shift 2
            ;;
        --sim-inferenceservice-b)
            SIM_INFERENCESERVICE_B="$2"
            shift 2
            ;;
        --sim-model-a)
            MODEL_NAME_A="$2"
            shift 2
            ;;
        --sim-model-b)
            MODEL_NAME_B="$2"
            shift 2
            ;;
        -s|--storage-class)
            STORAGE_CLASS="$2"
            shift 2
            ;;
        --models-pvc-size)
            MODELS_PVC_SIZE="$2"
            shift 2
            ;;
        --cache-pvc-size)
            CACHE_PVC_SIZE="$2"
            shift 2
            ;;
        --embedding-model)
            EMBEDDING_MODEL="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$NAMESPACE" ]; then
    echo -e "${RED}Error: Missing required arguments${NC}"
    usage
fi

if [ "$SIMULATOR" = false ]; then
    if [ -z "$INFERENCESERVICE_NAME" ] || [ -z "$MODEL_NAME" ]; then
        echo -e "${RED}Error: Missing required arguments${NC}"
        usage
    fi
fi

TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

# Banner
echo ""
echo "=================================================="
echo "  vLLM Semantic Router - KServe Deployment"
echo "=================================================="
echo ""

# Display configuration
echo -e "${BLUE}Configuration:${NC}"
echo "  Namespace:              $NAMESPACE"
if [ "$SIMULATOR" = true ]; then
    echo "  Simulator Mode:         true"
    echo "  LLMInferenceService A:  $SIM_INFERENCESERVICE_A"
    echo "  LLMInferenceService B:  $SIM_INFERENCESERVICE_B"
    echo "  Model A Name:           $MODEL_NAME_A"
    echo "  Model B Name:           $MODEL_NAME_B"
else
    echo "  Simulator Mode:         false"
    echo "  LLMInferenceService:    $INFERENCESERVICE_NAME"
    echo "  Model Name:             $MODEL_NAME"
fi
echo "  Embedding Model:        $EMBEDDING_MODEL"
echo "  Storage Class:          ${STORAGE_CLASS:-<cluster default>}"
echo "  Models PVC Size:        $MODELS_PVC_SIZE"
echo "  Cache PVC Size:         $CACHE_PVC_SIZE"
echo "  Dry Run:                $DRY_RUN"
echo ""

# Pre-deployment validation
if [ "$SKIP_VALIDATION" = false ]; then
    echo -e "${BLUE}Step 1: Validating prerequisites...${NC}"

    # Check oc command
    if ! command -v oc &> /dev/null; then
        echo -e "${RED}✗ Error: 'oc' command not found. Please install OpenShift CLI.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓${NC} OpenShift CLI found"

    # Check if logged in
    if ! oc whoami &> /dev/null; then
        echo -e "${RED}✗ Error: Not logged in to OpenShift. Run 'oc login' first.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓${NC} Logged in as $(oc whoami)"

    # Check if namespace exists
    if ! oc get namespace "$NAMESPACE" &> /dev/null; then
        echo -e "${YELLOW}⚠ Warning: Namespace '$NAMESPACE' does not exist.${NC}"
        read -p "Create namespace? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            oc create namespace "$NAMESPACE"
            echo -e "${GREEN}✓${NC} Created namespace: $NAMESPACE"
        else
            echo -e "${RED}✗ Aborted${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}✓${NC} Namespace exists: $NAMESPACE"
    fi

    validate_inferenceservice() {
        local name="$1"
        if ! oc get llminferenceservice "$name" -n "$NAMESPACE" &> /dev/null; then
            echo -e "${RED}✗ Error: LLMInferenceService '$name' not found in namespace '$NAMESPACE'${NC}"
            echo "  Please deploy your LLMInferenceService first."
            exit 1
        fi
        echo -e "${GREEN}✓${NC} LLMInferenceService exists: $name"

        local isvc_ready
        isvc_ready=$(oc get llminferenceservice "$name" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}')
        if [ "$isvc_ready" != "True" ]; then
            echo -e "${YELLOW}⚠ Warning: LLMInferenceService '$name' is not ready yet${NC}"
            echo "  Status: $(oc get llminferenceservice "$name" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Ready")].message}')"
            read -p "Continue anyway? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        else
            echo -e "${GREEN}✓${NC} LLMInferenceService is ready"
        fi

        local predictor_url
        predictor_url=$(oc get llminferenceservice "$name" -n "$NAMESPACE" -o jsonpath='{.status.url}' 2>/dev/null || echo "")
        if [ -n "$predictor_url" ]; then
            echo -e "${GREEN}✓${NC} Predictor URL: $predictor_url"
        fi
    }

    detect_predictor_selector_label() {
        local name="$1"
        local pods

        pods=$(oc get pods -n "$NAMESPACE" -l "serving.kserve.io/llm-inferenceservice=${name}" --no-headers 2>/dev/null || true)
        if [[ -n "$pods" ]]; then
            echo "serving.kserve.io/llm-inferenceservice"
            return
        fi

        pods=$(oc get pods -n "$NAMESPACE" -l "app.kubernetes.io/name=${name}" --no-headers 2>/dev/null || true)
        if [[ -n "$pods" ]]; then
            echo "app.kubernetes.io/name"
            return
        fi

        pods=$(oc get pods -n "$NAMESPACE" -l "serving.kserve.io/inferenceservice=${name}" --no-headers 2>/dev/null || true)
        if [[ -n "$pods" ]]; then
            echo "serving.kserve.io/inferenceservice"
            return
        fi

        pods=$(oc get pods -n "$NAMESPACE" -l "app=${name}" --no-headers 2>/dev/null || true)
        if [[ -n "$pods" ]]; then
            echo "app"
            return
        fi

        echo "serving.kserve.io/llm-inferenceservice"
    }

    create_stable_service() {
        local name="$1"
        local output
        local selector_label
        local selector_value

        echo "Creating stable ClusterIP service for predictor: $name" >&2
        selector_label=$(detect_predictor_selector_label "$name")
        selector_value="$name"
        if [ -f "$SCRIPT_DIR/service-predictor-stable.yaml" ]; then
            output="$TEMP_DIR/service-predictor-stable-${name}.yaml.tmp"
            sed -e "s|{{INFERENCESERVICE_NAME}}|$name|g" \
                -e "s|{{NAMESPACE}}|$NAMESPACE|g" \
                -e "s|{{PREDICTOR_SELECTOR_LABEL}}|$selector_label|g" \
                -e "s|{{PREDICTOR_SELECTOR_VALUE}}|$selector_value|g" \
                "$SCRIPT_DIR/service-predictor-stable.yaml" > "$output"
            oc apply -f "$output" -n "$NAMESPACE" > /dev/null 2>&1
            # Ensure selector is replaced (not merged) in case a previous selector exists.
            oc patch svc "${name}-predictor-stable" -n "$NAMESPACE" --type='json' \
                -p="[{'op':'replace','path':'/spec/selector','value':{'${selector_label}':'${selector_value}'}}]" \
                > /dev/null 2>&1 || true
        else
            cat <<EOF | oc apply -f - -n "$NAMESPACE" > /dev/null 2>&1
apiVersion: v1
kind: Service
metadata:
  name: ${name}-predictor-stable
  labels:
    app: ${name}
    component: predictor-stable
    managed-by: semantic-router-deploy
  annotations:
    description: "Stable ClusterIP service for semantic router (KServe headless service doesn't provide stable IP)"
spec:
  type: ClusterIP
  selector:
    ${selector_label}: ${selector_value}
  ports:
  - name: http
    port: 8000
    targetPort: 8000
    protocol: TCP
EOF
        fi

        local ip
        ip=$(oc get svc "${name}-predictor-stable" -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "")
        if [ -z "$ip" ]; then
            echo -e "${RED}✗ Error: Could not get predictor service ClusterIP for $name${NC}"
            echo "  The stable service was not created properly."
            exit 1
        fi
        echo "$ip"
    }

    if [ "$SIMULATOR" = true ]; then
        validate_inferenceservice "$SIM_INFERENCESERVICE_A"
        validate_inferenceservice "$SIM_INFERENCESERVICE_B"

        PREDICTOR_SERVICE_IP_A=$(create_stable_service "$SIM_INFERENCESERVICE_A")
        echo -e "${GREEN}✓${NC} Predictor service ClusterIP A: $PREDICTOR_SERVICE_IP_A (stable across pod restarts)"
        PREDICTOR_SERVICE_IP_B=$(create_stable_service "$SIM_INFERENCESERVICE_B")
        echo -e "${GREEN}✓${NC} Predictor service ClusterIP B: $PREDICTOR_SERVICE_IP_B (stable across pod restarts)"
    else
        validate_inferenceservice "$INFERENCESERVICE_NAME"

        PREDICTOR_SERVICE_IP=$(create_stable_service "$INFERENCESERVICE_NAME")
        echo -e "${GREEN}✓${NC} Predictor service ClusterIP: $PREDICTOR_SERVICE_IP (stable across pod restarts)"
    fi

    echo ""
else
    # When validation is skipped, we still need to get the service ClusterIPs
    # for the config template substitution
    echo -e "${BLUE}Skipping validation, retrieving existing service IPs...${NC}"

    get_existing_service_ip() {
        local name="$1"
        local ip
        ip=$(oc get svc "${name}-predictor-stable" -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "")
        if [ -z "$ip" ]; then
            echo -e "${YELLOW}⚠ Warning: Could not get ClusterIP for ${name}-predictor-stable${NC}" >&2
            echo "10.0.0.1"  # fallback
        else
            echo "$ip"
        fi
    }

    if [ "$SIMULATOR" = true ]; then
        PREDICTOR_SERVICE_IP_A=$(get_existing_service_ip "$SIM_INFERENCESERVICE_A")
        echo -e "${GREEN}✓${NC} Predictor service ClusterIP A: $PREDICTOR_SERVICE_IP_A"
        PREDICTOR_SERVICE_IP_B=$(get_existing_service_ip "$SIM_INFERENCESERVICE_B")
        echo -e "${GREEN}✓${NC} Predictor service ClusterIP B: $PREDICTOR_SERVICE_IP_B"
    else
        PREDICTOR_SERVICE_IP=$(get_existing_service_ip "$INFERENCESERVICE_NAME")
        echo -e "${GREEN}✓${NC} Predictor service ClusterIP: $PREDICTOR_SERVICE_IP"
    fi

    echo ""
fi

# Generate manifests
echo -e "${BLUE}Step 2: Generating manifests...${NC}"

CONFIGMAP_SRC="$SCRIPT_DIR/configmap-router-config.yaml"
ENVOY_CONFIG_SRC="$SCRIPT_DIR/configmap-envoy-config.yaml"
if [ "$SIMULATOR" = true ]; then
    CONFIGMAP_SRC="$SCRIPT_DIR/configmap-router-config-simulator.yaml"
    ENVOY_CONFIG_SRC="$SCRIPT_DIR/configmap-envoy-config-simulator.yaml"
fi

if [ -f "$CONFIGMAP_SRC" ]; then
    substitute_vars "$CONFIGMAP_SRC" "$TEMP_DIR/configmap-router-config.yaml"
    echo -e "${GREEN}✓${NC} Generated: configmap-router-config.yaml"
else
    echo -e "${YELLOW}⚠ Missing configmap source: $CONFIGMAP_SRC${NC}"
fi

if [ -f "$ENVOY_CONFIG_SRC" ]; then
    substitute_vars "$ENVOY_CONFIG_SRC" "$TEMP_DIR/configmap-envoy-config.yaml"
    echo -e "${GREEN}✓${NC} Generated: configmap-envoy-config.yaml"
else
    echo -e "${YELLOW}⚠ Missing envoy config source: $ENVOY_CONFIG_SRC${NC}"
fi

for file in serviceaccount.yaml pvc.yaml peerauthentication.yaml deployment.yaml service.yaml route.yaml; do
    if [ -f "$SCRIPT_DIR/$file" ]; then
        substitute_vars "$SCRIPT_DIR/$file" "$TEMP_DIR/$file"
        echo -e "${GREEN}✓${NC} Generated: $file"
    else
        echo -e "${YELLOW}⚠ Skipping missing file: $file${NC}"
    fi
done

echo ""

# Dry run - just show what would be deployed
if [ "$DRY_RUN" = true ]; then
    echo -e "${BLUE}Dry run mode - Generated manifests:${NC}"
    echo ""
    for file in "$TEMP_DIR"/*.yaml; do
        echo "--- $(basename "$file") ---"
        cat "$file"
        echo ""
    done

    echo -e "${YELLOW}Dry run complete. No resources were created.${NC}"
    echo "To deploy for real, run without --dry-run flag."
    exit 0
fi

# Deploy manifests
echo -e "${BLUE}Step 3: Deploying to OpenShift...${NC}"

# If we have RWO PVCs and an existing deployment, scale down first to avoid multi-attach.
if oc get deployment semantic-router-kserve -n "$NAMESPACE" &>/dev/null; then
    current_replicas=$(oc get deployment semantic-router-kserve -n "$NAMESPACE" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")
    models_mode=$(oc get pvc semantic-router-models -n "$NAMESPACE" -o jsonpath='{.spec.accessModes[0]}' 2>/dev/null || echo "")
    cache_mode=$(oc get pvc semantic-router-cache -n "$NAMESPACE" -o jsonpath='{.spec.accessModes[0]}' 2>/dev/null || echo "")
    if [[ "$current_replicas" -gt 0 ]] && { [[ "$models_mode" == "ReadWriteOnce" ]] || [[ "$cache_mode" == "ReadWriteOnce" ]]; }; then
        echo "Scaling down semantic-router to avoid RWO PVC multi-attach..."
        oc scale deployment/semantic-router-kserve -n "$NAMESPACE" --replicas=0 >/dev/null 2>&1 || true
        oc wait --for=delete pod -l app=semantic-router -n "$NAMESPACE" --timeout=3m >/dev/null 2>&1 || true
    fi
fi

oc apply -f "$TEMP_DIR/serviceaccount.yaml" -n "$NAMESPACE"
oc apply -f "$TEMP_DIR/pvc.yaml" -n "$NAMESPACE"
oc apply -f "$TEMP_DIR/configmap-router-config.yaml" -n "$NAMESPACE"
oc apply -f "$TEMP_DIR/configmap-envoy-config.yaml" -n "$NAMESPACE"
if oc get crd peerauthentications.security.istio.io &>/dev/null; then
    oc apply -f "$TEMP_DIR/peerauthentication.yaml" -n "$NAMESPACE"
else
    echo "Skipping PeerAuthentication (Istio CRD not found)."
fi
oc apply -f "$TEMP_DIR/deployment.yaml" -n "$NAMESPACE"
oc apply -f "$TEMP_DIR/service.yaml" -n "$NAMESPACE"
oc apply -f "$TEMP_DIR/route.yaml" -n "$NAMESPACE"

echo -e "${GREEN}✓${NC} Resources deployed successfully"
echo ""

# Wait for deployment
echo -e "${BLUE}Step 4: Waiting for deployment to be ready...${NC}"
echo "This may take a few minutes while models are downloaded..."
echo ""

# Monitor pod status
for i in {1..36}; do
    POD_STATUS=$(oc get pods -l app=semantic-router -n "$NAMESPACE" -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "")
    POD_NAME=$(oc get pods -l app=semantic-router -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

    if [ "$POD_STATUS" = "Running" ]; then
        READY=$(oc get pods -l app=semantic-router -n "$NAMESPACE" -o jsonpath='{.items[0].status.containerStatuses[*].ready}' 2>/dev/null || echo "")
        if [[ "$READY" == *"true true"* ]]; then
            echo -e "${GREEN}✓${NC} Pod is ready: $POD_NAME"
            break
        fi
    fi

    # Show init container progress
    INIT_STATUS=$(oc get pods -l app=semantic-router -n "$NAMESPACE" -o jsonpath='{.items[0].status.initContainerStatuses[0].state.running}' 2>/dev/null || echo "")
    if [ -n "$INIT_STATUS" ]; then
        echo "  Initializing... (downloading models)"
    else
        echo "  Waiting for pod... ($i/36)"
    fi

    if [ "$i" -eq 12 ]; then
        echo ""
        echo "  Quick status (init logs):"
        oc logs -l app=semantic-router -n "$NAMESPACE" -c model-downloader --tail=15 2>/dev/null || true
        echo ""
    fi

    sleep 5
done

echo ""

# Check final status
if ! oc get pods -l app=semantic-router -n "$NAMESPACE" -o jsonpath='{.items[0].status.containerStatuses[*].ready}' 2>/dev/null | grep -q "true true"; then
    echo -e "${YELLOW}Warning: Pod may not be fully ready yet${NC}"
    echo "  Check status with: oc get pods -l app=semantic-router -n $NAMESPACE"
    echo "  View logs with: oc logs -l app=semantic-router -c semantic-router -n $NAMESPACE"
    echo "  Init logs with: oc logs -l app=semantic-router -c model-downloader -n $NAMESPACE --tail=30"
fi

echo ""

# Get route URL
ROUTE_URL=$(oc get route semantic-router-kserve -n "$NAMESPACE" -o jsonpath='{.spec.host}' 2>/dev/null || echo "")
if [ -n "$ROUTE_URL" ]; then
    echo -e "${GREEN}✓${NC} External URL: https://$ROUTE_URL"
else
    echo -e "${YELLOW}⚠ Could not determine route URL${NC}"
fi

echo ""
echo "=================================================="
echo "  Deployment Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Set the route:"
echo "   ENVOY_ROUTE=$ROUTE_URL"
echo ""
echo "2. Test model auto-routing:"
echo "   curl -k -X POST https://${ROUTE_URL}/v1/chat/completions \\"
echo "     -H \"Content-Type: application/json\" \\"
echo "     -d '{\"model\":\"auto\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain the elements of a contract under common law and give a simple example.\"}]}'"
echo ""
echo "3. View logs:"
echo "   oc logs -l app=semantic-router -c semantic-router -n $NAMESPACE -f"
echo ""

echo ""
echo "For more information, see: $SCRIPT_DIR/README.md"
echo ""
