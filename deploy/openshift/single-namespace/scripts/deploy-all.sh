#!/bin/bash

# Script to deploy vLLM Semantic Router in single-namespace architecture
# All components (router, model-a, model-b) run in separate pods within the same namespace

set -e

NAMESPACE="vllm-semantic-router"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================="
echo "Deploying vLLM Semantic Router"
echo "Namespace: $NAMESPACE"
echo "Architecture: Single-namespace (separate pods)"
echo "========================================="

# Step 1: Create namespace
echo ""
echo "Step 1: Creating namespace..."
oc apply -f "$BASE_DIR/namespace.yaml"

# Wait for namespace to be ready
echo "Waiting for namespace to be ready..."
sleep 2

# Step 2: Create PVCs
echo ""
echo "Step 2: Creating PersistentVolumeClaims..."
oc apply -f "$BASE_DIR/pvcs.yaml"

# Step 3: Create Services (BEFORE ConfigMaps to get IPs)
echo ""
echo "Step 3: Creating Services..."
oc apply -f "$BASE_DIR/services.yaml"

# Wait for services to get ClusterIPs assigned
echo "Waiting for services to get ClusterIP addresses..."
sleep 3

# Step 3.1: Get dynamically assigned ClusterIP addresses
echo ""
echo "Step 3.1: Getting service ClusterIP addresses..."
MODEL_A_IP=$(oc get svc model-a -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
MODEL_B_IP=$(oc get svc model-b -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')

if [ -z "$MODEL_A_IP" ] || [ -z "$MODEL_B_IP" ]; then
    echo "ERROR: Failed to get service ClusterIP addresses"
    echo "Model-A IP: $MODEL_A_IP"
    echo "Model-B IP: $MODEL_B_IP"
    exit 1
fi

echo "  Model-A ClusterIP: $MODEL_A_IP"
echo "  Model-B ClusterIP: $MODEL_B_IP"

# Step 3.2: Update ConfigMap with actual ClusterIP addresses
echo ""
echo "Step 3.2: Updating router ConfigMap with ClusterIP addresses..."

# Create temporary copy of router config with updated IPs
TEMP_CONFIG="/tmp/configmap-router-updated.yaml"
cp "$BASE_DIR/configmap-router.yaml" "$TEMP_CONFIG"

# Replace the hardcoded IPs with actual ClusterIPs
# First occurrence (model-a-endpoint)
sed -i "0,/address: \"[0-9]\+\.[0-9]\+\.[0-9]\+\.[0-9]\+\"/s//address: \"$MODEL_A_IP\"/" "$TEMP_CONFIG"
# Second occurrence (model-b-endpoint)
sed -i "0,/address: \"[0-9]\+\.[0-9]\+\.[0-9]\+\.[0-9]\+\"/!s/address: \"[0-9]\+\.[0-9]\+\.[0-9]\+\.[0-9]\+\"/address: \"$MODEL_B_IP\"/" "$TEMP_CONFIG"

echo "  Updated model-a-endpoint address to: $MODEL_A_IP"
echo "  Updated model-b-endpoint address to: $MODEL_B_IP"

# Step 3.3: Apply updated ConfigMaps
echo ""
echo "Step 3.3: Creating ConfigMaps with updated IPs..."
oc apply -f "$TEMP_CONFIG"
oc apply -f "$BASE_DIR/configmap-envoy.yaml"

# Clean up temp file
rm -f "$TEMP_CONFIG"

# Step 5: Create ImageStreams
echo ""
echo "Step 5: Creating ImageStreams..."
oc apply -f "$BASE_DIR/imagestreams.yaml"

# Step 6: Create BuildConfigs
echo ""
echo "Step 6: Creating BuildConfigs..."
# Note: semantic-router uses pre-built image from ghcr.io, no build needed
oc apply -f "$BASE_DIR/buildconfig-llm-katan.yaml"

# Step 7: Start llm-katan build
echo ""
echo "Step 7: Building llm-katan image..."
echo "This will take several minutes (~5-10 min)."
echo ""

# Start llm-katan build (inline Dockerfile, no upload needed)
echo "Starting llm-katan build..."
oc start-build llm-katan -n $NAMESPACE --follow

# Check build status
echo ""
echo "Checking build status..."
oc get builds -n $NAMESPACE

# Step 8: Check GPU availability
echo ""
echo "Step 8: Checking GPU node availability..."
if bash "$SCRIPT_DIR/check-gpu-status.sh" "$NAMESPACE"; then
    echo "✓ GPU nodes are ready"
else
    echo "⚠️  WARNING: GPU nodes not ready. Model pods may remain in Pending state."
    read -p "Continue anyway? (yes/no): " CONTINUE
    if [ "$CONTINUE" != "yes" ]; then
        echo "Deployment cancelled. Fix GPU nodes and run again."
        exit 1
    fi
fi

# Step 9: Deploy applications
echo ""
echo "Step 9: Deploying applications..."
echo "Note: Deployments will wait for images to be available"
oc apply -f "$BASE_DIR/deployment-router.yaml"
oc apply -f "$BASE_DIR/deployment-model-a.yaml"
oc apply -f "$BASE_DIR/deployment-model-b.yaml"

# Step 9.1: Create route for external access
echo ""
echo "Step 9.1: Creating external route..."
oc apply -f "$BASE_DIR/route.yaml"

# Step 10: Monitor deployment
echo ""
echo "========================================="
echo "Deployment initiated!"
echo "========================================="
echo ""
echo "Monitoring deployment status..."
echo ""

# Function to check pod status
check_pods() {
    echo "Current pod status:"
    oc get pods -n $NAMESPACE -o wide
    echo ""
}

# Monitor for 2 minutes
for i in {1..8}; do
    check_pods
    echo "Waiting for pods to become ready... (check $i/8)"
    sleep 15
done

# Final status
echo ""
echo "========================================="
echo "Deployment Summary"
echo "========================================="
echo ""

echo "Namespace:"
oc get namespace $NAMESPACE
echo ""

echo "PVCs:"
oc get pvc -n $NAMESPACE
echo ""

echo "ConfigMaps:"
oc get configmap -n $NAMESPACE
echo ""

echo "Services:"
oc get svc -n $NAMESPACE
echo ""

echo "ImageStreams:"
oc get imagestream -n $NAMESPACE
echo ""

echo "BuildConfigs:"
oc get bc -n $NAMESPACE
echo ""

echo "Builds:"
oc get builds -n $NAMESPACE
echo ""

echo "Deployments:"
oc get deployment -n $NAMESPACE
echo ""

echo "Pods:"
oc get pods -n $NAMESPACE -o wide
echo ""

echo "========================================="
echo "Next Steps"
echo "========================================="
echo ""
echo "1. Check pod logs:"
echo "   oc logs -f deployment/semantic-router -c semantic-router -n $NAMESPACE"
echo "   oc logs -f deployment/model-a -n $NAMESPACE"
echo "   oc logs -f deployment/model-b -n $NAMESPACE"
echo ""
echo "2. Check build status:"
echo "   oc get builds -n $NAMESPACE"
echo "   oc logs -f build/llm-katan-1 -n $NAMESPACE"
echo ""
echo "3. Expose the service (if needed):"
echo "   oc expose svc/envoy-proxy -n $NAMESPACE"
echo ""
echo "4. Get the route URL:"
echo "   oc get route -n $NAMESPACE"
echo ""
echo "5. Test the deployment:"
echo "   curl -X POST http://\$(oc get route envoy-proxy -n $NAMESPACE -o jsonpath='{.spec.host}')/v1/chat/completions \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"model\": \"auto\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'"
echo ""
