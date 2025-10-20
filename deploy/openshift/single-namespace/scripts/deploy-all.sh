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

# Step 3: Create ConfigMaps
echo ""
echo "Step 3: Creating ConfigMaps..."
oc apply -f "$BASE_DIR/configmap-router.yaml"
oc apply -f "$BASE_DIR/configmap-envoy.yaml"

# Step 4: Create Services
echo ""
echo "Step 4: Creating Services..."
oc apply -f "$BASE_DIR/services.yaml"

# Step 5: Create ImageStreams
echo ""
echo "Step 5: Creating ImageStreams..."
oc apply -f "$BASE_DIR/imagestreams.yaml"

# Step 6: Create BuildConfigs
echo ""
echo "Step 6: Creating BuildConfigs..."
oc apply -f "$BASE_DIR/buildconfig-router.yaml"
oc apply -f "$BASE_DIR/buildconfig-llm-katan.yaml"

# Step 7: Start builds
echo ""
echo "Step 7: Starting container image builds..."
echo "This will take several minutes. Builds will run in the background."
echo ""

# Start semantic-router build from local directory
echo "Starting semantic-router build from local source..."
cd /home/szedan/semantic-router
oc start-build semantic-router --from-dir=. --follow -n $NAMESPACE &
ROUTER_BUILD_PID=$!

# Start llm-katan build from GitHub
echo "Starting llm-katan build from GitHub..."
oc start-build llm-katan -n $NAMESPACE &
KATAN_BUILD_PID=$!

# Wait for builds to complete
echo ""
echo "Waiting for builds to complete..."
echo "Router build PID: $ROUTER_BUILD_PID"
echo "Katan build PID: $KATAN_BUILD_PID"
echo ""
echo "You can monitor build progress in another terminal:"
echo "  oc logs -f bc/semantic-router -n $NAMESPACE"
echo "  oc logs -f bc/llm-katan -n $NAMESPACE"
echo ""

# Wait for the router build (we followed it)
wait $ROUTER_BUILD_PID || echo "Router build finished (check status with: oc get builds -n $NAMESPACE)"

# Check build status
echo ""
echo "Checking build status..."
oc get builds -n $NAMESPACE

# Step 8: Deploy applications
echo ""
echo "Step 8: Deploying applications..."
echo "Note: Deployments will wait for images to be available"
oc apply -f "$BASE_DIR/deployment-router.yaml"
oc apply -f "$BASE_DIR/deployment-model-a.yaml"
oc apply -f "$BASE_DIR/deployment-model-b.yaml"

# Step 9: Monitor deployment
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
echo "   oc logs -f build/semantic-router-1 -n $NAMESPACE"
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
