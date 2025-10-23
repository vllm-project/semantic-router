#!/bin/bash
set -e

# Script to build and deploy custom dashboard image to OpenShift
# This builds the dashboard with OpenWebUI integration patches for demo purposes

NAMESPACE="vllm-semantic-router-system"
IMAGE_NAME="dashboard-custom"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Building Custom Dashboard Image"
echo "=========================================="

# Get OpenShift internal registry
REGISTRY=$(oc get route default-route -n openshift-image-registry -o jsonpath='{.spec.host}' 2>/dev/null)
if [ -z "$REGISTRY" ]; then
    echo "Error: Could not find OpenShift internal registry route"
    echo "Creating registry route..."
    oc patch configs.imageregistry.operator.openshift.io/cluster --type merge -p '{"spec":{"defaultRoute":true}}'
    echo "Waiting for route to be created..."
    sleep 5
    REGISTRY=$(oc get route default-route -n openshift-image-registry -o jsonpath='{.spec.host}')
fi

echo "Registry: $REGISTRY"
echo "Namespace: $NAMESPACE"

# Login to registry
echo ""
echo "Logging in to OpenShift registry..."
TOKEN=$(oc whoami -t)
docker login -u "$(oc whoami)" -p "$TOKEN" "$REGISTRY"

# Prepare build directory
echo ""
echo "Preparing build directory with patched files..."
BUILD_DIR="/tmp/dashboard-build-$$"
mkdir -p $BUILD_DIR
cp -r dashboard $BUILD_DIR/

# Apply patches
echo "Applying OpenWebUI integration patches..."
if [ -f "$SCRIPT_DIR/main.go.patch" ]; then
    cp "$SCRIPT_DIR/main.go.patch" "$BUILD_DIR/dashboard/backend/main.go"
    echo "  ✓ Applied main.go patch (OpenWebUI proxy + auth)"
fi

if [ -f "$SCRIPT_DIR/PlaygroundPage.tsx.patch" ]; then
    cp "$SCRIPT_DIR/PlaygroundPage.tsx.patch" "$BUILD_DIR/dashboard/frontend/src/pages/PlaygroundPage.tsx"
    echo "  ✓ Applied PlaygroundPage.tsx patch (proxy path fix)"
fi

# Create Dockerfile
echo ""
echo "Creating Dockerfile..."
cat > $BUILD_DIR/dashboard/Dockerfile.custom <<'DOCKERFILE_EOF'
# Build frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Build backend
FROM golang:1.21-alpine AS backend-builder
WORKDIR /app/backend
COPY backend/go.* ./
RUN go mod download
COPY backend/ ./
RUN CGO_ENABLED=0 GOOS=linux go build -o dashboard-server .

# Final image
FROM alpine:3.18
RUN apk add --no-cache ca-certificates
WORKDIR /app
COPY --from=backend-builder /app/backend/dashboard-server .
COPY --from=frontend-builder /app/frontend/dist ./frontend
ENV DASHBOARD_STATIC_DIR=./frontend
EXPOSE 8700
CMD ["./dashboard-server"]
DOCKERFILE_EOF

# Ensure imagestream exists
echo ""
if ! oc get imagestream $IMAGE_NAME -n $NAMESPACE &>/dev/null; then
    echo "Creating imagestream $IMAGE_NAME..."
    oc create imagestream $IMAGE_NAME -n $NAMESPACE
else
    echo "Imagestream $IMAGE_NAME already exists"
fi

# Build image
echo ""
echo "Building docker image for linux/amd64 with no cache..."
cd $BUILD_DIR/dashboard
docker buildx build --no-cache --platform linux/amd64 -f Dockerfile.custom -t "$REGISTRY/$NAMESPACE/$IMAGE_NAME:latest" --load .

echo ""
echo "Pushing image to registry..."
docker push "$REGISTRY/$NAMESPACE/$IMAGE_NAME:latest"

# Cleanup
echo ""
echo "Cleaning up build directory..."
rm -rf $BUILD_DIR

# Apply deployment configuration if it doesn't exist
echo ""
if ! oc get deployment dashboard -n $NAMESPACE &>/dev/null; then
    echo "Dashboard deployment not found. Applying configuration..."
    oc apply -f "$SCRIPT_DIR/dashboard-deployment.yaml"
else
    echo "Dashboard deployment already exists. Updating image..."
    oc set image deployment/dashboard dashboard=image-registry.openshift-image-registry.svc:5000/$NAMESPACE/$IMAGE_NAME:latest -n $NAMESPACE
fi

echo ""
echo "Waiting for deployment to roll out..."
oc rollout status deployment/dashboard -n $NAMESPACE --timeout=5m

echo ""
echo "=========================================="
echo "Custom Dashboard Deployed Successfully!"
echo "=========================================="
echo ""
echo "Access the dashboard at:"
oc get route dashboard -n $NAMESPACE -o jsonpath='https://{.spec.host}'
echo ""
echo ""
echo "Patches applied:"
echo "  - OpenWebUI proxy path fix (PlaygroundPage.tsx)"
echo "  - OpenWebUI static assets proxying (main.go)"
echo "  - Smart API routing for OpenWebUI (main.go)"
echo "  - Authorization header forwarding (main.go)"
echo ""
