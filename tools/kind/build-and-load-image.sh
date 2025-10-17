#!/bin/bash
# Build and load semantic-router image locally into Kind cluster
# This is the most reliable solution for China mainland network issues

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

CLUSTER_NAME="semantic-router-cluster"
IMAGE_NAME="semantic-router-extproc"
IMAGE_TAG="local"

echo -e "${GREEN}=== Building and Loading Semantic Router Image into Kind ===${NC}"
echo ""

# Check if Kind cluster exists
if ! kind get clusters 2>/dev/null | grep -q "$CLUSTER_NAME"; then
    echo -e "${RED}Error: Cluster '${CLUSTER_NAME}' does not exist.${NC}"
    echo -e "${YELLOW}Please create it first using:${NC}"
    echo -e "  kind create cluster --name ${CLUSTER_NAME} --config tools/kind/kind-config.yaml"
    exit 1
fi

echo -e "${GREEN}Step 1: Building Docker image locally...${NC}"
echo -e "${YELLOW}This will use your local VPN connection${NC}"
echo ""

# Build the image locally
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -f Dockerfile.extproc .

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to build Docker image${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Step 2: Loading image into Kind cluster...${NC}"
echo ""

# Load the image into Kind
kind load docker-image ${IMAGE_NAME}:${IMAGE_TAG} --name ${CLUSTER_NAME}

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to load image into Kind cluster${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Step 3: Updating Kubernetes manifests...${NC}"
echo ""

# Update the kustomization.yaml to use the local image
cat > deploy/kubernetes/kustomization.yaml.new << EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

metadata:
  name: semantic-router

resources:
- namespace.yaml
- pvc.yaml
- deployment.yaml
- service.yaml

# Generate ConfigMap
configMapGenerator:
- name: semantic-router-config
  files:
  - config.yaml
  - tools_db.json

# Namespace for all resources
namespace: vllm-semantic-router-system

images:
- name: ghcr.io/vllm-project/semantic-router/extproc
  newName: ${IMAGE_NAME}
  newTag: ${IMAGE_TAG}

EOF

mv deploy/kubernetes/kustomization.yaml.new deploy/kubernetes/kustomization.yaml

echo -e "${GREEN}Updated kustomization.yaml to use local image${NC}"
echo ""

echo -e "${GREEN}Step 4: Applying changes to cluster...${NC}"
echo ""

# Delete the existing deployment if it exists
kubectl delete deployment semantic-router -n vllm-semantic-router-system --ignore-not-found=true

# Wait a moment for cleanup
sleep 5

# Reapply the manifests
kubectl apply -k deploy/kubernetes/

echo ""
echo -e "${GREEN}=== Setup Complete! ===${NC}"
echo ""
echo -e "${YELLOW}Monitor the deployment with:${NC}"
echo -e "  ${GREEN}kubectl get pods -n vllm-semantic-router-system -w${NC}"
echo ""
echo -e "${YELLOW}Check logs with:${NC}"
echo -e "  ${GREEN}kubectl logs -f deployment/semantic-router -n vllm-semantic-router-system${NC}"
echo ""
