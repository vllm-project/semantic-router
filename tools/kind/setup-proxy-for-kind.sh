#!/bin/bash
# Setup script to configure Kind cluster with proxy support for China mainland

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PROXY_HOST="host.docker.internal"
PROXY_PORT="7897"

echo -e "${GREEN}=== Kind Cluster Proxy Setup for China Mainland ===${NC}"
echo ""

# Get the host IP address that Kind containers can access
HOST_IP=$(ip route | grep default | awk '{print $3}' | head -n1)
if [ -z "$HOST_IP" ]; then
    HOST_IP=$(hostname -I | awk '{print $1}')
fi

echo -e "${YELLOW}Detected host IP: ${HOST_IP}${NC}"
echo -e "${YELLOW}Using proxy: http://${HOST_IP}:${PROXY_PORT}${NC}"
echo ""

# Function to configure Docker daemon in Kind nodes
configure_kind_node_proxy() {
    local node_name=$1
    echo -e "${GREEN}Configuring proxy for node: ${node_name}${NC}"
    
    # Create Docker daemon config with proxy
    docker exec "${node_name}" bash -c "cat > /etc/systemd/system/docker.service.d/http-proxy.conf << EOF
[Service]
Environment=\"HTTP_PROXY=http://${HOST_IP}:${PROXY_PORT}\"
Environment=\"HTTPS_PROXY=http://${HOST_IP}:${PROXY_PORT}\"
Environment=\"NO_PROXY=localhost,127.0.0.1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,.svc,.svc.cluster.local\"
EOF"
    
    # Reload and restart containerd
    docker exec "${node_name}" bash -c "systemctl daemon-reload"
    docker exec "${node_name}" bash -c "systemctl restart containerd || true"
    
    echo -e "${GREEN}Proxy configured for ${node_name}${NC}"
}

# Check if cluster exists
if ! kind get clusters 2>/dev/null | grep -q "semantic-router-cluster"; then
    echo -e "${YELLOW}Cluster 'semantic-router-cluster' does not exist.${NC}"
    echo -e "${YELLOW}Please create it first using: kind create cluster --config tools/kind/kind-config.yaml${NC}"
    exit 1
fi

echo -e "${GREEN}Step 1: Configuring Docker proxy in Kind nodes...${NC}"
echo ""

# Get all Kind nodes
NODES=$(kind get nodes --name semantic-router-cluster)

for node in $NODES; do
    configure_kind_node_proxy "$node"
done

echo ""
echo -e "${GREEN}Step 2: Alternative - Pre-load image from local Docker${NC}"
echo ""
echo -e "${YELLOW}If you have built the image locally, you can load it into Kind:${NC}"
echo -e "${YELLOW}  docker pull ghcr.io/vllm-project/semantic-router/extproc:latest${NC}"
echo -e "${YELLOW}  kind load docker-image ghcr.io/vllm-project/semantic-router/extproc:latest --name semantic-router-cluster${NC}"
echo ""

echo -e "${GREEN}Step 3: Use mirror registry (Recommended for China)${NC}"
echo ""
echo -e "${YELLOW}Consider using a mirror registry. Here's how:${NC}"
echo ""
echo -e "1. Edit deploy/kubernetes/kustomization.yaml and change the image:"
echo -e "   ${GREEN}images:${NC}"
echo -e "   ${GREEN}- name: ghcr.io/vllm-project/semantic-router/extproc${NC}"
echo -e "   ${GREEN}  newName: <your-mirror-registry>/semantic-router/extproc${NC}"
echo -e "   ${GREEN}  newTag: latest${NC}"
echo ""
echo -e "2. Or build and load locally:"
echo -e "   ${GREEN}docker build -t semantic-router:local -f Dockerfile.extproc .${NC}"
echo -e "   ${GREEN}kind load docker-image semantic-router:local --name semantic-router-cluster${NC}"
echo -e "   ${GREEN}# Then update deployment.yaml to use semantic-router:local${NC}"
echo ""

echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo -e "${YELLOW}Note: The proxy configuration requires:${NC}"
echo -e "  1. Your VPN/proxy running on localhost:${PROXY_PORT}"
echo -e "  2. Proxy accessible from ${HOST_IP}:${PROXY_PORT}"
echo -e "  3. You may need to restart your pods after this change"
echo ""
echo -e "${YELLOW}To restart the deployment:${NC}"
echo -e "  ${GREEN}kubectl rollout restart deployment/semantic-router -n vllm-semantic-router-system${NC}"
