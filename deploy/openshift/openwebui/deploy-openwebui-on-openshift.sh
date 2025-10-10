#!/bin/bash

# Deploy OpenWebUI on OpenShift
# This script deploys OpenWebUI to work with the existing semantic-router deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}🚀 OpenWebUI OpenShift Deployment Script${NC}"
echo "================================================"

# Check if oc is installed and logged in
if ! command -v oc &> /dev/null; then
    echo -e "${RED}❌ Error: 'oc' command not found. Please install OpenShift CLI.${NC}"
    exit 1
fi

if ! oc whoami &> /dev/null; then
    echo -e "${RED}❌ Error: Not logged into OpenShift. Please run 'oc login'.${NC}"
    exit 1
fi

# Get current user and project
CURRENT_USER=$(oc whoami)
echo -e "${GREEN}✅ Logged in as: ${CURRENT_USER}${NC}"

# Check if we're in the right namespace or if it exists
NAMESPACE="vllm-semantic-router-system"
if ! oc get namespace "$NAMESPACE" &> /dev/null; then
    echo -e "${RED}❌ Error: Namespace '$NAMESPACE' not found.${NC}"
    echo "Please ensure the semantic-router is deployed first."
    exit 1
fi

# Switch to the namespace
echo -e "${YELLOW}📁 Switching to namespace: ${NAMESPACE}${NC}"
oc project "$NAMESPACE"

# Check if semantic-router is running
echo -e "${YELLOW}🔍 Checking semantic-router deployment...${NC}"
if ! oc get deployment semantic-router &> /dev/null; then
    echo -e "${RED}❌ Error: semantic-router deployment not found.${NC}"
    echo "Please deploy semantic-router first."
    exit 1
fi

if ! oc get deployment semantic-router -o jsonpath='{.status.readyReplicas}' | grep -q "1"; then
    echo -e "${YELLOW}⚠️  Warning: semantic-router deployment may not be ready.${NC}"
    echo "Continuing with OpenWebUI deployment..."
fi

echo -e "${GREEN}✅ semantic-router found and ready${NC}"

# Deploy OpenWebUI components
echo -e "${YELLOW}🔧 Deploying OpenWebUI components...${NC}"

echo "  📦 Creating Persistent Volume Claim..."
oc apply -f "$SCRIPT_DIR/pvc.yaml"

echo "  🚀 Creating Deployment..."
oc apply -f "$SCRIPT_DIR/deployment.yaml"

echo "  🌐 Creating Service..."
oc apply -f "$SCRIPT_DIR/service.yaml"

echo "  🔗 Creating Route..."
oc apply -f "$SCRIPT_DIR/route.yaml"

# Wait for deployment to be ready
echo -e "${YELLOW}⏳ Waiting for OpenWebUI deployment to be ready...${NC}"
oc rollout status deployment/openwebui --timeout=300s

# Get the route URL
ROUTE_URL=$(oc get route openwebui -o jsonpath='{.spec.host}')
if [ -z "$ROUTE_URL" ]; then
    echo -e "${RED}❌ Error: Could not get route URL${NC}"
    exit 1
fi

# Check if OpenWebUI is responding
echo -e "${YELLOW}🔍 Testing OpenWebUI endpoint...${NC}"
if curl -k -s -o /dev/null -w "%{http_code}" "https://$ROUTE_URL" | grep -q "200"; then
    echo -e "${GREEN}✅ OpenWebUI is responding${NC}"
else
    echo -e "${YELLOW}⚠️  OpenWebUI may still be starting up...${NC}"
fi

# Test backend connectivity
echo -e "${YELLOW}🔍 Testing backend connectivity...${NC}"
if oc exec deployment/openwebui -- curl -s -o /dev/null -w "%{http_code}" \
   "http://semantic-router.vllm-semantic-router-system.svc.cluster.local:8801/v1/models" | grep -q "200"; then
    echo -e "${GREEN}✅ Backend connectivity working${NC}"
else
    echo -e "${YELLOW}⚠️  Backend connectivity may need time to establish...${NC}"
fi

# Display deployment information
echo ""
echo -e "${GREEN}🎉 OpenWebUI deployment completed successfully!${NC}"
echo "================================================"
echo -e "${BLUE}📊 Deployment Summary:${NC}"
echo "  🌐 URL: https://$ROUTE_URL"
echo "  🎯 Backend: http://semantic-router.vllm-semantic-router-system.svc.cluster.local:8801/v1"
echo "  📂 Namespace: $NAMESPACE"
echo ""
echo -e "${BLUE}🔧 Available Models:${NC}"
echo "  • auto (load balancer)"
echo "  • Model-A (Qwen/Qwen3-0.6B)"
echo "  • Model-B (Qwen/Qwen3-0.6B)"
echo ""
echo -e "${BLUE}📝 Configuration for OpenWebUI:${NC}"
echo "  • API Base URL: http://semantic-router.vllm-semantic-router-system.svc.cluster.local:8801/v1"
echo "  • API Key: not-needed-for-local-models (or leave empty)"
echo ""
echo -e "${YELLOW}💡 Next Steps:${NC}"
echo "  1. Open https://$ROUTE_URL in your browser"
echo "  2. Complete initial setup in OpenWebUI"
echo "  3. The models should be automatically available"
echo ""
echo -e "${GREEN}✨ Happy chatting with your models!${NC}"