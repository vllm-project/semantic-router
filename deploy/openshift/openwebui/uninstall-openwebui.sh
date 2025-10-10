#!/bin/bash

# Uninstall OpenWebUI from OpenShift
# This script safely removes OpenWebUI without affecting the semantic-router deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}🗑️  OpenWebUI OpenShift Uninstall Script${NC}"
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

# Get current user
CURRENT_USER=$(oc whoami)
echo -e "${GREEN}✅ Logged in as: ${CURRENT_USER}${NC}"

# Check if namespace exists
NAMESPACE="vllm-semantic-router-system"
if ! oc get namespace "$NAMESPACE" &> /dev/null; then
    echo -e "${YELLOW}⚠️  Namespace '$NAMESPACE' not found. Nothing to uninstall.${NC}"
    exit 0
fi

# Switch to the namespace
echo -e "${YELLOW}📁 Switching to namespace: ${NAMESPACE}${NC}"
oc project "$NAMESPACE"

# Check if OpenWebUI is deployed
if ! oc get deployment openwebui &> /dev/null; then
    echo -e "${YELLOW}⚠️  OpenWebUI deployment not found. Nothing to uninstall.${NC}"
    exit 0
fi

# Confirmation prompt
echo -e "${YELLOW}⚠️  This will remove OpenWebUI and ALL its data from OpenShift.${NC}"
echo -e "${YELLOW}   This includes persistent data, conversations, and configurations.${NC}"
echo -e "${YELLOW}   The semantic-router deployment will NOT be affected.${NC}"
echo ""
read -p "Are you sure you want to continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}❌ Uninstall cancelled.${NC}"
    exit 0
fi

# Store the user confirmation for later use
USER_CONFIRMED="yes"

echo ""
echo -e "${YELLOW}🗑️  Starting OpenWebUI uninstall...${NC}"

# Remove OpenWebUI components in reverse order
echo "  🔗 Removing Route..."
if oc get route openwebui &> /dev/null; then
    oc delete route openwebui
    echo -e "${GREEN}    ✅ Route removed${NC}"
else
    echo -e "${YELLOW}    ⚠️  Route not found${NC}"
fi

echo "  🌐 Removing Service..."
if oc get service openwebui &> /dev/null; then
    oc delete service openwebui
    echo -e "${GREEN}    ✅ Service removed${NC}"
else
    echo -e "${YELLOW}    ⚠️  Service not found${NC}"
fi

echo "  🚀 Removing Deployment..."
if oc get deployment openwebui &> /dev/null; then
    oc delete deployment openwebui
    echo -e "${GREEN}    ✅ Deployment removed${NC}"

    # Wait for pods to be terminated
    echo "  ⏳ Waiting for pods to terminate..."
    oc wait --for=delete pod -l app=openwebui --timeout=60s 2>/dev/null || true
    echo -e "${GREEN}    ✅ Pods terminated${NC}"
else
    echo -e "${YELLOW}    ⚠️  Deployment not found${NC}"
fi

# Remove PVC automatically since user confirmed complete uninstall
echo ""
echo -e "${YELLOW}📦 Removing Persistent Volume Claim and all data...${NC}"
if oc get pvc openwebui-data &> /dev/null; then
    echo "  📦 Removing Persistent Volume Claim..."
    oc delete pvc openwebui-data
    echo -e "${GREEN}    ✅ PVC and all data removed${NC}"
else
    echo -e "${YELLOW}    ⚠️  PVC not found${NC}"
fi

# Check remaining OpenWebUI resources
echo ""
echo -e "${YELLOW}🔍 Checking for remaining OpenWebUI resources...${NC}"

# Check for any resources with openwebui labels
REMAINING_RESOURCES=$(oc get all,pvc,configmap,secret -l app=openwebui -o name 2>/dev/null || true)
if [ -n "$REMAINING_RESOURCES" ]; then
    echo -e "${YELLOW}⚠️  Found remaining resources:${NC}"
    echo "$REMAINING_RESOURCES"
    echo ""
    read -p "Remove these resources too? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "$REMAINING_RESOURCES" | xargs oc delete 2>/dev/null || true
        echo -e "${GREEN}✅ Additional resources removed${NC}"
    fi
else
    echo -e "${GREEN}✅ No remaining OpenWebUI resources found${NC}"
fi

# Verify semantic-router is still running
echo ""
echo -e "${YELLOW}🔍 Verifying semantic-router is still running...${NC}"
if oc get deployment semantic-router &> /dev/null; then
    READY_REPLICAS=$(oc get deployment semantic-router -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
    if [ "$READY_REPLICAS" = "1" ]; then
        echo -e "${GREEN}✅ semantic-router is still running normally${NC}"
    else
        echo -e "${YELLOW}⚠️  semantic-router may need time to stabilize${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  semantic-router deployment not found${NC}"
fi

# Final summary
echo ""
echo -e "${GREEN}🎉 OpenWebUI uninstall completed!${NC}"
echo "================================================"
echo -e "${BLUE}📊 Uninstall Summary:${NC}"
echo "  ✅ OpenWebUI deployment removed"
echo "  ✅ Service and route removed"
echo "  ✅ Data PVC and all user data removed"
echo "  ✅ semantic-router deployment unaffected"
echo ""
echo -e "${BLUE}💡 To redeploy OpenWebUI:${NC}"
echo "  ./deploy-openwebui-on-openshift.sh"
echo ""
echo -e "${GREEN}✨ Cleanup completed successfully!${NC}"