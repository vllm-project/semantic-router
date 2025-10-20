#!/bin/bash

# Script to uninstall vLLM Semantic Router deployment
# Removes all components deployed by deploy-all.sh

set -e

NAMESPACE="vllm-semantic-router"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================="
echo "Uninstalling vLLM Semantic Router"
echo "Namespace: $NAMESPACE"
echo "========================================="
echo ""

# Confirm deletion
read -p "This will DELETE all resources in namespace '$NAMESPACE'. Are you sure? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "Uninstall cancelled."
    exit 0
fi

echo ""
echo "Starting uninstall process..."
echo ""

# Step 1: Delete Route
echo "Step 1: Deleting Route..."
if oc get route envoy-proxy -n $NAMESPACE &>/dev/null; then
    oc delete route envoy-proxy -n $NAMESPACE
    echo "  ✓ Route deleted"
else
    echo "  ⊘ Route not found (already deleted)"
fi

# Step 2: Delete Deployments
echo ""
echo "Step 2: Deleting Deployments..."
for deployment in semantic-router model-a model-b; do
    if oc get deployment $deployment -n $NAMESPACE &>/dev/null; then
        oc delete deployment $deployment -n $NAMESPACE
        echo "  ✓ Deployment '$deployment' deleted"
    else
        echo "  ⊘ Deployment '$deployment' not found"
    fi
done

# Wait for pods to terminate
echo ""
echo "Waiting for pods to terminate..."
sleep 5
PODS_REMAINING=$(oc get pods -n $NAMESPACE --no-headers 2>/dev/null | wc -l)
if [ "$PODS_REMAINING" -gt 0 ]; then
    echo "  Waiting for $PODS_REMAINING pod(s) to terminate..."
    sleep 10
fi

# Step 3: Delete Services
echo ""
echo "Step 3: Deleting Services..."
for service in envoy-proxy semantic-router model-a model-b; do
    if oc get service $service -n $NAMESPACE &>/dev/null; then
        oc delete service $service -n $NAMESPACE
        echo "  ✓ Service '$service' deleted"
    else
        echo "  ⊘ Service '$service' not found"
    fi
done

# Step 4: Delete ConfigMaps
echo ""
echo "Step 4: Deleting ConfigMaps..."
for configmap in semantic-router-config envoy-config; do
    if oc get configmap $configmap -n $NAMESPACE &>/dev/null; then
        oc delete configmap $configmap -n $NAMESPACE
        echo "  ✓ ConfigMap '$configmap' deleted"
    else
        echo "  ⊘ ConfigMap '$configmap' not found"
    fi
done

# Step 5: Delete Builds and BuildConfigs
echo ""
echo "Step 5: Deleting Builds and BuildConfigs..."

# Delete all builds
BUILDS=$(oc get builds -n $NAMESPACE --no-headers 2>/dev/null | awk '{print $1}')
if [ -n "$BUILDS" ]; then
    echo "  Deleting builds..."
    oc delete builds --all -n $NAMESPACE
    echo "  ✓ All builds deleted"
else
    echo "  ⊘ No builds found"
fi

# Delete BuildConfigs
for bc in semantic-router llm-katan; do
    if oc get buildconfig $bc -n $NAMESPACE &>/dev/null; then
        oc delete buildconfig $bc -n $NAMESPACE
        echo "  ✓ BuildConfig '$bc' deleted"
    else
        echo "  ⊘ BuildConfig '$bc' not found"
    fi
done

# Step 6: Delete ImageStreams
echo ""
echo "Step 6: Deleting ImageStreams..."
for imagestream in semantic-router llm-katan; do
    if oc get imagestream $imagestream -n $NAMESPACE &>/dev/null; then
        oc delete imagestream $imagestream -n $NAMESPACE
        echo "  ✓ ImageStream '$imagestream' deleted"
    else
        echo "  ⊘ ImageStream '$imagestream' not found"
    fi
done

# Step 7: Delete PVCs (with confirmation)
echo ""
echo "Step 7: Deleting PersistentVolumeClaims..."
echo "  WARNING: This will delete all persistent data including models!"
read -p "  Delete PVCs and all data? (yes/no): " DELETE_PVC

if [ "$DELETE_PVC" = "yes" ]; then
    for pvc in semantic-router-models semantic-router-cache; do
        if oc get pvc $pvc -n $NAMESPACE &>/dev/null; then
            oc delete pvc $pvc -n $NAMESPACE
            echo "  ✓ PVC '$pvc' deleted"
        else
            echo "  ⊘ PVC '$pvc' not found"
        fi
    done
else
    echo "  ⊘ PVCs preserved (not deleted)"
fi

# Step 8: Delete Namespace (with confirmation)
echo ""
echo "Step 8: Deleting Namespace..."
echo "  WARNING: This will delete the entire namespace '$NAMESPACE'!"
read -p "  Delete namespace? (yes/no): " DELETE_NS

if [ "$DELETE_NS" = "yes" ]; then
    if oc get namespace $NAMESPACE &>/dev/null; then
        oc delete namespace $NAMESPACE
        echo "  ✓ Namespace '$NAMESPACE' deleted"
        echo ""
        echo "Waiting for namespace to be fully removed..."
        while oc get namespace $NAMESPACE &>/dev/null; do
            echo -n "."
            sleep 2
        done
        echo ""
        echo "  ✓ Namespace fully removed"
    else
        echo "  ⊘ Namespace not found"
    fi
else
    echo "  ⊘ Namespace preserved (not deleted)"
fi

# Final status
echo ""
echo "========================================="
echo "Uninstall Summary"
echo "========================================="
echo ""

if [ "$DELETE_NS" = "yes" ]; then
    echo "✓ Complete uninstall: All resources deleted including namespace"
else
    echo "Partial uninstall: Deployments and services removed"
    echo ""
    echo "Remaining resources in namespace '$NAMESPACE':"
    oc get all,configmap,pvc -n $NAMESPACE 2>/dev/null || echo "  (namespace may have been deleted)"
fi

echo ""
echo "========================================="
echo "Uninstall Complete"
echo "========================================="
echo ""

if [ "$DELETE_NS" != "yes" ]; then
    echo "To completely remove the namespace, run:"
    echo "  oc delete namespace $NAMESPACE"
    echo ""
fi

# Clean up temporary files
if [ -f "/tmp/configmap-router-updated.yaml" ]; then
    rm -f /tmp/configmap-router-updated.yaml
    echo "Cleaned up temporary files"
fi

echo "Done!"
