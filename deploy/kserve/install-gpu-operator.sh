#!/bin/bash

# NVIDIA GPU Operator Installation Script for OpenShift
# This script installs NFD (Node Feature Discovery) and NVIDIA GPU Operator
# to enable GPU workloads on OpenShift clusters

set -e

echo "========================================="
echo "  NVIDIA GPU Operator Installation"
echo "========================================="
echo ""

# Check if running on OpenShift
if ! command -v oc &> /dev/null; then
    echo "Error: 'oc' command not found. This script requires OpenShift CLI."
    exit 1
fi

if ! oc whoami &> /dev/null; then
    echo "Error: Not logged in to OpenShift. Please login first:"
    echo "  oc login <your-openshift-server-url>"
    exit 1
fi

echo "Logged in as: $(oc whoami)"
echo ""

# Step 1: Install Node Feature Discovery (NFD) Operator
echo "1️⃣  Installing Node Feature Discovery (NFD) Operator..."

# Check if NFD operator is already installed
if oc get csv -n openshift-nfd 2>/dev/null | grep -q nfd; then
    echo "   ✅ NFD operator already installed"
else
    echo "   Creating openshift-nfd namespace..."
    cat <<EOF | oc apply -f -
apiVersion: v1
kind: Namespace
metadata:
  name: openshift-nfd
EOF

    echo "   Creating OperatorGroup..."
    cat <<EOF | oc apply -f -
apiVersion: operators.coreos.com/v1
kind: OperatorGroup
metadata:
  name: openshift-nfd
  namespace: openshift-nfd
spec:
  targetNamespaces:
  - openshift-nfd
EOF

    echo "   Creating Subscription..."
    cat <<EOF | oc apply -f -
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata:
  name: nfd
  namespace: openshift-nfd
spec:
  channel: stable
  name: nfd
  source: redhat-operators
  sourceNamespace: openshift-marketplace
EOF

    echo "   Waiting for NFD operator to be ready..."
    for i in {1..30}; do
        if oc get csv -n openshift-nfd 2>/dev/null | grep -q "Succeeded"; then
            echo "   ✅ NFD operator installed successfully"
            break
        fi
        echo "   Waiting for NFD operator CSV... ($i/30)"
        sleep 10
    done
fi

# Step 2: Create NFD Instance
echo ""
echo "2️⃣  Creating NFD Instance..."

if oc get nodefeaturediscovery -n openshift-nfd nfd-instance &>/dev/null; then
    echo "   ✅ NFD instance already exists"
else
    echo "   Creating NodeFeatureDiscovery CR..."
    cat <<EOF | oc apply -f -
apiVersion: nfd.openshift.io/v1
kind: NodeFeatureDiscovery
metadata:
  name: nfd-instance
  namespace: openshift-nfd
spec:
  operand:
    servicePort: 12000
  workerConfig:
    configData: |
      sources:
        pci:
          deviceClassWhitelist:
            - "0200"
            - "03"
            - "12"
          deviceLabelFields:
            - vendor
EOF

    echo "   Waiting for NFD worker pods..."
    for i in {1..30}; do
        READY_PODS=$(oc get pods -n openshift-nfd -l app.kubernetes.io/component=worker --no-headers 2>/dev/null | grep -c "Running" | head -1 || echo "0")
        if [[ "$READY_PODS" =~ ^[0-9]+$ ]] && [[ "$READY_PODS" -gt 0 ]]; then
            echo "   ✅ NFD worker pods running ($READY_PODS pods)"
            break
        fi
        echo "   Waiting for NFD worker pods... ($i/30)"
        sleep 10
    done
fi

# Step 3: Install NVIDIA GPU Operator
echo ""
echo "3️⃣  Installing NVIDIA GPU Operator..."

# Check if GPU operator is already installed
if oc get csv -n nvidia-gpu-operator 2>/dev/null | grep -q "gpu-operator"; then
    echo "   ✅ NVIDIA GPU operator already installed"
else
    echo "   Creating nvidia-gpu-operator namespace..."
    cat <<EOF | oc apply -f -
apiVersion: v1
kind: Namespace
metadata:
  name: nvidia-gpu-operator
EOF

    echo "   Creating OperatorGroup..."
    cat <<EOF | oc apply -f -
apiVersion: operators.coreos.com/v1
kind: OperatorGroup
metadata:
  name: nvidia-gpu-operator
  namespace: nvidia-gpu-operator
spec:
  targetNamespaces:
  - nvidia-gpu-operator
EOF

    echo "   Creating Subscription..."
    cat <<EOF | oc apply -f -
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata:
  name: gpu-operator-certified
  namespace: nvidia-gpu-operator
spec:
  channel: stable
  name: gpu-operator-certified
  source: certified-operators
  sourceNamespace: openshift-marketplace
EOF

    echo "   Waiting for GPU operator to be ready (this may take a few minutes)..."
    for i in {1..30}; do
        if oc get csv -n nvidia-gpu-operator 2>/dev/null | grep -q "Succeeded"; then
            echo "   ✅ NVIDIA GPU operator installed successfully"
            break
        fi
        echo "   Waiting for GPU operator CSV... ($i/30)"
        sleep 10
    done
fi

# Step 4: Create ClusterPolicy
echo ""
echo "4️⃣  Creating ClusterPolicy..."

# Wait for ClusterPolicy CRD to be available
echo "   Waiting for ClusterPolicy CRD to be available..."
for i in {1..30}; do
    if oc get crd clusterpolicies.nvidia.com &>/dev/null; then
        echo "   ✅ ClusterPolicy CRD is available"
        break
    fi
    echo "   Waiting for ClusterPolicy CRD... ($i/30)"
    sleep 10
done

if oc get clusterpolicy gpu-cluster-policy &>/dev/null; then
    echo "   ✅ ClusterPolicy already exists"
else
    echo "   Creating ClusterPolicy CR..."
    cat <<EOF | oc apply -f -
apiVersion: nvidia.com/v1
kind: ClusterPolicy
metadata:
  name: gpu-cluster-policy
spec:
  daemonsets: {}
  dcgm:
    enabled: true
  dcgmExporter:
    enabled: true
  devicePlugin:
    enabled: true
  driver:
    enabled: true
  gfd:
    enabled: true
  migManager:
    enabled: true
  nodeStatusExporter:
    enabled: true
  operator:
    defaultRuntime: crio
  toolkit:
    enabled: true
  validator:
    enabled: true
EOF

    echo "   Waiting for ClusterPolicy to be ready (this may take 5-10 minutes as drivers are compiled)..."
    for i in {1..60}; do
        STATE=$(oc get clusterpolicy gpu-cluster-policy -o jsonpath='{.status.state}' 2>/dev/null || echo "Unknown")
        if [[ "$STATE" == "ready" ]]; then
            echo "   ✅ ClusterPolicy is ready"
            break
        fi
        echo "   ClusterPolicy state: $STATE ($i/60)"
        sleep 10
    done
fi

# Step 5: Verify GPU Resources
echo ""
echo "========================================="
echo "📊 Verification"
echo "========================================="
echo ""

echo "NFD Pods:"
oc get pods -n openshift-nfd --no-headers 2>/dev/null | head -5 || echo "   No NFD pods found"

echo ""
echo "GPU Operator Pods:"
oc get pods -n nvidia-gpu-operator --no-headers 2>/dev/null | head -10 || echo "   No GPU operator pods found"

echo ""
echo "ClusterPolicy Status:"
oc get clusterpolicy gpu-cluster-policy -o jsonpath='{.status.state}' 2>/dev/null || echo "   Not found"
echo ""

echo ""
echo "GPU Resources on Nodes:"
GPU_FOUND=0
GPU_TABLE=$(oc get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.capacity.nvidia\.com/gpu}{"\t"}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' 2>/dev/null || true)
if [[ -z "$GPU_TABLE" ]]; then
    echo "   Could not retrieve GPU resources"
else
    echo -e "NAME\tGPU_CAPACITY\tGPU_ALLOCATABLE"
    echo "$GPU_TABLE" | awk -F'\t' '{cap=$2; alloc=$3; if (cap=="") cap="<none>"; if (alloc=="") alloc="<none>"; print $1"\t"cap"\t"alloc}'
    if echo "$GPU_TABLE" | awk -F'\t' '{if ($2+0>0 || $3+0>0) found=1} END{exit found?0:1}'; then
        echo "   ✅ Detected GPU resources on at least one node"
        GPU_FOUND=1
    else
        echo "   ⚠️  No GPU resources detected yet on nodes"
    fi
fi

echo ""
echo "========================================="
if [[ "$GPU_FOUND" -eq 1 ]]; then
    echo "✅ GPU Operator Installation Complete!"
else
    echo "⚠️  GPU Operator Installed, but GPUs not detected yet"
fi
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Wait for all GPU operator pods to be Running (may take 5-10 minutes)"
echo "2. Verify GPU resources appear on nodes:"
printf "   oc get nodes -o jsonpath='{range .items[*]}{.metadata.name}{\"\\t\"}{.status.capacity.nvidia\\.com/gpu}{\"\\t\"}{.status.allocatable.nvidia\\.com/gpu}{\"\\n\"}{end}'\n"
echo ""
echo "3. Deploy a GPU workload to test:"
echo "   kubectl apply -k deploy/kubernetes/llm-katan/base"
echo ""
echo "Troubleshooting:"
echo "- Check GPU operator pods: oc get pods -n nvidia-gpu-operator"
echo "- Check driver pod logs: oc logs -n nvidia-gpu-operator -l app=nvidia-driver-daemonset"
echo "- Check ClusterPolicy: oc describe clusterpolicy gpu-cluster-policy"
echo ""

if [[ "$GPU_FOUND" -ne 1 ]]; then
    echo "Error: GPU resources are not visible on any node yet."
    echo "       Wait a few minutes for driver pods to finish, then re-run the GPU check."
    exit 2
fi
