#!/bin/bash

# Script to check GPU node status and readiness
# This helps verify GPU availability before deploying GPU workloads

set -e

NAMESPACE="${1:-vllm-semantic-router}"

echo "========================================="
echo "GPU Node Status Check"
echo "========================================="
echo ""

# Check if any nodes have GPU labels
echo "Checking for GPU nodes..."
GPU_NODES=$(oc get nodes -l nvidia.com/gpu.present=true --no-headers 2>/dev/null | wc -l)

if [ "$GPU_NODES" -eq 0 ]; then
    echo "❌ No GPU nodes found in the cluster!"
    echo ""
    echo "GPU MachineSet status:"
    oc get machinesets -n openshift-machine-api | grep gpu || echo "  No GPU MachineSet found"
    echo ""
    echo "To create GPU nodes, check your MachineSet configuration."
    exit 1
fi

echo "✓ Found $GPU_NODES GPU node(s)"
echo ""

# Check GPU node readiness
echo "GPU Node Details:"
oc get nodes -l nvidia.com/gpu.present=true -o wide

echo ""
echo "GPU Node Status:"
READY_GPU_NODES=$(oc get nodes -l nvidia.com/gpu.present=true --no-headers 2>/dev/null | grep " Ready " | wc -l)

if [ "$READY_GPU_NODES" -eq 0 ]; then
    echo "⚠️  WARNING: No GPU nodes are Ready!"
    echo ""
    echo "Checking GPU machine status..."
    oc get machines -n openshift-machine-api | grep gpu
    echo ""

    # Check if machines are stopped (common in lab/demo environments)
    STOPPED_MACHINES=$(oc get machines -n openshift-machine-api -o json | grep -i '"instanceState": "stopped"' | wc -l)
    if [ "$STOPPED_MACHINES" -gt 0 ]; then
        echo "⚠️  GPU machine is STOPPED (common in lab environments to save costs)"
        echo ""
        echo "To restart the GPU node, run:"
        echo "  oc scale machineset cluster-*-gpu-worker-* --replicas=0 -n openshift-machine-api"
        echo "  sleep 10"
        echo "  oc scale machineset cluster-*-gpu-worker-* --replicas=1 -n openshift-machine-api"
        echo ""
        echo "Or use the actual machineset name:"
        GPU_MACHINESET=$(oc get machineset -n openshift-machine-api | grep gpu | awk '{print $1}')
        if [ -n "$GPU_MACHINESET" ]; then
            echo "  oc scale machineset $GPU_MACHINESET --replicas=0 -n openshift-machine-api"
            echo "  sleep 10"
            echo "  oc scale machineset $GPU_MACHINESET --replicas=1 -n openshift-machine-api"
        fi
        echo ""
    fi

    echo "GPU node may be starting. Wait a few minutes and run this script again."
    exit 1
fi

echo "✓ $READY_GPU_NODES GPU node(s) are Ready"
echo ""

# Check NVIDIA GPU Operator
echo "Checking NVIDIA GPU Operator..."
GPU_OPERATOR_NS=$(oc get pods -A | grep "nvidia-gpu-operator\|gpu-operator" | head -1 | awk '{print $1}')

if [ -z "$GPU_OPERATOR_NS" ]; then
    echo "❌ NVIDIA GPU Operator not found!"
    echo ""
    echo "Install the NVIDIA GPU Operator from OperatorHub:"
    echo "  1. Go to OpenShift Console > Operators > OperatorHub"
    echo "  2. Search for 'NVIDIA GPU Operator'"
    echo "  3. Install it"
    exit 1
fi

echo "✓ GPU Operator found in namespace: $GPU_OPERATOR_NS"
echo ""
echo "GPU Operator Pods:"
oc get pods -n "$GPU_OPERATOR_NS" -o wide

echo ""
echo "GPU Resources Available:"
oc describe nodes -l nvidia.com/gpu.present=true | grep -A 5 "Allocatable:" | grep nvidia.com/gpu || echo "  No GPU resources found"

echo ""
echo "========================================="
echo "GPU Status Summary"
echo "========================================="
echo ""

if [ "$READY_GPU_NODES" -gt 0 ]; then
    echo "✓ GPU nodes are ready for workloads"
    echo ""
    echo "You can now deploy GPU-enabled pods!"
    echo ""
    echo "To check GPU allocation:"
    echo "  oc describe nodes -l nvidia.com/gpu.present=true | grep nvidia.com/gpu"
    echo ""
    exit 0
else
    echo "⚠️  GPU nodes are not ready yet"
    echo ""
    echo "Wait a few minutes and run: $0"
    exit 1
fi
