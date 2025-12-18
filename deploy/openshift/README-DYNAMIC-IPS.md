# Dynamic IP Configuration for Cross-Cluster Deployments

## Overview

This deployment uses **dynamic IP configuration** to ensure portability across different OpenShift/Kubernetes clusters. Instead of hardcoding ClusterIPs, the deployment script automatically discovers service IPs at deployment time.

## Architecture

### Pod Structure

1. **semantic-router Pod**:
   - Container 1: `semantic-router` (ExtProc service)
   - Container 2: `envoy-proxy` (Proxy)

2. **vllm-model-a Pod**:
   - Container: `model-a` (llm-katan serving Qwen3-0.6B)

3. **vllm-model-b Pod**:
   - Container: `model-b` (llm-katan serving Qwen3-0.6B)

All pods run in the same namespace: `vllm-semantic-router-system`

### Dynamic IP Discovery Process

The `deploy-split.sh` script implements dynamic IP configuration:

```bash
# 1. Deploy vLLM model services first
oc apply -f deployment-split.yaml

# 2. Wait for services to get ClusterIPs
MODEL_A_IP=$(oc get svc vllm-model-a -o jsonpath='{.spec.clusterIP}')
MODEL_B_IP=$(oc get svc vllm-model-b -o jsonpath='{.spec.clusterIP}')

# 3. Generate config with actual IPs
sed "s/172.30.64.134/$MODEL_A_IP/g" config-split.yaml > dynamic-config.yaml

# 4. Create ConfigMap with dynamic config
oc create configmap semantic-router-config --from-file=dynamic-config.yaml
```

## Benefits

### ✅ Cross-Cluster Portability

- Works on any OpenShift/Kubernetes cluster
- No manual IP configuration needed
- IPs are discovered automatically

### ✅ Service-Based Routing

- Uses Kubernetes ClusterIP services
- Automatic service discovery
- Load balancing handled by Kubernetes

### ✅ Separation of Concerns

- vLLM models in separate pods
- Independent scaling
- Better resource isolation

## Deployment

### Quick Deploy

```bash
cd deploy/openshift/single-namespace
./deploy-split.sh
```

### What Happens

1. **Namespace Creation**: `vllm-semantic-router-system`
2. **Image Build**: `llm-katan` image (if not exists)
3. **PVC Creation**: Persistent volumes for models and cache
4. **Service Deployment**: vLLM model services created first
5. **IP Discovery**: Script queries ClusterIPs dynamically
6. **Config Generation**: Creates config with actual IPs
7. **Router Deployment**: semantic-router deployed with dynamic config
8. **Route Creation**: OpenShift routes for external access

### Verification

```bash
# Check all pods are running
oc get pods -n vllm-semantic-router-system

# Verify services have ClusterIPs
oc get svc -n vllm-semantic-router-system

# Test Model-A endpoint
oc exec deployment/semantic-router -c semantic-router -- \
  curl -s http://$(oc get svc vllm-model-a -o jsonpath='{.spec.clusterIP}'):8000/v1/models

# Test Model-B endpoint
oc exec deployment/semantic-router -c semantic-router -- \
  curl -s http://$(oc get svc vllm-model-b -o jsonpath='{.spec.clusterIP}'):8001/v1/models
```

## Configuration Files

### Template: config-split.yaml

Contains **placeholder IPs** that get replaced:

```yaml
vllm_endpoints:
  - name: "model-a-endpoint"
    address: "172.30.64.134"  # PLACEHOLDER - replaced at deploy time
    port: 8000
  - name: "model-b-endpoint"
    address: "172.30.116.177"  # PLACEHOLDER - replaced at deploy time
    port: 8001
```

### Generated: ConfigMap

Contains **actual ClusterIPs** discovered during deployment:

```yaml
vllm_endpoints:
  - name: "model-a-endpoint"
    address: "172.30.64.134"  # Actual ClusterIP from cluster
    port: 8000
  - name: "model-b-endpoint"
    address: "172.30.116.177"  # Actual ClusterIP from cluster
    port: 8001
```

## Testing on Different Clusters

### Scenario: Deploy to New Cluster

```bash
# 1. Login to new cluster
oc login https://new-cluster-api.example.com:6443

# 2. Run deploy script (IPs auto-discovered)
cd deploy/openshift/single-namespace
./deploy-split.sh

# 3. Verify new ClusterIPs
oc get svc -n vllm-semantic-router-system
# vllm-model-a   ClusterIP   10.96.10.50   <none>   8000/TCP
# vllm-model-b   ClusterIP   10.96.20.80   <none>   8001/TCP

# 4. Check config has new IPs
oc get configmap semantic-router-config -o yaml | grep address:
#     address: "10.96.10.50"  # New cluster IP for Model-A
#     address: "10.96.20.80"  # New cluster IP for Model-B
```

## Troubleshooting

### Issue: Classification Errors

If you see classification errors, verify model connectivity:

```bash
# From semantic-router pod, test Model-A
oc exec deployment/semantic-router -c semantic-router -- \
  curl http://$(oc get svc vllm-model-a -o jsonpath='{.spec.clusterIP}'):8000/health

# Test Model-B
oc exec deployment/semantic-router -c semantic-router -- \
  curl http://$(oc get svc vllm-model-b -o jsonpath='{.spec.clusterIP}'):8001/health
```

### Issue: IP Discovery Fails

If the script fails to get ClusterIPs:

```bash
# Check services exist
oc get svc -n vllm-semantic-router-system

# Manually verify ClusterIPs
oc get svc vllm-model-a -o jsonpath='{.spec.clusterIP}'
oc get svc vllm-model-b -o jsonpath='{.spec.clusterIP}'
```

### Issue: ConfigMap Not Updated

Restart semantic-router to pick up new config:

```bash
oc rollout restart deployment/semantic-router -n vllm-semantic-router-system
oc rollout status deployment/semantic-router -n vllm-semantic-router-system
```

## Comparison: Alternative Approaches

### ❌ Hardcoded IPs (Original)

```yaml
address: "172.30.64.134"  # Works only on specific cluster
```

### ❌ Localhost (Sidecar Pattern)

```yaml
address: "127.0.0.1"  # Requires all containers in same pod
```

### ✅ Dynamic IPs (Current Solution)

```yaml
address: "$DISCOVERED_IP"  # Works on any cluster
```

### 🚀 DNS Names (Future Enhancement)

```yaml
address: "vllm-model-a.vllm-semantic-router-system.svc.cluster.local"
```

**Note**: Requires Go code changes to accept DNS names (see `src/semantic-router/pkg/config/validator.go`)

## Future Improvements

1. **DNS-Based Routing**: Modify validator to accept Kubernetes service DNS names
2. **Multi-Cluster Support**: Deploy across multiple clusters with federation
3. **Auto-Scaling**: Horizontal pod autoscaling based on traffic
4. **Health Checks**: Enhanced health probes for better reliability

## Related Files

- `deploy-split.sh`: Main deployment script with dynamic IP logic (deploy/openshift/single-namespace/deploy-split.sh:109-164)
- `config-split.yaml`: Configuration template with placeholder IPs (deploy/openshift/single-namespace/config-split.yaml:30-41)
- `deployment-split.yaml`: Kubernetes manifests for split architecture (deploy/openshift/single-namespace/deployment-split.yaml)
- `validator.go`: IP validation code (requires modification for DNS support) (src/semantic-router/pkg/config/validator.go:20-51)
