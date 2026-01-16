# OpenShift Quick Start Guide

This guide will help you deploy the Semantic Router Operator on Red Hat OpenShift.

## Prerequisites

- OpenShift 4.12 or later
- Cluster admin access (for operator installation)
- `oc` CLI tool installed and configured
- Access to pull images from ghcr.io (or your internal registry)

## Installation Methods

### Method 1: OperatorHub (Recommended for Production)

Once the operator is published to OperatorHub:

1. Log into OpenShift Console
2. Navigate to **Operators** → **OperatorHub**
3. Search for "Semantic Router"
4. Click **Install**
5. Choose installation mode:
   - **All namespaces**: Operator can manage resources cluster-wide
   - **Specific namespace**: Operator only manages one namespace
6. Click **Install** and wait for completion

### Method 2: Manual Installation from Source

#### Step 1: Build and Push Images

```bash
# Clone repository
git clone https://github.com/vllm-project/semantic-router
cd semantic-router/operator

# Login to your registry
podman login quay.io  # or your registry

# Build operator image using UBI base
make podman-build IMG=quay.io/<your-org>/semantic-router-operator:latest

# Push operator image
make podman-push IMG=quay.io/<your-org>/semantic-router-operator:latest

# Build OLM bundle
make bundle-podman-build BUNDLE_IMG=quay.io/<your-org>/semantic-router-operator-bundle:latest

# Push bundle
make bundle-podman-push BUNDLE_IMG=quay.io/<your-org>/semantic-router-operator-bundle:latest
```

#### Step 2: Deploy via OLM

```bash
# Login to OpenShift
oc login --server=https://api.your-cluster.com:6443

# Create CatalogSource
cat <<EOF | oc apply -f -
apiVersion: operators.coreos.com/v1alpha1
kind: CatalogSource
metadata:
  name: semantic-router-catalog
  namespace: openshift-marketplace
spec:
  sourceType: grpc
  image: quay.io/<your-org>/semantic-router-operator-bundle:latest
  displayName: Semantic Router Operator
  publisher: Your Organization
  updateStrategy:
    registryPoll:
      interval: 10m
EOF

# Wait for catalog to be ready
oc get catalogsource -n openshift-marketplace semantic-router-catalog -w

# Create namespace for operator
oc create namespace semantic-router-operator-system

# Create OperatorGroup
cat <<EOF | oc apply -f -
apiVersion: operators.coreos.com/v1
kind: OperatorGroup
metadata:
  name: semantic-router-operator-group
  namespace: semantic-router-operator-system
spec:
  targetNamespaces:
  - semantic-router-operator-system
EOF

# Create Subscription
cat <<EOF | oc apply -f -
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata:
  name: semantic-router-operator
  namespace: semantic-router-operator-system
spec:
  channel: stable
  name: semantic-router-operator
  source: semantic-router-catalog
  sourceNamespace: openshift-marketplace
EOF

# Check operator status
oc get csv -n semantic-router-operator-system
oc get pods -n semantic-router-operator-system
```

## Deploy a SemanticRouter Instance

### Step 1: Create Project

```bash
oc new-project semantic-router-demo
```

### Step 2: Create HuggingFace Token Secret (Optional)

For downloading gated models:

```bash
oc create secret generic hf-token-secret \
  --from-literal=token=YOUR_HUGGINGFACE_TOKEN \
  -n semantic-router-demo
```

### Step 3: Create SemanticRouter Resource

Create a file `semantic-router.yaml`:

```yaml
apiVersion: vllm.ai/v1alpha1
kind: SemanticRouter
metadata:
  name: demo-router
  namespace: semantic-router-demo
spec:
  replicas: 1

  # Use RHEL-based image
  image:
    repository: ghcr.io/vllm-project/semantic-router/extproc
    tag: latest
    pullPolicy: IfNotPresent

  # Resource allocation
  resources:
    limits:
      memory: "7Gi"
      cpu: "2"
    requests:
      memory: "3Gi"
      cpu: "1"

  # Persistent storage for ML models
  persistence:
    enabled: true
    storageClassName: "gp3-csi"  # Adjust for your cluster
    size: 10Gi

  # Security context for OpenShift
  securityContext:
    runAsNonRoot: false  # Models need write access during download
    allowPrivilegeEscalation: false

  # Environment variables
  env:
  - name: LD_LIBRARY_PATH
    value: "/app/lib"
  - name: HF_TOKEN
    valueFrom:
      secretKeyRef:
        name: hf-token-secret
        key: token
        optional: true

  # Startup probe - model download can take time
  startupProbe:
    enabled: true
    periodSeconds: 10
    timeoutSeconds: 5
    failureThreshold: 360  # 60 min timeout

  # Configuration
  config:
    semantic_cache:
      enabled: true
      backend_type: "memory"
      max_entries: 1000

    tools:
      enabled: true
      top_k: 3

    prompt_guard:
      enabled: true
      threshold: 0.7
```

Apply the configuration:

```bash
oc apply -f semantic-router.yaml
```

### Step 4: Monitor Deployment

```bash
# Watch the SemanticRouter status
oc get semanticrouter demo-router -w

# Check pods
oc get pods -l app.kubernetes.io/instance=demo-router

# View logs (model download progress)
oc logs -f deployment/demo-router

# Check events
oc get events --sort-by='.lastTimestamp' | grep demo-router
```

### Step 5: Access the Service

#### Internal Access (within cluster)

Services are automatically created:

```bash
# List services
oc get svc

# gRPC service: demo-router:50051
# HTTP API: demo-router:8080
# Metrics: demo-router:9190
```

#### External Access via Route

Create a route for external access:

```bash
# Expose HTTP API
oc expose svc demo-router --port=8080

# Get route URL
oc get route demo-router

# Access the API
curl http://$(oc get route demo-router -o jsonpath='{.spec.host}')/health
```

#### Port Forwarding for Testing

```bash
# Forward ports to local machine
oc port-forward svc/demo-router 50051:50051 8080:8080 9190:9190

# In another terminal:
# gRPC: localhost:50051
# API: http://localhost:8080
# Metrics: http://localhost:9190/metrics
```

## Configuration for OpenShift

### Storage Classes

OpenShift uses different storage classes. Check available options:

```bash
oc get storageclass
```

Common OpenShift storage classes:

- `gp3-csi` - AWS EBS GP3
- `thin` - vSphere thin provisioned
- `ocs-storagecluster-ceph-rbd` - OpenShift Container Storage

Update your SemanticRouter spec:

```yaml
spec:
  persistence:
    storageClassName: "gp3-csi"  # Use your cluster's storage class
```

### Security Context Constraints (SCC)

OpenShift uses SCCs for pod security. **The operator automatically detects if it's running on OpenShift** and configures security contexts appropriately:

#### Automatic Platform Detection

The operator automatically detects the platform at startup:

- **On OpenShift**:
  - Does NOT set `runAsUser`, `runAsGroup`, or `fsGroup` in pod security context
  - Allows OpenShift SCCs to assign UIDs/GIDs from the namespace's allowed range
  - Sets `allowPrivilegeEscalation: false` and drops ALL capabilities
  - Compatible with `restricted` SCC and custom SCCs

- **On Standard Kubernetes**:
  - Sets secure defaults: `runAsUser: 1000`, `fsGroup: 1000`, `runAsNonRoot: true`
  - Sets `allowPrivilegeEscalation: false` and drops ALL capabilities

#### Override Security Context

You can override the automatic security context in your SemanticRouter CR:

```yaml
spec:
  # Container security context (optional)
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    allowPrivilegeEscalation: false
    capabilities:
      drop:
        - ALL

  # Pod security context (optional)
  podSecurityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 1000
```

**Note**: When running on OpenShift, it's recommended to omit `runAsUser` and `fsGroup` from your CR and let the SCCs handle UID/GID assignment automatically.

#### Custom SCC (Advanced)

The operator works with OpenShift's default SCCs, but for custom scenarios:

```bash
# Check SCC being used
oc get pod <pod-name> -o jsonpath='{.metadata.annotations.openshift\.io/scc}'

# If needed, create custom SCC
oc create -f - <<EOF
apiVersion: security.openshift.io/v1
kind: SecurityContextConstraints
metadata:
  name: semantic-router-scc
allowHostDirVolumePlugin: false
allowHostIPC: false
allowHostNetwork: false
allowHostPID: false
allowHostPorts: false
allowPrivilegedContainer: false
allowedCapabilities: []
defaultAddCapabilities: []
fsGroup:
  type: RunAsAny
readOnlyRootFilesystem: false
requiredDropCapabilities:
- ALL
runAsUser:
  type: RunAsAny
seLinuxContext:
  type: RunAsAny
supplementalGroups:
  type: RunAsAny
volumes:
- configMap
- downwardAPI
- emptyDir
- persistentVolumeClaim
- projected
- secret
EOF
```

### Network Policies

For secure multi-tenant clusters:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: semantic-router-netpol
  namespace: semantic-router-demo
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/instance: demo-router
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: envoy-gateway
    ports:
    - protocol: TCP
      port: 50051
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: TCP
      port: 443  # HuggingFace downloads
  - to:
    - namespaceSelector:
        matchLabels:
          name: vllm-backends
    ports:
    - protocol: TCP
      port: 8000
```

## Production Deployment

### High Availability Setup

```yaml
apiVersion: vllm.ai/v1alpha1
kind: SemanticRouter
metadata:
  name: prod-router
spec:
  replicas: 3

  # Anti-affinity for HA
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchLabels:
            app.kubernetes.io/instance: prod-router
        topologyKey: kubernetes.io/hostname

  # Autoscaling
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 20
    targetCPUUtilizationPercentage: 70

  # Production resources
  resources:
    limits:
      memory: "10Gi"
      cpu: "4"
    requests:
      memory: "5Gi"
      cpu: "2"

  # Fast storage
  persistence:
    enabled: true
    storageClassName: "gp3-csi"
    size: 20Gi

  # Strict probes
  livenessProbe:
    enabled: true
    initialDelaySeconds: 60
    periodSeconds: 30
    failureThreshold: 3

  # Pod disruption budget
  podAnnotations:
    pod-disruption-budget.kubernetes.io/max-unavailable: "1"
```

Create PodDisruptionBudget:

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: prod-router-pdb
spec:
  maxUnavailable: 1
  selector:
    matchLabels:
      app.kubernetes.io/instance: prod-router
```

### Monitoring with OpenShift Monitoring

Enable ServiceMonitor for Prometheus:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: semantic-router-metrics
  namespace: semantic-router-demo
spec:
  selector:
    matchLabels:
      app.kubernetes.io/instance: demo-router
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

View in OpenShift Console:

- **Observe** → **Metrics** → Query for `semantic_router_*`

## Troubleshooting

### Check Operator Logs

```bash
oc logs -n semantic-router-operator-system \
  deployment/semantic-router-operator-controller-manager
```

### Check Resource Status

```bash
# Detailed resource info
oc describe semanticrouter demo-router

# Check all related resources
oc get all -l app.kubernetes.io/instance=demo-router
```

### Common Issues

**Issue**: Pod stuck in `ImagePullBackOff`

```bash
# Check image pull secret
oc get pod <pod-name> -o jsonpath='{.spec.imagePullSecrets}'

# Create pull secret if needed
oc create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io \
  --docker-username=<username> \
  --docker-password=<token>

# Update SemanticRouter to use secret
spec:
  imagePullSecrets:
  - name: ghcr-secret
```

**Issue**: PVC stuck in `Pending`

```bash
# Check PVC status
oc get pvc

# Check storage class exists
oc get storageclass

# Check events
oc get events | grep PersistentVolumeClaim
```

**Issue**: Models not downloading

```bash
# Check HF token secret
oc get secret hf-token-secret -o yaml

# Check pod logs for download errors
oc logs deployment/demo-router | grep -i download

# Manually test HF access
oc run -it --rm debug --image=python:3.11 --restart=Never -- \
  bash -c "pip install huggingface_hub && huggingface-cli login"
```

## Cleanup

```bash
# Delete SemanticRouter instance
oc delete semanticrouter demo-router

# Delete project
oc delete project semantic-router-demo

# Uninstall operator (if needed)
make openshift-undeploy
```

## Next Steps

- [Full Configuration Reference](README.md)
- [Operator Development Guide](README.md#development)
- [Main Project Documentation](https://vllm-semantic-router.com)

## Support

For OpenShift-specific issues:

- GitHub Issues: https://github.com/vllm-project/semantic-router/issues
- Tag issues with `openshift` label
