# Semantic Router Integration with OpenShift AI KServe

Deploy vLLM Semantic Router as an intelligent gateway for your OpenShift AI KServe LLMInferenceServices.

> **Deployment Focus**: This guide is specifically for deploying semantic router on **OpenShift AI with KServe**.
>
> **Learn about features?** See links to feature documentation throughout this guide.

## Overview

The semantic router acts as an intelligent API gateway that provides:

- **Intelligent Model Selection**: Automatically routes requests to the best model based on semantic understanding
- **PII Detection & Protection**: Blocks or redacts sensitive information before sending to models
- **Prompt Guard**: Detects and blocks jailbreak attempts
- **Semantic Caching**: Reduces latency and costs through intelligent response caching
- **Category-Specific Prompts**: Injects domain-specific system prompts for better results
- **Tools Auto-Selection**: Automatically selects relevant tools for function calling

## Prerequisites

Before deploying, ensure you have:

1. **OpenShift Cluster** with OpenShift AI (RHOAI) installed
2. **KServe LLMInferenceService** already deployed and running
3. **OpenShift CLI (oc)** installed and logged in
4. **Cluster admin or namespace admin** permissions

## Quick Deployment

Use the `deploy.sh` script for automated deployment. It handles validation, model downloads, and resource creation:

```bash
./deploy.sh --namespace <namespace> --inferenceservice <name> --model <model>
```

**Example:**

```bash
./deploy.sh -n semantic -i granite32-8b -m granite32-8b
```

The script validates prerequisites, creates a stable service for your predictor, downloads classification models (~2-3 min), and deploys all resources. Optional flags include `--embedding-model`, `--storage-class`, `--models-pvc-size`, and `--cache-pvc-size`. For manual step-by-step deployment, continue reading below.

## Manual Deployment

### Step 1: Verify LLMInferenceService

Check that your LLMInferenceService is deployed and ready:

```bash
NAMESPACE=<your-namespace>
INFERENCESERVICE_NAME=<your-llminferenceservice-name>

# List LLMInferenceServices
oc get llminferenceservice -n $NAMESPACE

# Create stable ClusterIP service for predictor
cat <<EOF | oc apply -f - -n $NAMESPACE
apiVersion: v1
kind: Service
metadata:
  name: ${INFERENCESERVICE_NAME}-predictor-stable
spec:
  type: ClusterIP
  selector:
    serving.kserve.io/llm-inferenceservice: ${INFERENCESERVICE_NAME}
  ports:
  - name: http
    port: 8000
    targetPort: 8000
EOF

# Get the stable ClusterIP
PREDICTOR_SERVICE_IP=$(oc get svc "${INFERENCESERVICE_NAME}-predictor-stable" -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
echo "Predictor service ClusterIP: $PREDICTOR_SERVICE_IP"
```

### Step 2: Configure Router Settings

Edit `configmap-router-config.yaml`:

1. Update `vllm_endpoints` with your predictor service ClusterIP
2. Configure `model_config` with your model name and PII policies
3. Update `categories` with model scores for routing
4. Set `default_model` to your model name

Edit `configmap-envoy-config.yaml`:

1. Update `kserve_dynamic_cluster` address to: `<llminferenceservice>-predictor.<namespace>.svc.cluster.local`

### Step 3: Deploy Resources

Apply manifests in order:

```bash
NAMESPACE=<your-namespace>

# Deploy resources
oc apply -f serviceaccount.yaml -n $NAMESPACE
oc apply -f pvc.yaml -n $NAMESPACE
oc apply -f configmap-router-config.yaml -n $NAMESPACE
oc apply -f configmap-envoy-config.yaml -n $NAMESPACE
oc apply -f peerauthentication.yaml -n $NAMESPACE
oc apply -f deployment.yaml -n $NAMESPACE
oc apply -f service.yaml -n $NAMESPACE
oc apply -f route.yaml -n $NAMESPACE
```

### Step 4: Wait for Ready

Monitor deployment progress:

```bash
# Watch pod status
oc get pods -l app=semantic-router -n $NAMESPACE -w

# Check logs
oc logs -l app=semantic-router -c semantic-router -n $NAMESPACE -f
```

The pod will download models, then start serving traffic.

## Accessing Services

Get the route URL:

```bash
ROUTER_URL=$(oc get route semantic-router-kserve -n $NAMESPACE -o jsonpath='{.spec.host}')
echo "External URL: https://$ROUTER_URL"
```

Test the deployment:

```bash
# Test models endpoint
curl -k "https://$ROUTER_URL/v1/models"

# Domain-specific auto-routing (law -> Model-B)
curl -k -X POST "https://$ROUTER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Explain the elements of a contract under common law and give a simple example."}]}'

# Domain-specific auto-routing (math -> Model-A)
curl -k -X POST "https://$ROUTER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"What is 2+2?"}]}'
```

Run validation tests:

```bash
# Auto-detect configuration
./test-semantic-routing.sh

# Or specify explicitly
NAMESPACE=$NAMESPACE MODEL_NAME=<model> ./test-semantic-routing.sh
```

## Monitoring

### Check Deployment Status

```bash
# Check pods
oc get pods -l app=semantic-router -n $NAMESPACE

# Check services
oc get svc -n $NAMESPACE

# Check routes
oc get routes -n $NAMESPACE
```

### View Logs

```bash
# Router logs (models are downloaded automatically at startup)
oc logs -l app=semantic-router -c semantic-router -n $NAMESPACE -f

# Envoy logs
oc logs -l app=semantic-router -c envoy-proxy -n $NAMESPACE -f
```

### Metrics

```bash
# Port-forward metrics endpoint
POD=$(oc get pods -l app=semantic-router -n $NAMESPACE -o jsonpath='{.items[0].metadata.name}')
oc port-forward $POD 9190:9190 -n $NAMESPACE

# View metrics
curl http://localhost:9190/metrics
```

## Cleanup

Remove all deployed resources:

```bash
NAMESPACE=<your-namespace>

oc delete route semantic-router-kserve -n $NAMESPACE
oc delete service semantic-router-kserve -n $NAMESPACE
oc delete deployment semantic-router-kserve -n $NAMESPACE
oc delete configmap semantic-router-kserve-config semantic-router-envoy-kserve-config -n $NAMESPACE
oc delete pvc semantic-router-models semantic-router-cache -n $NAMESPACE
oc delete peerauthentication semantic-router-kserve-permissive -n $NAMESPACE
oc delete serviceaccount semantic-router -n $NAMESPACE
```

**Warning**: Deleting PVCs will remove downloaded models and cache data. To preserve data, skip PVC deletion.

## Troubleshooting

### Pod Not Starting

```bash
# Check pod status and events
oc get pods -l app=semantic-router -n $NAMESPACE
oc describe pod -l app=semantic-router -n $NAMESPACE

# Check init container logs (model download)
oc logs -l app=semantic-router -c model-downloader -n $NAMESPACE
```

**Common causes:**

- Network issues downloading models
- PVC not bound - check storage class
- Insufficient memory - increase init container resources

### Router Container Crashing

```bash
# Check router logs
oc logs -l app=semantic-router -c semantic-router -n $NAMESPACE --previous
```

**Common causes:**

- Configuration error - validate YAML syntax
- Invalid IP address - use ClusterIP not DNS in `vllm_endpoints.address`
- Missing models - verify init container completed

### Cannot Connect to LLMInferenceService

```bash
# Test from router pod
POD=$(oc get pods -l app=semantic-router -n $NAMESPACE -o jsonpath='{.items[0].metadata.name}')
oc exec $POD -c semantic-router -n $NAMESPACE -- \
  curl -v http://<llminferenceservice>-predictor.$NAMESPACE.svc.cluster.local:8000/v1/models
```

**Common causes:**

- LLMInferenceService not ready - check `oc get llminferenceservice -n $NAMESPACE`
- Wrong DNS name - verify format: `<llminferenceservice>-predictor.<namespace>.svc.cluster.local`
- Network policy blocking traffic
- mTLS mode mismatch - ensure PERMISSIVE mode in PeerAuthentication

## Configuration

For detailed configuration options, see the main project documentation:

- **Category Classification**: Train custom models at [Category Classifier Training](../../src/training/classifier_model_fine_tuning/)
- **PII Detection**: Train custom models at [PII Detection Training](../../src/training/pii_model_fine_tuning/)
- **Prompt Guard**: Train custom models at [Prompt Guard Training](../../src/training/prompt_guard_fine_tuning/)

## Related Documentation

### Within This Repository

- **[Category Classifier Training](../../src/training/classifier_model_fine_tuning/)** - Train custom category classification models
- **[PII Detector Training](../../src/training/pii_model_fine_tuning/)** - Train custom PII detection models
- **[Prompt Guard Training](../../src/training/prompt_guard_fine_tuning/)** - Train custom jailbreak detection models

### Other Deployment Options

- **[OpenShift Deployment](../openshift/)** - Deploy with standalone vLLM containers (not KServe)
- *This directory* - OpenShift AI KServe deployment (you are here)

### External Resources

- **Main Project**: https://github.com/vllm-project/semantic-router
- **Full Documentation**: https://vllm-semantic-router.com
- **KServe Docs**: https://kserve.github.io/website/
