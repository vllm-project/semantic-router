# Quick Start Guide - Semantic Router with KServe

**🚀 Automated deployment in under 5 minutes using the helper script.**

> **Need more control?** See [README.md](./README.md) for comprehensive manual deployment and configuration.
>
> This quick start uses the automated `deploy.sh` script for the fastest path to deployment.

## Prerequisites Checklist

- [ ] OpenShift cluster with OpenShift AI installed
- [ ] At least one KServe InferenceService deployed and ready
- [ ] OpenShift CLI (`oc`) installed
- [ ] Logged in to your cluster (`oc login`)
- [ ] Sufficient permissions in your namespace

## 5-Minute Deployment

### Step 1: Verify Your Model

```bash
# Set your namespace
NAMESPACE=<your-namespace>

# List your InferenceServices
oc get inferenceservice -n $NAMESPACE

# Note the InferenceService name and verify it's READY=True
```

### Step 2: Deploy Semantic Router

```bash
cd deploy/kserve

# Deploy with one command
./deploy.sh \
  --namespace <your-namespace> \
  --inferenceservice <your-inferenceservice-name> \
  --model <your-model-name>
```

**Example:**

```bash
./deploy.sh --namespace semantic --inferenceservice granite32-8b --model granite32-8b
```

### Step 3: Wait for Ready

The script will:

- ✓ Validate your environment
- ✓ Download classification models (~2-3 minutes)
- ✓ Start the semantic router
- ✓ Provide your external URL

### Step 4: Test It

```bash
# Use the URL provided by the deployment script
ROUTER_URL=<your-route-url>

# Quick test
curl -k "https://$ROUTER_URL/v1/models"

# Try a chat completion
curl -k "https://$ROUTER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<your-model>",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }'
```

## Common Scenarios

### Scenario 1: Basic Deployment (Default Settings)

Just need semantic routing with defaults:

```bash
./deploy.sh -n myproject -i mymodel -m mymodel
```

### Scenario 2: Custom Storage and Embedding Model

Using a specific storage class, larger PVCs, and custom embedding model:

```bash
./deploy.sh \
  -n myproject \
  -i mymodel \
  -m mymodel \
  -s gp3-csi \
  --models-pvc-size 20Gi \
  --cache-pvc-size 10Gi \
  --embedding-model all-mpnet-base-v2
```

**Available Embedding Models:**

- `all-MiniLM-L12-v2` (default) - Balanced speed/quality (~90MB)
- `all-mpnet-base-v2` - Higher quality, larger (~420MB)
- `all-MiniLM-L6-v2` - Faster, smaller (~80MB)
- `paraphrase-multilingual-MiniLM-L12-v2` - Multilingual support

### Scenario 3: Preview Before Deploying

Want to see what will be created first:

```bash
./deploy.sh -n myproject -i mymodel -m mymodel --dry-run
```

## What You Get

Once deployed, you have:

✅ **Intelligent Routing** - Requests route based on semantic understanding
✅ **PII Protection** - Sensitive data detection and blocking
✅ **Semantic Caching** - ~50% faster responses for similar queries
✅ **Jailbreak Detection** - Security against prompt injection
✅ **OpenAI Compatible API** - Drop-in replacement for OpenAI endpoints
✅ **Production Ready** - Monitoring, logging, and metrics included

## Accessing Your Deployment

### External URL

```bash
# Get your route
oc get route semantic-router-kserve -n <namespace>

# Access via HTTPS
ROUTER_URL=$(oc get route semantic-router-kserve -n <namespace> -o jsonpath='{.spec.host}')
echo "https://$ROUTER_URL"
```

### Logs

```bash
# View router logs
oc logs -l app=semantic-router -c semantic-router -n <namespace> -f

# View all logs
oc logs -l app=semantic-router --all-containers -n <namespace> -f
```

### Metrics

```bash
# Port-forward metrics endpoint
POD=$(oc get pods -l app=semantic-router -n <namespace> -o jsonpath='{.items[0].metadata.name}')
oc port-forward $POD 9190:9190 -n <namespace>

# View in browser
open http://localhost:9190/metrics
```

## Integration Examples

### Python (OpenAI SDK)

```python
from openai import OpenAI

# Point to your semantic router
client = OpenAI(
    base_url="https://<your-router-url>/v1",
    api_key="not-needed"  # KServe doesn't require API key by default
)

# Use like normal OpenAI
response = client.chat.completions.create(
    model="<your-model>",
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)

print(response.choices[0].message.content)
```

### cURL

```bash
curl -k "https://<your-router-url>/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<your-model>",
    "messages": [
      {"role": "user", "content": "Write a Python function to calculate fibonacci"}
    ],
    "max_tokens": 500
  }'
```

### LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="https://<your-router-url>/v1",
    model="<your-model>",
    api_key="not-needed"
)

response = llm.invoke("What are the benefits of semantic routing?")
print(response.content)
```

## Troubleshooting Quick Fixes

### Pod Not Starting

```bash
# Check pod status
oc get pods -l app=semantic-router -n <namespace>

# View events
oc describe pod -l app=semantic-router -n <namespace>

# Check init container logs (model download)
oc logs -l app=semantic-router -c model-downloader -n <namespace>
```

### Can't Connect to InferenceService

```bash
# Test connectivity from router pod
POD=$(oc get pods -l app=semantic-router -o jsonpath='{.items[0].metadata.name}')
oc exec $POD -c semantic-router -n <namespace> -- \
  curl http://<inferenceservice>-predictor.<namespace>.svc.cluster.local:8080/v1/models
```

### Predictor Pod Restarted (IP Changed)

Simply redeploy:

```bash
./deploy.sh -n <namespace> -i <inferenceservice> -m <model>
```

## Next Steps

1. **Run validation tests**:

   ```bash
   # Set namespace and model name
   NAMESPACE=<namespace> MODEL_NAME=<model> ./test-semantic-routing.sh

   # Or let the script auto-detect from your deployment
   cd deploy/kserve
   ./test-semantic-routing.sh
   ```

2. **Customize configuration**: See [README.md](./README.md) for detailed configuration options:
   - Adjust category scores and routing logic
   - Configure PII policies and prompt guards
   - Tune semantic caching parameters
   - Set up multi-model routing
   - Configure monitoring and tracing

3. **Advanced topics**: [README.md](./README.md) covers:
   - Multi-model configuration
   - Horizontal and vertical scaling
   - Troubleshooting guides
   - Monitoring and observability
   - Production hardening

## Getting Help

- 📖 **Manual Deployment & Configuration**: [README.md](./README.md) - comprehensive guide
- 🌐 **Project Website**: https://vllm-semantic-router.com
- 💬 **GitHub Issues**: https://github.com/vllm-project/semantic-router/issues
- 📚 **KServe Docs**: https://kserve.github.io/website/

## Want More Control?

This quick start uses the automated `deploy.sh` script for simplicity. If you need:

- Manual step-by-step deployment
- Deep understanding of configuration options
- Advanced customization
- Troubleshooting guidance
- Production hardening tips

**See the comprehensive [README.md](./README.md) guide.**

## Cleanup

To remove the deployment:

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

---

**Questions?** Check the [README.md](./README.md) for detailed documentation or open an issue on GitHub.
