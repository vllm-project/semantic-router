# HuggingChat ChatUI Deployment for OpenShift

This directory contains OpenShift deployment manifests for HuggingFace ChatUI, which provides the HuggingChat interface accessible through the dashboard.

## Overview

ChatUI is deployed as a containerized web application that provides a chat interface similar to HuggingChat. It connects to the semantic-router service for LLM inference and uses MongoDB for data persistence.

## Components

- **Deployment**: ChatUI application container
- **Service**: ClusterIP service for internal access
- **Route**: OpenShift route for external access with TLS
- **Dependencies**: MongoDB (deployed separately)

## Configuration

The ChatUI deployment is configured via environment variables:

- `OPENAI_BASE_URL`: Points to semantic-router service (http://semantic-router:8801/v1)
- `MONGODB_URL`: MongoDB connection string (mongodb://mongo:27017)
- `PUBLIC_APP_NAME`: Display name ("HuggingChat")
- `LOG_LEVEL`: Logging level (info)

## Deployment

ChatUI is deployed automatically when running the full OpenShift deployment script:

```bash
cd deploy/openshift
./deploy-to-openshift.sh
```

### Manual Deployment

If deploying manually, ensure MongoDB is deployed first:

```bash
# Deploy MongoDB
oc apply -f deploy/openshift/mongo/deployment.yaml

# Deploy ChatUI
oc apply -f deploy/openshift/chatui/deployment.yaml
```

## Accessing ChatUI

### Through Dashboard

Access ChatUI through the semantic-router dashboard at `/huggingchat`:

```bash
DASHBOARD_URL=$(oc get route dashboard -n vllm-semantic-router-system -o jsonpath='{.spec.host}')
echo "HuggingChat: https://$DASHBOARD_URL/huggingchat"
```

### Direct Access

Get the ChatUI route URL:

```bash
CHATUI_URL=$(oc get route chatui -n vllm-semantic-router-system -o jsonpath='{.spec.host}')
echo "ChatUI direct URL: https://$CHATUI_URL"
```

## Integration with Dashboard

The dashboard backend proxies ChatUI at `/embedded/chatui/` and the frontend displays it in an iframe at `/huggingchat`. The integration requires:

1. **TARGET_CHATUI_URL environment variable** in dashboard ConfigMap pointing to `http://chatui:3000`
2. **ChatUI service** running and accessible within the cluster
3. **MongoDB service** running for ChatUI data persistence

## Troubleshooting

### ChatUI Not Loading

1. Check if ChatUI pod is running:

   ```bash
   oc get pods -l app=chatui -n vllm-semantic-router-system
   ```

2. Check ChatUI logs:

   ```bash
   oc logs -f deployment/chatui -n vllm-semantic-router-system
   ```

3. Verify MongoDB connection:

   ```bash
   oc get pods -l app=mongo -n vllm-semantic-router-system
   ```

### Dashboard Shows "HuggingChat not configured" Error

This means the `TARGET_CHATUI_URL` environment variable is not set in the dashboard deployment. Verify the dashboard ConfigMap:

```bash
oc get configmap dashboard-config -n vllm-semantic-router-system -o yaml | grep CHATUI
```

If missing, redeploy the dashboard with the updated configuration.

## Security

- **OpenShift Security Contexts**: Runs with restricted security contexts (no privilege escalation, dropped capabilities)
- **HTTPS Access**: Secure external access via OpenShift routes with TLS edge termination
- **Network Policies**: Internal cluster communication only (no direct external access to backend services)

## Resource Requirements

- **CPU**: 100m request, 500m limit
- **Memory**: 256Mi request, 1Gi limit
- **Depends on**: MongoDB (additional resources)

## Notes

- ChatUI uses the same semantic-router backend as OpenWebUI
- Both chat interfaces (ChatUI and OpenWebUI) are available through the dashboard
- ChatUI requires MongoDB while OpenWebUI has its own data storage
- The dashboard automatically configures the proxy routing based on the TARGET_CHATUI_URL environment variable
