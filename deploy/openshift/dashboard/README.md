# Custom Dashboard with OpenWebUI Playground for OpenShift

This directory contains the OpenShift deployment configuration and custom build for the dashboard with OpenWebUI playground integration.

## Files

- `dashboard-deployment.yaml` - Kubernetes resources (Deployment, Service, Route, ConfigMap)
- `build-custom-dashboard.sh` - Builds custom dashboard image with OpenWebUI integration patches
- `PlaygroundPage.tsx.patch` - Frontend patch for OpenShift hostname-aware OpenWebUI URL construction
- `README.md` - This file

## Quick Start

### Prerequisites

1. OpenShift cluster with `oc` CLI configured
2. Semantic router and OpenWebUI already deployed in `vllm-semantic-router-system` namespace
3. Docker configured to access OpenShift internal registry

### Deploy Dashboard

**Single command deployment:**

```bash
./deploy/openshift/dashboard/build-custom-dashboard.sh
```

This script automatically:

1. Creates the `dashboard-custom` imagestream if needed
2. Builds the patched dashboard image with OpenWebUI integration
3. Pushes the image to the OpenShift internal registry
4. Applies the deployment YAML if the dashboard doesn't exist, or updates the image if it does
5. Waits for the rollout to complete

### Access the Dashboard

```bash
# Get the dashboard URL
oc get route dashboard -n vllm-semantic-router-system -o jsonpath='https://{.spec.host}'
```

Navigate to `/playground` to access the OpenWebUI playground.

## How It Works

### The PlaygroundPage.tsx Patch

OpenShift uses route-based URLs for services. The patch enables the frontend to:

1. Detect when running in OpenShift (by checking the hostname)
2. Dynamically construct the correct OpenWebUI route URL
3. Load OpenWebUI in the iframe using the direct route instead of an embedded proxy path

**Before (doesn't work in OpenShift):**

```javascript
const openWebUIUrl = '/embedded/openwebui/'
```

**After (works in OpenShift):**

```javascript
const getOpenWebUIUrl = () => {
  const hostname = window.location.hostname
  if (hostname.includes('dashboard-vllm-semantic-router-system')) {
    return hostname.replace('dashboard-vllm-semantic-router-system', 'openwebui-vllm-semantic-router-system')
  }
  return '/embedded/openwebui/'
}
```

### Why a Custom Build?

The upstream dashboard uses `localhost:3001` for local OpenWebUI development. In OpenShift:

- Services are accessed via routes with unique hostnames
- The OpenWebUI URL must be dynamically constructed based on the deployment environment
- The patch is applied during build time to inject this logic

## Notes

- Patches are maintained separately and not committed to dashboard/
- Only used for OpenShift demo deployment
- Original dashboard code remains untouched for upstream compatibility
