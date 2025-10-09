# OpenWebUI OpenShift Integration

This directory contains the OpenShift deployment manifests for OpenWebUI, integrated with the existing semantic-router deployment.

## Architecture

- **Namespace**: `vllm-semantic-router-system` (same as semantic-router)
- **Backend Integration**: Connects to Envoy proxy endpoint with load balancing
- **External Access**: Available via OpenShift Route with HTTPS
- **Storage**: Persistent volume for user data and configurations

## Quick Deployment

### Using Scripts (Recommended)

```bash
# Deploy OpenWebUI with full validation and setup
./deploy-openwebui-on-openshift.sh

# Uninstall OpenWebUI (preserves data by default)
./uninstall-openwebui.sh
```

### Using Kubernetes Manifests

```bash
# Deploy OpenWebUI manifests individually
oc apply -f pvc.yaml
oc apply -f deployment.yaml
oc apply -f service.yaml
oc apply -f route.yaml

# Check deployment status
oc get pods -n vllm-semantic-router-system -l app=openwebui

# Get the external URL
oc get route openwebui -n vllm-semantic-router-system -o jsonpath='{.spec.host}'
```

## Configuration

OpenWebUI is configured to connect to the Envoy proxy automatically:

- **Backend URL**: `http://semantic-router.vllm-semantic-router-system.svc.cluster.local:8801/v1`
- **Available Models**: `auto` (load balancer), `Model-A`, `Model-B`
- **Port**: Service exposed on port 3000, mapped to container port 8080
- **Storage**: 2Gi persistent volume for user data

### OpenWebUI Settings

When configuring OpenWebUI in the interface:

- **API Base URL**: `http://semantic-router.vllm-semantic-router-system.svc.cluster.local:8801/v1`
- **API Key**: `not-needed-for-local-models` (or leave empty)
- **Models**: Will auto-discover `auto`, `Model-A`, and `Model-B`

## Files

- `deploy-openwebui-on-openshift.sh` - Complete deployment script with validation
- `uninstall-openwebui.sh` - Safe uninstall script with data preservation options
- `deployment.yaml` - OpenWebUI deployment with OpenShift security contexts
- `service.yaml` - ClusterIP service exposing port 3000
- `route.yaml` - OpenShift route for external HTTPS access
- `pvc.yaml` - Persistent volume claim for data storage
- `kustomization.yaml` - Kustomize configuration for easy deployment

## Usage

1. **Deploy**: Run `./deploy-openwebui-on-openshift.sh`
2. **Access**: Open the provided HTTPS URL in your browser
3. **Configure**: Models are pre-configured and auto-discovered
4. **Chat**: Start conversations with Model-A, Model-B, or auto (load balanced)

## Cleanup

```bash
# Safe uninstall with data preservation option
./uninstall-openwebui.sh

# Or remove all resources immediately
oc delete -f route.yaml -f service.yaml -f deployment.yaml -f pvc.yaml
```

## Features

- **Zero-config Setup**: Automatically connects to semantic-router
- **Load Balancing**: Access both models through Envoy proxy
- **Persistent Data**: User conversations and settings preserved
- **OpenShift Security**: Runs with restricted security contexts
- **HTTPS Access**: Secure external access via OpenShift routes
- **Health Monitoring**: Built-in health checks and monitoring

## Troubleshooting

- **503 Errors**: Check if service endpoints are available with `oc get endpoints openwebui`
- **Connection Issues**: Verify semantic-router is running with `oc get pods -l app=semantic-router`
- **Model Discovery**: Test backend connectivity with the deployment script validation
