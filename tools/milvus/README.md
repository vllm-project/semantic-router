# Milvus Installation Validation Script

Validates commands in `website/docs/installation/milvus.md`.

## Features

1. Prerequisites check (kubectl, kind, helm)
2. Create Kind cluster (`make create-cluster`)
3. Deploy Milvus (Standalone or Cluster mode)
4. Verify deployment
5. Apply client config & network policies
6. Connection tests

## Deployment Modes

| Mode           | Use Case            |
| -------------- | ------------------- |
| **Standalone** | Development/testing |
| **Cluster**    | Production (HA)     |

## Usage

**Interactive:**

```bash
./tools/milvus/test-milvus-deployment.sh
```

**Non-Interactive (CI/CD):**

```bash
MILVUS_MODE=standalone RECREATE_CLUSTER=false CLEANUP=false ./tools/milvus/test-milvus-deployment.sh
```

### Environment Variables

| Variable           | Values                  | Description                     |
| ------------------ | ----------------------- | ------------------------------- |
| `MILVUS_MODE`      | `standalone`, `cluster` | Deployment mode                 |
| `RECREATE_CLUSTER` | `true`, `false`         | Recreate Kind cluster if exists |
| `CLEANUP`          | `true`, `false`         | Cleanup after test              |

## Troubleshooting

**ServiceMonitor CRD Not Found:**

```bash
# Add: --set metrics.serviceMonitor.enabled=false
```

**Both Pulsar versions running:**

```bash
# Add: --set pulsar.enabled=false --set pulsarv3.enabled=true
```
