# MongoDB Deployment for OpenShift

This directory contains OpenShift deployment manifests for MongoDB, which provides data persistence for the ChatUI (HuggingChat) application.

## Overview

MongoDB is deployed as a stateful database service that stores ChatUI conversation history, user sessions, and application data.

## Components

- **PersistentVolumeClaim**: 5Gi storage for MongoDB data
- **Deployment**: MongoDB 7 container
- **Service**: ClusterIP service for internal access

## Configuration

- **Image**: mongo:7
- **Port**: 27017
- **Storage**: 5Gi persistent volume mounted at `/data/db`
- **fsGroup**: 999 (MongoDB user)

## Deployment

MongoDB is deployed automatically when running the full OpenShift deployment script:

```bash
cd deploy/openshift
./deploy-to-openshift.sh
```

### Manual Deployment

```bash
oc apply -f deploy/openshift/mongo/deployment.yaml
```

## Verification

Check MongoDB deployment status:

```bash
# Check pod status
oc get pods -l app=mongo -n vllm-semantic-router-system

# Check service
oc get service mongo -n vllm-semantic-router-system

# Check PVC
oc get pvc mongo-data -n vllm-semantic-router-system

# View MongoDB logs
oc logs -f deployment/mongo -n vllm-semantic-router-system
```

## Health Checks

MongoDB includes liveness and readiness probes using the `mongosh` command:

```bash
# Liveness probe (every 10s after 30s delay)
mongosh --eval "db.adminCommand('ping')"

# Readiness probe (every 5s after 10s delay)
mongosh --eval "db.adminCommand('ping')"
```

## Connectivity Test

Test MongoDB connectivity from within the cluster:

```bash
# From another pod
oc run -it --rm mongo-test --image=mongo:7 --restart=Never -- \
  mongosh --host mongo.vllm-semantic-router-system.svc.cluster.local --eval "db.adminCommand('ping')"
```

## Security

- **OpenShift Security Contexts**: Runs with restricted security contexts
  - No privilege escalation allowed
  - All capabilities dropped
  - seccompProfile: RuntimeDefault
- **fsGroup**: Set to 999 for proper file permissions
- **Network Access**: Internal cluster access only (ClusterIP service)

## Storage

- **Volume Type**: PersistentVolumeClaim (RWO - ReadWriteOnce)
- **Storage Size**: 5Gi
- **Mount Path**: `/data/db`
- **Storage Class**: Uses OpenShift default storage class

To change storage size, edit the PVC in deployment.yaml before deploying:

```yaml
resources:
  requests:
    storage: 10Gi  # Change from 5Gi to 10Gi
```

## Resource Limits

- **CPU Request**: 100m
- **CPU Limit**: 500m
- **Memory Request**: 256Mi
- **Memory Limit**: 1Gi

## Backup and Recovery

For production deployments, consider implementing MongoDB backup strategies:

1. **Volume Snapshots**: Use OpenShift volume snapshot capabilities
2. **mongodump**: Regular database exports
3. **Persistent Volume Backups**: Backup the PV data

### Example mongodump

```bash
oc exec deployment/mongo -- mongodump --archive=/data/db/backup-$(date +%Y%m%d).archive --gzip
```

## Troubleshooting

### Pod Not Starting

1. Check pod events:

   ```bash
   oc describe pod -l app=mongo -n vllm-semantic-router-system
   ```

2. Check PVC status:

   ```bash
   oc get pvc mongo-data -n vllm-semantic-router-system
   oc describe pvc mongo-data -n vllm-semantic-router-system
   ```

3. Check security context constraints:

   ```bash
   oc get pod -l app=mongo -n vllm-semantic-router-system -o yaml | grep -A 10 securityContext
   ```

### Connection Issues

1. Verify service endpoints:

   ```bash
   oc get endpoints mongo -n vllm-semantic-router-system
   ```

2. Test from ChatUI pod:

   ```bash
   oc exec deployment/chatui -- sh -c 'nc -zv mongo.vllm-semantic-router-system.svc.cluster.local 27017'
   ```

### Performance Issues

1. Check resource usage:

   ```bash
   oc adm top pod -l app=mongo -n vllm-semantic-router-system
   ```

2. Review MongoDB logs for slow queries:

   ```bash
   oc logs deployment/mongo -n vllm-semantic-router-system | grep slow
   ```

## Dependencies

- **ChatUI**: Requires MongoDB for data persistence
  - Connection string: `mongodb://mongo.vllm-semantic-router-system.svc.cluster.local:27017`
  - Database name: `chatui`

## Notes

- MongoDB 7 is used for compatibility with ChatUI requirements
- No authentication is configured (acceptable for internal cluster use)
- For production, consider enabling MongoDB authentication and TLS
- The deployment uses `emptyDir` in Kubernetes but PVC in OpenShift for data persistence
