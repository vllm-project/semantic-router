# Backend address discovery on OpenShift

[`config-openshift.yaml`](config-openshift.yaml) is a template. Its
`DYNAMIC_MODEL_A_IP` and `DYNAMIC_MODEL_B_IP` values are replaced by
[`deploy-to-openshift.sh`](deploy-to-openshift.sh) after the selected backend
Services exist.

```text
deploy backend Services
  -> read Service addresses
  -> render a temporary canonical Router config
  -> create or update the semantic-router-config ConfigMap
  -> roll out the Router
```

This keeps cluster-specific addresses out of Git, but it does not make a
ClusterIP a durable service-discovery contract. If your environment supports
stable Service DNS from the Router namespace, prefer DNS names in a maintained
deployment overlay.

## Verify the rendered config

```bash
oc get services --namespace vllm-semantic-router-system
oc get configmap semantic-router-config \
  --namespace vllm-semantic-router-system \
  -o jsonpath='{.data.config\.yaml}'
```

Confirm that:

- no `DYNAMIC_*` placeholder remains;
- each endpoint port matches its Service;
- the model names match the backend's served-model names;
- the Router pod mounts the current ConfigMap.

Probe the Services from inside the namespace if the Router cannot connect:

```bash
oc run endpoint-check --rm -i --restart=Never \
  --namespace vllm-semantic-router-system \
  --image=curlimages/curl -- \
  curl --fail-with-body http://MODEL_SERVICE:PORT/v1/models
```

Replace `MODEL_SERVICE:PORT` with the Service DNS name and port. Do not paste a
credential into this command.

## Common failures

- **Placeholder remains:** the deployment script did not find the expected
  Service; check the selected backend mode and namespace.
- **Address changed after deployment:** rerun the config render or move to
  Service DNS.
- **ConfigMap changed but Router did not:** inspect the deployment's volume and
  restart policy.
- **Connection refused:** check Service endpoints, target port, and backend
  readiness before changing Router policy.

The generated file is temporary deployment state. Do not commit it as a new
canonical config.
