# Helm deployment

The chart in [`semantic-router/`](semantic-router/) deploys the Router and its
optional Dashboard, ingress, autoscaling, persistence, and observability
dependencies. Use this README when developing the chart. For an end-user
comparison of deployment methods, see
[Deployment options](../../website/docs/installation/deployment-options.md).

## Choose an install path

The CLI is the shortest path when you already have a canonical Router config:

```bash
vllm-sr serve --target k8s --profile dev --config config/config.yaml
vllm-sr status --target k8s
```

Use Helm directly when a GitOps or platform workflow owns releases and values:

```bash
helm upgrade --install semantic-router ./deploy/helm/semantic-router \
  --namespace vllm-semantic-router-system \
  --create-namespace
```

The CLI translates canonical Router YAML into chart values and invokes Helm.
Helm users can instead set `configOverride` to a complete canonical Router
document. Do not maintain a second, hand-converted configuration.

```yaml
configOverride:
  version: v0.3
  providers:
    defaults:
      default_model: my-model
    models:
      - name: my-model
        backend_refs:
          - name: primary
            endpoint: my-vllm.default.svc.cluster.local:8000
  routing:
    modelCards:
      - name: my-model
    decisions: []
```

Replace the model and endpoint and add the signals, decisions, algorithms, and
plugins required by your route. Validate the canonical document before a
release rather than using Helm templating to invent a second schema.

## Values and profiles

- [`semantic-router/values.yaml`](semantic-router/values.yaml) contains defaults.
- [`semantic-router/values-dev.yaml`](semantic-router/values-dev.yaml) is a
  smaller local-development profile.
- [`semantic-router/values-prod.yaml`](semantic-router/values-prod.yaml) enables
  the production-oriented replica, autoscaling, storage, and security defaults
  defined by this repository. Review them against your cluster requirements.
- [`semantic-router/values.schema.json`](semantic-router/values.schema.json)
  rejects invalid public value types before rendering.
- [`semantic-router/README.md`](semantic-router/README.md) is the generated value
  reference.

Create a separate values file for environment-specific endpoints, images,
resources, Secret references, storage classes, and ingress. Avoid editing
`values.yaml` for one cluster.

## Credentials

The CLI places `HF_TOKEN`, `OPENAI_API_KEY`, and `ANTHROPIC_API_KEY` in the
`vllm-sr-env-secrets` Secret instead of plain-text Helm values. With direct
Helm, create a Secret and reference it:

```bash
kubectl create secret generic vllm-sr-env-secrets \
  --namespace vllm-semantic-router-system \
  --from-literal=HF_TOKEN="$HF_TOKEN"
```

```yaml
envFromSecrets:
  - vllm-sr-env-secrets
```

Use an external secret controller in shared or production clusters. Do not
commit credentials to a values file.

## Operate a release

Repository Make targets provide stable wrappers around common Helm commands:

```bash
make helm-install-or-upgrade
make helm-status
make helm-logs
make helm-port-forward-api
make helm-rollback
make helm-uninstall
```

Run `make help` for supported variables such as release, namespace, context,
and values-file overrides. The chart's `NOTES.txt` prints endpoints for the
rendered release.

## Validate chart changes

```bash
make helm-lint
make helm-template
make helm-ci-validate HELM_REPO_UPDATE=false
make helm-safety-validate HELM_REPO_UPDATE=false
```

`helm-ci-validate` resolves dependencies and renders the maintained profiles.
`helm-safety-validate` checks value-schema and local-state safety guards, such
as unsupported shared use of replica-local learning state.

When a chart value changes, update `values.yaml`, `values.schema.json`, the
affected templates, and the generated
[`semantic-router/README.md`](semantic-router/README.md) together.

## Common failures

- **Image pull errors:** confirm the tag, registry credentials, and any
  `global.imageRegistry` override.
- **Model download errors:** check the token Secret, network policy, writable
  cache volume, and Router logs.
- **Pending pods:** inspect events and verify node resources, PVC binding, and
  scheduling constraints.
- **No service endpoints:** compare pod labels with the rendered Service
  selectors using `make helm-template`.

Configuration ownership, upgrades, and security controls are covered in the
[configuration workflow](../../website/docs/installation/configuration-workflows.md),
[upgrade and rollback](../../website/docs/installation/upgrade-rollback.md),
and [security hardening](../../website/docs/installation/security-hardening.md)
guides.
