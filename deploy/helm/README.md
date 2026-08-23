# Helm deployment

The chart in [`semantic-router/`](semantic-router/) deploys the Router and its
optional Dashboard, ingress, autoscaling, persistence, and observability
dependencies. Use this README when developing the chart. For an end-user
comparison of deployment methods, see
[Deployment options](../../website/docs/installation/deployment-options.md).

## Choose an install path

The CLI is the shortest path when you already have a canonical Router config:

```bash
vllm-sr serve --target k8s --profile dev
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

The Router image also carries the canonical built-in Recipe distribution. In
managed mode every replica reconciles it through PostgreSQL before becoming
ready, and a periodic idempotent worker installs it for later Namespaces. The
chart intentionally creates no Recipe catalog ConfigMap, volume, CRD, init
container, or Dashboard copy; the image and Router Management API remain the
only distribution and read paths.

Managed mode uses one stateless Router Deployment rather than separate control-
and data-plane workloads. A public ExtProc Service, private HTTPS Management
Service, and internal backend-dispatch Service select the same Pods. Durable
claims and consumer groups coordinate projectors and usage workers across HPA
replicas. The inference path reads applied routing, ProviderCredential, access,
and quota state from Valkey and never queries PostgreSQL.

A fresh managed installation is intentionally two-phase. Install the release without
`--wait`, then use the private Management Service to bootstrap identity and publish the
first complete routing revision. That Service can reach live Router Pods before their
inference readiness probe succeeds; it remains private, authenticated, TLS protected,
and NetworkPolicy scoped. After publication, `/ready` succeeds and normal Helm rollout
waiting applies. Do not disable the Router probes or expose the Management Service to
break the bootstrap gate.

```yaml
configOverride:
  version: v0.4
  models:
    - name: my-model
      card:
        capabilities: [chat]
      connections:
        - provider: vllm
          endpoint: http://my-vllm.default.svc.cluster.local:8000/v1
          model: my-model
  recipes:
    - name: default
      document:
        decisions:
          - name: default
            priority: 100
            rules: {}
  entrypoints:
    - name: vllm-sr/auto
      aliases: [auto]
      recipe: default
      assignments:
        default:
          models: [{model: my-model}]
```

Replace the Model and endpoint, then add the Recipe and Entrypoint behavior
required by your route. In managed mode, keep bootstrap configuration free of
inline Models, Recipes, and Entrypoints and publish those resources through the
Management API. Validate the canonical document before a release rather than
using Helm templating to invent a second schema.

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
