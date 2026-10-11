# Semantic Router Operator Helm chart

Deploys the Semantic Router operator — the controller manager that watches
`SemanticRouter` custom resources and reconciles Router installs — with
Helm, as an alternative to the make-based install documented in
[`deploy/operator`](../../operator/README.md) (`make install` +
`make deploy`, which apply `kustomize build config/crd` and
`kustomize build config/default`).

This chart is an example proposed for
[vllm-project/semantic-router#4342](https://github.com/vllm-project/semantic-router/issues/4342).
It renders the same workload as the kustomize build: the CRD, the manager
Deployment, its ConfigMap, the leader-election Role, and the ClusterRole
with the operator's full rule set. Intentional differences are listed under
[Equivalence with the make install](#equivalence-with-the-make-install).

## Install

```bash
helm upgrade --install semantic-router-operator ./deploy/helm/semantic-router-operator \
  --namespace semantic-router-operator-system \
  --create-namespace
```

Pin a released operator image instead of the default main-channel tag:

```bash
helm upgrade --install semantic-router-operator ./deploy/helm/semantic-router-operator \
  --namespace semantic-router-operator-system \
  --create-namespace \
  --set image.tag=v0.4.0
```

## Upgrade

```bash
helm upgrade semantic-router-operator ./deploy/helm/semantic-router-operator \
  --namespace semantic-router-operator-system
```

## Uninstall

```bash
helm uninstall semantic-router-operator --namespace semantic-router-operator-system
```

Uninstalling removes the manager and its RBAC. It does **not** remove the
`SemanticRouter` CRD or any `SemanticRouter` custom resources (see below),
and resources the operator created on behalf of a `SemanticRouter` are left
for their owner to clean up, exactly as with `make undeploy`.

## CRD lifecycle

The `semanticrouters.vllm.ai` CRD ships verbatim from
`deploy/operator/config/crd/bases/` in the chart's `crds/` directory,
following the convention of the Router chart in
[`../semantic-router`](../semantic-router/). Standard Helm semantics apply:

- the CRD is installed when the chart is first installed;
- Helm does **not** upgrade the CRD when the chart is upgraded, and does
  **not** delete it when the chart is uninstalled.

To upgrade the CRD by hand, apply the file from the newer chart (or run
`make install` from `deploy/operator` at the same revision):

```bash
kubectl apply -f deploy/helm/semantic-router-operator/crds/vllm.ai_semanticrouters.yaml
```

Keeping the CRD in step with `config/crd/bases/` is a release-time step:
copy the generated base into `crds/` whenever it changes, the same way the
Router chart syncs its CRDs.

## Values

| Value | Default | Purpose |
| --- | --- | --- |
| `replicaCount` | `1` | Manager replicas. Keep at 1; leader election (below) arbitrates the active replica if raised. |
| `image.repository` | `ghcr.io/vllm-project/semantic-router-operator` | Manager image, the operator Makefile's `IMG` name. |
| `image.tag` | `""` (chart `appVersion`, currently `latest`) | Image tag; the Makefile default is likewise `latest`. Pin a release tag in production. |
| `image.pullPolicy` | `IfNotPresent` | As in the kustomize deployment. |
| `imagePullSecrets` | `[]` | Registry secrets for private mirrors. |
| `nameOverride` / `fullnameOverride` | `""` | Standard Helm naming overrides. |
| `serviceAccount.create` | `true` | Create the manager's ServiceAccount. |
| `serviceAccount.name` | `""` (generated: `<fullname>-controller-manager`) | ServiceAccount name. |
| `serviceAccount.annotations` | `{}` | ServiceAccount annotations. |
| `leaderElection.enabled` | `true` | Pass `--leader-elect` and install the leader-election Role/RoleBinding. |
| `extraArgs` | `[]` | Extra manager arguments, after `--leader-elect`. |
| `podAnnotations` | `kubectl.kubernetes.io/default-container: manager` | As in the kustomize deployment. |
| `podLabels` | `{}` | Extra pod labels (selector labels are fixed). |
| `podSecurityContext` | `runAsNonRoot: true`, `seccompProfile: RuntimeDefault` | As in the kustomize deployment. |
| `securityContext` | `allowPrivilegeEscalation: false`, `capabilities.drop: [ALL]` | Container security context, as in the kustomize deployment. |
| `resources` | limits `500m`/`512Mi`, requests `100m`/`128Mi` | As in the kustomize deployment. |
| `nodeSelector` / `tolerations` / `affinity` | `{}` / `[]` / `{}` | Standard scheduling controls. |

## Equivalence with the make install

Verified by rendering both at the same revision
(`helm template semantic-router-operator . --namespace semantic-router-operator-system`
against `kustomize build config/default` + `config/crd`):

- Same kinds with the same specs: Deployment (image, command, args,
  probes, resources, security contexts), ConfigMap content, leader-election
  Role/RoleBinding rules, ClusterRole rules (copied verbatim from
  `config/rbac/role.yaml`), and the CRD (byte-identical in `crds/`).
- **Names.** Helm names derive from the release: with the release name
  above, the Deployment is `semantic-router-operator-controller-manager`.
  Note the kustomize build currently produces doubled prefixes
  (`semantic-router-operator-semantic-router-operator-controller-manager`)
  because the source manifests already carry the
  `semantic-router-operator-` prefix and `config/default` adds it again as
  `namePrefix`. The chart deliberately renders the single-prefix names the
  source manifests intend; switching an existing make-installed operator to
  Helm therefore recreates the manager under the corrected names.
- **Namespace.** The kustomize build contains a `Namespace` object; Helm
  installs into the release namespace instead (`--create-namespace` creates
  it), per Helm convention.
- **Labels.** Resources carry the standard Helm labels in addition to the
  kustomize set (`app.kubernetes.io/name`, `app.kubernetes.io/component:
  manager`, `app.kubernetes.io/part-of: semantic-router`); the Deployment
  selector labels are identical, so pod identity is unchanged.

Webhooks and cert-manager are out of scope: the operator's default
kustomize build does not enable them either (the `[WEBHOOK]` and
`[CERTMANAGER]` sections of `config/crd` and `config/default` are scaffold
comments, and `config/webhook` is not part of `config/default`).
