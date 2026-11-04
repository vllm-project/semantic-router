---
title: Deployment Architecture
description: The stable deployment contract for Semantic Router — component responsibilities, configuration ownership, Kubernetes APIs, versioning, and lifecycle semantics across Docker, Helm, the Operator, and OpenShift.
---

# Deployment Architecture

Every supported way of running Semantic Router assembles the same pieces:
one Router serving one canonical configuration, optionally behind a gateway,
in front of model backends the Router does not own. This page is the stable
contract those pieces follow. Installation guides show how to run each
stack; [Deployment Support](support-matrix) states which stacks the project
maintains and what evidence backs each claim; this page defines the
responsibilities, ownership, and versioning rules that keep the stacks
from drifting into incompatible variants. How the serving path itself
composes — frontend, decision engine, and model runtime — is covered in
[Component Architecture](../overview/component-architecture).

## Components and responsibilities

| Component | Owns | Does not own |
| --- | --- | --- |
| Router | Routing decisions, the canonical configuration document, configuration validation and activation, its inference and management listeners | Model serving, backend lifecycle, gateway traffic policy |
| Model runtime | Serving the Router's own decision, classifier, embedding, and reranking models, as managed or attached deployments | Chat/generation backends — those remain external providers reached through canonical backend references |
| CLI (`vllm-sr`) | The local Docker stack, local configuration files, host-side preflight | Cluster resources; it is not a Kubernetes installer |
| Dashboard | Editing and submitting canonical configuration through the Router/CLI management surfaces | The configuration schema — it renders the generated contract and never maintains a copy |
| Helm chart (Router) | The Router workload, Service, ConfigMap-seeded configuration, the `IntelligentPool`/`IntelligentRoute` CRDs, optional Dashboard and observability resources | Gateways, storage classes, and model servers, which stay external |
| Operator | Reconciling each `SemanticRouter` resource into a Router workload, Service, generated configuration, and optional autoscaling, ingress, and storage | Model servers (it discovers or references them), gateways (it integrates with, but does not install, them), and the `IntelligentPool`/`IntelligentRoute` APIs |
| Router Kubernetes reconciler | Watching `IntelligentPool`/`IntelligentRoute` in one namespace and activating the merged configuration in that Router process | Kubernetes workloads — it runs inside the Router it configures |
| Gateway (Envoy-based) | Client traffic entry, transport policy, and ext_proc invocation in `extproc` mode | Routing decisions and Router configuration |
| Inference platform (KServe, llm-d, AIBrix, Dynamo, …) | Model deployment, replicas, and scheduling for its backends | Router policy; the Router selects among the platform's targets |
| Storage dependency (Valkey, Redis, Milvus, Qdrant, …) | Durability, replication, and backup of the data it stores | Router semantics; the Router owns references to the store, not the store |

Two boundary rules follow from this table:

- The Operator is not an installer for external dependencies. It wires the
  Router to gateways, inference platforms, and stores that the environment
  already provides.
- No deployment surface creates a second configuration schema. Helm values,
  the `SemanticRouter` spec, and the Dashboard are producers of — or views
  onto — the one canonical document defined by the
  [Configuration Contract](configuration-contract).

## One canonical configuration, one source of truth per Router

The canonical document is versioned independently of any deployment API:
its current version is `v0.3`, its source of truth is the Go configuration
types, and a generated JSON Schema is the only copy consumers read.

Each Router instance has exactly one authoritative configuration source:

| Stack | Source of truth | How changes activate |
| --- | --- | --- |
| Local Docker | The configuration file managed by the CLI | CLI/Router apply flow; changes the running Router cannot take report `RESTART_REQUIRED` and apply on the next `serve` |
| Helm, file mode | The chart-managed ConfigMap, seeded from chart values on first install | Chart values seed the first install; later management-API writes persist to the ConfigMap and take effect on rollout, and Helm preserves the live document on upgrade by default |
| Helm, Kubernetes mode | `IntelligentPool` + `IntelligentRoute` in the Router's watched namespace | The in-Router reconciler validates the complete merged candidate, then activates it; CR-managed configuration is read-only through the management API |
| Operator | The `SemanticRouter` spec, translated by the controller into a generated canonical document | Reconciliation regenerates the workload configuration; convergence is visible in `status.observedGeneration` |

Mixing sources for one Router — for example, editing a ConfigMap that the
Operator generates, or pointing one Router at both a file and Kubernetes
CRs — is not supported. Choose the source when you choose the stack.

Configuration references are environment-neutral: backends are named
through canonical provider `backend_refs`, credentials are referenced
(Secret names, environment-provided values) rather than embedded per
environment, and [Backend Target Compatibility](backend-target-compatibility)
defines which target forms every producer must preserve.

## Supported topologies

Topology is two independent choices, covered in
[Gateway Modes](gateway-modes) and [Choose a Deployment](deployment-options):

- **Traffic entry:** `standalone`, where the Router serves the
  OpenAI-compatible API on its own listeners, or `extproc`, where an
  Envoy-based gateway in front of the Router invokes it over ext_proc.
  Both modes run the same routing core.
- **Target:** local Docker managed by the CLI, or Kubernetes through the
  Helm chart or the Operator. OpenShift is the Operator path with
  platform-specific defaults and optional Route creation; the shipped
  OpenShift manifests remain an experimental example in the
  [support matrix](support-matrix), not a separate topology.

Hardware never creates a topology. CPU, NVIDIA, and AMD support are
profiles overlaid on these stacks; see
[Hardware overlays](#hardware-profiles-are-overlays).

Scaling is per-Router: replica counts and autoscaling belong to the stack
that owns the workload (chart values and HPA, or `spec.replicas` /
`spec.autoscaling` on `SemanticRouter`). The Operator manager itself runs
as a single replica with leader election available for its supported
install paths. Multi-replica Dashboard state coordination is not yet part
of the maintained contract; it is tracked separately in
[issue #3168](https://github.com/vllm-project/semantic-router/issues/3168).

## Kubernetes APIs

Kubernetes has two distinct CRD families in the same API group. They are
not interchangeable, and each has exactly one consumer.

| | `SemanticRouter` | `IntelligentPool` + `IntelligentRoute` |
| --- | --- | --- |
| Group / version | `vllm.ai/v1alpha1` | `vllm.ai/v1alpha1` |
| Scope | Namespaced | Namespaced |
| Consumer | The Operator controller | The Router's in-process reconciler |
| Cardinality | One resource per Router deployment; multiple resources in a namespace or cluster are independent deployments | Exactly one pool and exactly one route per watched namespace; duplicates are reported as a `Conflict` condition and nothing is applied |
| Composition | One resource composes the whole deployment: workload, Service, ServiceAccount, generated ConfigMap, and optional PVC, HPA, and Ingress, all owned by the resource | Pool (providers and models) and route (signals and decisions) compose one Router configuration, merged with the Router's static global settings |
| Backend references | `spec.vllmEndpoints[]` discovery adapters (Kubernetes Service, KServe `InferenceService`, Llama Stack label selection), translated to canonical `backend_refs` and model cards | Canonical provider/model references in the pool; route decisions reference pool models |
| Configuration role | `spec.config` carries canonical routing plus Operator adapters for global settings, translated by the controller — the translation logic lives in the controller, not the API types | The pair *is* the routing configuration in Kubernetes form; global settings stay in the Router's static configuration |
| Status | `conditions`, `observedGeneration`, replica and ready-replica counts, `phase`, and the effective `gatewayMode` | A `Ready` condition per resource: `Ready=True` acknowledges activation on the reconciling replica; an invalid candidate reports `ValidationFailed`, a failed activation `ActivationFailed`, and the previously active configuration keeps serving in both cases |
| Admission | Validating webhook plus CEL validation on the CRD | Structural schema validation on the CRDs; complete-config semantic validation happens in the reconciler before activation |

Ownership rule: the Operator owns the deployment lifecycle of the Router;
the Router owns the routing configuration it activates. A Router deployed
by the Operator takes its configuration from `SemanticRouter`; a Router
deployed by the chart in Kubernetes mode takes its routing configuration
from the pool/route pair. Neither family is a translation of the other,
and neither may silently drop fields the other produces — a target form
with no handling path is documented as unsupported in
[Backend Target Compatibility](backend-target-compatibility) instead of
being lost.

## API versioning, compatibility, and deprecation

- **CRDs are alpha.** Both families publish a single version, `v1alpha1`,
  which is both served and storage. No conversion webhook exists, so
  there is currently no in-place conversion between versions: an
  incompatible CRD change must first ship as a new version with a
  documented conversion or migration path, and promotion beyond alpha
  requires that path to exist. Within alpha, breaking changes are called
  out in release notes and the shipped CRDs and samples are regenerated
  in the same change.
- **Unknown fields.** The CRD schemas are structural, so the API server
  prunes fields the schema does not declare. Subtrees explicitly marked
  to preserve unknown fields hold free-form canonical payloads; those
  payloads are not exempt from validation — the family's admission
  checks (webhook and CEL rules where defined) and the Router's semantic
  validators reject invalid content there.
- **Configuration versioning is separate.** The canonical document
  version (`v0.3`) evolves on its own cadence; its compatibility,
  migration, and retired-field rules are defined in the
  [Configuration Contract](configuration-contract#add-or-change-a-field).
  A CRD version and a configuration version never substitute for each
  other.
- **Management API.** The Router's management surface is versioned in
  its path (`/api/v1/…`), and the configuration contract it serves is
  identified by its schema identity and `ETag`, so consumers pin to a
  contract rather than to a deployment stack.
- **One release set.** Support covers artifacts taken from a single
  release — CLI, chart, CRDs, controller, and images together — as
  stated in [Deployment Support](support-matrix#use-one-tested-version-set).
  Mixing versions across a stack is outside the maintained contract.
- **Deprecation.** A deployment asset becomes **Deprecated** only with a
  documented replacement or removal path recorded in the support
  matrix. No shipped asset is currently deprecated.

## Lifecycle semantics

The lifecycle terms mean the same thing on every maintained stack:

| Stage | Contract on every stack |
| --- | --- |
| Install | Ends only when the Router reports ready with the intended configuration active — a running process with a failed or pending configuration is not installed |
| Upgrade | Moves one release set at a time; the previous configuration and image remain recoverable until the new generation reports ready |
| Configuration change | Validated as a complete candidate before activation; a rejected candidate leaves the active configuration serving |
| Drain | Stops new work being routed to the departing generation while in-flight requests finish, before the workload is replaced or removed |
| Rollback | Returns to the last known-good release set or configuration generation, using the stack's recorded state (`helm rollback`, the prior `SemanticRouter` spec, or the CLI's saved configuration) |
| Recovery | A restarted Router or controller resumes from its source of truth (file, ConfigMap, CR spec, or pool/route pair) without manual reconstruction |

The per-stack procedures live in [Upgrade and Rollback](upgrade-rollback)
and [Operate an Operator Deployment](k8s/operator-operations). Mapping each
maintained stack to reproducible install, upgrade, drain, rollback, and
recovery evidence is tracked in
[issue #3191](https://github.com/vllm-project/semantic-router/issues/3191);
the versioned health and partial-deployment status this lifecycle consumes
is tracked in
[issue #3189](https://github.com/vllm-project/semantic-router/issues/3189).

## Hardware profiles are overlays

CPU, NVIDIA CUDA, AMD ROCm, and Arm64 support qualify a stack; they never
fork it. A hardware profile may change image selection, device exposure,
and Router-side model placement, but it must not change configuration
semantics, CRD shapes, or lifecycle behavior. Profiles that cannot meet
that rule stay experimental in the
[support matrix](support-matrix#hardware-overlays) — currently the AMD AI
PC/NPU and NVIDIA DGX Spark profiles — until they qualify against the same
contract as the maintained stacks.

## Classification and review

Every deployment asset carries exactly one classification — maintained
reference stack, supported integration, experimental example, or
deprecated — recorded in [Deployment Support](support-matrix), and
repository validation (`tools/ci/check_deployment_support_matrix.py`)
fails when a deployment surface is unclassified. A change that adds a
deployment asset, a component responsibility, a CRD, or a configuration
source must update this page and the matrix in the same change; a change
that alters any rule on this page requires workgroup review, because
other stacks are entitled to rely on it.
