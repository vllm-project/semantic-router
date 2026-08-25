---
title: Router-Native Management Authorization Appendix
description: Specifies Management permissions, role presets, scope containment, safe delegation, and operation authorization.
created: 2026-08-22
status: Proposal
---

> **Status:** Proposal appendix · **Created:** 2026-08-22

This appendix is normative for Management authorization in
[Router-Native Access Control and Quota Accounting](./router-native-access-control).
The [Management API appendix](./router-native-access-control-management-api) owns
authentication, endpoints, requests, and responses. The
[resource contract](./router-native-access-control-contracts) owns persisted role and
scope shapes.

## Permission registry

Management authorization is a permission plus a resource scope. Role names are
presets; permissions are authoritative and may be composed into custom roles.

Authorization also requires the target action's typed authentication predicate.
Human sessions satisfy only the configured AAL/AMR/freshness branch; service-token or
mTLS sessions satisfy only the workload-class/source-age branch. Neither evidence
kind is coerced into the other. RBAC and scope are evaluated first, then the relevant
cluster or namespace predicate; satisfying stronger authentication never adds a
permission or widens scope.

| Permission | Boundary |
| --- | --- |
| <code>self.read</code>, <code>self.manage</code> | Intrinsic, non-delegable access to the current principal, its sessions, and its namespace-linked User; never another subject. |
| <code>cluster.read</code>, <code>cluster.manage</code> | Installation state, namespace lifecycle, and bootstrap-owned global resources. |
| <code>namespace.read</code>, <code>namespace.manage</code> | Namespace metadata and self-service defaults. |
| <code>identity_issuer.read</code>, <code>identity_issuer.manage</code> | Trusted issuer and exact mTLS identity-mapping metadata, key refresh, and lifecycle. |
| <code>principal.read</code>, <code>principal.manage</code> | Cluster-scoped lifecycle of global Management principals. |
| <code>principal_directory.read</code>, <code>principal_link.read</code>, <code>principal_link.manage</code> | Safe principal lookup and namespace-local Principal/User links without global lifecycle authority. |
| <code>management_role.read</code>, <code>management_role.manage</code> | Built-in role discovery and custom role definitions. |
| <code>role_binding.read</code>, <code>role_binding.manage</code> | Administrative role assignments. |
| <code>service_account.read</code>, <code>service_account.manage</code> | Automation identities and their credentials. |
| <code>invitation.read</code>, <code>invitation.manage</code>, <code>onboarding.manage</code> | One-time invitations and privileged onboarding. |
| <code>user.read</code>, <code>user.manage</code> | Router Users within scope. |
| <code>team.read</code>, <code>team.manage</code>, <code>membership.manage</code> | Teams and memberships within scope. |
| <code>key.read</code>, <code>key.manage</code>, <code>key.reveal</code> | Logical keys, lifecycle, and separately privileged reveal. |
| <code>delegation.use</code>, <code>delegation.manage</code> | Create/list/revoke only the actor's own short-lived inference sessions, or administer every session for an authorized key. |
| <code>access_policy.read</code>, <code>access_policy.manage</code> | Grants and access bindings. |
| <code>rate_policy.read</code>, <code>rate_policy.manage</code> | Quota rules and rate bindings. |
| <code>routing_context.read</code>, <code>routing_context.manage</code> | Typed Key/User/Team routing values and namespace claim schema; never model access. |
| <code>routing.read</code>, <code>routing.manage</code> | Models, Recipes, Entrypoints, topology, and probes. |
| <code>routing.publish</code>, <code>evaluation.run</code> | Explicit publication authority and evaluation execution. |
| <code>agent.read</code>, <code>agent.use</code>, <code>agent.manage</code> | Scoped Agent discovery/session use and Agent definition administration. |
| <code>tool.read</code>, <code>tool.invoke</code>, <code>tool.manage</code> | Scoped tool discovery/use and Tool Source administration. |
| <code>provider_catalog.read</code> | Registered provider schemas and capabilities; never credentials. |
| <code>provider_credential.read</code>, <code>provider_credential.manage</code>, <code>provider_credential.use</code> | Backend credential metadata, lifecycle, and separately authorized use by discovery or Model backends; never reveal. |
| <code>quota.read</code>, <code>quota.reconcile</code> | Live meters and privileged unknown-usage resolution. |
| <code>usage.read</code>, <code>usage.internal_dimensions.read</code> | Scoped aggregates and separately protected backend/provider/dispatch dimensions. |
| <code>log.read</code>, <code>log_payload.read</code>, <code>audit.read</code> | Request metadata, separately protected payloads, and audit. |
| <code>operation.read</code>, <code>operation.manage</code>, <code>health.read</code> | Async progress/cancel and authenticated runtime diagnostics. |

## Exact built-in roles

Built-in roles are immutable and expand to these exact permission sets; there are no
hidden wildcard permissions:

~~~yaml
cluster_admin:
  [cluster.read, cluster.manage, namespace.read, namespace.manage,
   identity_issuer.read, identity_issuer.manage, principal.read, principal.manage,
   management_role.read, management_role.manage, role_binding.read,
   role_binding.manage, service_account.read, service_account.manage, audit.read,
   health.read]
platform_admin:
  [namespace.read, namespace.manage, principal_directory.read, principal_link.read,
   principal_link.manage,
   management_role.read, management_role.manage, role_binding.read,
   role_binding.manage, service_account.read, service_account.manage,
   invitation.read, invitation.manage, onboarding.manage, user.read, user.manage,
   team.read, team.manage, membership.manage, key.read, key.manage,
   delegation.use, delegation.manage, access_policy.read, access_policy.manage, rate_policy.read,
   rate_policy.manage, routing_context.read, routing_context.manage, routing.read,
   routing.manage, routing.publish, evaluation.run,
   agent.read, agent.use, agent.manage, tool.read, tool.invoke, tool.manage,
   provider_catalog.read,
   provider_credential.read, provider_credential.manage, provider_credential.use,
   quota.read, quota.reconcile, usage.read, usage.internal_dimensions.read,
   log.read, log_payload.read, audit.read, operation.read, operation.manage,
   health.read]
operator:
  [namespace.read, routing.read, routing.manage, routing.publish, evaluation.run,
   agent.read, agent.use, agent.manage, tool.read, tool.invoke, tool.manage,
   provider_catalog.read,
   provider_credential.read, provider_credential.manage, provider_credential.use,
   access_policy.read, rate_policy.read, routing_context.read,
   routing_context.manage, quota.read, usage.read,
   usage.internal_dimensions.read, log.read, log_payload.read, operation.read,
   operation.manage, health.read]
access_admin:
  [namespace.read, principal_directory.read, principal_link.read,
   principal_link.manage, invitation.read, invitation.manage,
   onboarding.manage, user.read, user.manage, team.read, team.manage,
   membership.manage, key.read, key.manage, delegation.use, delegation.manage, access_policy.read,
   access_policy.manage, rate_policy.read, rate_policy.manage, routing_context.read,
   routing_context.manage, routing.read, quota.read,
   quota.reconcile, usage.read, audit.read, operation.read, operation.manage]
credential_revealer: [key.read, key.reveal]
analyst: [namespace.read, user.read, team.read, key.read, quota.read, usage.read,
          log.read, audit.read]
viewer: [routing.read, provider_catalog.read, agent.read, tool.read]
consumer: [user.read, team.read, key.read, delegation.use,
           access_policy.read, rate_policy.read, routing_context.read, quota.read,
           usage.read, operation.read, agent.read, agent.use, tool.read,
           tool.invoke]
~~~

`key.reveal` is absent from every broad administrator preset and must be assigned
deliberately. A `consumer` binding is always User-scoped and deliberately excludes
`key.manage`: it can inspect its own key, effective policy, quota, usage, and Operations
and can create delegated Playground sessions. Its key-scoped `routing-catalog` view
is compiled from that key's applied grants and exposes only discoverable Models,
Entrypoints, and their safe topology metadata; it does not require or imply
namespace-wide `routing.read`. The consumer cannot create, rotate, reassign, or delete
a key or alter its access/rate bindings. An invitation may materialize the
namespace's automatic first key in its single authorized acceptance transaction;
later key lifecycle requires a separately delegated, scope-contained permission.

ManagementPrincipal bindings are the Dashboard and Management API authority. Access
Users, Teams, memberships, policies, budgets, and API keys are inference-service
resources. A Principal/User link composes the two without merging them: the Management
role controls which console surfaces and actions are visible, while the linked User,
optional Team membership, and the API key's effective policy control which Models can
be called and which quotas are charged. The `viewer` preset is therefore only a safe
routing/catalog view. The User-scoped `consumer` preset adds the caller's own account
views; intrinsic `self.read` and `self.manage` come from the authenticated principal,
not from either preset and never authorize another User.

The installation bootstrap creates the first `cluster_admin` binding with a fixed
delegation ceiling containing the complete registered permission set, including
`key.reveal`; that ceiling grants no usable permission. It lets the cluster
administrator deliberately establish namespace administrators and a narrowly scoped
`credential_revealer` without gaining reveal capability itself. Later bindings have
an empty ceiling unless an authorized grantor explicitly supplies a contained one.

## Scope containment and safe delegation

A role-binding scope is a discriminated object, not an untyped ID:

~~~text
cluster
namespace(namespace_id)
team(namespace_id, team_id)
user(namespace_id, user_id)
resource(namespace_id, resource_type, resource_id)
~~~

Cluster contains every namespace and every global principal, issuer, mTLS mapping,
and cluster-owned service account. Namespace does not contain global principal
lifecycle; it contains its Principal/User links, namespace-scoped role bindings and
roles, and namespace-owned service accounts whose principals cannot receive authority
elsewhere. Team contains the Team, its memberships, Team-owned keys and their child
credentials/delegations, Team-subject policy bindings, Team aggregate usage, and
operations whose complete target set stays inside those resources. It does not
contain a member's User-owned keys, User-level bindings, raw request payloads, or
other member-private records. User contains that User, its membership rows,
User-owned keys and children, User-subject policy bindings, User-attributed usage/log
metadata, and operations confined to those resources. Resource scope contains the
exact typed resource and only schema-declared children: an API key contains its
credential versions, delegated sessions, effective-policy, quota, and usage views;
a policy contains its rules/grants but not a binding to an unrelated subject. Every
other resource scope is exact. Operations inherit the intersection of all domain
targets rather than becoming a new authority island.

An action that relates resources requires every permission/scope operand. For example,
creating a binding requires policy manage on the policy and subject manage on the
target subject; key reassignment requires key manage on the key and subject manage on
both old and new owners; a Model backend reference requires routing manage on the
Model and provider-credential use on the credential; and unknown-fence reconciliation
requires quota reconcile over every affected binding. Bulk operations authorize the
whole requested set at enqueue and each item again at execution.

A role binding stores an explicit `delegation_ceiling` permission set, separate from
the role's usable permissions. Creating a custom role requires
`management_role.manage` and one source ceiling containing the custom set. Creating
or changing a binding requires `role_binding.manage` plus one active source binding
whose ceiling contains both the target role's entire permission set and any target
ceiling, and whose scope contains the target scope. A caller cannot combine disjoint
sources, use a ceiling as runtime authority, or delegate intrinsic `self.*` or
TeamRole entitlements. The same test runs in the database transaction and again
before publication.

Custom-role permission sets are immutable. PATCH changes descriptive metadata only;
changing permissions creates a new role and reauthorizes every replacement binding.
An active binding prevents role deletion, so a role edit can never widen old scopes.

## Endpoint authorization

The OpenAPI operation metadata and server middleware use this authoritative mapping:

| Endpoint family | Read/list | Create/update/delete/actions |
| --- | --- | --- |
| Namespaces | <code>cluster.read</code> or scoped <code>namespace.read</code> | Namespace create/delete requires <code>cluster.manage</code>; namespace update requires <code>namespace.manage</code>. |
| Management session/security policy | Cluster policy reads require <code>cluster.read</code>; namespace policy reads require <code>namespace.read</code> in that namespace. | Cluster policy updates require <code>cluster.manage</code>; namespace policy updates require <code>namespace.manage</code> in that namespace. |
| Trusted issuers | Cluster-scoped <code>identity_issuer.read</code> | Cluster-scoped <code>identity_issuer.manage</code>. |
| mTLS identity mappings | Cluster-scoped <code>identity_issuer.read</code> | Lifecycle requires cluster-scoped <code>identity_issuer.manage</code> **and** <code>principal.manage</code> on every current and target principal. |
| Global principals | Cluster-scoped <code>principal.read</code> | Cluster-scoped <code>principal.manage</code>, including authentication-identity attachment and session revoke. |
| Principal directory and User links | <code>principal_directory.read</code> or <code>principal_link.read</code> in namespace | <code>principal_link.manage</code> in namespace plus <code>user.manage</code> on every current and target User; no global principal lifecycle. |
| Roles, role bindings, service accounts | Corresponding <code>*.read</code> | Corresponding <code>*.manage</code>; namespace service-account operations require matching immutable owner namespace, cluster-owned accounts require cluster scope. |
| Invitations | <code>invitation.read</code> | Create, delete, and token rotation require <code>invitation.manage</code>. Every delegated role and scope must fit one active delegation ceiling; assigning a TeamRole additionally requires <code>membership.manage</code> on that Team. |
| Privileged onboarding | No list or read surface. | <code>POST /onboarding</code> requires <code>onboarding.manage</code> plus every ordinary permission needed for the User/link, role binding, optional Team membership, key, and any no-Team default policy bindings materialized by that request. |
| Users, Teams, memberships | <code>user.read</code> or <code>team.read</code> | <code>user.manage</code>, <code>team.manage</code>, or <code>membership.manage</code>. |
| Keys and delegations | <code>key.read</code>; self eligible-key discovery uses intrinsic <code>self.read</code> plus <code>delegation.use</code> | <code>key.manage</code>, <code>key.reveal</code>, <code>delegation.use</code>, or <code>delegation.manage</code> for the named action. |
| Access/rate policies and bindings | Corresponding <code>*.read</code> | Corresponding <code>*.manage</code>. |
| Routing context | <code>routing_context.read</code> over the named subject | Schema or subject-value writes require <code>routing_context.manage</code> plus namespace or subject manage respectively. |
| Access decision simulation | <code>POST /access:check</code> requires <code>access_policy.read</code> over both subject and resource plus <code>routing_context.read</code> on the subject; explicit context overrides also require <code>routing_context.manage</code>. | Read-only simulation; routing expansion separately requires <code>routing.read</code> over every disclosed dependency. |
| Live quota and unknown-usage fences | <code>quota.read</code> | Reconciliation requires <code>quota.reconcile</code>. |
| Provider catalog | <code>provider_catalog.read</code> | Read-only. |
| Routing and provider credentials | <code>routing.read</code> or <code>provider_credential.read</code> | Corresponding <code>*.manage</code>; Model reference, discovery, probe through a provider, and bulk import also require <code>provider_credential.use</code> on every credential. |
| Usage, logs, audit | <code>usage.read</code>, <code>log.read</code>, or <code>audit.read</code> | Internal dimensions and payload fields additionally require their dedicated permissions. |
| Operations and diagnostics | <code>operation.read</code> or <code>health.read</code> | Cancel requires <code>operation.manage</code> plus the original domain mutation permission. |

Each OpenAPI operation carries a machine-checked `x-router-permission-expression`.
The cross-family endpoints use these exact conjunctions (`AND`), not whichever family
happens to route the request:

| Operation | Required expression and scope |
| --- | --- |
| <code>GET /&#123;users&#124;teams&#124;api-keys&#125;/&#123;id&#125;/effective-policy</code> | Subject/key read **and** `access_policy.read` **and** `rate_policy.read`, all covering the named subject/key. |
| <code>GET /&#123;users&#124;teams&#124;api-keys&#125;/&#123;id&#125;/quota</code> | Subject/key read **and** `quota.read`, both covering the named subject/key and every returned binding. |
| <code>GET /&#123;users&#124;teams&#124;api-keys&#125;/&#123;id&#125;/usage</code> | Subject/key read **and** `usage.read`, with the same subject filter injected server-side. |
| Namespace self-service-policy update | `namespace.manage` **and** corresponding policy-manage on every current default being removed/replaced and every target default. Each referenced policy must be active and in the path namespace; clearing a default still requires authority over its current policy. |
| mTLS identity-mapping lifecycle | `identity_issuer.manage` on the mapping **and** cluster-scoped `principal.manage` on its current and target principals. Create/rebind/delete/status changes validate exact certificate uniqueness and install authentication-source session barriers; rebind is delete-plus-create, never PATCH. |
| Namespace principal directory/link routes | `principal_directory.read` or `principal_link.read` in the path namespace. Link create requires `principal_link.manage` there **and** `user.manage` on the target User; relink/delete also requires `user.manage` on the current User. A link cannot change while a principal has a User-scoped role binding unless an authorized transaction replaces/removes that binding. Global principal/all-links routes require cluster-scoped `principal.read` or `principal.manage`. |
| Invitation create/delete/rotate-token | `invitation.manage`; create applies the delegation ceiling and cannot select a policy override. A no-Team invitation pins active same-namespace self-service policy IDs/revisions; a Team invitation pins the membership path and accept rechecks that the Team remains active with an Access layer and Rate allocation. A TeamRole also requires `membership.manage`; lifecycle confers no privileged-onboarding authority. |
| `POST /onboarding` | `onboarding.manage` **and** `principal_link.manage` **and** `user.manage`, plus `role_binding.manage` and the normal delegation-ceiling test for every role binding, `membership.manage` for an optional Team membership, and `key.manage` for an optional first key. Corresponding policy-manage plus subject-manage permissions apply only when no Team is selected and default User policy bindings are materialized. It confers no invitation lifecycle authority. |
| Team membership list | `team.read` on the Team; member mutation requires `membership.manage` on the Team. The server validates the named User in the same namespace without granting a User-directory read. |
| User membership list | `user.read` on the User; each returned row additionally requires `team.read` on that Team and contains safe Team/TeamRole metadata only. Server-side User/Team scope filters and keyset indexes run before serialization. |
| API-key credential list/rotate/delete/reveal | `key.read`; rotation/deletion additionally require `key.manage`; reveal additionally requires `key.reveal` and satisfaction of the action's typed authentication predicate, all on the logical key. |
| API-key delegated-session list/revoke/revoke-all | `key.read` **and** `delegation.manage` on the logical key. Self-session routes use `self.*` plus `delegation.use` instead. |
| Principal Management-session list/revoke-all | Cluster-scoped `principal.read`; revocation also requires cluster-scoped `principal.manage`. |
| API-key create/reassign/delete | Inherited-only create requires `key.manage` on the owner. Existing explicit bindings also need manage on each policy; inline rules need `rate_policy.manage` on the namespace and atomically return an ordinary policy/binding. Reassign needs key manage plus subject manage on both owners; delete needs key manage. Invitation-driven automatic first-key provisioning is inherited-only; the `consumer` preset itself has no key-lifecycle authority. |
| Team create | `team.manage` on the new Team scope **and** `access_policy.manage` plus `rate_policy.manage` on every selected policy, including policies selected through namespace defaults. Team, AccessPolicy bindings, and the RateLimit allocation are one transaction: all become active together or nothing is created. |
| Access/rate binding create/update/delete | Corresponding policy manage on the policy **and** subject manage for the key, User, or Team on the binding subject. |
| Access-policy grant mutation | `access_policy.manage` on the policy **and** `routing.read` on every referenced Model/Entrypoint; the transaction rejects absent, deleting, cross-namespace, or wrong-kind targets. |
| Access-check response fields | Base access authority returns only decision, matched policy grant, and source binding. Recipe, Entrypoint rule, assignments, and resolver explanation require `routing.read` over the Entrypoint and every disclosed dependency; provider/backend detail also requires `usage.internal_dimensions.read`. Without the full conjunction the whole routing-detail object is omitted, never partially redacted. |
| Entrypoint read/list, snapshot, export | Entrypoint-scoped `routing.read` returns identity/lifecycle only. Rules, Recipe revision, assignments, and topology require `routing.read` on every dependency; list omits that expansion per item. Snapshot members/export require namespace-wide `routing.read`; the path namespace is injected and out-of-scope revisions return nondisclosing `404`. |
| Request-log detail | `log.read` over the path namespace and attributed subject. Lookup uses `(namespace_id, admission_id)`; out-of-scope and absent rows return the same `404` before field serialization. |
| Entrypoint create/update/publish | `routing.manage` on the target Entrypoint scope **and** `routing.read` on the exact Recipe revision and every assigned Model, all in the same namespace. Publish rechecks every pinned dependency and fails atomically if scope, lifecycle, or revision changed. |
| Entrypoint `:resolve` | `routing.read` on the Entrypoint, exact selected Recipe revision, and every returned Model, all same-namespace; it returns no partial topology. Native access also needs `routing_context.read` when a subject is supplied and `routing_context.manage` for an override. Routing-only accepts neither subject nor override. |
| Subject routing-context read/write | Subject read plus `routing_context.read`; writes require subject manage plus `routing_context.manage`. Namespace schema writes require `namespace.manage` plus `routing_context.manage`. |
| Provider discovery | `provider_catalog.read`; with `credentialId`, also credential read/use; without one, namespace `routing.manage`. Connection fields must match provider and deployment egress schemas. No routing resource permission is required for credential-backed read-only discovery. |
| Model create/update/probe and bulk import | `routing.manage` on affected/new routing resources **and** `provider_credential.use` on every actually referenced backend credential; bulk import also requires `provider_catalog.read`. |
| Model/provider-credential response fields | `routing.read` alone exposes safe provider/catalog capability and `credentialConfigured`; credential UID, normalized origin, version/status, and sensitive connection fields require `provider_credential.read` on that exact credential. The same omission applies to Models, snapshots, resolve, Operations, audit detail, and errors. |
| Unknown-fence list/detail/reconcile | List/detail requires `quota.read` on **every** affected binding and never returns partial fences. Internal dispatch/provider/pricing fields additionally require `usage.internal_dimensions.read`, evidence payload requires `log_payload.read`, and actor/audit fields require `audit.read` or `quota.reconcile`. Reconcile requires `quota.reconcile` on every binding; `actual` also requires internal-dimension read. |
| Operation read/cancel | Cross-actor access needs `operation.read`; the originator may use intrinsic `self.read`. Both recheck original domain authority. Cancel also needs `self.manage` or `operation.manage` plus mutation permission. Operations never expose a secret-claim action. |
| Request-log payload | `log.read` **and** `log_payload.read` over the same attributed subject scope. |

`GET /me` and self-session listing require intrinsic `self.read`; revoking one's own
Management session requires intrinsic `self.manage`, not `principal.manage`.
Self-inference-session list/create/delete requires `self.read` or `self.manage` for the
named action plus `delegation.use` over the selected User/key and the self-service
policy. Administrator session revocation still requires `principal.manage`. A filter
outside scope is rejected with `403 invalid_scope`; a detail resource outside scope
returns the same nondisclosing `404` as an absent resource. Server-side field
authorization runs after row scoping and before serialization.

## Team roles

TeamRole does not create a ManagementRole record. The evaluator synthesizes a fixed,
non-delegable Team-scoped entitlement while the membership is active:

- `member` receives `team.read`, `access_policy.read`, `rate_policy.read`,
  `quota.read`, and `usage.read` for that Team's shared summaries. When self-service
  policy permits Team-key Playground use, it also receives `delegation.use` limited to
  its own sessions on eligible Team-owned keys.
- `admin` receives the member set plus `key.read`; namespace self-service policy may
  additionally synthesize `membership.manage` and `key.manage` for that Team only.

These entitlements disappear atomically when membership is disabled. They never grant
role administration, policy mutation, quota expansion, key reveal, member-private
keys, raw logs, or payloads. Wider Team administration requires an explicit scoped
ManagementRole binding.
