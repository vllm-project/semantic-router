---
title: Router-Native Access Control Management API Appendix
description: Specifies management identity exchange, delegated inference credentials, API resources, effective policy responses, and authorization.
created: 2026-08-22
status: Proposal
---

> **Status:** Proposal appendix · **Created:** 2026-08-22

This appendix is normative for Management authentication, endpoints, requests, and responses in [Router-Native Access Control and Quota Accounting](./router-native-access-control).
The [resource contract](./router-native-access-control-contracts) owns storage and policy; [authorization](./router-native-access-control-authorization) owns permissions and scopes; [deployment](./router-native-access-control-deployment) owns bootstrap, topology, readiness, and recovery.

## Management identity exchange

The Dashboard is an optional browser client, not a trusted session database for the Router. A browser cookie is never accepted directly by the Management API.

Every authenticated Management request uses a Router-issued access token. External OIDC/local assertions, service credentials, and mTLS identities are bootstrap evidence only; none is accepted directly as a Management bearer token.

An OIDC client may perform the exchange itself. For local login, the Dashboard is one configured issuer and signs a short-lived JWS containing issuer, immutable subject, Management audience/session, verified attributes, authentication time/methods/assurance, expiry, token ID, and one-time nonce. Both use:

~~~text
POST /management/v1/auth/exchange-challenges
POST /management/v1/auth/token-exchange
~~~

Before starting login, the client requests an exchange challenge with `issuerId`. The
Router returns a random nonce, opaque `exchangeChallengeId`, and short expiry and
stores only the bound nonce hash/state in Valkey. The client supplies that nonce to
its OIDC authorization request or local assertion signer. The token-exchange JSON is
a discriminated union with `issuerId`, `exchangeChallengeId`, `subjectToken`, and
`subjectTokenType` equal to `oidc_id_token` or `router_local_assertion`; it may also
carry `invitationToken` on a fresh exchange. The Router loads the expected nonce,
validates configured issuer, audience, signature algorithm and keys, exact token nonce,
issuer session state, `iat`, `exp`, and token ID, and atomically consumes the challenge
before resolving `(issuer, subject)`. Lost/expired challenges restart login; challenge
creation is rate-limited and grants no identity information. OIDC authorization-code
and PKCE handling remains in the client; only the resulting ID token crosses the
exchange. Email and display name are attributes, never identity keys. A trusted issuer
cannot silently create a principal: first exchange requires an invitation,
pre-created principal, or one-time bootstrap.

Automation uses a distinct credential `vsm_<credential-id>_<256-bit-secret>`.
`POST /management/v1/auth/service-token` accepts it only through
`Authorization: VSR-Service <credential>` over TLS. The same endpoint accepts a
configured mTLS peer when no service credential header is present. The Router performs
an O(1) credential-ID lookup and HMAC verification, or maps the verified certificate
identity, checks principal and role-binding status, and creates a bounded service
Management session. Service credentials are never valid Bearer tokens and cannot call
inference. mTLS uses only exact, active `mTLSIdentityMapping` resources after the
listener validates the client chain against its configured trust bundle; wildcard or
request-header certificate identities are forbidden. The Router chooses one selector
class deterministically: a single verified SPIFFE ID, else SAN URI, else SAN DNS, else
the normalized subject-DN hash. It evaluates only that class. Zero matches denies;
multiple matching values or mappings fail closed and emit an ambiguity audit rather
than using list order or another selector class.

Clean installation uses `POST /management/v1/auth/bootstrap` with
`Authorization: VSR-Bootstrap <secret>` on the private Management listener. The
secret is compared in constant time with `bootstrap.token_file`. Its discriminated
body selects either a first service-account principal or an external principal plus
the trusted-issuer definition needed to verify that subject. One serializable
PostgreSQL transaction locks the installation singleton, proves that no durable
cluster administrator or consumed marker exists, creates the issuer when requested,
principal, fixed `cluster_admin` role binding and delegation ceiling, audit/outbox,
and permanent consumed marker. It requires the versioned cluster session-policy seed;
invalid state commits nothing and fails readiness. Service-account bootstrap creates its first
credential and returns it through the normal idempotent secret envelope. The response
contains an immutable bootstrap receipt. This security transaction always waits for
local WAL durability and, in HA, synchronous database quorum regardless of the normal
profile; failure commits nothing. The consumed marker rejects new bootstrap mutations
on that database timeline. Public readiness remains blocked until the operator removes
the bootstrap secret from the deployment, eliminating reuse after an older database
restore; the receipt reports this required finalization.

The consumed check has one non-mutating exception for delivery recovery: the same
bootstrap token, `Idempotency-Key`, and normalized request digest may replay the stored
encrypted receipt and first credential during its bounded envelope window. It cannot
change the request or create another resource. After expiry it returns
`410 bootstrap_result_expired`, and recovery is the only repair path. Recovery stores
the same idempotency/request digests and envelope before spending its in-process token,
so an exact recovery call can replay its committed credential without executing twice.

Break-glass uses `POST /management/v1/auth/recovery` with
`Authorization: VSR-Recovery <secret>`. The route is registered only when recovery was
explicitly enabled at process start, is reachable only from the Router's loopback
interface, and uses a separate root-readable token file. Its bounded body can only
create or restore one `cluster_admin` binding and fixed delegation ceiling for a named
principal, optionally creating a recovery service account and one credential. It
cannot read or mutate keys, policies, routing, usage, or secrets. One serializable
transaction records the reason, actor target, one-use recovery nonce, binding, and
audit; the credential uses the secret envelope. The in-process token is spent after
one successful call, and normal operation requires restart with recovery disabled.

Both token endpoints commit a Management session and return `{accessToken, tokenType: "Bearer", expiresIn, managementSessionId}`. The Router-signed JWT has fixed Management audience and `sub`, `sid`, `jti`, `iat`, `exp`, `auth_source_kind`, stable source ID, and exactly one evidence object: human `auth_time|aal|amr` or workload `class|source_assured_at`. The server derives these claims from the persisted issuer, credential, or mTLS mapping; clients cannot supply them. Every request validates signature/audience and principal, authentication-source, and session deny projections. Issuer back-channel
logout, client logout, credential disablement, or administrator revocation installs
that barrier before reporting success. When a Dashboard backend or another broker
performs the human exchange, its service account remains in the actor chain but does
not replace the human principal. Audit records both actors.

Management sessions are global. Cluster policy alone sets JWT/session TTL, active-
session limits, and typed cluster-action authentication predicates; namespace policy
sets typed action predicates. Each request checks its target's applied rule. A human
failure returns a step-up challenge; a workload failure returns
`403 source_assurance_insufficient` and requires credential rotation or mapping
re-registration. A restrictive change installs a barrier without weakening other scopes.

Disabling one service credential or mTLS mapping first installs its authentication-
source deny barrier, enumerates the source's indexed active Management sessions, and
revokes them before reporting success. It does not revoke sessions created by another
credential of the same service account. Disabling the service-account principal uses
the broader principal barrier and revokes every source.

`POST /management/v1/auth/backchannel-logout` is authenticated only by a signed logout
token from the named trusted issuer, never by a browser cookie or Management bearer
token. The Router validates configured `iss`, Management client `aud`, signature,
`iat`, `jti`, and the back-channel logout event claim; `nonce` is forbidden and at
least one of `sid` or `sub` is required. A `sid` revokes matching issuer sessions; a
`sub` without `sid` revokes every session for that `(issuer, subject)`. The hashed
`jti` is retained through token expiry for replay rejection. Repeating an already
applied valid logout is an idempotent `200`; a reused ID with different claims is rejected.

Any fresh verified token exchange may carry a matching invitation, whether the exact
`(issuer, subject)` principal is new or active. One CAS transaction consumes it,
creates or reuses only that principal, and creates the missing namespace User/link,
roles, optional TeamMembership, audit/outbox, and optional inherited first key. It
never merges by email. A conflicting existing link/User or already-onboarded scope
returns a deterministic conflict without consuming the invitation. Exact idempotent
replay preserves its one-use result. The token response nests either the applied
one-time key result or pending `onboardingOperation` and `secretResultClaim`; it is
never a bare Operation, so the new session can poll and claim; failure creates nothing.

Pending first-key creation records a one-time onboarding-claim capability bound to the
exact principal, issuing authentication source, evidence kind, pinned typed authentication
predicate, Operation, logical key, claim HMAC, and delivery TTL. It permits that principal
to reauthenticate through the same source and claim from any
current non-revoked session meeting the pinned same-kind predicate. It grants no key
management and is erased on delivery or expiry.

~~~text
GET /management/v1/me
~~~

The response includes the immutable principal ID, current Management session,
cluster-scoped permissions, every authorized namespace and its role bindings,
namespace-specific linked User, Team memberships, self-service policy, desired
revision, and applied revision. A Principal/User link is unique per principal and
namespace, not globally. Navigation may render from this response, but navigation is
never an authorization boundary.

## Delegated inference credentials

Playground and other first-party experiences call the public inference listener.
They do not use a Dashboard proxy or a shared service API key. An authenticated
principal linked to a User may request a short-lived delegated inference credential:

~~~text
GET    /management/v1/self/inference-keys
GET    /management/v1/self/inference-sessions
POST   /management/v1/self/inference-sessions
DELETE /management/v1/self/inference-sessions/{sessionId}
~~~

Creation names a namespace and selects either a key owned by the linked User or a
Team-owned key for an active membership when self-service policy permits that use.
`GET /self/inference-keys` returns only safe ID/name/owner/expiry metadata for those
eligible keys; it never reveals a key secret or grants general Team-key read access.
The response is the distinct, non-revealable delegated-credential format specified
by the resource contract, bound to key ID, User ID, Team context, audience, and
Management session with a short expiry. Each inference request resolves the selected
key's current policy revision, uses the same binding-owned counters, and records the
same key, User, Team, Entrypoint, model, and dispatch usage as a direct API-key
request. Disabling the key, User, Team membership, principal, or Management session
invalidates the delegation. Listing returns metadata only.

An administrator may test a Single Model only when the selected key has an explicit
invoke grant for that Model. The request uses the normal public inference path and
therefore cannot bypass discovery, authorization, quota, logs, or usage accounting.

## Router Management API

The Router publishes one generated OpenAPI contract under
<code>/management/v1</code>. Version 0.4 has no parallel
<code>/api/v1/access-control</code> surface. These mutation resources exist only in
managed mode; standalone does not start a mutable Management control plane.

### Identity and namespace resources

~~~text
GET    /management/v1/me
POST   /management/v1/auth/bootstrap
POST   /management/v1/auth/exchange-challenges
POST   /management/v1/auth/token-exchange
POST   /management/v1/auth/service-token
POST   /management/v1/auth/recovery
GET    /management/v1/self/management-sessions
DELETE /management/v1/self/management-sessions/{sessionId}
POST   /management/v1/management-sessions/{sessionId}:revoke
POST   /management/v1/auth/backchannel-logout
GET    /management/v1/management-session-policy
PATCH  /management/v1/management-session-policy
GET    /management/v1/namespaces
POST   /management/v1/namespaces
GET    /management/v1/namespaces/{namespaceId}
PATCH  /management/v1/namespaces/{namespaceId}
DELETE /management/v1/namespaces/{namespaceId}
GET    /management/v1/namespaces/{namespaceId}/self-service-policy
PATCH  /management/v1/namespaces/{namespaceId}/self-service-policy
GET    /management/v1/namespaces/{namespaceId}/management-security-policy
PATCH  /management/v1/namespaces/{namespaceId}/management-security-policy
GET    /management/v1/namespaces/{namespaceId}/routing-claim-schema
PATCH  /management/v1/namespaces/{namespaceId}/routing-claim-schema
GET    /management/v1/trusted-identity-issuers
POST   /management/v1/trusted-identity-issuers
GET    /management/v1/trusted-identity-issuers/{issuerId}
PATCH  /management/v1/trusted-identity-issuers/{issuerId}
DELETE /management/v1/trusted-identity-issuers/{issuerId}
POST   /management/v1/trusted-identity-issuers/{issuerId}:refresh-keys
GET    /management/v1/mtls-identity-mappings
POST   /management/v1/mtls-identity-mappings
GET    /management/v1/mtls-identity-mappings/{mappingId}
PATCH  /management/v1/mtls-identity-mappings/{mappingId}
DELETE /management/v1/mtls-identity-mappings/{mappingId}
GET    /management/v1/management-roles
POST   /management/v1/management-roles
GET    /management/v1/management-roles/{roleId}
PATCH  /management/v1/management-roles/{roleId}
DELETE /management/v1/management-roles/{roleId}
GET    /management/v1/management-principals
POST   /management/v1/management-principals
GET    /management/v1/management-principals/{principalId}
PATCH  /management/v1/management-principals/{principalId}
DELETE /management/v1/management-principals/{principalId}
GET    /management/v1/management-principals/{principalId}/user-links
GET    /management/v1/management-principals/{principalId}/management-sessions
POST   /management/v1/management-principals/{principalId}/management-sessions:revoke-all
GET    /management/v1/namespaces/{namespaceId}/principal-directory
GET    /management/v1/namespaces/{namespaceId}/principal-directory/{principalId}
GET    /management/v1/namespaces/{namespaceId}/principal-user-links
PUT    /management/v1/namespaces/{namespaceId}/principal-user-links/{principalId}
DELETE /management/v1/namespaces/{namespaceId}/principal-user-links/{principalId}
GET    /management/v1/role-bindings
POST   /management/v1/role-bindings
GET    /management/v1/role-bindings/{bindingId}
PATCH  /management/v1/role-bindings/{bindingId}
DELETE /management/v1/role-bindings/{bindingId}
GET    /management/v1/service-accounts
POST   /management/v1/service-accounts
GET    /management/v1/service-accounts/{serviceAccountId}
PATCH  /management/v1/service-accounts/{serviceAccountId}
DELETE /management/v1/service-accounts/{serviceAccountId}
GET    /management/v1/service-accounts/{serviceAccountId}/credentials
POST   /management/v1/service-accounts/{serviceAccountId}/credentials:rotate
DELETE /management/v1/service-accounts/{serviceAccountId}/credentials/{credentialId}
GET    /management/v1/invitations
POST   /management/v1/invitations
GET    /management/v1/invitations/{invitationId}
DELETE /management/v1/invitations/{invitationId}
POST   /management/v1/invitations/{invitationId}:rotate-token
POST   /management/v1/onboarding
~~~

Namespace create atomically seeds restrictive SelfServicePolicy and
ManagementSecurityPolicy rows; absence is an error, never an implicit fallback.

ManagementPrincipal identity, authentication attachment, and lifecycle are global and
require cluster-scoped `principal.*`. Only invitation and bootstrap may create or
attach one inside their constrained transactions. Namespace administrators use
`principal_link.*` only for namespace User links. Role bindings and links are scoped.
Built-ins are immutable; a namespace administrator may
create a custom role only from its delegation ceiling. A custom role's permission set
is immutable after creation; PATCH changes display metadata only. Permission changes
create a new role ID and explicitly replace each binding through normal ceiling/scope
authorization. Deleting a role with an active binding is rejected.

A namespace link mutation requires authority over both the link and every current or
target User. A principal's User-scoped role binding must name its linked User. Ordinary
link PUT/DELETE rejects while such a binding exists; onboarding may create the link
and initial binding atomically, and an explicitly authorized role-binding transaction
may replace or remove both under one CAS. Relinking never silently carries consumer
authority to another User.

A service account is a ManagementPrincipal with a reserved issuer, immutable
`cluster|namespace` owner scope, and independently rotated HMAC credential. A
namespace-owned service-account principal and all its role bindings are restricted to
that owner namespace; a cluster-owned one requires cluster authority. It receives no
authority outside explicit role bindings.

The namespace principal directory requires its path namespace and returns only
principal ID, display name, verified email when policy permits, status usable for
linking, and whether a link already exists there. It never returns issuer subject,
attributes, sessions, credentials, role bindings, or links in another namespace.
Search is bounded and audited. Namespace link list/detail likewise injects the path
scope and cannot be widened by a query filter. The global principal and all-links
endpoints remain cluster-only. Principal-user-link lists accept indexed
`principalId|userId`; role-binding lists accept indexed `principalId` plus typed
scope filters. User detail first resolves the indexed User link and then queries
bindings by principal. Every filter is conjunctive and keyset paginated.

An mTLS mapping maps one exact normalized certificate identity to one pre-created
ManagementPrincipal. Lifecycle needs cluster-scoped `identity_issuer.manage` and
`principal.manage` over every current/target principal. It validates uniqueness and
installs the source session barrier; changing principal is delete-plus-create.

The one-time bootstrap credential can create the first cluster administrator and
trusted issuer only while no durable cluster administrator exists. It is then
permanently disabled. Recovery uses a separately enabled, loopback-only break-glass
mode and never reactivates bootstrap.

An invitation stores token HMAC, expected identity, grants, optional TeamRole, expiry,
and an immutable onboarding snapshot. It pins active same-namespace Access/Rate policy
IDs/revisions from self-service defaults; invitations cannot override them. This
namespace-authorized capability is consumable with `invitation.manage`; TeamRole also
needs membership manage. Create/rotate returns the secret once. Accept verifies
identity, delegation ceiling, and unchanged policy revisions in one CAS; stale or
conflict neither reveals nor consumes it.

The namespace self-service policy defines maximum logical keys per User,
delegated-session count and TTL, whether active members may use Team-owned keys,
automatic first-key behavior, Team-admin capabilities, and default AccessPolicy and
RateLimitPolicy IDs. A write authorizes both current and target defaults and accepts
only active same-namespace policies. Onboarding materializes them as real bindings;
there is no hidden namespace layer in runtime inheritance.
`POST /onboarding` is the privileged, idempotent administrative form of the same
transaction for a pre-created external principal.

The namespace routing-claim schema defines the bounded names and value types accepted
by the Key/User/Team routing-context endpoints. A subject update requires both subject
manage and `routing_context.manage`, validates against that schema, increments the
subject and namespace routing-context revision, audits the before/after values, and
publishes every affected key projection before reporting applied. Reads return stored
and effective values separately with source subject and revision. The values are not
accepted on public inference requests.

Management tokens carry one typed human or workload evidence object. Key reveal
requires `key.reveal` and the namespace action predicate. Insufficient human evidence
returns a machine-readable step-up challenge; insufficient workload evidence returns
`source_assurance_insufficient` rather than pretending automation can perform a human
step-up.

### Users and Teams

~~~text
GET    /management/v1/users
POST   /management/v1/users
GET    /management/v1/users/{userId}
PATCH  /management/v1/users/{userId}
DELETE /management/v1/users/{userId}
GET    /management/v1/users/{userId}/effective-policy
GET    /management/v1/users/{userId}/routing-context
PUT    /management/v1/users/{userId}/routing-context
GET    /management/v1/users/{userId}/quota
GET    /management/v1/users/{userId}/usage
GET    /management/v1/users/{userId}/memberships
GET    /management/v1/teams
POST   /management/v1/teams
GET    /management/v1/teams/{teamId}
PATCH  /management/v1/teams/{teamId}
DELETE /management/v1/teams/{teamId}
POST   /management/v1/teams/{teamId}:activate
POST   /management/v1/teams/{teamId}:disable
GET    /management/v1/teams/{teamId}/effective-policy
GET    /management/v1/teams/{teamId}/routing-context
PUT    /management/v1/teams/{teamId}/routing-context
GET    /management/v1/teams/{teamId}/quota
GET    /management/v1/teams/{teamId}/usage
GET    /management/v1/teams/{teamId}/members
PUT    /management/v1/teams/{teamId}/members/{userId}
PATCH  /management/v1/teams/{teamId}/members/{userId}
DELETE /management/v1/teams/{teamId}/members/{userId}
~~~

User memberships are indexed/keyset-paginated and return safe Team/TeamRole fields
within caller scope. Team members use the inverse index; neither scans all Teams.

A Team begins in <code>draft</code>. Its activate action atomically materializes an
AccessPolicy binding and a RateLimitPolicy allocation binding selected explicitly or
from namespace onboarding defaults. This avoids unbounded keys and hidden runtime
inheritance. TeamRole remains separate from ManagementRole. Activation requires
manage authority over the Team and both policies even when defaults select them.

### API keys, access, and quota

~~~text
GET    /management/v1/api-keys
POST   /management/v1/api-keys
GET    /management/v1/api-keys/{keyId}
PATCH  /management/v1/api-keys/{keyId}
DELETE /management/v1/api-keys/{keyId}
POST   /management/v1/api-keys/{keyId}:enable
POST   /management/v1/api-keys/{keyId}:disable
POST   /management/v1/api-keys/{keyId}:renew
POST   /management/v1/api-keys/{keyId}:reassign
GET    /management/v1/api-keys/{keyId}/credentials
POST   /management/v1/api-keys/{keyId}/credentials:rotate
POST   /management/v1/api-keys/{keyId}/credentials/{credentialId}:reveal
DELETE /management/v1/api-keys/{keyId}/credentials/{credentialId}
GET    /management/v1/api-keys/{keyId}/inference-sessions
DELETE /management/v1/api-keys/{keyId}/inference-sessions/{sessionId}
POST   /management/v1/api-keys/{keyId}/inference-sessions:revoke-all
GET    /management/v1/api-keys/{keyId}/effective-policy
GET    /management/v1/api-keys/{keyId}/routing-context
PUT    /management/v1/api-keys/{keyId}/routing-context
GET    /management/v1/api-keys/{keyId}/quota
GET    /management/v1/api-keys/{keyId}/usage
GET    /management/v1/access-policies
POST   /management/v1/access-policies
GET    /management/v1/access-policies/{policyId}
PATCH  /management/v1/access-policies/{policyId}
DELETE /management/v1/access-policies/{policyId}
GET    /management/v1/access-policy-bindings
POST   /management/v1/access-policy-bindings
GET    /management/v1/access-policy-bindings/{bindingId}
PATCH  /management/v1/access-policy-bindings/{bindingId}
DELETE /management/v1/access-policy-bindings/{bindingId}
POST   /management/v1/access-policy-bindings:bulk-apply
GET    /management/v1/rate-limit-policies
POST   /management/v1/rate-limit-policies
GET    /management/v1/rate-limit-policies/{policyId}
PATCH  /management/v1/rate-limit-policies/{policyId}
DELETE /management/v1/rate-limit-policies/{policyId}
GET    /management/v1/rate-limit-bindings
POST   /management/v1/rate-limit-bindings
GET    /management/v1/rate-limit-bindings/{bindingId}
PATCH  /management/v1/rate-limit-bindings/{bindingId}
DELETE /management/v1/rate-limit-bindings/{bindingId}
POST   /management/v1/rate-limit-bindings:bulk-apply
POST   /management/v1/access:check
GET    /management/v1/unknown-usage-fences
GET    /management/v1/unknown-usage-fences/{fenceId}
POST   /management/v1/unknown-usage-fences/{fenceId}:reconcile
~~~

API-key ownership is a required one-of choice: User or Team. Key-level access and
rate-limit allocations are optional overrides. Without them, a User-owned key
inherits User then Team allocations; a Team-owned key inherits Team allocations.
Every shared hard cap remains cumulative. Administrators can expand quota at Key,
User, or Team subject to broader caps. Inherited-only create needs owner key-manage;
explicit bindings also need referenced policy-manage. Consumer self-service remains
inherited-only. Optional `rateLimitOverride` is exactly one of `policyId` or
`inlinePolicy`; inline rules atomically create/bind an ordinary reusable Budget, and
the idempotent response returns key, optional policy/binding, and created flag.

`PATCH /api-keys/{keyId}` changes descriptive metadata only. Reassignment is an
explicit restrictive operation: install a deny barrier, require zero in-flight
admissions for the key, change owner/context, preserve immutable historical
attribution, resolve the new inherited bindings, publish, and then remove the
barrier. Key-owned counters remain with the key; inherited User/Team counters never
move. Deleting the final active credential of an enabled logical key returns
`409 last_active_credential`; disabling or deleting the logical key revokes all
versions.

Administrative inference-session reads return metadata only and require key read plus
`delegation.manage`. Single revoke installs the session barrier before success.
Revoke-all atomically increments the logical key's delegation epoch, so every older
session fails immediately without scanning, then cleans projections asynchronously;
it does not disable the API key. Principal session list/revoke-all is cluster-scoped,
requires `principal.read|manage`, and uses the principal barrier/index before success.

`access:check` accepts subject/resource IDs, permission, and optional path; it derives
trusted context from stored state and never accepts a raw credential. A validated
override needs `routing_context.manage` and is labeled `simulation`. Access-policy
read returns only decision, matched policy grant, and source binding. Recipe, rule,
assignment, and resolver detail needs `routing.read` over every dependency. Provider
detail also needs internal-dimension permission. Unauthorized routing detail is
omitted as one object; out-of-scope and absent resources return `404`.

Unknown-fence list returns a fence only when `quota.read` covers every affected
binding; filters cannot widen scope and a fence is never partially redacted into a
misleading subset. The quota-reader detail contains opaque fence/admission IDs,
lifecycle/ETag, sanitized reason, visible affected meters, aggregate known charge,
and desired/applied timestamps. Backend/provider/dispatch/pricing fields require
`usage.internal_dimensions.read`; evidence payloads require `log_payload.read`; actor
and audit detail require `audit.read` or `quota.reconcile`. Reconciliation requires
`If-Match`, `Idempotency-Key`, a strategy of
`actual|conservative_debit|waive`, evidence references, and a mandatory reason:

- `actual` supplies one canonical dispatch-usage record for every unknown dispatch.
  Dispatch IDs, token arithmetic, usage states, and evidence must validate against the
  immutable dispatch ledger; missing or duplicate dispatches are rejected. It also
  requires internal-dimension permission, and payload evidence requires payload-read.
- `conservative_debit` accepts no caller-provided amount. Requests charge one if not
  already charged; token meters use the immutable input and generated-token,
  multimodal, context, concurrency, and maximum-dispatch bounds recorded at admission.
  If a metric lacked a provable bound, the deterministic debit equals that rule's
  immutable admission-time limit, never a later edited limit. Concurrency leases are
  released and are not converted into token usage.
- `waive` applies zero unknown charge, requires `quota.reconcile` and satisfaction of
  the action's typed authentication predicate, and records justification and evidence.

The response is an Operation. The Router records one immutable reconciliation plan in
PostgreSQL, atomically applies every binding delta under one Valkey reconciliation ID,
persists the correction UsageEvent and audit, and only then removes that fence from
all affected bindings. Until all steps finish, lifecycle remains `reconciling` and
every affected binding stays frozen. Retrying the same idempotency key returns the
same Operation; stale ETags return `412`, another plan for an open/reconciling fence
returns `409 reconciliation_conflict`, and a new request for a resolved fence returns
`409 fence_resolved`.

An open/reconciling fence prevents deletion of its bindings/rules and any semantic
rule change. Limit-only publication may proceed, but the admission-time rule tombstone
and limit remain through resolution, and its fixed debit uses the same counter lineage.

### Routing and provider resources

~~~text
GET    /management/v1/routing/models
POST   /management/v1/routing/models
POST   /management/v1/routing/models:bulk-import
GET    /management/v1/routing/models/{modelId}
PATCH  /management/v1/routing/models/{modelId}
DELETE /management/v1/routing/models/{modelId}
POST   /management/v1/routing/models/{modelId}:probe
GET    /management/v1/routing/recipes
POST   /management/v1/routing/recipes
GET    /management/v1/routing/recipes/{recipeId}
PATCH  /management/v1/routing/recipes/{recipeId}
DELETE /management/v1/routing/recipes/{recipeId}
GET    /management/v1/routing/entrypoints
POST   /management/v1/routing/entrypoints
GET    /management/v1/routing/entrypoints/{entrypointId}
PATCH  /management/v1/routing/entrypoints/{entrypointId}
DELETE /management/v1/routing/entrypoints/{entrypointId}
POST   /management/v1/routing/entrypoints/{entrypointId}:publish
POST   /management/v1/routing/entrypoints/{entrypointId}:unpublish
POST   /management/v1/routing/entrypoints/{entrypointId}:resolve
GET    /management/v1/namespaces/{namespaceId}/routing/snapshots
GET    /management/v1/namespaces/{namespaceId}/routing/snapshots/{routingRevision}
GET    /management/v1/providers
GET    /management/v1/providers/{providerId}
POST   /management/v1/providers/{providerId}:discover-models
GET    /management/v1/provider-credentials
POST   /management/v1/provider-credentials
GET    /management/v1/provider-credentials/{credentialId}
PATCH  /management/v1/provider-credentials/{credentialId}
DELETE /management/v1/provider-credentials/{credentialId}
POST   /management/v1/provider-credentials/{credentialId}:rotate
~~~

Routing APIs expose Model, Recipe, and Entrypoint only; assignments live in rule
actions. Publish alone activates a snapshot; unpublish uses deny barriers. Writes
require read on the exact Recipe/Models and publish rechecks them. Entrypoint-scoped
read returns identity/lifecycle only; topology requires every dependency. Lists omit
unauthorized topology per item, while snapshot members/export need namespace-wide
read. There is no ModelPool, Mixture, model-binding, or detached assignment API.
Without exact credential read, Models show safe capability and `credentialConfigured`;
one projector omits credential metadata from all responses. Secrets are never read.

Entrypoint `:resolve` accepts path and optional subject. In managed access a subject
is required only for claim rules and loads stored effective context; an override is
a privileged simulation. Its full atomic result requires `routing.read` on the
Entrypoint, exact Recipe revision, and every returned Model; there is no partial
topology response. Managed routing-only rejects subject/override fields.

The read-only provider catalog is generated from the Router's registered adapters. A
provider entry has a stable `providerId`, display metadata and icon key, API dialect,
default base URL when one is public and stable, typed authentication and optional
connection-field JSON Schemas, discovery support, streaming/tool/multimodal
capabilities, and whether a caller-supplied base URL is required. Public providers
with no stable default URL are not advertised as key-only integrations. Private and
OpenAI-compatible adapters explicitly mark base URL as required. ProviderCredential
create references one catalog `providerId` and is rejected when its fields do not
validate against that provider's schema; consoles never hard-code an internal adapter
list. Authentication schemas explicitly distinguish `none` from secret-bearing
methods. An unauthenticated local backend references no ProviderCredential; use
authorization applies only to credentials a backend actually references.

ProviderCredential create binds its immutable `providerId` and normalized
scheme/host/port/base-path origin before encrypting the secret. Fixed public origins
come from the catalog; custom origins pass egress validation. Rotation changes only
secret version. Moving an origin requires a new credential, and Model/discovery/probe
validation rejects provider/origin mismatch or non-schema connection overrides.

Create and rotate accept a secret over the private Management transport when the
provider authentication schema requires one and return metadata only. Discover-models
accepts schema-validated connection fields and an optional `credentialId`, then uses
provider metadata and that credential, when present, to return a
bounded, normalized remote catalog without persisting Models, and is available only
when the provider advertises discovery. Its request accepts bounded `pageSize`, search,
and an opaque provider cursor; each response returns stable catalog item IDs,
`nextCursor`, and a `discoveryRevision` with expiry. Bulk-import accepts selected IDs,
that revision, and non-security Model overrides. The server-held/signed,
namespace-bound revision pins provider catalog version, normalized connection/origin
digest, optional credential/version, item IDs, and expiry. Import derives backends
only from it, forbids origin/credential overrides, and rechecks egress plus credential
status/use at enqueue and per-item execution. Expiry/change returns `stale_catalog`;
success returns an Operation. Creating or updating a Model backend requires `routing.manage` on the Model and
`provider_credential.use` on every referenced ProviderCredential. Discovery requires
`provider_catalog.read` plus the conditional authority below. Bulk import additionally
requires routing manage on its target scope. Routing authority alone cannot spend or
probe an unrelated credential.

Every discovery target passes the deployment backend-egress policy before any socket
opens and again after DNS resolution or redirect. Scheme, host, port, resolved IP,
redirect, private/loopback, and metadata ranges are fail-closed and the normalized
target is audited. Secret-bearing discovery/probe/dispatch never follows redirects;
no-auth redirects, if an adapter permits them, are revalidated without secret headers.
Credential-backed discovery requires credential read/use; no-auth
discovery additionally requires namespace `routing.manage`. Thus catalog read alone
never grants arbitrary Router-side network access.

Model create, update, probe, bulk-import overrides, standalone manifests, and exports
share the [Model runtime contract](./router-native-access-control-model-runtime). The Model revision contains `execution.maxRetries` (0-5
additional attempts governed by Router's fixed safe-retry predicate), duration strings
`execution.requestTimeout` and `execution.streamTimeout` (1 second-24 hours), plus
four nullable non-negative decimal-string prices per million tokens:
`inputCostPerMillionTokens`, `outputCostPerMillionTokens`,
`cacheReadCostPerMillionTokens`, and `cacheWriteCostPerMillionTokens`. Cache prices
inherit input when omitted; explicit zero means free, while absent input/output means
unpriced. The namespace supplies the immutable billing currency. Reads return both
configured and effective defaults, and probe uses the saved execution revision.

Model PATCH uses ETag, creates an immutable revision, audits the diff, and enters the
normal routing publication gate. Discovery suggestions are never another authority.
The Dashboard keeps provider, credential/base URL, and discovered Model selection in
the primary flow. A collapsed **Advanced settings** section contains Max retries,
Request timeout, Stream timeout, and Input/Output/Cache Read/Cache Write Cost, each
labeled **per 1M tokens** with inline defaults and validation. It edits the Model
value directly; there is no separate pricing, retry, or timeout resource. List/detail
show effective values, while Usage/Insights expose cost, currency, and incomplete-cost
state instead of treating missing prices or usage as zero.

### Usage, logs, audit, and bulk operations

~~~text
GET    /management/v1/usage
GET    /management/v1/usage/series
GET    /management/v1/usage/breakdowns
GET    /management/v1/request-logs
GET    /management/v1/namespaces/{namespaceId}/request-logs/{admissionId}
GET    /management/v1/audit-events
GET    /management/v1/runtime-diagnostics
POST   /management/v1/api-keys:bulk-create
GET    /management/v1/operations
GET    /management/v1/operations/{operationId}
POST   /management/v1/operations/{operationId}:cancel
POST   /management/v1/operations/{operationId}:claim-secret-result
~~~

Usage queries accept a bounded start and end time, IANA time zone, automatic or
explicit minute/hour/day grain, keyset cursor, and server-authorized filters for
namespace, Team, User, API key, Entrypoint, Recipe, logical Model, backend Model,
provider, status, and dispatch type. Responses include <code>asOf</code>, rollup
grain, ledger watermark, ingestion lag, and whether the result is final. Long ranges
read rollups; request detail is keyed by path namespace plus Router `admission_id`,
reads the immutable ledger, and returns external IDs only as correlation. Request
payload capture is off by default and separately permissioned.

Every usage total, series point, and breakdown row returns `costs: CostSummary[]`,
including `[]` when no cost-bearing dispatch exists. Entries use CurrencyDecimal,
carry completeness and dispatch counts, sort by currency, and never mix currencies.
Partial amount is a lower bound, never a total. Cost meters use the quota appendix's
public decimal shape and share the token quota `asOf` contract.

Without `usage.internal_dimensions.read`, responses omit logical/backend Model,
provider, internal dispatch, and hidden assignment fields and reject filters on those
dimensions. The sole exception is a direct-Model request whose Model is discoverable
to the caller; it may show that public Model only. `log.read` does not expose payloads;
`log_payload.read` is a separate field-level permission. Team administrators receive
Team aggregates by default and cannot inspect another member's raw log or payload
without an explicit User-scoped permission.

Domain actions create asynchronous Operations instead of one transaction over 10,000
resources. There is no generic task-submission endpoint. API-key bulk-create,
binding bulk-apply, and Model bulk-import each validate their own domain permission
and subject scope at enqueue and again for every item at execution. An Operation
exposes progress, item failures, desired revision, publication revision, and applied
revision without secret values. Cancel stops pending work but never deletes completed
results or audit.

An Operation records its originating principal and complete target scope. That actor
may read its own Operation with intrinsic `self.read` while its original domain
authority remains valid, including TeamRole/self-service operations; cross-actor reads
require `operation.read`. The same distinction applies to cancel and secret claim, so
a Team-only user never needs a broad operations role to retrieve its own result.

A secret-generating request that returns `202` also returns a random 256-bit
`secretResultClaim` exactly once. PostgreSQL stores only its HMAC and a separately
envelope-encrypted copy of the claim alongside the separately encrypted response,
both bound to operation ID, original principal, normalized request digest, originating
idempotency-key digest, and issuing authentication context. The claim remains valid
through the bounded maximum Operation lifetime; delivery TTL starts only at terminal
success. Replaying the request with the same `Idempotency-Key` reproduces the identical
`202` and claim, so a lost initial response does not strand the result. When the
Operation reaches `succeeded` or `partially_succeeded`, it seals an envelope containing
IDs and one-time secrets for successful items plus non-secret errors for failed items,
and starts the delivery TTL. Once any secret-bearing item commits, the Operation must
finish `succeeded` or `partially_succeeded` and seal every committed item, even after
cancel or worker recovery. `failed` and `cancelled` are legal only when no item
committed and therefore seal an empty result. The original principal presents the
claim to `:claim-secret-result` from any current non-revoked session with the original
domain permission, issuing evidence kind, and pinned typed authentication predicate.
The endpoint atomically marks the envelope delivered and returns that sealed result.
Invitation onboarding uses its capability from any current non-revoked session for the
exact principal, same authentication source and evidence kind, and pinned typed predicate.
It needs no `key.manage` and survives session renewal. Claim atomically rechecks the unchanged
active Principal/User link, key owner/context, deliverable credential,
User/Team/membership, and deny barriers. Restrictive lifecycle mutation erases each
affected envelope and claim before success; re-enable cannot resurrect it.
The same `Idempotency-Key` may replay the byte-identical response during a short
delivery-recovery window; another key, actor, or payload gets no secret.
After the recovery window the encrypted claim and response envelope are
cryptographically erased and the endpoint returns `410 secret_result_expired`.
Normal Operation reads and audit never expose claim, envelope, or secret. This covers
onboarding, key create/rotate, delegated sessions, and `api-keys:bulk-create`.

## API contract rules

- Creates/actions accept <code>Idempotency-Key</code> and replay one result only for an
  identical request. Synchronous key, invitation, service-account, and delegated-
  session calls retain one encrypted response for a bounded window; asynchronous
  calls use the claim protocol. A different payload is rejected; ProviderCredential
  writes never echo input secrets. Every secret-bearing success, partial, error, or
  replay uses `Cache-Control: no-store`, never redirects, excludes secrets from URL,
  history, compression, logs, and traces, and uses `Vary: Authorization` when authenticated.
- Router <code>admission_id</code> identifies admission/settlement. Client idempotency
  is scoped to logical key and request digest and never grants a free dispatch.
- Mutations return <code>ETag</code>; updates require <code>If-Match</code> and reject lost
  updates with <code>412 Precondition Failed</code>.
- Lists use opaque keyset cursors, stable sort, explicit filters, and bounded pages;
  large offsets are unsupported.
- Publication responses distinguish desired, staged, publication, and applied
  revisions. Restriction waits for its deny barrier; expansion waits for its gate.
- Delete first denies credentials, then removes mutable links under referential
  checks; immutable usage/audit references remain.
- Management authentication uses Router-issued tokens obtained from verified external
  assertions, scoped service credentials, or mTLS mappings. Inference credentials
  have no Management API authority.
- Every mutation, secret reveal, role change, access check, and reconciliation stores
  actor chain, request ID, resource, before/after revision, outcome, and reason with
  no raw secret.
- Dashboard, CLI, and custom consoles use generated OpenAPI clients, never Router
  database or internal package types.

## Effective policy and quota responses

~~~json
{
  "keyId": "key_01...",
  "revision": 42,
  "appliedRevision": 42,
  "access": {
    "grants": [
      {
        "resourceType": "entrypoint",
        "resourceId": "ep_blend_01...",
        "permissions": ["discover", "invoke"],
        "source": {"subjectType": "user", "subjectId": "usr_01...", "bindingId": "ab_01..."}
      }
    ]
  },
  "quota": {
    "meters": [
      {
        "policyId": "rlp_developer_01...",
        "ruleId": "rpm_rolling",
        "bindingId": "rb_key_01...",
        "source": {"subjectType": "api_key", "subjectId": "key_01..."},
        "counterOwner": "rb_key_01...",
        "metric": "requests",
        "algorithm": "sliding_log",
        "accounting": "request",
        "enforcement": "enforce",
        "window": "PT1M",
        "limit": "12",
        "used": "2",
        "remaining": "10",
        "resetAt": "2026-08-22T12:01:04Z",
        "state": "known", "capacityState": "available",
        "activeFenceIds": [],
        "freshness": {"source": "live", "asOf": "2026-08-22T12:00:05Z"}
      }
    ],
    "limitingRuleId": "rpm_rolling",
    "unknownUsageFences": [],
    "asOf": "2026-08-22T12:00:05Z"
  }
}
~~~

Each grant names its source and binding. Each non-cost meter returns whole-unit
limit/used/remaining; cost meters use the public decimal shape in the
[quota runtime](./router-native-access-control-quota-runtime). All include live reset,
freshness, and known/capacity state from the same admission Functions. The response
identifies the limiting rule; analytics lag cannot alter live quota.

## Management authorization

The normative permission registry, exact built-in roles, scope-containment graph,
delegation ceiling, per-operation expressions, intrinsic self permissions, and
TeamRole entitlements live in the
[Management authorization appendix](./router-native-access-control-authorization).

Invitation email delivery, welcome-page presentation, password setup, and browser
session presentation remain issuer/client UX. The Router owns the invitation's
one-time authorization, expiry, target roles, Team, onboarding transaction, and
audit. Automatic first-key creation uses the same logical key, Team inheritance,
publication gate, and secret-envelope path as an administrator or custom console.
Every Management endpoint enforces these permissions even when the Dashboard hides
the corresponding navigation or control.
