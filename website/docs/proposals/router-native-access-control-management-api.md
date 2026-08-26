---
title: Router-Native Access Control Management API Appendix
description: Specifies management identity exchange, delegated inference credentials, API resources, effective policy responses, and authorization.
created: 2026-08-22
status: Proposal
---

> **Status:** Proposal appendix · **Created:** 2026-08-22

This appendix is normative for Management authentication, endpoints, requests, and responses in [Router-Native Access Control and Quota Accounting](./router-native-access-control).
The [resource contract](./router-native-access-control-contracts) owns storage and
policy; [Provider catalog](./router-native-access-control-provider-catalog) owns the
Integration Registry, compiler, and adapter boundaries;
[authorization](./router-native-access-control-authorization) owns permissions and
scopes; [deployment](./router-native-access-control-deployment) owns bootstrap,
topology, readiness, and recovery.

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
stores the issuer, nonce digest, rate-identity digest, expiry, and one-time consumption
state in PostgreSQL. The raw nonce is never persisted. The client supplies that nonce
to its OIDC authorization request or local assertion signer. The token-exchange JSON
is a discriminated union with `issuerId`, `exchangeChallengeId`, `subjectToken`, and
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

A `router_local_assertion` is a one-time exchange assertion, not the lifetime
evidence for the resulting Management session. Its `exp` remains short and bounds
replay exposure. It must also carry integer `source_session_exp`, copied exactly from
the broker's already verified source session. The Router rejects a missing or expired
source expiry, one earlier than the assertion expiry, or one more than 30 days in the
future, and persists it as the Management session's evidence-expiry ceiling. OIDC
exchanges continue to use the verified ID token's own `exp` as that ceiling. A local
bootstrap or invitation flow must durably define the prospective source-session
expiry before exchange and issue the resulting browser session with that exact expiry;
it may rotate expired pending evidence but must never invent an unpersisted fallback.

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
`410 bootstrap_result_expired`; that first credential is no longer recoverable through
bootstrap. An operator may restore an existing administrator binding through the
separate break-glass route and then use ordinary credential rotation.

Break-glass uses `POST /management/v1/auth/recovery` with
`Authorization: VSR-Recovery <secret>`. The route is registered only when recovery was
explicitly enabled at process start, is reachable only from the Router's loopback
interface, and uses a separate root-readable token file. Its bounded body names one
existing durable principal and a required reason. It can only reactivate that principal
and restore its built-in `cluster_admin` binding with the fixed delegation ceiling; it
cannot create principals or credentials, or read or mutate keys, policies, routing,
usage, or secrets. One serializable transaction records the target, one-use recovery
nonce, normalized request digest, binding, receipt, and audit. An exact idempotent replay
can recover the non-secret receipt during its bounded window. The token is spent after
one successful request, and normal operation requires restart with recovery disabled.

The token-exchange and service-token endpoints commit a Management session and return `{accessToken, tokenType: "Bearer", expiresIn, managementSessionId}`. Bootstrap returns its bounded receipt and, for service-account bootstrap, the first credential; recovery returns only a non-secret receipt. Neither is a Management-session authentication source. The Router-signed JWT has fixed Management audience and `sub`, `sid`, `jti`, `iat`, `exp`, `auth_source_kind`, stable source ID, and exactly one evidence object: human `auth_time|aal|amr` or workload `class|source_assured_at`. The server derives these claims from the persisted issuer, credential, or mTLS mapping; clients cannot supply them. Every request validates signature/audience and principal, authentication-source, and session deny projections. Issuer back-channel
logout, client logout, credential disablement, or administrator revocation installs
that barrier before reporting success. When a Dashboard backend or another broker
performs the human exchange, its service account remains in the actor chain but does
not replace the human principal. Audit records both actors.

Access tokens have no refresh token and an old bearer is never sufficient to renew
itself. Before access-token expiry, a client repeats token exchange with a fresh
verified issuer assertion or repeats service-token authentication with the live
credential or mTLS identity. For issuer exchange, the Router reuses an active durable
session only when principal, source kind/ID, issuer session ID, audience,
authentication time, and the complete evidence object match exactly. That reissue
keeps the durable session ID and `jti` stable, does not extend durable session or
evidence expiry, and returns another short-lived token bounded by the same expiry.
Previously issued tokens for that exact session may overlap until their own bounded
expiry; independently cached tokens on several Dashboard replicas therefore do not
invalidate one another. A changed source, issuer session, audience, authentication
time, or evidence creates a separate bounded session subject to the active-session
limit. Serializable exchange and a principal/session row lock ensure concurrent
first exchange or reissue converges to one durable result across replicas.

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
least one of `sid` or `sub` is required. In one serializable transaction, a `sid`
revokes matching issuer sessions and installs a durable issuer/SID tombstone; a `sub`
without `sid` revokes sessions authenticated no later than the logout event and
installs a durable issuer/subject authentication-time watermark. Selector values are
stored only as domain-separated digests. Exchange locks and checks both applicable
selectors before reading or creating a session, so a logout that arrives before or
concurrently with exchange cannot be bypassed by a late commit. A SID cannot be
reissued after its tombstone. Subject authentication evidence strictly newer than the
logout watermark may create a new session, while older or equal evidence is denied.
The hashed logout-token `jti` is retained through token expiry for replay rejection.
Repeating an already applied valid logout is an idempotent `200`; a reused ID with
different claims is rejected.

Any fresh verified token exchange may carry a matching invitation, whether the exact
`(issuer, subject)` principal is new or active. One CAS transaction consumes it,
creates or reuses only that principal, and creates the missing namespace User/link,
roles, optional TeamMembership, audit/outbox, and optional inherited first key. It
never merges by email. A conflicting existing link/User or already-onboarded scope
returns a deterministic conflict without consuming the invitation. Exact idempotent
replay preserves its bounded one-use result. When automatic first-key creation is
enabled, the token response nests the applied one-time key envelope directly; it never
creates a secret-bearing asynchronous Operation. Failure creates nothing.

A Dashboard invitation materializes two explicit role bindings when the invited
person is also a model consumer: one scope-contained ManagementRole selected by the
inviter for Dashboard/Management API authority, and one User-scoped `consumer` role
for delegated inference and read-only account access. Neither role implies the other.
This prevents a read-only Dashboard member from acquiring platform mutation authority
while ensuring that the inherited first key and Playground work through the same
Router-owned inference boundary as a direct client.

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

Playground and other first-party experiences call the public inference listener with
a short-lived delegated inference credential. An authenticated principal linked to a
User may request that credential through these operations:

~~~text
GET    /management/v1/self/inference-keys
GET    /management/v1/self/inference-keys/{keyId}
GET    /management/v1/self/inference-sessions
POST   /management/v1/self/inference-sessions
DELETE /management/v1/self/inference-sessions/{sessionId}
~~~

Creation names a namespace and selects either a key owned by the linked User or a
Team-owned key for an active membership when self-service policy permits that use.
`GET /self/inference-keys` returns only safe ID/name/owner/expiry metadata for those
eligible keys. Its scoped by-ID companion returns the same metadata for one eligible
key and otherwise returns not found; neither operation reveals a key secret or grants
general Team-key read access.
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
<code>/management/v1</code>. It is the sole authority for dynamic mutations;
file-only deployments do not start a mutable Management API.

This is an optional capability of the same v0.3 bootstrap, not a `managed` deployment
mode. `global.stores.management` selects durable desired state, while
`global.services.management_api.enabled` exposes its authorized HTTP mutation and
query surface. With the listener disabled, PostgreSQL publication can remain active
without a public Management endpoint. With no Management store, the static manifest
remains the sole routing authority.

### Version negotiation

The Management API version is independent from the top-level Router manifest
version. Clients select `/management/v1` and
`application/vnd.vllm-semantic-router.management.v1+json` explicitly in both
`Accept` and `Content-Type`; there is no release-number negotiation or silent
fallback. Version 1 may add response fields and optional request fields with
documented defaults. Removing, renaming, reinterpreting, or changing a default
requires `/management/v2`, a new media type, and a separately published OpenAPI
document. Unknown request fields are rejected; clients ignore unknown response
fields.

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
and an immutable onboarding snapshot. A Team invitation pins the Team assignment and
requires that Team to retain an active AccessPolicy layer and RateLimit allocation at
acceptance; it does not create a User-level policy override. An invitation without a
Team instead pins active same-namespace Access/Rate policy IDs and revisions from the
self-service defaults. Invitations cannot choose either policy path directly. This
namespace-authorized capability is consumable with `invitation.manage`; TeamRole also
needs membership manage. Create/rotate returns the secret once. Accept verifies
identity, delegation ceiling, and the selected inheritance path in one CAS; stale or
conflict neither reveals nor consumes it.

The namespace self-service policy defines maximum logical keys per User,
delegated-session count and TTL, whether active members may use Team-owned keys,
automatic first-key behavior, Team-admin capabilities, and default AccessPolicy and
RateLimitPolicy IDs. A write authorizes both current and target defaults and accepts
only active same-namespace policies. Onboarding without a Team materializes them as
real User bindings; Team onboarding creates only the membership so the User and first
key continue to inherit current Team policy. There is no hidden namespace layer in
runtime inheritance.
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

User and Team lists accept an optional bounded `search` term over canonical display
fields. Search executes after authorization scope is applied and before keyset
pagination. The opaque cursor binds the normalized search, status filter, and
authorization-scope digest, so it cannot be replayed against a broader or different
query. Empty search is the ordinary stable list; implementations use suitable
indexes and reject terms that exceed the documented bound.

A Team is created together with its AccessPolicy binding and RateLimitPolicy
allocation binding, selected explicitly or from namespace onboarding defaults.
There is no partially configured Team and no hidden runtime inheritance. TeamRole
remains separate from ManagementRole. Creation requires manage authority over the
Team and both selected policies; lifecycle changes use the ordinary revision-checked
Team update.

`POST /teams` accepts optional `accessPolicyIds` and `rateLimitPolicyId` selections.
Omitting a field selects its current namespace default; an explicitly empty AccessPolicy
list or blank RateLimitPolicy ID is invalid. The Router resolves omitted selections before
authorization, requires authority over every resolved policy, and includes the sorted,
unique AccessPolicy IDs plus the RateLimitPolicy ID in the idempotency digest. The create
transaction locks any default revision it used, verifies every selected policy is active
and belongs to the same namespace in its typed policy store, then creates the Team, all
AccessPolicy bindings, the single RateLimit allocation binding, audit event, outbox event,
and command result together. Any failed validation rolls back the whole operation.

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
GET    /management/v1/api-keys/{keyId}/routing-catalog
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

API-key, Access Policy, and Rate Limit Policy lists expose the same bounded
server-side `search` contract as Users and Teams. API-key search covers name and the
public key identifier only; it never searches or reveals credential material.
Policy search covers name and description. Every cursor binds the normalized search,
filters, result scope, and resource kind.

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
GET    /management/v1/routing/model-cards
POST   /management/v1/routing/imports
GET    /management/v1/routing/exports/current
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
POST   /management/v1/provider-catalog:bootstrap
POST   /management/v1/provider-catalog:activate
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
read. Decision assignments are embedded in Entrypoint writes and reads; there is no
detached assignment API.

The API-key `routing-catalog` endpoint is the consumer projection for Routing,
Topology, and Playground. It joins that key's exact applied access projection to the
immutable routing snapshot pinned by the same publication, then returns only
discoverable Models and Entrypoints, the referenced Recipe metadata, and assignments
to discoverable Models. It cannot represent Provider backends, credentials, or Recipe
source. A User-scoped `consumer` may read the catalog only for a key under that User;
switching among owned keys changes the effective projection but never broadens it.

`POST /routing/imports` is the only bridge from a later static manifest to an
initialized Management store. It accepts one strict v0.3 manifest, `If-Match`, and
`Idempotency-Key`; validates the complete Model/modelCard/Recipe/Entrypoint closure;
returns a typed diff whose arrays contain stable resource names; and creates one
Operation whose publication is atomic. Referenced credentials require
`provider_credential.use` on every exact credential and are checked active in the
same PostgreSQL transaction. It never
imports Users, Teams, keys, policies, counters, credentials, or generated IDs. A
dry-run request performs the same compilation without writing. Current export emits
readable v0.3 routing source with secrets represented only by references, requires
namespace-wide Routing and Provider Credential read, and omits
default-only fields. Startup uses the same importer exactly once when the target
Namespace store is genuinely empty; a nonempty store is never reconciled implicitly.

The snapshot collection is an immutable, newest-first audit surface. Its signed
cursor is bound to the path Namespace and returns revision, content digest,
lifecycle, member count, and publication timestamps without loading compiled
payloads. Snapshot detail returns the exact ordered Model, Recipe, and Entrypoint
member revisions plus the validated self-contained routing export stored for that
revision. The Router verifies the export digest and member closure before returning
it; corruption fails closed, and neither response contains credential material.

`Recipe.document` is authoring source: Decisions are addressed only by their readable
`name`, and an authored `id` is rejected. The response's separate `decisions` metadata
contains the compiler-owned stable IDs used by Entrypoint assignment maps and immutable
snapshots. Those IDs are never written back into YAML or DSL.
Without exact credential read, Models show safe capability and `credentialConfigured`;
one projector omits credential metadata from all responses. Secrets are never read.

The ModelCard endpoint is a read-only semantic projection for Recipe and
Entrypoint authoring, assignment, and topology views. It exposes a model's
human-readable identity and capability metadata, while excluding connection
details, provider credentials, compiled backends, provider-catalog state,
invocation control, pricing, health state, revisions, and timestamps. The view is
searchable and cursor-paginated under `routing.read`; it is never mutated as an
independent resource.

Each decision key in `rules[].action.assignments` maps to a
`{models, fallback?}` value. A Model reference accepts `modelId`, optional priority,
weight, LoRA name, and reasoning controls. Priority defaults to zero. Fallback is a
closed `priority` strategy with a bounded trigger set; enabling it requires a
single-dispatch Recipe decision and at least two valid tiers. The OpenAPI schema does
not expose a free-form status-code list, gateway retry knobs, or another fallback
resource. Create, update, resolve, topology, snapshot, import, and export use this one
shape.

Entrypoint `:resolve` accepts path and optional subject. With native access a subject
is required only for claim rules and loads stored effective context; an override is
a privileged simulation. Its full atomic result requires `routing.read` on the
Entrypoint, exact Recipe revision, and every returned Model; there is no partial
topology response. Deployments without native access reject subject/override fields.

The read-only provider catalog is the active, content-addressed value produced by
the application-installed control-plane Integration Registry. It is not generated
from a hard-coded Dashboard product list or user-authored Router configuration. A
provider entry has a stable `providerId` and Definition revision, display metadata
and a validated control-plane-owned icon descriptor, fixed or user-supplied origin prompts, explicit
`none|optional|required` credential UX, typed non-secret connection fields,
discovery availability, streaming/tool/multimodal capabilities, and catalog
revision. The Management representation omits protocol, credential, and discovery adapter IDs,
invocation paths, and internal headers; those remain compiler inputs. Public
providers with no stable default URL are not advertised as key-only integrations.
Private and compatible providers explicitly require a base URL. ProviderCredential
create references one catalog `providerId`; the server copies its credential adapter and
canonical origin into immutable credential metadata. Consoles render the returned
definition and never hard-code products, authentication headers, or provider-specific
forms. An unauthenticated local backend references no ProviderCredential; use
authorization applies only to credentials a backend actually references.

Every catalog response returns `catalogRevision`; pagination cursors bind that
revision. Provider Integrations are application composition, not Management CRUD
resources. The coordinator validates installed compiler, protocol, credential, and
discovery adapter IDs, stores the immutable catalog by digest, and activates it only
after the declared control-plane and data-plane rollout groups report compatibility.

On a genuinely empty durable store, Router replicas automatically stage and activate
the unique application-installed Integration Registry through the same
compare-and-swap and rollout-gate contract. Concurrent startup is idempotent: each
replica ACKs its declared groups, conflicts are reread, and a missing peer ACK leaves
the catalog unready rather than inventing a smaller gate. A different desired or active
revision is never changed by this cold-start path.

The explicit lifecycle operations remain available to authorized automation.
`POST /management/v1/provider-catalog:bootstrap` accepts the positive decimal-string
`expectedGeneration` and stages only the immutable application-installed snapshot; it
never accepts Provider definitions. Replaying the same Registry revision and rollout
gate is idempotent, while another desired revision or generation returns a conflict.
`POST /management/v1/provider-catalog:activate` accepts the exact `revision` and
`expectedGeneration` and remains blocked until every configured rollout group holds a
compatible, unexpired capability lease. Both operations require `cluster.manage`.
Catalog read, discovery, and Model mutation remain unavailable until activation.

A rolling deployment may serve an older catalog revision for an already issued
cursor or discovery claim, but it cannot
silently reinterpret it. Removed definitions stop new writes; published Model
snapshots continue to carry their already compiled adapter and connection values.

ProviderCredential create binds its immutable `providerId`,
`credentialAdapterId`, catalog revision, and normalized
scheme/host/port/base-path origin before encrypting the secret. Fixed public origins
come from the definition; user-supplied origins pass egress validation. Rotation
changes only secret version. Moving an origin or changing credential adapter requires
a new credential, and Model/discovery/probe validation rejects
provider/adapter/origin mismatch or non-schema connection overrides.

Create and rotate accept a secret over the private Management transport when the
provider authentication schema requires one and return metadata only. Discover-models
accepts schema-validated connection fields and an optional `credentialId`, then uses
the definition's control-plane discovery adapter and that credential, when present,
to return a
bounded, normalized remote catalog without persisting Models, and is available only
when the provider advertises discovery. Its request accepts bounded `pageSize`, search,
and an opaque provider cursor; each response returns stable catalog item IDs,
`nextCursor`, and a `discoveryRevision` with expiry. Bulk-import accepts selected IDs,
that revision, and non-security Model overrides. The server-held/signed,
namespace-bound revision pins provider catalog and Definition revisions, normalized connection/origin
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
target is audited. Discovery never follows redirects. Secret-bearing probe/dispatch
never follows redirects; a future no-auth redirect policy would require explicit
adapter support and full target revalidation without secret headers.
Credential-backed discovery requires credential read/use; no-auth
discovery additionally requires namespace `routing.manage`. Thus catalog read alone
never grants arbitrary Router-side network access.

Model create, update, probe, bulk-import overrides, bootstrap manifests, and exports
share the [Model control contract](./router-native-access-control-model-runtime). The
Model revision contains `control.retry.count` (0-5 additional attempts governed by
Router's fixed safe-retry predicate), a duplicate-free closed
`control.retry.on` list, duration strings `control.timeout.request` and
`control.timeout.stream` (1 second-24 hours), plus
four nullable non-negative decimal-string prices per million tokens:
`inputCostPerMillionTokens`, `outputCostPerMillionTokens`,
`cacheReadCostPerMillionTokens`, and `cacheWriteCostPerMillionTokens`. Cache prices
inherit input when omitted; explicit zero means free, while absent input/output means
unpriced. The namespace supplies the immutable billing currency. Management reads
return the effective immutable revision; readable file export omits values that were
only defaulted. Probe uses the saved retry policy, while its control-plane operation
deadline is the smaller of the saved request timeout and five minutes so a long
inference deadline cannot make model verification unbounded.

The Model create, update, and read surface uses one nested value; flattened retry or
timeout fields are not accepted:

~~~json
{
  "control": {
    "retry": {"count": 2, "on": ["unavailable", "timeout"]},
    "timeout": {"request": "60s", "stream": "10m"}
  }
}
~~~

Provider-specific `connectionFields` are the one deliberately dynamic request
object in Model and discovery writes. Their effective schema comes from the exact
active Provider Definition returned by `/providers`, not from Dashboard code. The
server rejects undeclared, missing required, incorrectly typed, or out-of-range
values before a compiler runs, and the published backend contains only the compiler's
closed non-secret output. This OpenAPI envelope therefore does not create an
open-ended data-plane configuration map.

Model PATCH is sparse: omitted fields retain their server-owned value. Changing
control, pricing, name, aliases, capabilities, reasoning, or LoRAs therefore never
requires a client to read or resubmit backend origins, compiled connection values, or
ProviderCredential references. Supplying `backends` is an explicit whole-list
replacement and conditionally requires `provider_credential.use` on every referenced
credential. An empty patch is invalid. Every successful PATCH uses ETag, creates an
immutable revision, audits the diff, and enters the normal routing publication gate.
Discovery suggestions are never another authority.
The Dashboard keeps provider, credential/base URL, and discovered Model selection in
the primary flow. A collapsed **Advanced settings** section contains Max retries,
Retry on, Request timeout, Stream timeout, and Input/Output/Cache Read/Cache Write
Cost, each labeled **per 1M tokens** with inline defaults and validation. It edits the Model
value directly; there is no separate pricing, retry, or timeout resource. List/detail
show effective values, while Usage/Insights expose cost, currency, and incomplete-cost
state instead of treating missing prices or usage as zero.

The Dashboard is a replaceable client of these resources. Its Models inventory and
Recipe-scoped Signals, Projections, and Decisions editors read and mutate this API
with resource revisions; they never edit Router YAML or keep a private Recipe draft
table. Built-in Recipes are immutable and can be duplicated into a custom Recipe.
Decision authoring contains no physical Model picker: Model priority, weighting,
reasoning controls, and fallback are configured only in the Entrypoint assignment
that turns a Recipe into a callable Mixture-of-Models.

### Router-native Agent resources

~~~text
GET    /management/v1/agent-profiles
POST   /management/v1/agent-profiles
GET    /management/v1/agent-profiles/{profile}
PATCH  /management/v1/agent-profiles/{profile}
DELETE /management/v1/agent-profiles/{profile}
GET    /management/v1/agent-skills
POST   /management/v1/agent-skills
GET    /management/v1/agent-skills/{skill}
PATCH  /management/v1/agent-skills/{skill}
DELETE /management/v1/agent-skills/{skill}
GET    /management/v1/agent-tools
GET    /management/v1/agent-tool-credentials
POST   /management/v1/agent-tool-credentials
GET    /management/v1/agent-tool-credentials/{credential}
PATCH  /management/v1/agent-tool-credentials/{credential}
DELETE /management/v1/agent-tool-credentials/{credential}
POST   /management/v1/agent-tool-credentials/{credential}:rotate
GET    /management/v1/agent-tool-sources
POST   /management/v1/agent-tool-sources
GET    /management/v1/agent-tool-sources/{source}
PATCH  /management/v1/agent-tool-sources/{source}
DELETE /management/v1/agent-tool-sources/{source}
POST   /management/v1/agent-tool-sources/{source}:test
POST   /management/v1/agent-tool-sources/{source}:approve
GET    /management/v1/agent-sessions
POST   /management/v1/agent-sessions
GET    /management/v1/agent-sessions/{session}
PATCH  /management/v1/agent-sessions/{session}
DELETE /management/v1/agent-sessions/{session}
POST   /management/v1/agent-sessions/{session}/turns
GET    /management/v1/agent-sessions/{session}/turns
GET    /management/v1/agent-sessions/{session}/events
POST   /management/v1/agent-sessions/{session}/turns/{turn}:cancel
GET    /management/v1/agent-artifacts/{artifact}
GET    /management/v1/agent-artifacts/{artifact}/content
POST   /management/v1/publication-plans/{plan}:commit
~~~

These routes are the Router-owned durable Agent surface. Profiles select bounded
capabilities and versioned Skills; Sessions pin one authorized inference target;
Turns, Events, Artifacts, and publication plans remain namespace- and
subject-scoped. Tool Sources reference write-only credential resources. Publication
preparation may produce an immutable plan, but only the separate human-authorized
commit endpoint may publish it. The Dashboard is only one generated client of this
surface and owns no parallel Agent workflow state.

### Usage, logs, audit, and operations

~~~text
GET    /management/v1/statistics
GET    /management/v1/usage
GET    /management/v1/usage/series
GET    /management/v1/usage/breakdowns
GET    /management/v1/request-logs
GET    /management/v1/namespaces/{namespaceId}/request-logs/{admissionId}
GET    /management/v1/audit-events
GET    /management/v1/runtime-diagnostics
GET    /management/v1/operations
GET    /management/v1/operations/{operationId}
POST   /management/v1/operations/{operationId}:cancel
~~~

`GET /management/v1/statistics` is the bounded control-plane companion to Usage.
It returns `asOf`, an inclusive `expiringBefore` boundary fixed by the server, and
exact decimal-string counts for Users, Teams, active and soon-expiring API keys,
Access Policies, and active Rate Limit Policies. The operation first requires
`usage.read`; each optional field is then independently projected through
`user.read`, `team.read`, `key.read`, `access_policy.read`, or
`rate_policy.read`. A denied field is omitted, never reported as zero. Failure to
resolve any non-denied scope fails the request closed.

The repository executes one fixed-shape PostgreSQL statement whose indexed
subqueries receive the authorized User, Team, API-key, or typed resource IDs. It
does not enumerate list pages, return entity rows, or use offset pagination, so
the response and query count remain constant at 10,000 keys and beyond. This
endpoint contains no request, token, cost, or latency aggregates: clients compose
it with the Usage endpoints below, which remain the sole ledger authority.

Usage queries accept a bounded start and end time, IANA time zone, automatic or
explicit minute/hour/day grain, keyset cursor, and server-authorized filters for
namespace, Team, User, API key, Entrypoint, Recipe, logical Model, backend Model,
provider, status, and dispatch type. Responses include <code>asOf</code>, rollup
grain, ledger watermark, ingestion lag, and whether the result is final. Long ranges
read rollups; request detail is keyed by path namespace plus Router `admission_id`,
reads the immutable ledger, and returns external IDs only as correlation. Request
payload capture is off by default and separately permissioned.

The User, Team, and API-key detail endpoints expose the same exact usage summary
contract through `/{resourceId}/usage`. They first resolve the resource without
revealing whether a denied ID exists, authorize both the resource read and
`usage.read`, and then constrain the ledger query to that exact subject. A caller
cannot widen the query with a second filter for the same dimension. Time range,
grain, and the remaining authorized dimensions behave exactly like the namespace
usage endpoint, so control-plane clients do not need a separate accounting path for
detail pages.

API-key cost is available in both the exact key summary and the API-key breakdown of
the main Usage view. The API-key detail client composes
`/api-keys/{keyId}/usage` with `/api-keys/{keyId}/quota`: immutable ledger cost shows
what was spent, while a live `cost` meter shows its limit, exact used/remaining,
currency, reset, completeness, and capacity state. For example, an eight-hour budget
is an ordinary response-actual sliding rule:

~~~json
{
  "metric": "cost",
  "limit": "20",
  "algorithm": "sliding_log",
  "window": "PT8H",
  "accounting": "response_actual",
  "enforcement": "enforce"
}
~~~

The Dashboard presents this as `8h` and submits the canonical `PT8H` duration; it
does not maintain a second fixed RPM/TPM budget schema.

The same rule may bind directly to a key or be inherited from its User or Team. The
Dashboard labels currency and reset time explicitly and never calculates remaining
from delayed Usage rollups.

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

`GET /management/v1/runtime-diagnostics` is a cluster-scoped `health.read`
operation and never accepts a namespace Management session. With no query it
returns only sanitized PostgreSQL/Valkey state and the registered namespace
count. An optional exact canonical-UUID `namespaceId` selector adds the coupled
access/routing publication heads, active-replica acknowledgements, projector
lag, quota recovery state, pending-admission counts, expired-pending count,
oldest pending deadline, usage-stream backlog, configured backlog limit, and
whether admission is currently blocked by that limit. The summary uses the
Valkey directory cardinality and the exact view uses a single directory lookup;
neither path downloads or scans all namespace policies. A dependency fault is
returned as a typed `200 degraded` snapshot when a trustworthy partial view is
available, an unknown selector is nondisclosing `404`, and failure to construct
a trustworthy view is `503`. Responses never contain connection strings,
credentials, raw runtime keys, policy documents, routing topology, or secrets.

Unbounded domain actions use asynchronous Operations instead of one transaction over
large resource sets. There is no generic task-submission endpoint. Binding bulk-apply
validates domain permission and subject scope at enqueue and again for every item at
execution. Model bulk-import is capped at 200 selections, validates the complete batch,
commits it atomically, and returns a terminal Operation receipt. An Operation exposes
progress, item failures, desired revision, publication revision, and applied revision
without secret values. Cancel stops pending work but never deletes completed results
or audit.

An Operation records its originating principal and complete target scope. That actor
may read its own Operation with intrinsic `self.read` while its original domain
authority remains valid, including TeamRole/self-service operations; cross-actor reads
require `operation.read`. The same distinction applies to cancel, so a Team-only user
never needs a broad operations role to inspect its own non-secret result. Operations
never carry generated credentials. Secret-producing actions remain synchronous and
use their resource-specific bounded one-time response envelope.

## API contract rules

- Creates/actions accept <code>Idempotency-Key</code> and replay one result only for an
  identical request. Key, invitation, service-account, and delegated-session calls
  retain one encrypted response for a bounded window. Asynchronous Operations never
  contain generated credentials. A different payload is rejected; ProviderCredential
  writes never echo input secrets. Every secret-bearing success, partial, error, or
  replay uses `Cache-Control: no-store`, never redirects, excludes secrets from URL,
  history, compression, logs, and traces, and uses `Vary: Authorization` when authenticated.
- Router <code>admission_id</code> identifies admission/settlement. Client idempotency
  is scoped to logical key and request digest and never grants a free dispatch.
- Mutations return <code>ETag</code>; updates require <code>If-Match</code> and reject lost
  updates with <code>412 Precondition Failed</code>.
- Lists use opaque keyset cursors, stable sort, explicit filters, and bounded pages;
  large offsets are unsupported. Directory lists also provide bounded server-side
  search whose normalized value is cryptographically bound into the cursor.
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
  "subject": {
    "type": "api_key",
    "id": "74b7bdf1-69e1-40c7-a1af-92e9e4d5f252"
  },
  "revision": 42,
  "appliedRevision": 42,
  "access": {
    "grants": [
      {
        "resourceType": "entrypoint",
        "resourceId": "ep_blend_01...",
        "permissions": ["discover", "invoke"],
        "effect": "allow",
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
        "source": {
          "subjectType": "api_key",
          "subjectId": "key_01...",
          "bindingId": "rb_key_01..."
        },
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
        "completeness": "complete", "knownDispatches": "2",
        "incompleteDispatches": "0", "capacityState": "available",
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
freshness, completeness, and capacity state from the same admission runtime. The response
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
The bundled Dashboard carries the one-time secret only through an in-memory handoff
to its standard API-key delivery flow and never writes it to browser storage. Every
Management endpoint enforces these permissions even when a client hides the
corresponding navigation or control.
