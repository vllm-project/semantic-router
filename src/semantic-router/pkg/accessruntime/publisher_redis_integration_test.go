package accessruntime

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessprojection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	managementauthredis "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth/redis"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

const (
	publisherDelegationNamespaceID       = "namespace-delegated-runtime"
	publisherDelegationPartition         = "partition-delegated-runtime"
	publisherDelegationUserID            = "user-delegated-runtime"
	publisherDelegationKeyID             = "key-delegated-runtime"
	publisherDelegationEntrypointID      = "entry-delegated-chat"
	publisherDelegationAllowedModelID    = "model-delegated-allowed"
	publisherDelegationHiddenModelID     = "model-delegated-hidden"
	publisherDelegationManagementSession = "11111111-1111-4111-8111-111111111111"
	publisherDelegationPrincipalID       = "22222222-2222-4222-8222-222222222222"
	publisherDelegationAudience          = "vllm-sr-inference"
)

func TestPublisherRedisRuntimeAuthenticatesUserOwnedDelegationWithoutTeam(t *testing.T) {
	fixture := newPublisherDelegationFixture(t)
	authentication := authenticatePublisherDelegation(t, fixture)
	assertPublisherDelegationAuthorization(t, fixture, authentication)
	discoveryRequest := publisherDelegationDiscoveryRequest(authentication.Session)
	assertPublisherDelegationDiscovery(t, fixture, discoveryRequest)
	assertPublisherDelegationRevocationGates(t, fixture, discoveryRequest)
}

type publisherDelegationFixture struct {
	ctx           context.Context
	client        *redis.Client
	prefix        string
	issued        accesscredential.Issued
	publicationID string
	runtime       *Runtime
	barriers      *managementauthredis.Store
}

func newPublisherDelegationFixture(t *testing.T) publisherDelegationFixture {
	t.Helper()
	address := os.Getenv("ACCESSRUNTIME_TEST_REDIS_ADDR")
	if address == "" {
		address = os.Getenv("ACCESSPUBLISHER_REDIS_ADDR")
	}
	if address == "" {
		t.Skip("ACCESSRUNTIME_TEST_REDIS_ADDR or ACCESSPUBLISHER_REDIS_ADDR is not configured")
	}

	client := redis.NewClient(&redis.Options{Addr: address})
	t.Cleanup(func() { _ = client.Close() })
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)
	if err := client.Ping(ctx).Err(); err != nil {
		t.Fatalf("ping Redis: %v", err)
	}
	prefix := "access-publisher-delegation-it:" + uuid.NewString()
	t.Cleanup(func() { deletePublisherDelegationPrefix(context.Background(), client, prefix+":*") })

	keyring := accesscredential.PepperKeyring{
		ActiveVersion: "delegation-pepper-1",
		Keys: map[string][]byte{
			"delegation-pepper-1": []byte("0123456789abcdef0123456789abcdef"),
		},
	}
	issued, err := keyring.Issue(accesscredential.KindDelegation, "delegatedpublisher01")
	if err != nil {
		t.Fatalf("issue delegated credential: %v", err)
	}
	state := publisherDelegationDesiredState(issued, time.Now().UTC().Add(-time.Minute).Truncate(time.Millisecond))
	published := publishPublisherDelegationState(t, ctx, client, prefix, state)
	runtime, barriers := newPublisherDelegationRuntime(t, ctx, client, prefix, keyring)
	return publisherDelegationFixture{
		ctx: ctx, client: client, prefix: prefix, issued: issued,
		publicationID: published.PublicationID, runtime: runtime, barriers: barriers,
	}
}

func publishPublisherDelegationState(
	t *testing.T,
	ctx context.Context,
	client *redis.Client,
	prefix string,
	state accesspublisher.DesiredState,
) accesspublisher.ProcessResult {
	t.Helper()
	outbox := &publisherDelegationOutbox{batch: accesspublisher.OutboxBatch{
		NamespaceID:     publisherDelegationNamespaceID,
		DesiredRevision: state.Revision,
		RuntimeEpoch:    state.Namespace.RuntimeEpoch,
		QuotaPartition:  publisherDelegationPartition,
		RowIDs:          []string{"outbox-delegated-runtime"},
	}}
	store, err := accesspublisher.NewRedisStore(accesspublisher.RedisStoreOptions{
		Client: client, KeyPrefix: prefix,
	})
	if err != nil {
		t.Fatalf("create publisher Redis store: %v", err)
	}
	publisher, err := accesspublisher.NewEngine(accesspublisher.EngineOptions{
		Outbox: outbox,
		Desired: publisherDelegationDesiredReader{
			state: state,
		},
		Runtime: store, WorkerID: "delegation-runtime-integration",
		ClaimLease: 10 * time.Second, RetryDelay: time.Millisecond, CompactionBatch: 10,
	})
	if err != nil {
		t.Fatalf("create publisher: %v", err)
	}
	published, err := publisher.ProcessOnce(ctx)
	if err != nil || published.Disposition != accesspublisher.ProcessApplied || published.PublicationID == "" {
		t.Fatalf("ProcessOnce() = %+v, %v", published, err)
	}
	if !outbox.fenced || outbox.released || outbox.failed {
		t.Fatalf("outbox lifecycle = fenced %t, released %t, failed %t", outbox.fenced, outbox.released, outbox.failed)
	}
	if len(outbox.staged.Credentials) != 1 {
		t.Fatalf("published credentials = %d, want 1", len(outbox.staged.Credentials))
	}
	delegated := outbox.staged.Credentials[0]
	if delegated.Kind != accesspublisher.CredentialKindDelegation ||
		delegated.Projection.UserID != publisherDelegationUserID || delegated.Projection.TeamID != "" {
		t.Fatalf(
			"published delegated authority mismatch: kind=%q user_bound=%t team_empty=%t",
			delegated.Kind,
			delegated.Projection.UserID == publisherDelegationUserID,
			delegated.Projection.TeamID == "",
		)
	}
	return published
}

func newPublisherDelegationRuntime(
	t *testing.T,
	ctx context.Context,
	client *redis.Client,
	prefix string,
	keyring accesscredential.PepperKeyring,
) (*Runtime, *managementauthredis.Store) {
	t.Helper()
	reader, err := NewRedisProjectionReader(RedisProjectionReaderOptions{Client: client, KeyPrefix: prefix})
	if err != nil {
		t.Fatalf("create runtime projection reader: %v", err)
	}
	quotaEngine, err := quotaruntime.NewRedisEngine(client, quotaruntime.RedisEngineOptions{KeyPrefix: prefix})
	if err != nil {
		t.Fatalf("create runtime access engine: %v", err)
	}
	barriers, err := managementauthredis.New(managementauthredis.Options{
		Client: client, KeyPrefix: prefix, Loader: publisherDelegationBarrierLoader{},
	})
	if err != nil {
		t.Fatalf("create management delegation barriers: %v", err)
	}
	if err := barriers.Rebuild(ctx); err != nil {
		t.Fatalf("initialize management delegation barriers: %v", err)
	}
	runtime, err := New(RuntimeOptions{
		Reader: reader, Engine: quotaEngine, APIKeyPeppers: keyring, DelegationPeppers: keyring,
		DelegationAudience: publisherDelegationAudience, DelegationBarriers: barriers, KeyPrefix: prefix,
	})
	if err != nil {
		t.Fatalf("create access runtime: %v", err)
	}
	return runtime, barriers
}

func authenticatePublisherDelegation(
	t *testing.T,
	fixture publisherDelegationFixture,
) Authentication {
	t.Helper()
	authentication, err := fixture.runtime.Authenticate(
		fixture.ctx,
		AuthenticationRequest{Credential: fixture.issued.Plaintext},
	)
	if err != nil || !authentication.Result.Allowed() {
		t.Fatalf("Authenticate() = %+v, %v", authentication, err)
	}
	if authentication.Source != AuthenticationSourceDelegated ||
		authentication.Tenant.NamespaceID != publisherDelegationNamespaceID ||
		authentication.Tenant.APIKeyID != publisherDelegationKeyID ||
		authentication.Tenant.UserID != publisherDelegationUserID || authentication.Tenant.TeamID != "" ||
		authentication.Tenant.PublicationID != fixture.publicationID {
		t.Fatalf("authenticated delegated tenant = %+v, source = %q", authentication.Tenant, authentication.Source)
	}
	return authentication
}

func assertPublisherDelegationAuthorization(
	t *testing.T,
	fixture publisherDelegationFixture,
	authentication Authentication,
) {
	t.Helper()
	authorization, err := fixture.runtime.Authorize(fixture.ctx, AuthorizationRequest{
		Session: authentication.Session,
		Target: Target{
			ResourceType: accesscontrol.GrantResourceEntrypoint,
			ResourceID:   publisherDelegationEntrypointID,
			Permission:   accesscontrol.GrantPermissionInvoke,
		},
	})
	if err != nil || !authorization.Result.Allowed() || authorization.Tenant.TeamID != "" {
		t.Fatalf("Authorize() = %+v, %v", authorization, err)
	}
	admission, err := fixture.runtime.Admit(fixture.ctx, AdmissionRequest{
		Session: authentication.Session,
		Target: Target{
			ResourceType: accesscontrol.GrantResourceEntrypoint,
			ResourceID:   publisherDelegationEntrypointID,
			Permission:   accesscontrol.GrantPermissionInvoke,
		},
		AdmissionID:   "admission-delegated-runtime",
		RequestDigest: strings.Repeat("d", 64),
		LeaseDuration: time.Minute,
	})
	if err != nil || !admission.Result.Allowed() || admission.Result.PlanDigest == "" || admission.Tenant.TeamID != "" {
		t.Fatalf("Admit() = %+v, %v", admission, err)
	}
}

func publisherDelegationDiscoveryRequest(session Session) CatalogDiscoveryRequest {
	return CatalogDiscoveryRequest{
		Session: session,
		Queries: []DiscoveryQuery{
			{ResourceType: accesscontrol.GrantResourceEntrypoint, Permission: accesscontrol.GrantPermissionDiscover},
			{ResourceType: accesscontrol.GrantResourceModel, Permission: accesscontrol.GrantPermissionDiscover},
		},
	}
}

func assertPublisherDelegationDiscovery(
	t *testing.T,
	fixture publisherDelegationFixture,
	discoveryRequest CatalogDiscoveryRequest,
) {
	t.Helper()
	discovery, err := fixture.runtime.DiscoverCatalog(fixture.ctx, discoveryRequest)
	if err != nil || !discovery.Result.Allowed() {
		t.Fatalf("DiscoverCatalog() = %+v, %v", discovery, err)
	}
	entrypoints := discovery.Resources[accesscontrol.GrantResourceEntrypoint]
	models := discovery.Resources[accesscontrol.GrantResourceModel]
	if len(entrypoints) != 1 || entrypoints[0] != publisherDelegationEntrypointID ||
		len(models) != 1 || models[0] != publisherDelegationAllowedModelID {
		t.Fatalf("discovered resources = %+v", discovery.Resources)
	}
}

func assertPublisherDelegationRevocationGates(
	t *testing.T,
	fixture publisherDelegationFixture,
	discoveryRequest CatalogDiscoveryRequest,
) {
	t.Helper()
	projectionKeys, err := quotaruntime.NewAccessProjectionKeyspaceWithPrefix(
		fixture.prefix,
		publisherDelegationPartition,
	)
	if err != nil {
		t.Fatalf("create access projection keyspace: %v", err)
	}
	modelDenyKey := projectionKeys.Deny(string(accesscontrol.GrantResourceModel), publisherDelegationAllowedModelID)
	if err := fixture.client.SAdd(
		fixture.ctx,
		modelDenyKey,
		"model-discovery-regression",
	).Err(); err != nil {
		t.Fatalf("install model deny gate: %v", err)
	}
	blocked, err := fixture.runtime.DiscoverCatalog(fixture.ctx, discoveryRequest)
	if err != nil || blocked.Result.Disposition != quotaruntime.AdmissionForbidden || blocked.Result.Reason != "resource_denied" {
		t.Fatalf("DiscoverCatalog() behind model deny gate = %+v, %v", blocked, err)
	}

	if err := fixture.client.SRem(fixture.ctx, modelDenyKey, "model-discovery-regression").Err(); err != nil {
		t.Fatalf("clear model deny gate: %v", err)
	}
	if err := fixture.barriers.InstallDeny(
		fixture.ctx, managementauth.BarrierManagementSession, publisherDelegationManagementSession,
	); err != nil {
		t.Fatalf("install management session deny: %v", err)
	}
	revoked, err := fixture.runtime.DiscoverCatalog(fixture.ctx, discoveryRequest)
	if err != nil || revoked.Result.Disposition != quotaruntime.AdmissionForbidden || revoked.Result.Reason != "management_session_denied" {
		t.Fatalf("DiscoverCatalog() behind management session deny = %+v, %v", revoked, err)
	}
}

type publisherDelegationDesiredReader struct {
	state accesspublisher.DesiredState
}

func (reader publisherDelegationDesiredReader) LoadDesiredState(
	_ context.Context,
	namespaceID string,
	revision uint64,
) (accesspublisher.DesiredState, error) {
	if namespaceID != string(reader.state.Namespace.ID) || revision != reader.state.Revision {
		return accesspublisher.DesiredState{}, fmt.Errorf("unexpected desired state %s at revision %d", namespaceID, revision)
	}
	return reader.state, nil
}

type publisherDelegationOutbox struct {
	batch    accesspublisher.OutboxBatch
	staged   accesspublisher.Publication
	applied  accesspublisher.AppliedState
	claimed  bool
	fenced   bool
	released bool
	failed   bool
}

func (outbox *publisherDelegationOutbox) ClaimLatest(
	_ context.Context,
	workerID string,
	_ time.Duration,
) (accesspublisher.OutboxBatch, error) {
	if outbox.claimed {
		return accesspublisher.OutboxBatch{}, accesspublisher.ErrNoWork
	}
	outbox.claimed = true
	outbox.batch.WorkerID = workerID
	outbox.batch.ClaimedAt = time.Now().UTC()
	return outbox.batch, nil
}

func (outbox *publisherDelegationOutbox) RecordStaged(
	_ context.Context,
	_ accesspublisher.OutboxBatch,
	publication accesspublisher.Publication,
) error {
	outbox.staged = publication
	return nil
}

func (outbox *publisherDelegationOutbox) Release(
	_ context.Context,
	_ accesspublisher.OutboxBatch,
	_ error,
	_ time.Duration,
) error {
	outbox.released = true
	return nil
}

func (outbox *publisherDelegationOutbox) Fail(
	_ context.Context,
	_ accesspublisher.OutboxBatch,
	_ error,
) error {
	outbox.failed = true
	return nil
}

func (outbox *publisherDelegationOutbox) WithRevisionFence(
	ctx context.Context,
	batch accesspublisher.OutboxBatch,
	activate func(context.Context) error,
) error {
	if batch.NamespaceID != outbox.batch.NamespaceID || batch.DesiredRevision != outbox.batch.DesiredRevision {
		return accesspublisher.ErrSuperseded
	}
	if err := activate(ctx); err != nil {
		return err
	}
	outbox.fenced = true
	outbox.applied = accesspublisher.AppliedState{
		NamespaceID:     batch.NamespaceID,
		QuotaPartition:  batch.QuotaPartition,
		RuntimeEpoch:    batch.RuntimeEpoch,
		DesiredRevision: batch.DesiredRevision,
		PublicationID:   outbox.staged.ID,
		AccessDigest:    outbox.staged.Digest,
		RoutingDigest:   outbox.staged.Routing.Digest,
	}
	return nil
}

func (outbox *publisherDelegationOutbox) Applied(
	_ context.Context,
	namespaceID string,
) (accesspublisher.AppliedState, error) {
	if namespaceID != outbox.applied.NamespaceID {
		return accesspublisher.AppliedState{}, accesspublisher.ErrNoWork
	}
	return outbox.applied, nil
}

type publisherDelegationBarrierLoader struct{}

func (publisherDelegationBarrierLoader) LoadRevocationBarriers(
	context.Context,
) ([]managementauth.RevocationBarrier, error) {
	return nil, nil
}

func publisherDelegationDesiredState(
	issued accesscredential.Issued,
	now time.Time,
) accesspublisher.DesiredState {
	const revision = 1
	expiresAt := now.Add(time.Hour)
	namespace := accesscontrol.Namespace{
		ID: publisherDelegationNamespaceID, Name: "Delegated runtime", QuotaPartitionID: publisherDelegationPartition,
		BillingCurrency: "USD", Status: accesscontrol.NamespaceStatusActive, Revision: revision, RuntimeEpoch: 1,
		CreatedAt: now, UpdatedAt: now,
	}
	user := accesscontrol.User{
		NamespaceID: namespace.ID, ID: publisherDelegationUserID, Email: "delegated-runtime@example.com",
		DisplayName: "Delegated runtime user", Status: accesscontrol.UserStatusActive,
		CreatedAt: now, UpdatedAt: now,
	}
	key := accesscontrol.APIKey{
		NamespaceID: namespace.ID, ID: publisherDelegationKeyID, Name: "Delegated runtime key",
		Owner: user.SubjectRef(), Status: accesscontrol.APIKeyStatusActive,
		PolicyEpoch: 1, DelegationEpoch: 1, Revision: revision, CreatedAt: now, UpdatedAt: now,
	}
	policy := accesscontrol.AccessPolicy{
		NamespaceID: namespace.ID, ID: "policy-delegated-runtime", DisplayName: "Delegated runtime access",
		Status: accesscontrol.PolicyStatusActive, Revision: revision, CreatedAt: now, UpdatedAt: now,
		Grants: []accesscontrol.AccessPolicyGrant{
			{
				PolicyID:   "policy-delegated-runtime",
				Resource:   accesscontrol.GrantResource{Type: accesscontrol.GrantResourceEntrypoint, ID: publisherDelegationEntrypointID},
				Permission: accesscontrol.GrantPermissionDiscover, Effect: accesscontrol.GrantEffectAllow,
			},
			{
				PolicyID:   "policy-delegated-runtime",
				Resource:   accesscontrol.GrantResource{Type: accesscontrol.GrantResourceEntrypoint, ID: publisherDelegationEntrypointID},
				Permission: accesscontrol.GrantPermissionInvoke, Effect: accesscontrol.GrantEffectAllow,
			},
			{
				PolicyID:   "policy-delegated-runtime",
				Resource:   accesscontrol.GrantResource{Type: accesscontrol.GrantResourceModel, ID: publisherDelegationAllowedModelID},
				Permission: accesscontrol.GrantPermissionDiscover, Effect: accesscontrol.GrantEffectAllow,
			},
		},
	}
	candidate := accessprojection.Candidate{
		Revision: revision, Namespace: namespace, Key: key,
		Relationships: accesscontrol.APIKeyRelationships{OwnerUser: &user},
		UserAccessBindings: []accesscontrol.AccessPolicyBinding{{
			ID: "binding-delegated-runtime", NamespaceID: namespace.ID, Subject: user.SubjectRef(),
			PolicyID: policy.ID, Status: accesscontrol.BindingStatusActive, Revision: revision,
		}},
		AccessPolicies: map[accesscontrol.AccessPolicyID]accesscontrol.AccessPolicy{policy.ID: policy},
	}
	credential := accesscontrol.CredentialVersion{
		ID: "credential-delegated-runtime", APIKeyID: key.ID, KID: issued.Digest.PublicID,
		SecretHMAC: issued.Digest.HMAC, PepperVersion: issued.Digest.PepperVersion,
		Status: accesscontrol.CredentialStatusActive, NotBefore: now, ExpiresAt: &expiresAt, CreatedAt: now,
	}
	routing := publisherDelegationRouting(namespace, revision)
	return accesspublisher.DesiredState{
		Namespace: namespace, Revision: revision, RevisionTime: now,
		Keys: []accessprojection.Candidate{candidate},
		Credentials: []accesspublisher.CredentialCandidate{{
			Kind: accesspublisher.CredentialKindDelegation, Credential: credential,
			Delegation: &accessprojection.DelegationContext{
				ManagementSessionID: publisherDelegationManagementSession,
				PrincipalID:         publisherDelegationPrincipalID,
				DelegationEpoch:     1, UserID: publisherDelegationUserID, TeamID: "",
				Audience: publisherDelegationAudience,
			},
		}},
		Routing: routing,
	}
}

func publisherDelegationRouting(
	namespace accesscontrol.Namespace,
	revision uint64,
) routingsnapshot.Bundle {
	return routingsnapshot.Bundle{
		NamespaceID: string(namespace.ID), Revision: int64(revision), Currency: "USD",
		Models: []routingsnapshot.Model{
			publisherDelegationModel(publisherDelegationAllowedModelID, "a"),
			publisherDelegationModel(publisherDelegationHiddenModelID, "b"),
		},
		Recipes: []routingsnapshot.Recipe{{
			ID: "recipe-delegated-runtime", Revision: 1, Name: "Delegated runtime recipe",
			Decisions: []routingsnapshot.Decision{{
				ID: "decision-delegated-runtime", Name: "Delegated runtime decision",
				DispatchCardinality: routingsnapshot.DispatchCardinalitySingle,
			}},
			Document: json.RawMessage(`{"signals":[],"decisions":[]}`),
		}},
		Entrypoints: []routingsnapshot.Entrypoint{{
			ID: publisherDelegationEntrypointID, Revision: 1, Name: "Delegated runtime chat",
			Aliases: []string{"vllm-sr/delegated-runtime"},
			Rules: []routingsnapshot.EntrypointRule{{
				ID: "rule-delegated-runtime", Name: "Delegated runtime rule",
				RecipeID: "recipe-delegated-runtime", RecipeRevision: 1,
				Assignments: map[string]routingsnapshot.AssignmentSet{
					"decision-delegated-runtime": {
						Models: []routingsnapshot.Assignment{{
							ModelID: publisherDelegationAllowedModelID, ModelRevision: 1, Weight: "1",
						}},
					},
				},
			}},
		}},
	}
}

func publisherDelegationModel(id, digestCharacter string) routingsnapshot.Model {
	return routingsnapshot.Model{
		ID: id, Revision: 1, CatalogRevision: "sha256:" + strings.Repeat(digestCharacter, 64),
		Name: "local/" + id, Capabilities: []string{"text"},
		Execution: routingsnapshot.ModelExecution{RequestTimeout: "30s", StreamTimeout: "60s"},
		Backends: []routingsnapshot.Backend{{
			ID: "backend-" + id, ProviderID: "openai-compatible", WireFormat: "openai.chat.v1",
			Origin: "https://models.example/v1", ProviderModelID: id,
			Connection: routingsnapshot.BackendConnection{Path: "/chat/completions"}, Weight: "1",
		}},
	}
}

func deletePublisherDelegationPrefix(ctx context.Context, client *redis.Client, pattern string) {
	var cursor uint64
	for {
		keys, next, err := client.Scan(ctx, cursor, pattern, 100).Result()
		if err != nil {
			return
		}
		if len(keys) > 0 {
			_ = client.Del(ctx, keys...).Err()
		}
		cursor = next
		if cursor == 0 {
			return
		}
	}
}
