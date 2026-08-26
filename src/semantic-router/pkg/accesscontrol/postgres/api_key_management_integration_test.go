package postgres

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"net/netip"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apikeymanagement"
	controlpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

func TestAPIKeyManagementPostgresAtomicPolicyOverridesAndReplicaReplay(t *testing.T) {
	ctx, cancel, db := apiKeyManagementIntegrationDatabase(t)
	defer cancel()

	const (
		namespaceID   = "11111111-1111-4111-8111-111111111111"
		principalID   = "22222222-2222-4222-8222-222222222222"
		ownerUserID   = "33333333-3333-4333-8333-333333333333"
		secondUserID  = "44444444-4444-4444-8444-444444444444"
		existingKeyID = "55555555-5555-4555-8555-555555555555"
		accessPolicy  = "66666666-6666-4666-8666-666666666666"
		ratePolicy    = "77777777-7777-4777-8777-777777777777"
		rateRule      = "88888888-8888-4888-8888-888888888888"
	)
	seedPolicyManagement(t, ctx, db, namespaceID, principalID, ownerUserID, secondUserID, existingKeyID, "test_model")
	seedAPIKeyPolicyOverrides(t, ctx, db, namespaceID, accessPolicy, ratePolicy, rateRule)

	repository := newPostgresAPIKeyManagementRepository(t, db)
	services := []*apikeymanagement.Service{
		newAPIKeyIntegrationService(t, repository),
		newAPIKeyIntegrationService(t, repository),
	}
	actor := apikeymanagement.Actor{
		PrincipalID: principalID, ActorChain: []string{principalID},
		RequestID: "api-key-integration", SourceIP: netip.MustParseAddr("192.0.2.12"),
	}
	request := apikeymanagement.CreateRequest{
		NamespaceID: namespaceID, Name: "Atomic developer key",
		Owner:             apikeymanagement.Owner{Kind: accesscontrol.SubjectKindUser, ID: ownerUserID},
		AccessPolicyIDs:   []string{accessPolicy},
		RateLimitOverride: &apikeymanagement.RateLimitOverrideInput{PolicyID: ratePolicy},
		IdempotencyKey:    "atomic-api-key-create-0001", Actor: actor,
	}

	result := createAPIKeyAcrossReplicas(t, ctx, services, request)
	issued := assertIssuedAPIKeyPolicies(t, result, accessPolicy, ratePolicy)

	keyID := string(result.Key.ID)
	accessBindingID := issued.AccessPolicyBindings[0].BindingID
	rateBindingID := issued.RateLimitOverride.BindingID
	assertCreatedAPIKeyRows(t, ctx, db, namespaceID, keyID, accessBindingID, accessPolicy, rateBindingID, ratePolicy)
	assertSearchedAPIKeyPage(t, ctx, services[0], namespaceID, ownerUserID, result.Key.ID)
	assertPermissionFilteredAPIKeyPage(t, ctx, services[0], namespaceID, ownerUserID, accessPolicy)
}

func TestAPIKeyManagementPostgresListsExcludeDeletedKeysByDefault(t *testing.T) {
	ctx, cancel, db := apiKeyManagementIntegrationDatabase(t)
	defer cancel()

	const (
		namespaceID = "11111111-1111-4111-8111-111111111111"
		principalID = "22222222-2222-4222-8222-222222222222"
		ownerUserID = "33333333-3333-4333-8333-333333333333"
		secondUser  = "44444444-4444-4444-8444-444444444444"
		apiKeyID    = "55555555-5555-4555-8555-555555555555"
	)
	seedPolicyManagement(t, ctx, db, namespaceID, principalID, ownerUserID, secondUser, apiKeyID, "test_model")
	repository := newPostgresAPIKeyManagementRepository(t, db)
	service := newAPIKeyIntegrationService(t, repository)
	actor := apikeymanagement.Actor{
		PrincipalID: principalID, ActorChain: []string{principalID},
		RequestID: "api-key-delete-list", SourceIP: netip.MustParseAddr("192.0.2.12"),
	}
	assertDeletedAPIKey(t, ctx, service, namespaceID, apiKeyID, actor)

	scope := accesscontrol.ResultScope{NamespaceID: namespaceID, All: true}
	assertEmptyAPIKeyPage(t, ctx, service, apikeymanagement.ListKeysRequest{
		NamespaceID: namespaceID, PageSize: 10, IncludeTotal: true, Scope: scope,
	}, "default API-key page after delete")
	assertEmptyAPIKeyPage(t, ctx, service, apikeymanagement.ListKeysRequest{
		NamespaceID: namespaceID, Search: "EXISTING", PageSize: 10, IncludeTotal: true, Scope: scope,
	}, "default searched API-key page after delete")
	assertDeletedAPIKeyPage(t, ctx, service, apiKeyID, apikeymanagement.ListKeysRequest{
		NamespaceID: namespaceID, Status: accesscontrol.APIKeyStatusDeleted,
		PageSize: 10, IncludeTotal: true, Scope: scope,
	})
}

func apiKeyManagementIntegrationDatabase(t *testing.T) (context.Context, context.CancelFunc, *sql.DB) {
	t.Helper()
	dsn := os.Getenv("VLLM_SR_CONTROL_PLANE_TEST_DATABASE_URL")
	if dsn == "" {
		dsn = os.Getenv("VLLM_SR_ACCESS_CONTROL_TEST_DATABASE_URL")
	}
	if dsn == "" {
		t.Skip("PostgreSQL API-key Management test database is not configured")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	db := isolatedPolicyDatabase(t, ctx, dsn)
	if migrationErr := (controlpostgres.Migrator{DB: db}).Apply(ctx); migrationErr != nil {
		cancel()
		t.Fatal(migrationErr)
	}
	return ctx, cancel, db
}

func seedAPIKeyPolicyOverrides(
	t *testing.T,
	ctx context.Context,
	db *sql.DB,
	namespaceID, accessPolicy, ratePolicy, rateRule string,
) {
	t.Helper()
	for _, statement := range []struct {
		query string
		args  []any
	}{
		{`INSERT INTO access_policies
  (id,namespace_id,name,status) VALUES ($1,$2,'API key access','active')`, []any{accessPolicy, namespaceID}},
		{`INSERT INTO rate_limit_policies
  (id,namespace_id,name,status) VALUES ($1,$2,'API key budget','active')`, []any{ratePolicy, namespaceID}},
		{`INSERT INTO rate_limit_rules
  (id,policy_id,metric,algorithm,limit_value,window_seconds,accounting,enforcement,ordinal)
VALUES ($1,$2,'requests','sliding_log',12,60,'request','enforce',0)`, []any{rateRule, ratePolicy}},
	} {
		if _, execErr := db.ExecContext(ctx, statement.query, statement.args...); execErr != nil {
			t.Fatal(execErr)
		}
	}
}

func newPostgresAPIKeyManagementRepository(t *testing.T, db *sql.DB) apikeymanagement.Repository {
	t.Helper()
	store, storeErr := New(db)
	if storeErr != nil {
		t.Fatal(storeErr)
	}
	repository, repositoryErr := NewAPIKeyManagementRepository(store)
	if repositoryErr != nil {
		t.Fatal(repositoryErr)
	}
	return repository
}

func assertIssuedAPIKeyPolicies(
	t *testing.T,
	result apikeymanagement.SecretMutationResult,
	accessPolicy, ratePolicy string,
) apikeymanagement.IssuedSecret {
	t.Helper()
	var issued apikeymanagement.IssuedSecret
	if decodeErr := json.Unmarshal(result.CanonicalJSON, &issued); decodeErr != nil {
		t.Fatal(decodeErr)
	}
	if len(issued.AccessPolicyBindings) != 1 || issued.AccessPolicyBindings[0].PolicyID != accessPolicy ||
		issued.RateLimitOverride == nil || issued.RateLimitOverride.PolicyID != ratePolicy ||
		issued.RateLimitOverride.Created {
		t.Fatalf("one-time policy receipts = %#v / %#v", issued.AccessPolicyBindings, issued.RateLimitOverride)
	}
	return issued
}

func assertSearchedAPIKeyPage(
	t *testing.T,
	ctx context.Context,
	service *apikeymanagement.Service,
	namespaceID, ownerUserID string,
	keyID accesscontrol.APIKeyID,
) {
	t.Helper()
	searched, listErr := service.List(ctx, apikeymanagement.ListKeysRequest{
		NamespaceID: namespaceID, Search: "ATOMIC DEV", PageSize: 1,
		OwnerKind: accesscontrol.SubjectKindUser, OwnerID: ownerUserID, IncludeTotal: true,
		Scope: accesscontrol.ResultScope{NamespaceID: accesscontrol.NamespaceID(namespaceID), All: true},
	})
	if listErr != nil || len(searched.Items) != 1 || searched.Items[0].ID != keyID || searched.HasMore ||
		searched.TotalCount == nil || *searched.TotalCount != 1 {
		t.Fatalf("searched API-key page = %#v, %v", searched, listErr)
	}
}

func assertPermissionFilteredAPIKeyPage(
	t *testing.T,
	ctx context.Context,
	service *apikeymanagement.Service,
	namespaceID, ownerUserID, accessPolicy string,
) {
	t.Helper()
	hidden, listErr := service.List(ctx, apikeymanagement.ListKeysRequest{
		NamespaceID: namespaceID, OwnerKind: accesscontrol.SubjectKindUser, OwnerID: ownerUserID,
		PageSize: 1, IncludeTotal: true,
		Scope: accesscontrol.ResultScope{
			NamespaceID: accesscontrol.NamespaceID(namespaceID),
			APIKeyIDs:   []accesscontrol.APIKeyID{accesscontrol.APIKeyID(accessPolicy)},
		},
	})
	if listErr != nil || len(hidden.Items) != 0 || hidden.TotalCount == nil || *hidden.TotalCount != 0 {
		t.Fatalf("permission-filtered API-key page = %#v, %v", hidden, listErr)
	}
}

func assertDeletedAPIKey(
	t *testing.T,
	ctx context.Context,
	service *apikeymanagement.Service,
	namespaceID, apiKeyID string,
	actor apikeymanagement.Actor,
) {
	t.Helper()
	deleted, deleteErr := service.Delete(ctx, apikeymanagement.LifecycleRequest{
		NamespaceID: namespaceID, KeyID: apiKeyID, ExpectedRevision: 1, Actor: actor,
	})
	if deleteErr != nil || deleted.HTTPStatus != 204 || deleted.Key.Status != accesscontrol.APIKeyStatusDeleted {
		t.Fatalf("delete API key = %#v, %v", deleted, deleteErr)
	}
}

func assertEmptyAPIKeyPage(
	t *testing.T,
	ctx context.Context,
	service *apikeymanagement.Service,
	request apikeymanagement.ListKeysRequest,
	label string,
) {
	t.Helper()
	page, listErr := service.List(ctx, request)
	if listErr != nil || len(page.Items) != 0 || page.TotalCount == nil || *page.TotalCount != 0 {
		t.Fatalf("%s = %#v, %v", label, page, listErr)
	}
}

func assertDeletedAPIKeyPage(
	t *testing.T,
	ctx context.Context,
	service *apikeymanagement.Service,
	apiKeyID string,
	request apikeymanagement.ListKeysRequest,
) {
	t.Helper()
	tombstones, listErr := service.List(ctx, request)
	if listErr != nil || len(tombstones.Items) != 1 || tombstones.Items[0].ID != accesscontrol.APIKeyID(apiKeyID) ||
		tombstones.Items[0].Status != accesscontrol.APIKeyStatusDeleted || tombstones.TotalCount == nil ||
		*tombstones.TotalCount != 1 {
		t.Fatalf("deleted API-key page = %#v, %v", tombstones, listErr)
	}
}

func createAPIKeyAcrossReplicas(
	t *testing.T,
	ctx context.Context,
	services []*apikeymanagement.Service,
	request apikeymanagement.CreateRequest,
) apikeymanagement.SecretMutationResult {
	t.Helper()
	const replicas = 8
	results := make([]apikeymanagement.SecretMutationResult, replicas)
	errors := make([]error, replicas)
	start := make(chan struct{})
	var wait sync.WaitGroup
	for index := 0; index < replicas; index++ {
		wait.Add(1)
		go func(index int) {
			defer wait.Done()
			<-start
			results[index], errors[index] = services[index%len(services)].Create(ctx, request)
		}(index)
	}
	close(start)
	wait.Wait()
	for index, err := range errors {
		if err != nil {
			t.Fatalf("replica %d create: %v", index, err)
		}
		if results[index].Secret != results[0].Secret ||
			!bytes.Equal(results[index].CanonicalJSON, results[0].CanonicalJSON) ||
			results[index].Key.ID != results[0].Key.ID {
			t.Fatalf("replica %d did not receive exact replay", index)
		}
	}
	return results[0]
}

func assertCreatedAPIKeyRows(
	t *testing.T,
	ctx context.Context,
	db interface {
		QueryRowContext(context.Context, string, ...any) *sql.Row
	},
	namespaceID, keyID, accessBindingID, accessPolicy, rateBindingID, ratePolicy string,
) {
	t.Helper()
	var keys, credentials, accessBindings, rateBindings, desiredRevisions, outboxRows int
	err := db.QueryRowContext(ctx, `SELECT
  (SELECT count(*) FROM access_api_keys WHERE namespace_id=$1 AND name='Atomic developer key'),
  (SELECT count(*) FROM access_api_key_credentials WHERE namespace_id=$1 AND api_key_id=$2 AND status='active'),
  (SELECT count(*) FROM access_policy_bindings WHERE namespace_id=$1 AND id=$3 AND subject_id=$2 AND policy_id=$4),
  (SELECT count(*) FROM rate_limit_bindings WHERE namespace_id=$1 AND id=$5 AND subject_id=$2
     AND policy_id=$6 AND binding_mode='allocation' AND quota_partition_id='policy-test-partition'),
	  (SELECT count(DISTINCT desired_revision) FROM policy_outbox
	     WHERE namespace_id=$1 AND aggregate_id IN ($2::text,$3::text,$5::text)),
	  (SELECT count(*) FROM policy_outbox WHERE namespace_id=$1 AND aggregate_id IN ($2::text,$3::text,$5::text))`,
		namespaceID, keyID, accessBindingID, accessPolicy, rateBindingID, ratePolicy,
	).Scan(&keys, &credentials, &accessBindings, &rateBindings, &desiredRevisions, &outboxRows)
	if err != nil {
		t.Fatal(err)
	}
	if keys != 1 || credentials != 1 || accessBindings != 1 || rateBindings != 1 ||
		desiredRevisions != 1 || outboxRows != 3 {
		t.Fatalf("atomic rows key/credential/access/rate/revision/outbox = %d/%d/%d/%d/%d/%d",
			keys, credentials, accessBindings, rateBindings, desiredRevisions, outboxRows)
	}
}

func newAPIKeyIntegrationService(t *testing.T, repository apikeymanagement.Repository) *apikeymanagement.Service {
	t.Helper()
	commands, err := managementcommand.NewCodec(securitykeyring.Symmetric{
		ActiveVersion: "command-v1", Keys: map[string][]byte{"command-v1": []byte(strings.Repeat("c", 32))},
	})
	if err != nil {
		t.Fatal(err)
	}
	service, err := apikeymanagement.NewService(apikeymanagement.Options{
		Repository: repository, Waiter: immediateAPIKeyPublicationWaiter{}, Commands: commands,
		CursorKeyring: securitykeyring.Symmetric{ActiveVersion: "cursor-v1", Keys: map[string][]byte{
			"cursor-v1": []byte(strings.Repeat("u", 32)),
		}},
		APIKeyPeppers: accesscredential.PepperKeyring{ActiveVersion: "pepper-v1", Keys: map[string][]byte{
			"pepper-v1": []byte(strings.Repeat("p", 32)),
		}},
		ResponseKEK: accesscredential.KEKKeyring{ActiveVersion: "response-v1", Keys: map[string][]byte{
			"response-v1": []byte(strings.Repeat("r", 32)),
		}},
		IdempotencyTTL: time.Hour, SecretDeliveryTTL: 10 * time.Minute,
		PublicationTimeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(service.Close)
	return service
}

type immediateAPIKeyPublicationWaiter struct{}

func (immediateAPIKeyPublicationWaiter) WaitAPIKeyActive(context.Context, string, string, string) error {
	return nil
}
