package accessruntime

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"os"
	"strconv"
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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
)

func TestRedisReaderClassifiesStagedCredentialAsPending(t *testing.T) {
	address := os.Getenv("ACCESSRUNTIME_TEST_REDIS_ADDR")
	if address == "" {
		t.Skip("ACCESSRUNTIME_TEST_REDIS_ADDR is not configured")
	}
	client := redis.NewClient(&redis.Options{Addr: address})
	t.Cleanup(func() { _ = client.Close() })
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	if err := client.Ping(ctx).Err(); err != nil {
		t.Fatalf("ping Redis: %v", err)
	}
	prefix := "access-pending-it:" + uuid.NewString()
	t.Cleanup(func() { deletePrefix(context.Background(), client, prefix+":*") })
	const (
		namespaceID = "11111111-1111-4111-8111-111111111111"
		partition   = "tenant-pending"
		publicID    = "pendingcredential0001"
		pendingID   = "publication-pending"
		activeID    = "publication-active"
	)
	directory, err := quotaruntime.CredentialDirectoryKeyWithPrefix(
		prefix, string(accesscredential.KindAPIKey), publicID,
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := client.HSet(ctx, directory, map[string]any{
		"pending_publication_id": pendingID,
		"pending_state":          "active",
		"pending_partition":      partition,
		"pending_namespace_id":   namespaceID,
		"pending_kind":           string(accesscredential.KindAPIKey),
		"pending_public_id":      publicID,
	}).Err(); err != nil {
		t.Fatal(err)
	}
	reader, err := NewRedisProjectionReader(RedisProjectionReaderOptions{
		Client: client, KeyPrefix: prefix,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := reader.LocateCredential(ctx, accesscredential.KindAPIKey, publicID); !errors.Is(err, ErrPublicationPending) {
		t.Fatalf("first staged publication = %v, want pending", err)
	}
	keys, err := accesspublisher.NewKeyspace(prefix, namespaceID, partition)
	if err != nil {
		t.Fatal(err)
	}
	if err := client.HSet(ctx, keys.AccessGate(), map[string]any{
		"publication_id": activeID, "revision": 1, "runtime_epoch": 1,
		"publication_digest": strings.Repeat("a", 64),
		"manifest_digest":    strings.Repeat("b", 64),
	}).Err(); err != nil {
		t.Fatal(err)
	}
	if err := client.HSet(ctx, keys.RoutingGate(), map[string]any{
		"publication_id": activeID, "revision": 1, "runtime_epoch": 1,
		"publication_digest": strings.Repeat("a", 64),
		"snapshot_digest":    strings.Repeat("c", 64),
		"snapshot_key":       keys.RoutingSnapshot(1),
	}).Err(); err != nil {
		t.Fatal(err)
	}
	if _, err := reader.LocateCredential(ctx, accesscredential.KindAPIKey, publicID); !errors.Is(err, ErrPublicationPending) {
		t.Fatalf("next staged publication = %v, want pending", err)
	}
	if err := client.HSet(ctx, directory,
		"publication_id", "unexpected-active", "state", "active",
	).Err(); err != nil {
		t.Fatal(err)
	}
	if _, err := reader.LocateCredential(ctx, accesscredential.KindAPIKey, publicID); !errors.Is(err, ErrRuntimeCorrupt) {
		t.Fatalf("mismatched active pointer = %v, want corrupt", err)
	}
}

func TestRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPath(t *testing.T) {
	address := os.Getenv("ACCESSRUNTIME_TEST_REDIS_ADDR")
	if address == "" {
		t.Skip("ACCESSRUNTIME_TEST_REDIS_ADDR is not configured")
	}
	client := redis.NewClient(&redis.Options{Addr: address})
	t.Cleanup(func() { _ = client.Close() })
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	if err := client.Ping(ctx).Err(); err != nil {
		t.Fatalf("ping Redis: %v", err)
	}
	prefix := "access-it:" + uuid.NewString()
	t.Cleanup(func() { deletePrefix(context.Background(), client, prefix+":*") })

	fixture := publishRuntimeProjection(t, ctx, client, prefix)
	keyring := fixture.keyring
	issued := fixture.issued
	projection := fixture.projection
	keyspace := fixture.keyspace
	publicationKeys := fixture.publicationKeys

	reader, testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr := NewRedisProjectionReader(RedisProjectionReaderOptions{Client: client, KeyPrefix: prefix})
	if testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr != nil {
		t.Fatal(testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr)
	}
	engine, testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr := quotaruntime.NewRedisEngine(client, quotaruntime.RedisEngineOptions{KeyPrefix: prefix})
	if testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr != nil {
		t.Fatal(testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr)
	}
	runtime, testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr := New(RuntimeOptions{
		Reader: reader, Engine: engine, APIKeyPeppers: keyring, DelegationPeppers: keyring,
		DelegationAudience: "vllm-sr-inference",
		DelegationBarriers: &fakeDelegationBarriers{state: managementauth.DelegationBarrierState{Ready: true}},
		KeyPrefix:          prefix,
	})
	if testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr != nil {
		t.Fatal(testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr)
	}
	authentication, testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr := runtime.Authenticate(ctx, AuthenticationRequest{Credential: issued.Plaintext})
	if testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr != nil || !authentication.Result.Allowed() || authentication.Tenant.APIKeyID == "" {
		t.Fatalf("Authenticate() = %+v, %v", authentication, testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr)
	}
	discovered, testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr := runtime.Discover(ctx, DiscoveryRequest{
		Session: authentication.Session, ResourceType: accesscontrol.GrantResourceEntrypoint,
		Permission: accesscontrol.GrantPermissionInvoke,
	})
	if testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr != nil || !discovered.Result.Allowed() || len(discovered.ResourceIDs) != 1 {
		t.Fatalf("Discover() = %+v, %v", discovered, testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr)
	}
	catalog, testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr := runtime.DiscoverCatalog(ctx, CatalogDiscoveryRequest{
		Session: authentication.Session,
		Queries: []DiscoveryQuery{
			{ResourceType: accesscontrol.GrantResourceEntrypoint, Permission: accesscontrol.GrantPermissionDiscover},
			{ResourceType: accesscontrol.GrantResourceModel, Permission: accesscontrol.GrantPermissionDiscover},
		},
	})
	if testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr != nil || !catalog.Result.Allowed() || catalog.Tenant.APIKeyID == "" ||
		len(catalog.Resources[accesscontrol.GrantResourceEntrypoint]) != 1 ||
		len(catalog.Resources[accesscontrol.GrantResourceModel]) != 1 {
		t.Fatalf("DiscoverCatalog() = %+v, %v", catalog, testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr)
	}
	if err := client.HSet(ctx, keyspace.LogicalKey(projection.KeyID), "status", "disabled").Err(); err != nil {
		t.Fatal(err)
	}
	checked, testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr := runtime.Authorize(ctx, AuthorizationRequest{
		Session: authentication.Session,
		Target:  Target{ResourceType: accesscontrol.GrantResourceEntrypoint, ResourceID: "entry-chat", Permission: accesscontrol.GrantPermissionInvoke},
	})
	if testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr != nil || checked.Result.Disposition != quotaruntime.AdmissionUnauthenticated {
		t.Fatalf("disabled logical key check = %+v, %v", checked, testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr)
	}
	if err := client.HSet(ctx, keyspace.LogicalKey(projection.KeyID), "status", string(projection.KeyStatus)).Err(); err != nil {
		t.Fatal(err)
	}
	if err := client.HSet(ctx, publicationKeys.RoutingGate(), "publication_id", "publication-integration-2").Err(); err != nil {
		t.Fatal(err)
	}
	checked, testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr = runtime.Authorize(ctx, AuthorizationRequest{
		Session: authentication.Session,
		Target:  Target{ResourceType: accesscontrol.GrantResourceEntrypoint, ResourceID: "entry-chat", Permission: accesscontrol.GrantPermissionInvoke},
	})
	if testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr != nil || checked.Result.Disposition != quotaruntime.AdmissionUnavailable || checked.Result.Reason != "routing_publication_changed" {
		t.Fatalf("switched routing publication check = %+v, %v", checked, testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr)
	}
}

func TestRedisRuntimeAuthenticatesAcrossActivationBeforePointerCompaction(t *testing.T) {
	address := os.Getenv("ACCESSRUNTIME_TEST_REDIS_ADDR")
	if address == "" {
		t.Skip("ACCESSRUNTIME_TEST_REDIS_ADDR is not configured")
	}
	client := redis.NewClient(&redis.Options{Addr: address})
	t.Cleanup(func() { _ = client.Close() })
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	if err := client.Ping(ctx).Err(); err != nil {
		t.Fatalf("ping Redis: %v", err)
	}
	prefix := "access-compaction-it:" + uuid.NewString()
	t.Cleanup(func() { deletePrefix(context.Background(), client, prefix+":*") })

	fixture := publishRuntimeProjection(t, ctx, client, prefix)
	next := testProjectionAtRevision(t, fixture.projection.Revision+1)
	publicationID := "publication-integration-2"
	stageActivatedProjection(t, ctx, client, fixture, next, publicationID)

	reader, err := NewRedisProjectionReader(RedisProjectionReaderOptions{Client: client, KeyPrefix: prefix})
	if err != nil {
		t.Fatal(err)
	}
	engine, err := quotaruntime.NewRedisEngine(client, quotaruntime.RedisEngineOptions{KeyPrefix: prefix})
	if err != nil {
		t.Fatal(err)
	}
	runtime, err := New(RuntimeOptions{
		Reader: reader, Engine: engine,
		APIKeyPeppers: fixture.keyring, DelegationPeppers: fixture.keyring,
		DelegationAudience: "vllm-sr-inference",
		DelegationBarriers: &fakeDelegationBarriers{state: managementauth.DelegationBarrierState{Ready: true}},
		KeyPrefix:          prefix,
	})
	if err != nil {
		t.Fatal(err)
	}

	authentication, err := runtime.Authenticate(ctx, AuthenticationRequest{Credential: fixture.issued.Plaintext})
	if err != nil || !authentication.Result.Allowed() || authentication.Tenant.PublicationID != publicationID ||
		authentication.Tenant.PolicyRevision != next.Revision {
		t.Fatalf("Authenticate() during pointer compaction = %+v, %v", authentication, err)
	}
	assertCompactionAdmission(t, ctx, runtime, authentication, "admission-before-compaction")

	promoteActivatedProjection(t, ctx, client, fixture, publicationID)
	assertCompactionAdmission(t, ctx, runtime, authentication, "admission-after-compaction")
	compacted, err := runtime.Authenticate(ctx, AuthenticationRequest{Credential: fixture.issued.Plaintext})
	if err != nil || !compacted.Result.Allowed() || compacted.Tenant.PublicationID != publicationID {
		t.Fatalf("Authenticate() after pointer compaction = %+v, %v", compacted, err)
	}
}

func assertCompactionAdmission(
	t *testing.T,
	ctx context.Context,
	runtime *Runtime,
	authentication Authentication,
	admissionID string,
) {
	t.Helper()
	target := Target{
		ResourceType: accesscontrol.GrantResourceEntrypoint,
		ResourceID:   "entry-chat", Permission: accesscontrol.GrantPermissionInvoke,
	}
	authorization, err := runtime.Authorize(ctx, AuthorizationRequest{Session: authentication.Session, Target: target})
	if err != nil || !authorization.Result.Allowed() {
		t.Fatalf("Authorize(%s) = %+v, %v", admissionID, authorization, err)
	}
	admission, err := runtime.Admit(ctx, AdmissionRequest{
		Session: authentication.Session, Target: target, AdmissionID: admissionID,
		RequestDigest: strings.Repeat("d", 64), LeaseDuration: time.Minute,
	})
	if err != nil || !admission.Result.Allowed() {
		t.Fatalf("Admit(%s) = %+v, %v", admissionID, admission, err)
	}
}

func stageActivatedProjection(
	t *testing.T,
	ctx context.Context,
	client *redis.Client,
	fixture runtimeProjectionFixture,
	projection accessprojection.Projection,
	publicationID string,
) {
	t.Helper()
	directory, err := fixture.publicationKeys.CredentialDirectory(
		string(accesscredential.KindAPIKey), fixture.issued.Digest.PublicID,
	)
	if err != nil {
		t.Fatal(err)
	}
	document, err := json.Marshal(projection)
	if err != nil {
		t.Fatal(err)
	}
	credential := fixture.keyspace.Credential(string(accesscredential.KindAPIKey), fixture.issued.Digest.PublicID)
	logical := fixture.keyspace.LogicalKey(projection.KeyID)
	active := fixture.keyspace.Active(projection.KeyID)
	now := time.Now().UTC()
	pipeline := client.TxPipeline()
	pipeline.HSet(ctx, directory, map[string]any{
		"pending_publication_id": publicationID, "pending_state": "active",
		"pending_revision": projection.Revision, "pending_partition": projection.QuotaPartition,
		"pending_namespace_id": projection.NamespaceID, "pending_kind": string(accesscredential.KindAPIKey),
		"pending_public_id": fixture.issued.Digest.PublicID,
	})
	pipeline.HSet(ctx, credential, map[string]any{
		"pending_publication_id": publicationID, "pending_state": "active", "pending_revision": projection.Revision,
		"pending_kind": string(accesscredential.KindAPIKey), "pending_kid": fixture.issued.Digest.PublicID,
		"pending_key_id":         projection.KeyID,
		"pending_secret_hmac":    base64.RawURLEncoding.EncodeToString(fixture.issued.Digest.HMAC),
		"pending_pepper_version": fixture.issued.Digest.PepperVersion,
		"pending_status":         string(accesscontrol.CredentialStatusActive),
		"pending_not_before_ms":  now.Add(-time.Minute).UnixMilli(),
	})
	pipeline.HSet(ctx, logical, map[string]any{
		"pending_publication_id": publicationID, "pending_state": "active", "pending_revision": projection.Revision,
		"pending_status": string(projection.KeyStatus), "pending_policy_epoch": projection.PolicyEpoch,
		"pending_delegation_epoch": projection.DelegationEpoch,
		"pending_expires_at_ms":    projection.KeyExpiresAt.UnixMilli(),
	})
	pipeline.HSet(ctx, active, map[string]any{
		"pending_publication_id": publicationID, "pending_state": "active",
		"pending_revision": projection.Revision, "pending_digest": projection.Digest,
	})
	pipeline.HSet(ctx, fixture.keyspace.Policy(projection.KeyID, strconv.FormatUint(projection.Revision, 10)), map[string]any{
		"publication_id": publicationID, "digest": projection.Digest, "document": string(document),
	})
	pipeline.HSet(ctx, fixture.publicationKeys.AccessGate(), map[string]any{
		"publication_id": publicationID, "revision": projection.Revision, "runtime_epoch": 1,
		"publication_digest": strings.Repeat("d", 64), "manifest_digest": strings.Repeat("e", 64),
	})
	pipeline.HSet(ctx, fixture.publicationKeys.RoutingGate(), map[string]any{
		"publication_id": publicationID, "revision": projection.Revision, "runtime_epoch": 1,
		"publication_digest": strings.Repeat("d", 64), "snapshot_digest": strings.Repeat("f", 64),
		"snapshot_key": fixture.publicationKeys.RoutingSnapshot(projection.Revision),
	})
	if _, err := pipeline.Exec(ctx); err != nil {
		t.Fatalf("stage activated projection: %v", err)
	}
}

func promoteActivatedProjection(
	t *testing.T,
	ctx context.Context,
	client *redis.Client,
	fixture runtimeProjectionFixture,
	publicationID string,
) {
	t.Helper()
	directory, err := fixture.publicationKeys.CredentialDirectory(
		string(accesscredential.KindAPIKey), fixture.issued.Digest.PublicID,
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{
		directory,
		fixture.keyspace.Credential(string(accesscredential.KindAPIKey), fixture.issued.Digest.PublicID),
		fixture.keyspace.LogicalKey(fixture.projection.KeyID),
		fixture.keyspace.Active(fixture.projection.KeyID),
	} {
		values, readErr := client.HGetAll(ctx, key).Result()
		if readErr != nil {
			t.Fatal(readErr)
		}
		selected, state, selectErr := accesspublisher.SelectPointer(values, publicationID)
		if selectErr != nil || state != accesspublisher.PointerStateActive {
			t.Fatalf("select pending pointer %s = %s, %v", key, state, selectErr)
		}
		if err := client.Del(ctx, key).Err(); err != nil {
			t.Fatal(err)
		}
		if err := client.HSet(ctx, key, selected).Err(); err != nil {
			t.Fatal(err)
		}
	}
}

type runtimeProjectionFixture struct {
	keyring         accesscredential.PepperKeyring
	issued          accesscredential.Issued
	projection      accessprojection.Projection
	keyspace        quotaruntime.AccessProjectionKeyspace
	publicationKeys accesspublisher.Keyspace
	publicationID   string
}

func publishRuntimeProjection(
	t *testing.T,
	ctx context.Context,
	client *redis.Client,
	prefix string,
) runtimeProjectionFixture {
	t.Helper()
	pepper := []byte("0123456789abcdef0123456789abcdef")
	fixture := runtimeProjectionFixture{
		keyring: accesscredential.PepperKeyring{
			ActiveVersion: "pepper-1", Keys: map[string][]byte{"pepper-1": pepper},
		},
		projection:    testProjection(t),
		publicationID: "publication-integration-1",
	}
	var err error
	fixture.issued, err = fixture.keyring.Issue(accesscredential.KindAPIKey, "publicid0001")
	if err != nil {
		t.Fatal(err)
	}
	fixture.keyspace, err = quotaruntime.NewAccessProjectionKeyspaceWithPrefix(prefix, fixture.projection.QuotaPartition)
	if err != nil {
		t.Fatal(err)
	}
	directory, err := quotaruntime.CredentialDirectoryKeyWithPrefix(prefix, string(accesscredential.KindAPIKey), fixture.issued.Digest.PublicID)
	if err != nil {
		t.Fatal(err)
	}
	document, err := json.Marshal(fixture.projection)
	if err != nil {
		t.Fatal(err)
	}
	fixture.publicationKeys, err = accesspublisher.NewKeyspace(prefix, fixture.projection.NamespaceID, fixture.projection.QuotaPartition)
	if err != nil {
		t.Fatal(err)
	}
	publishRuntimeProjectionHashes(t, ctx, client, directory, document, fixture)
	return fixture
}

func publishRuntimeProjectionHashes(
	t *testing.T,
	ctx context.Context,
	client *redis.Client,
	directory string,
	document []byte,
	fixture runtimeProjectionFixture,
) {
	t.Helper()
	now := time.Now().UTC()
	publicationDigest := strings.Repeat("a", 64)
	manifestDigest := strings.Repeat("b", 64)
	snapshotDigest := strings.Repeat("c", 64)
	projection := fixture.projection
	keyspace := fixture.keyspace
	publicationKeys := fixture.publicationKeys
	issued := fixture.issued
	publicationID := fixture.publicationID
	pipeline := client.TxPipeline()
	pipeline.HSet(ctx, directory, map[string]any{
		"publication_id": publicationID, "state": "active", "revision": projection.Revision,
		"partition": projection.QuotaPartition, "namespace_id": projection.NamespaceID,
		"kind": string(accesscredential.KindAPIKey), "public_id": issued.Digest.PublicID,
	})
	pipeline.HSet(ctx, publicationKeys.AccessGate(), map[string]any{
		"publication_id": publicationID, "revision": projection.Revision, "runtime_epoch": 1,
		"publication_digest": publicationDigest, "manifest_digest": manifestDigest,
	})
	pipeline.HSet(ctx, publicationKeys.RoutingGate(), map[string]any{
		"publication_id": publicationID, "revision": projection.Revision, "runtime_epoch": 1,
		"publication_digest": publicationDigest, "snapshot_digest": snapshotDigest,
		"snapshot_key": publicationKeys.RoutingSnapshot(projection.Revision),
	})
	pipeline.HSet(ctx, keyspace.Credential(string(accesscredential.KindAPIKey), issued.Digest.PublicID), map[string]any{
		"publication_id": publicationID, "state": "active", "revision": projection.Revision,
		"kind": string(accesscredential.KindAPIKey), "kid": issued.Digest.PublicID, "key_id": projection.KeyID,
		"secret_hmac":    base64.RawURLEncoding.EncodeToString(issued.Digest.HMAC),
		"pepper_version": issued.Digest.PepperVersion, "status": "active",
		"not_before_ms": now.Add(-time.Minute).UnixMilli(),
	})
	pipeline.HSet(ctx, keyspace.LogicalKey(projection.KeyID), map[string]any{
		"publication_id": publicationID, "state": "active", "revision": projection.Revision,
		"status": string(projection.KeyStatus), "policy_epoch": projection.PolicyEpoch,
		"delegation_epoch": projection.DelegationEpoch, "expires_at_ms": projection.KeyExpiresAt.UnixMilli(),
	})
	pipeline.HSet(ctx, keyspace.Active(projection.KeyID), map[string]any{
		"publication_id": publicationID, "state": "active",
		"revision": projection.Revision, "digest": projection.Digest,
	})
	pipeline.HSet(ctx, keyspace.Policy(projection.KeyID, "7"), map[string]any{
		"publication_id": publicationID, "digest": projection.Digest, "document": string(document),
	})
	if _, err := pipeline.Exec(ctx); err != nil {
		t.Fatalf("publish test projection: %v", err)
	}
}

func deletePrefix(ctx context.Context, client *redis.Client, pattern string) {
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
