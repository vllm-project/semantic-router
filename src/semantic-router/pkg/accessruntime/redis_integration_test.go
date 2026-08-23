package accessruntime

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
)

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

	pepper := []byte("0123456789abcdef0123456789abcdef")
	keyring := accesscredential.PepperKeyring{
		ActiveVersion: "pepper-1", Keys: map[string][]byte{"pepper-1": pepper},
	}
	issued, testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr := keyring.Issue(accesscredential.KindAPIKey, "publicid0001")
	if testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr != nil {
		t.Fatal(testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr)
	}
	projection := testProjection(t)
	keyspace, testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr := quotaruntime.NewAccessProjectionKeyspaceWithPrefix(prefix, projection.QuotaPartition)
	if testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr != nil {
		t.Fatal(testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr)
	}
	directory, testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr := quotaruntime.CredentialDirectoryKeyWithPrefix(prefix, string(accesscredential.KindAPIKey), issued.Digest.PublicID)
	if testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr != nil {
		t.Fatal(testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr)
	}
	document, testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr := json.Marshal(projection)
	if testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr != nil {
		t.Fatal(testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr)
	}
	now := time.Now().UTC()
	publicationID := "publication-integration-1"
	publicationDigest := strings.Repeat("a", 64)
	manifestDigest := strings.Repeat("b", 64)
	snapshotDigest := strings.Repeat("c", 64)
	publicationKeys, testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr := accesspublisher.NewKeyspace(prefix, projection.NamespaceID, projection.QuotaPartition)
	if testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr != nil {
		t.Fatal(testRedisRuntimeUsesOnePrefixedAtomicAccessAndQuotaPathErr)
	}
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
