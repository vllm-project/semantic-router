//go:build integration

package sessiontools

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func newRedisIntegrationStore(
	t *testing.T,
	maxSessions int,
	maxSessionsByIdentity int,
	ttlSeconds int,
) (*RedisStore, config.ToolSessionStoreConfig) {
	t.Helper()
	if os.Getenv("SKIP_REDIS_TESTS") == "true" {
		t.Skip("Redis integration tests disabled")
	}
	host := os.Getenv("REDIS_HOST")
	if host == "" {
		host = "localhost"
	}
	port := 6379
	if configured := os.Getenv("REDIS_PORT"); configured != "" {
		parsed, err := strconv.Atoi(configured)
		if err != nil {
			t.Fatalf("REDIS_PORT: %v", err)
		}
		port = parsed
	}
	maxStateBytes := 4096
	timeoutMillis := 1000
	prefix := fmt.Sprintf("vsr:test:session-tools:%d:", time.Now().UnixNano())
	cfg := config.ToolSessionStoreConfig{
		Backend:               config.ToolSessionStoreBackendRedis,
		TTLSeconds:            &ttlSeconds,
		MaxSessions:           &maxSessions,
		MaxSessionsByIdentity: &maxSessionsByIdentity,
		MaxStateBytes:         &maxStateBytes,
		TimeoutMs:             &timeoutMillis,
		Redis: &config.ToolSessionRedisConfig{
			Address:   fmt.Sprintf("%s:%d", host, port),
			Database:  15,
			KeyPrefix: prefix,
		},
	}
	store, err := NewRedisStore(cfg)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := store.client.Ping(ctx).Err(); err != nil {
		_ = store.Close()
		t.Skipf("Redis unavailable: %v", err)
	}

	cleanupClient := redis.NewClient(&redis.Options{
		Addr:     cfg.Redis.Address,
		DB:       cfg.Redis.Database,
		Password: cfg.Redis.Password,
	})
	t.Cleanup(func() {
		cleanupRedisIntegrationPrefix(cleanupClient, prefix)
		_ = cleanupClient.Close()
		_ = store.Close()
	})
	return store, cfg
}

func cleanupRedisIntegrationPrefix(client *redis.Client, prefix string) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	iterator := client.Scan(ctx, 0, prefix+"*", 100).Iterator()
	for iterator.Next(ctx) {
		_ = client.Del(ctx, iterator.Val()).Err()
	}
}

func redisIntegrationState(policy string) State {
	now := time.Now().UTC()
	return State{
		SchemaVersion:         SchemaVersion,
		PolicyFingerprint:     policy,
		CatalogFingerprint:    "catalog",
		CapabilityFingerprint: "capability",
		Tools: []ToolState{
			{Name: "search", DefinitionFingerprint: "definition", FirstSeenTurn: 0},
		},
		CreatedAt:  now,
		LastSeenAt: now,
		ExpiresAt:  now.Add(time.Minute),
	}
}

func TestRedisStoreIntegrationSharedReplicaAndRestart(t *testing.T) {
	ctx := context.Background()
	storeA, cfg := newRedisIntegrationStore(t, 10, 5, 60)
	storeB, err := NewRedisStore(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = storeB.Close() })
	quota := QuotaKey{Principal: "principal", Namespace: "recipe"}

	applied, err := storeA.CompareAndSwap(ctx, "session", 0, redisIntegrationState("policy-1"), time.Minute, quota)
	if err != nil || !applied {
		t.Fatalf("create: applied=%v err=%v", applied, err)
	}
	loaded, metadata, err := storeB.LoadWithMetadata(ctx, "session")
	if err != nil || !loaded.Found {
		t.Fatalf("cross-replica load: found=%v err=%v", loaded.Found, err)
	}
	if loaded.State.Revision != 1 || metadata.ObservedGeneration == 0 {
		t.Fatalf("loaded state=%+v metadata=%+v", loaded.State, metadata)
	}

	next := loaded.State.Clone()
	next.PolicyFingerprint = "policy-2"
	applied, err = storeB.CompareAndSwap(ctx, "session", loaded.State.Revision, next, time.Minute, quota)
	if err != nil || !applied {
		t.Fatalf("cross-replica update: applied=%v err=%v", applied, err)
	}
	updated, err := storeA.Load(ctx, "session")
	if err != nil || !updated.Found || updated.State.Revision != 2 || updated.State.PolicyFingerprint != "policy-2" {
		t.Fatalf("updated state = %+v, err=%v", updated, err)
	}

	if err := storeA.Close(); err != nil {
		t.Fatal(err)
	}
	restarted, err := NewRedisStore(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = restarted.Close() })
	afterRestart, err := restarted.Load(ctx, "session")
	if err != nil || !afterRestart.Found || afterRestart.State.Revision != 2 {
		t.Fatalf("load after store restart = %+v, err=%v", afterRestart, err)
	}
}

func TestRedisStoreIntegrationCASAndGenerationAwareDelete(t *testing.T) {
	ctx := context.Background()
	store, _ := newRedisIntegrationStore(t, 100, 100, 60)
	quota := QuotaKey{Principal: "principal", Namespace: "recipe"}

	const workers = 20
	var wait sync.WaitGroup
	winners := make(chan struct{}, workers)
	errs := make(chan error, workers)
	for range workers {
		wait.Add(1)
		go func() {
			defer wait.Done()
			applied, err := store.CompareAndSwap(ctx, "contended", 0, redisIntegrationState("policy"), time.Minute, quota)
			if err != nil && !errors.Is(err, ErrRevisionMismatch) {
				errs <- err
				return
			}
			if applied {
				winners <- struct{}{}
			}
		}()
	}
	wait.Wait()
	close(winners)
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	if got := len(winners); got != 1 {
		t.Fatalf("concurrent create winners = %d, want 1", got)
	}

	_, oldToken, err := store.LoadWithMetadata(ctx, "contended")
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Delete(ctx, "contended"); err != nil {
		t.Fatal(err)
	}
	applied, err := store.CompareAndSwap(ctx, "contended", 0, redisIntegrationState("recreated"), time.Minute, quota)
	if err != nil || !applied {
		t.Fatalf("recreate: applied=%v err=%v", applied, err)
	}
	deleted, err := store.DeleteIfToken(ctx, "contended", StateToken{
		Revision:   oldToken.ObservedRevision,
		Generation: oldToken.ObservedGeneration,
	})
	if err != nil {
		t.Fatal(err)
	}
	if deleted {
		t.Fatal("a token from before delete-and-recreate must not delete the new state")
	}
	current, currentToken, err := store.LoadWithMetadata(ctx, "contended")
	if err != nil || !current.Found {
		t.Fatalf("recreated load: found=%v err=%v", current.Found, err)
	}
	if currentToken.ObservedGeneration == oldToken.ObservedGeneration {
		t.Fatal("a recreated key must receive a new generation")
	}
}

func TestRedisStoreIntegrationQuotasTTLAndCorruption(t *testing.T) {
	ctx := context.Background()
	store, _ := newRedisIntegrationStore(t, 3, 2, 1)
	quota := QuotaKey{Principal: "principal-a", Namespace: "recipe"}
	otherQuota := QuotaKey{Principal: "principal-b", Namespace: "recipe"}

	create := func(key string, identity QuotaKey, ttl time.Duration) {
		t.Helper()
		applied, err := store.CompareAndSwap(ctx, key, 0, redisIntegrationState(key), ttl, identity)
		if err != nil || !applied {
			t.Fatalf("create %s: applied=%v err=%v", key, applied, err)
		}
		time.Sleep(2 * time.Millisecond)
	}
	create("identity-old", quota, time.Second)
	create("identity-new", quota, time.Second)
	create("identity-admitted", quota, time.Second)
	if exists := store.client.Exists(ctx, store.stateKey("identity-old")).Val(); exists != 0 {
		t.Fatal("per-identity admission must evict that identity's oldest session")
	}
	create("global-other", otherQuota, time.Second)
	create("global-admitted", otherQuota, time.Second)
	if exists := store.client.Exists(ctx, store.stateKey("identity-new")).Val(); exists != 0 {
		t.Fatal("global admission must evict the globally oldest remaining session")
	}
	if count := store.client.ZCard(ctx, store.globalLRUKey()).Val(); count != 3 {
		t.Fatalf("global session index size = %d, want 3", count)
	}

	create("expires", otherQuota, 50*time.Millisecond)
	time.Sleep(80 * time.Millisecond)
	expired, err := store.Load(ctx, "expires")
	if err != nil || expired.Found {
		t.Fatalf("expired load: found=%v err=%v", expired.Found, err)
	}

	create("corrupt", otherQuota, time.Second)
	if err := store.client.HSet(ctx, store.stateKey("corrupt"), redisStatePayloadField, `{`).Err(); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Load(ctx, "corrupt"); !errors.Is(err, ErrStateCorrupted) {
		t.Fatalf("corrupt load: err=%v, want ErrStateCorrupted", err)
	}
	if exists := store.client.Exists(ctx, store.stateKey("corrupt")).Val(); exists != 0 {
		t.Fatal("a corrupt value must be removed conditionally")
	}

	if err := store.client.Set(ctx, store.stateKey("wrong-type"), "invalid", time.Second).Err(); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Load(ctx, "wrong-type"); !errors.Is(err, ErrStateCorrupted) {
		t.Fatalf("wrong-type load: err=%v, want ErrStateCorrupted", err)
	}
	if exists := store.client.Exists(ctx, store.stateKey("wrong-type")).Val(); exists != 0 {
		t.Fatal("a wrong-type value must be removed atomically")
	}
}

func TestRedisStoreIntegrationWrongTypeIndexesRecover(t *testing.T) {
	ctx := context.Background()
	store, _ := newRedisIntegrationStore(t, 10, 10, 60)
	quota := QuotaKey{Principal: "wrong-type-principal", Namespace: "recipe"}
	const session = "wrong-type-index"
	applied, err := store.CompareAndSwap(ctx, session, 0, redisIntegrationState("policy"), time.Minute, quota)
	if err != nil || !applied {
		t.Fatalf("create: applied=%v err=%v", applied, err)
	}

	quotaLRU, quotaExpiry := store.quotaIndexKeys(quota)
	indexKeys := []string{
		store.globalLRUKey(),
		store.globalExpiryKey(),
		quotaLRU,
		quotaExpiry,
	}
	for _, indexKey := range indexKeys {
		if err := store.client.Set(ctx, indexKey, "wrong-type", time.Minute).Err(); err != nil {
			t.Fatalf("corrupt %s: %v", indexKey, err)
		}
		loaded, loadErr := store.Load(ctx, session)
		if loadErr != nil || !loaded.Found {
			t.Fatalf("load after corrupting %s: found=%v err=%v", indexKey, loaded.Found, loadErr)
		}
	}

	if err := store.client.Set(ctx, store.globalLRUKey(), "wrong-type", time.Minute).Err(); err != nil {
		t.Fatal(err)
	}
	if err := store.client.Set(ctx, store.globalExpiryKey(), "wrong-type", time.Minute).Err(); err != nil {
		t.Fatal(err)
	}
	if err := store.client.Set(ctx, quotaLRU, "wrong-type", time.Minute).Err(); err != nil {
		t.Fatal(err)
	}
	if err := store.client.Set(ctx, quotaExpiry, "wrong-type", time.Minute).Err(); err != nil {
		t.Fatal(err)
	}
	if err := store.Delete(ctx, session); err != nil {
		t.Fatalf("delete with wrong-type indexes: %v", err)
	}
	if exists := store.client.Exists(ctx, store.stateKey(session)).Val(); exists != 0 {
		t.Fatal("delete must remove the state despite wrong-type indexes")
	}
}

func TestRedisStoreIntegrationStaleIndexMemberIsReclaimed(t *testing.T) {
	ctx := context.Background()
	store, _ := newRedisIntegrationStore(t, 2, 1, 60)
	quota := QuotaKey{Principal: "stale-index-principal", Namespace: "recipe"}

	applied, err := store.CompareAndSwap(ctx, "stale", 0, redisIntegrationState("policy"), time.Minute, quota)
	if err != nil || !applied {
		t.Fatalf("create stale member: applied=%v err=%v", applied, err)
	}
	// Simulate independent state-key expiry while the quota indexes still
	// retain their member. Admission must reclaim the stale member before it
	// counts against the per-identity limit.
	if err := store.client.Del(ctx, store.stateKey("stale")).Err(); err != nil {
		t.Fatal(err)
	}
	applied, err = store.CompareAndSwap(ctx, "replacement", 0, redisIntegrationState("replacement"), time.Minute, quota)
	if err != nil || !applied {
		t.Fatalf("reclaim stale member: applied=%v err=%v", applied, err)
	}
	replacement, err := store.Load(ctx, "replacement")
	if err != nil || !replacement.Found {
		t.Fatalf("replacement load: found=%v err=%v", replacement.Found, err)
	}
	if exists := store.client.Exists(ctx, store.stateKey("stale")).Val(); exists != 0 {
		t.Fatal("stale state must remain absent")
	}
}

func TestRedisStoreIntegrationRejectsSelfReferentialQuotaIndexes(t *testing.T) {
	ctx := context.Background()
	store, _ := newRedisIntegrationStore(t, 10, 10, 60)
	const session = "self-referential-quota"
	quota := QuotaKey{Principal: "self-reference-principal", Namespace: "recipe"}
	applied, err := store.CompareAndSwap(ctx, session, 0, redisIntegrationState("policy"), time.Minute, quota)
	if err != nil || !applied {
		t.Fatalf("create: applied=%v err=%v", applied, err)
	}
	stateKey := store.stateKey(session)
	if err := store.client.HSet(ctx, stateKey,
		redisStateQuotaLRUField, stateKey,
		redisStateQuotaExpiryField, stateKey,
	).Err(); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Load(ctx, session); !errors.Is(err, ErrStateCorrupted) {
		t.Fatalf("self-referential quota fields: err=%v, want ErrStateCorrupted", err)
	}
	if exists := store.client.Exists(ctx, stateKey).Val(); exists != 0 {
		t.Fatal("a state with self-referential quota indexes must be removed")
	}
}

func TestRedisStoreIntegrationPersistsOnlyIdentityState(t *testing.T) {
	ctx := context.Background()
	store, _ := newRedisIntegrationStore(t, 10, 5, 60)
	const session = "raw-session"
	quota := QuotaKey{Principal: "raw-principal", Namespace: "raw-namespace"}
	applied, err := store.CompareAndSwap(ctx, session, 0, redisIntegrationState("policy"), time.Minute, quota)
	if err != nil || !applied {
		t.Fatalf("create: applied=%v err=%v", applied, err)
	}

	fields, err := store.client.HGetAll(ctx, store.stateKey(session)).Result()
	if err != nil {
		t.Fatal(err)
	}
	wantFields := map[string]struct{}{
		redisStatePayloadField:     {},
		redisStateRevisionField:    {},
		redisStateGenerationField:  {},
		redisStateExpiresField:     {},
		redisStateQuotaLRUField:    {},
		redisStateQuotaExpiryField: {},
	}
	for field := range fields {
		if _, allowed := wantFields[field]; !allowed {
			t.Fatalf("unexpected stored field %q", field)
		}
	}
	encoded := strings.Join([]string{
		store.stateKey(session),
		fields[redisStatePayloadField],
		fields[redisStateQuotaLRUField],
		fields[redisStateQuotaExpiryField],
	}, "\n")
	for _, forbidden := range []string{session, quota.Principal, quota.Namespace, "description", "parameters", "arguments", "results", "prompt"} {
		if strings.Contains(encoded, forbidden) {
			t.Fatalf("stored Redis data contains forbidden content %q", forbidden)
		}
	}
}
