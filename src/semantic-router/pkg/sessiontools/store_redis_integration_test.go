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
	if loaded.State.Revision == 0 || metadata.ObservedGeneration == 0 {
		t.Fatalf("loaded state=%+v metadata=%+v", loaded.State, metadata)
	}
	createdRevision := loaded.State.Revision

	next := loaded.State.Clone()
	next.PolicyFingerprint = "policy-2"
	applied, err = storeB.CompareAndSwap(ctx, "session", loaded.State.Revision, next, time.Minute, quota)
	if err != nil || !applied {
		t.Fatalf("cross-replica update: applied=%v err=%v", applied, err)
	}
	updated, err := storeA.Load(ctx, "session")
	if err != nil || !updated.Found || updated.State.Revision == createdRevision || updated.State.PolicyFingerprint != "policy-2" {
		t.Fatalf("updated state = %+v, err=%v", updated, err)
	}
	updatedRevision := updated.State.Revision

	if err := storeA.Close(); err != nil {
		t.Fatal(err)
	}
	restarted, err := NewRedisStore(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = restarted.Close() })
	afterRestart, err := restarted.Load(ctx, "session")
	if err != nil || !afterRestart.Found || afterRestart.State.Revision != updatedRevision {
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

func TestRedisStoreIntegrationStaleRevisionRejectedAfterRecreate(t *testing.T) {
	for _, expire := range []bool{false, true} {
		name := "delete"
		if expire {
			name = "expiry"
		}
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			store, _ := newRedisIntegrationStore(t, 10, 10, 60)
			quota := QuotaKey{Principal: "stale-writer", Namespace: "recipe"}

			createdRevision, applied, err := store.CompareAndSwapWithRevision(
				ctx,
				"session",
				0,
				redisIntegrationState("writer-a"),
				time.Minute,
				quota,
			)
			if err != nil || !applied {
				t.Fatalf("create: revision=%d applied=%v err=%v", createdRevision, applied, err)
			}
			staleRevision := createdRevision
			if expire {
				staleRevision, applied, err = store.CompareAndSwapWithRevision(
					ctx,
					"session",
					createdRevision,
					redisIntegrationState("writer-a-updated"),
					40*time.Millisecond,
					quota,
				)
				if err != nil || !applied {
					t.Fatalf("update before expiry: revision=%d applied=%v err=%v", staleRevision, applied, err)
				}
				time.Sleep(70 * time.Millisecond)
			} else if err := store.Delete(ctx, "session"); err != nil {
				t.Fatal(err)
			}

			recreatedRevision, applied, err := store.CompareAndSwapWithRevision(
				ctx,
				"session",
				0,
				redisIntegrationState("writer-b"),
				time.Minute,
				quota,
			)
			if err != nil || !applied {
				t.Fatalf("recreate: revision=%d applied=%v err=%v", recreatedRevision, applied, err)
			}
			if recreatedRevision == staleRevision {
				t.Fatalf("recreated revision %d reused stale token", recreatedRevision)
			}

			_, applied, err = store.CompareAndSwapWithRevision(
				ctx,
				"session",
				staleRevision,
				redisIntegrationState("writer-a-stale"),
				time.Minute,
				quota,
			)
			if applied || !errors.Is(err, ErrRevisionMismatch) {
				t.Fatalf("stale CAS: applied=%v err=%v, want ErrRevisionMismatch", applied, err)
			}
			current, err := store.Load(ctx, "session")
			if err != nil || !current.Found || current.State.PolicyFingerprint != "writer-b" {
				t.Fatalf("state after stale CAS: found=%v policy=%q err=%v", current.Found, current.State.PolicyFingerprint, err)
			}
		})
	}
}

func TestRedisStoreIntegrationLoadDoesNotRefreshStateOrIndexes(t *testing.T) {
	ctx := context.Background()
	store, _ := newRedisIntegrationStore(t, 10, 10, 60)
	quota := QuotaKey{Principal: "read-only-load", Namespace: "recipe"}
	const session = "read-only-load"
	if applied, err := store.CompareAndSwap(
		ctx,
		session,
		0,
		redisIntegrationState("policy"),
		2*time.Second,
		quota,
	); err != nil || !applied {
		t.Fatalf("create: applied=%v err=%v", applied, err)
	}

	stateKey := store.stateKey(session)
	quotaLRU, quotaExpiry := store.quotaIndexKeys(quota)
	indexes := []string{store.globalLRUKey(), store.globalExpiryKey(), quotaLRU, quotaExpiry}
	beforeScores := make(map[string]float64, len(indexes))
	beforeTTLs := make(map[string]time.Duration, len(indexes))
	for _, indexKey := range indexes {
		score, err := store.client.ZScore(ctx, indexKey, stateKey).Result()
		if err != nil {
			t.Fatalf("read score for %s: %v", indexKey, err)
		}
		beforeScores[indexKey] = score
		beforeTTLs[indexKey] = store.client.PTTL(ctx, indexKey).Val()
	}
	beforeStateTTL := store.client.PTTL(ctx, stateKey).Val()
	beforeExpiry, err := store.client.HGet(ctx, stateKey, redisStateExpiresField).Result()
	if err != nil {
		t.Fatal(err)
	}

	time.Sleep(25 * time.Millisecond)
	loaded, err := store.Load(ctx, session)
	if err != nil || !loaded.Found {
		t.Fatalf("load: found=%v err=%v", loaded.Found, err)
	}
	afterExpiry, err := store.client.HGet(ctx, stateKey, redisStateExpiresField).Result()
	if err != nil {
		t.Fatal(err)
	}
	if afterExpiry != beforeExpiry {
		t.Fatalf("load changed expiry: before=%s after=%s", beforeExpiry, afterExpiry)
	}
	for _, indexKey := range indexes {
		score, err := store.client.ZScore(ctx, indexKey, stateKey).Result()
		if err != nil {
			t.Fatalf("read score for %s after load: %v", indexKey, err)
		}
		if score != beforeScores[indexKey] {
			t.Fatalf("load changed %s score: before=%v after=%v", indexKey, beforeScores[indexKey], score)
		}
		if ttl := store.client.PTTL(ctx, indexKey).Val(); ttl <= 0 || ttl > beforeTTLs[indexKey] {
			t.Fatalf("load refreshed %s TTL from %v to %v", indexKey, beforeTTLs[indexKey], ttl)
		}
	}
	if ttl := store.client.PTTL(ctx, stateKey).Val(); ttl <= 0 || ttl > beforeStateTTL {
		t.Fatalf("load refreshed state TTL from %v to %v", beforeStateTTL, ttl)
	}

	next := loaded.State.Clone()
	next.PolicyFingerprint = "updated"
	if applied, err := store.CompareAndSwap(ctx, session, loaded.State.Revision, next, 4*time.Second, quota); err != nil || !applied {
		t.Fatalf("refresh CAS: applied=%v err=%v", applied, err)
	}
	for _, indexKey := range indexes {
		if ttl := store.client.PTTL(ctx, indexKey).Val(); ttl <= 3*time.Second {
			t.Fatalf("CAS did not refresh %s TTL: %v", indexKey, ttl)
		}
		score, err := store.client.ZScore(ctx, indexKey, stateKey).Result()
		if err != nil {
			t.Fatalf("read score for %s after CAS: %v", indexKey, err)
		}
		if score == beforeScores[indexKey] {
			t.Fatalf("CAS did not refresh %s score %v", indexKey, score)
		}
	}
	if ttl := store.client.PTTL(ctx, stateKey).Val(); ttl <= 3*time.Second {
		t.Fatalf("CAS did not refresh state TTL: %v", ttl)
	}
}

func TestRedisStoreIntegrationRevisionRemainsExactAboveLuaIntegerPrecision(t *testing.T) {
	ctx := context.Background()
	store, _ := newRedisIntegrationStore(t, 10, 10, 60)
	quota := QuotaKey{Principal: "large-revision", Namespace: "recipe"}
	const startingRevision uint64 = 9007199254740992
	if err := store.client.MSet(ctx,
		store.generationKey(), "1",
		store.revisionKey(), strconv.FormatUint(startingRevision, 10),
	).Err(); err != nil {
		t.Fatal(err)
	}

	revision, applied, err := store.CompareAndSwapWithRevision(
		ctx,
		"session",
		0,
		redisIntegrationState("policy"),
		time.Minute,
		quota,
	)
	if err != nil || !applied {
		t.Fatalf("create: revision=%d applied=%v err=%v", revision, applied, err)
	}
	wantRevision := startingRevision + 1
	if revision != wantRevision {
		t.Fatalf("committed revision = %d, want %d", revision, wantRevision)
	}
	storedRevision, err := store.client.HGet(ctx, store.stateKey("session"), redisStateRevisionField).Result()
	if err != nil {
		t.Fatal(err)
	}
	if storedRevision != strconv.FormatUint(wantRevision, 10) {
		t.Fatalf("stored revision = %q, want %d", storedRevision, wantRevision)
	}
	loaded, metadata, err := store.LoadWithMetadata(ctx, "session")
	if err != nil || !loaded.Found {
		t.Fatalf("load: found=%v err=%v", loaded.Found, err)
	}
	if loaded.State.Revision != wantRevision || metadata.ObservedRevision != wantRevision {
		t.Fatalf("loaded revision=%d metadata=%d, want %d", loaded.State.Revision, metadata.ObservedRevision, wantRevision)
	}
}

func TestRedisStoreIntegrationRejectsNonCanonicalSequenceAndStateTokens(t *testing.T) {
	for _, sequence := range []string{"generation", "revision"} {
		for _, value := range []string{"0", "01"} {
			t.Run("sequence-"+sequence+"-"+value, func(t *testing.T) {
				ctx := context.Background()
				store, _ := newRedisIntegrationStore(t, 10, 10, 60)
				if err := store.client.MSet(ctx,
					store.generationKey(), "1",
					store.revisionKey(), "1",
				).Err(); err != nil {
					t.Fatal(err)
				}
				sequenceKey := store.generationKey()
				if sequence == "revision" {
					sequenceKey = store.revisionKey()
				}
				if err := store.client.Set(ctx, sequenceKey, value, 0).Err(); err != nil {
					t.Fatal(err)
				}
				applied, err := store.CompareAndSwap(
					ctx,
					"session",
					0,
					redisIntegrationState("policy"),
					time.Minute,
					QuotaKey{Principal: "non-canonical-sequence", Namespace: "recipe"},
				)
				if applied || !errors.Is(err, ErrStateCorrupted) {
					t.Fatalf("create: applied=%v err=%v, want ErrStateCorrupted", applied, err)
				}
				if exists := store.client.Exists(ctx, store.stateKey("session")).Val(); exists != 0 {
					t.Fatal("a non-canonical sequence must not admit state")
				}
			})
		}
	}

	for _, field := range []string{redisStateRevisionField, redisStateGenerationField} {
		for _, value := range []string{"0", "01"} {
			t.Run("state-"+field+"-"+value, func(t *testing.T) {
				ctx := context.Background()
				store, _ := newRedisIntegrationStore(t, 10, 10, 60)
				quota := QuotaKey{Principal: "non-canonical-state", Namespace: "recipe"}
				if applied, err := store.CompareAndSwap(
					ctx,
					"session",
					0,
					redisIntegrationState("policy"),
					time.Minute,
					quota,
				); err != nil || !applied {
					t.Fatalf("create: applied=%v err=%v", applied, err)
				}
				if err := store.client.HSet(ctx, store.stateKey("session"), field, value).Err(); err != nil {
					t.Fatal(err)
				}
				if _, err := store.Load(ctx, "session"); !errors.Is(err, ErrStateCorrupted) {
					t.Fatalf("load: err=%v, want ErrStateCorrupted", err)
				}
			})
		}
	}
}

func TestRedisStoreIntegrationCounterCorruptionFailsClosed(t *testing.T) {
	for _, phase := range []string{"create", "update"} {
		for _, sequence := range []string{"generation", "revision"} {
			for _, corruption := range []string{"missing", "wrong-type"} {
				t.Run(phase+"-"+sequence+"-"+corruption, func(t *testing.T) {
					ctx := context.Background()
					store, _ := newRedisIntegrationStore(t, 10, 10, 60)
					quota := QuotaKey{Principal: "counter-corruption", Namespace: "recipe"}
					expectedRevision := uint64(0)
					if phase == "update" {
						var applied bool
						var err error
						expectedRevision, applied, err = store.CompareAndSwapWithRevision(
							ctx,
							"session",
							0,
							redisIntegrationState("original"),
							time.Minute,
							quota,
						)
						if err != nil || !applied {
							t.Fatalf("create: revision=%d applied=%v err=%v", expectedRevision, applied, err)
						}
					} else if err := store.client.MSet(ctx,
						store.generationKey(), "1",
						store.revisionKey(), "1",
					).Err(); err != nil {
						t.Fatal(err)
					}

					sequenceKey := store.generationKey()
					if sequence == "revision" {
						sequenceKey = store.revisionKey()
					}
					if err := store.client.Del(ctx, sequenceKey).Err(); err != nil {
						t.Fatal(err)
					}
					if corruption == "wrong-type" {
						if err := store.client.HSet(ctx, sequenceKey, "sentinel", "value").Err(); err != nil {
							t.Fatal(err)
						}
					}

					revision, applied, err := store.CompareAndSwapWithRevision(
						ctx,
						"session",
						expectedRevision,
						redisIntegrationState("updated"),
						time.Minute,
						quota,
					)
					if revision != 0 || applied || !errors.Is(err, ErrStateCorrupted) {
						t.Fatalf("CAS: revision=%d applied=%v err=%v, want ErrStateCorrupted", revision, applied, err)
					}
					if phase == "create" {
						if exists := store.client.Exists(ctx, store.stateKey("session")).Val(); exists != 0 {
							t.Fatal("counter corruption must not admit state")
						}
						return
					}
					current, err := store.Load(ctx, "session")
					if err != nil || !current.Found || current.State.PolicyFingerprint != "original" {
						t.Fatalf("state after failed update: found=%v policy=%q err=%v", current.Found, current.State.PolicyFingerprint, err)
					}
				})
			}
		}
	}
}

func TestRedisStoreIntegrationUpdateRequiresEveryIndexMember(t *testing.T) {
	for _, indexName := range []string{"global-lru", "global-expiry", "quota-lru", "quota-expiry"} {
		t.Run(indexName, func(t *testing.T) {
			ctx := context.Background()
			store, _ := newRedisIntegrationStore(t, 10, 10, 60)
			quota := QuotaKey{Principal: "missing-index-member", Namespace: "recipe"}
			const session = "missing-index-member"
			revision, applied, err := store.CompareAndSwapWithRevision(
				ctx,
				session,
				0,
				redisIntegrationState("original"),
				time.Minute,
				quota,
			)
			if err != nil || !applied {
				t.Fatalf("create: revision=%d applied=%v err=%v", revision, applied, err)
			}
			if applied, err := store.CompareAndSwap(
				ctx,
				"keeper",
				0,
				redisIntegrationState("keeper"),
				time.Minute,
				quota,
			); err != nil || !applied {
				t.Fatalf("create keeper: applied=%v err=%v", applied, err)
			}

			stateKey := store.stateKey(session)
			quotaLRU, quotaExpiry := store.quotaIndexKeys(quota)
			indexes := map[string]string{
				"global-lru":    store.globalLRUKey(),
				"global-expiry": store.globalExpiryKey(),
				"quota-lru":     quotaLRU,
				"quota-expiry":  quotaExpiry,
			}
			if err := store.client.ZRem(ctx, indexes[indexName], stateKey).Err(); err != nil {
				t.Fatal(err)
			}
			committedRevision, applied, err := store.CompareAndSwapWithRevision(
				ctx,
				session,
				revision,
				redisIntegrationState("updated"),
				time.Minute,
				quota,
			)
			if committedRevision != 0 || applied || !errors.Is(err, ErrStateCorrupted) {
				t.Fatalf("update: revision=%d applied=%v err=%v, want ErrStateCorrupted", committedRevision, applied, err)
			}
			if exists := store.client.Exists(ctx, stateKey).Val(); exists != 0 {
				t.Fatal("a live state with incomplete indexes must be invalidated")
			}
			if exists := store.client.Exists(ctx, store.stateKey("keeper")).Val(); exists != 1 {
				t.Fatal("invalidating one state must not delete another indexed state")
			}
			for name, indexKey := range indexes {
				if _, err := store.client.ZScore(ctx, indexKey, stateKey).Result(); !errors.Is(err, redis.Nil) {
					t.Fatalf("%s still contains invalidated state: %v", name, err)
				}
			}
		})
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
	if exists := store.client.Exists(ctx, store.stateKey("wrong-type")).Val(); exists != 1 {
		t.Fatal("read-only load must not delete a wrong-type value")
	}
}

func TestRedisStoreIntegrationWrongTypeIndexesFailClosed(t *testing.T) {
	for _, indexName := range []string{"global-lru", "global-expiry", "quota-lru", "quota-expiry"} {
		t.Run(indexName, func(t *testing.T) {
			ctx := context.Background()
			store, _ := newRedisIntegrationStore(t, 10, 10, 60)
			quota := QuotaKey{Principal: "wrong-type-principal", Namespace: "recipe"}
			const session = "wrong-type-index"
			applied, err := store.CompareAndSwap(ctx, session, 0, redisIntegrationState("policy"), time.Minute, quota)
			if err != nil || !applied {
				t.Fatalf("create: applied=%v err=%v", applied, err)
			}
			loaded, err := store.Load(ctx, session)
			if err != nil || !loaded.Found {
				t.Fatalf("load: found=%v err=%v", loaded.Found, err)
			}

			quotaLRU, quotaExpiry := store.quotaIndexKeys(quota)
			indexKey := map[string]string{
				"global-lru":    store.globalLRUKey(),
				"global-expiry": store.globalExpiryKey(),
				"quota-lru":     quotaLRU,
				"quota-expiry":  quotaExpiry,
			}[indexName]
			if err := store.client.Set(ctx, indexKey, "wrong-type", time.Minute).Err(); err != nil {
				t.Fatalf("corrupt %s: %v", indexKey, err)
			}

			next := loaded.State.Clone()
			next.PolicyFingerprint = "updated"
			if _, err := store.CompareAndSwap(ctx, session, loaded.State.Revision, next, time.Minute, quota); !errors.Is(err, ErrStateCorrupted) {
				t.Fatalf("CAS with wrong-type %s: err=%v, want ErrStateCorrupted", indexName, err)
			}
			if value, err := store.client.Get(ctx, indexKey).Result(); err != nil || value != "wrong-type" {
				t.Fatalf("wrong-type index was modified: value=%q err=%v", value, err)
			}
			if err := store.Delete(ctx, session); err != nil {
				t.Fatalf("delete with wrong-type index: %v", err)
			}
			if exists := store.client.Exists(ctx, store.stateKey(session)).Val(); exists != 0 {
				t.Fatal("delete must remove the state despite a wrong-type index")
			}
			if value, err := store.client.Get(ctx, indexKey).Result(); err != nil || value != "wrong-type" {
				t.Fatalf("delete modified the wrong-type index: value=%q err=%v", value, err)
			}
		})
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

func TestRedisStoreIntegrationRejectsExternalQuotaIndexesWithoutTouchingThem(t *testing.T) {
	ctx := context.Background()
	store, _ := newRedisIntegrationStore(t, 10, 10, 60)
	const session = "external-quota-index"
	const externalIndex = "other-store:identity:lru"
	quota := QuotaKey{Principal: "external-index-principal", Namespace: "recipe"}
	applied, err := store.CompareAndSwap(ctx, session, 0, redisIntegrationState("policy"), time.Minute, quota)
	if err != nil || !applied {
		t.Fatalf("create: applied=%v err=%v", applied, err)
	}
	if err := store.client.ZAdd(ctx, externalIndex, redis.Z{Score: 1, Member: "sentinel"}).Err(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = store.client.Del(context.Background(), externalIndex, externalIndex+":expiry").Err()
	})
	if err := store.client.HSet(ctx, store.stateKey(session),
		redisStateQuotaLRUField, externalIndex,
		redisStateQuotaExpiryField, externalIndex+":expiry",
	).Err(); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Load(ctx, session); !errors.Is(err, ErrStateCorrupted) {
		t.Fatalf("external quota fields: err=%v, want ErrStateCorrupted", err)
	}
	if exists := store.client.Exists(ctx, externalIndex).Val(); exists != 1 {
		t.Fatal("an external quota index must not be deleted")
	}
	if members := store.client.ZCard(ctx, externalIndex).Val(); members != 1 {
		t.Fatalf("external quota index members = %d, want 1", members)
	}
}

func TestRedisStoreIntegrationRejectsCrossIdentityQuotaPointers(t *testing.T) {
	ctx := context.Background()
	store, _ := newRedisIntegrationStore(t, 20, 20, 60)
	victimQuota := QuotaKey{Principal: "cross-pointer-victim", Namespace: "recipe"}
	tamperedQuota := QuotaKey{Principal: "cross-pointer-tampered", Namespace: "recipe"}

	const victimSession = "cross-pointer-victim-state"
	applied, err := store.CompareAndSwap(
		ctx,
		victimSession,
		0,
		redisIntegrationState("victim"),
		time.Minute,
		victimQuota,
	)
	if err != nil || !applied {
		t.Fatalf("create victim: applied=%v err=%v", applied, err)
	}
	victimStateKey := store.stateKey(victimSession)
	victimLRU, victimExpiry := store.quotaIndexKeys(victimQuota)
	victimLRUScore, err := store.client.ZScore(ctx, victimLRU, victimStateKey).Result()
	if err != nil {
		t.Fatalf("victim quota LRU score: %v", err)
	}
	victimExpiryScore, err := store.client.ZScore(ctx, victimExpiry, victimStateKey).Result()
	if err != nil {
		t.Fatalf("victim quota expiry score: %v", err)
	}

	// Corrupt a live state so its stored quota pointers name a different,
	// valid identity bucket. Keep a member in the victim indexes to prove that
	// corruption cleanup never follows an untrusted state-derived pointer.
	prepareTampered := func(t *testing.T, session string) (StateToken, string, string, string) {
		t.Helper()
		applied, err := store.CompareAndSwap(
			ctx,
			session,
			0,
			redisIntegrationState(session),
			time.Minute,
			tamperedQuota,
		)
		if err != nil || !applied {
			t.Fatalf("create tampered state %s: applied=%v err=%v", session, applied, err)
		}
		stateKey := store.stateKey(session)
		_, metadata, err := store.LoadWithMetadata(ctx, session)
		if err != nil {
			t.Fatalf("load tampered state %s: %v", session, err)
		}
		payload, err := store.client.HGet(ctx, stateKey, redisStatePayloadField).Result()
		if err != nil {
			t.Fatalf("read tampered payload %s: %v", session, err)
		}
		revision, err := store.client.HGet(ctx, stateKey, redisStateRevisionField).Result()
		if err != nil {
			t.Fatalf("read tampered revision %s: %v", session, err)
		}
		generation, err := store.client.HGet(ctx, stateKey, redisStateGenerationField).Result()
		if err != nil {
			t.Fatalf("read tampered generation %s: %v", session, err)
		}
		if err := store.client.HSet(ctx, stateKey,
			redisStateQuotaLRUField, victimLRU,
			redisStateQuotaExpiryField, victimExpiry,
		).Err(); err != nil {
			t.Fatalf("tamper quota pointers %s: %v", session, err)
		}
		if err := store.client.ZAdd(ctx, victimLRU, redis.Z{Score: 2, Member: stateKey}).Err(); err != nil {
			t.Fatalf("add tampered victim LRU member %s: %v", session, err)
		}
		if err := store.client.ZAdd(ctx, victimExpiry, redis.Z{Score: 2, Member: stateKey}).Err(); err != nil {
			t.Fatalf("add tampered victim expiry member %s: %v", session, err)
		}
		return StateToken{
			Revision:   metadata.ObservedRevision,
			Generation: metadata.ObservedGeneration,
		}, payload, revision, generation
	}

	assertVictimIndexesUnchanged := func(t *testing.T, tamperedStateKey string) {
		t.Helper()
		if exists := store.client.Exists(ctx, victimStateKey).Val(); exists != 1 {
			t.Fatalf("victim state exists = %d, want 1", exists)
		}
		for _, index := range []struct {
			name  string
			key   string
			score float64
		}{
			{name: "victim LRU", key: victimLRU, score: victimLRUScore},
			{name: "victim expiry", key: victimExpiry, score: victimExpiryScore},
		} {
			got, err := store.client.ZScore(ctx, index.key, victimStateKey).Result()
			if err != nil {
				t.Fatalf("%s victim member: %v", index.name, err)
			}
			if got != index.score {
				t.Fatalf("%s victim score = %v, want %v", index.name, got, index.score)
			}
			got, err = store.client.ZScore(ctx, index.key, tamperedStateKey).Result()
			if err != nil {
				t.Fatalf("%s tampered member: %v", index.name, err)
			}
			if got != 2 {
				t.Fatalf("%s tampered score = %v, want 2", index.name, got)
			}
		}
	}

	const casSession = "cross-pointer-cas"
	casToken, _, _, _ := prepareTampered(t, casSession)
	if _, err := store.CompareAndSwap(
		ctx,
		casSession,
		casToken.Revision,
		redisIntegrationState("cas-update"),
		time.Minute,
		tamperedQuota,
	); !errors.Is(err, ErrStateCorrupted) {
		t.Fatalf("CAS with a cross-identity pointer: err=%v, want ErrStateCorrupted", err)
	}
	if exists := store.client.Exists(ctx, store.stateKey(casSession)).Val(); exists != 0 {
		t.Fatal("a corrupted CAS state must be removed")
	}
	assertVictimIndexesUnchanged(t, store.stateKey(casSession))

	const deleteSession = "cross-pointer-delete"
	_, _, _, _ = prepareTampered(t, deleteSession)
	if err := store.Delete(ctx, deleteSession); err != nil {
		t.Fatalf("delete with a cross-identity pointer: %v", err)
	}
	assertVictimIndexesUnchanged(t, store.stateKey(deleteSession))

	const tokenSession = "cross-pointer-token"
	token, _, _, _ := prepareTampered(t, tokenSession)
	deleted, err := store.DeleteIfToken(ctx, tokenSession, token)
	if err != nil || !deleted {
		t.Fatalf("conditional delete with a cross-identity pointer: deleted=%v err=%v", deleted, err)
	}
	assertVictimIndexesUnchanged(t, store.stateKey(tokenSession))

	const rawSession = "cross-pointer-raw"
	_, payload, revision, generation := prepareTampered(t, rawSession)
	if err := store.deleteRawIfCurrent(
		ctx,
		store.stateKey(rawSession),
		payload,
		revision,
		generation,
	); err != nil {
		t.Fatalf("raw conditional delete with a cross-identity pointer: %v", err)
	}
	assertVictimIndexesUnchanged(t, store.stateKey(rawSession))
}

func TestRedisStoreIntegrationReclaimsMalformedStateMembersWithoutDeletingThem(t *testing.T) {
	ctx := context.Background()
	store, _ := newRedisIntegrationStore(t, 1, 1, 60)
	malformedStateKey := store.keyPrefix + "state:sentinel"
	quota := QuotaKey{Principal: "external-member-principal", Namespace: "recipe"}
	if err := store.client.Set(ctx, malformedStateKey, "sentinel", time.Minute).Err(); err != nil {
		t.Fatal(err)
	}
	if err := store.client.ZAdd(ctx, store.globalLRUKey(), redis.Z{Score: 1, Member: malformedStateKey}).Err(); err != nil {
		t.Fatal(err)
	}
	if err := store.client.ZAdd(ctx, store.globalExpiryKey(), redis.Z{Score: 1, Member: malformedStateKey}).Err(); err != nil {
		t.Fatal(err)
	}
	applied, err := store.CompareAndSwap(ctx, "owned-replacement", 0, redisIntegrationState("policy"), time.Minute, quota)
	if err != nil || !applied {
		t.Fatalf("admission with malformed member: applied=%v err=%v", applied, err)
	}
	if exists := store.client.Exists(ctx, malformedStateKey).Val(); exists != 1 {
		t.Fatal("a same-prefix non-state key must not be deleted")
	}
	if members := store.client.ZCard(ctx, store.globalLRUKey()).Val(); members != 1 {
		t.Fatalf("global index members = %d, want the admitted owned state only", members)
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
