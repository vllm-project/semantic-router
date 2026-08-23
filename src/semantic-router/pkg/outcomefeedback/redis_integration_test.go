package outcomefeedback

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"
)

func TestRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicas(t *testing.T) {
	redisURL := os.Getenv("VLLM_SR_OUTCOME_TEST_REDIS_URL")
	if redisURL == "" {
		redisURL = os.Getenv("VLLM_SR_USAGE_LEDGER_TEST_REDIS_URL")
	}
	if redisURL == "" {
		t.Skip("outcome Redis integration store is not configured")
	}
	options, testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr := redis.ParseURL(redisURL)
	if testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr != nil {
		t.Fatal(testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr)
	}
	replicaA := redis.NewClient(options)
	replicaB := redis.NewClient(options)
	t.Cleanup(func() {
		_ = replicaA.Close()
		_ = replicaB.Close()
	})
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	if err := replicaA.Ping(ctx).Err(); err != nil {
		t.Fatal(err)
	}
	prefix := "outcome-it:" + strings.ReplaceAll(uuid.NewString(), "-", "")
	t.Cleanup(func() { deleteOutcomeRedisPrefix(replicaA, prefix) })

	limiterA, testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr := NewRedisAbuseLimiter(RedisAbuseLimiterOptions{
		Client: replicaA, KeyPrefix: prefix, Limit: 2, Window: time.Minute,
	})
	if testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr != nil {
		t.Fatal(testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr)
	}
	limiterB, testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr := NewRedisAbuseLimiter(RedisAbuseLimiterOptions{
		Client: replicaB, KeyPrefix: prefix, Limit: 2, Window: time.Minute,
	})
	if testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr != nil {
		t.Fatal(testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr)
	}
	caller := validCaller()
	quotaSentinel := prefix + ":quota:sentinel"
	if err := replicaA.Set(ctx, quotaSentinel, "unchanged", 0).Err(); err != nil {
		t.Fatal(err)
	}
	first, testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr := limiterA.Allow(ctx, caller)
	if testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr != nil || !first.Allowed {
		t.Fatalf("first Allow() = (%+v, %v)", first, testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr)
	}
	second, testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr := limiterB.Allow(ctx, caller)
	if testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr != nil || !second.Allowed {
		t.Fatalf("second Allow() = (%+v, %v)", second, testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr)
	}
	denied, testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr := limiterA.Allow(ctx, caller)
	if testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr != nil || denied.Allowed || denied.RetryAfter <= 0 {
		t.Fatalf("global denied Allow() = (%+v, %v)", denied, testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr)
	}
	otherKey := caller
	otherKey.APIKeyID = "00000000-0000-4000-8000-000000000099"
	independent, testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr := limiterB.Allow(ctx, otherKey)
	if testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr != nil || !independent.Allowed {
		t.Fatalf("independent key Allow() = (%+v, %v)", independent, testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr)
	}
	if value, err := replicaB.Get(ctx, quotaSentinel).Result(); err != nil || value != "unchanged" {
		t.Fatalf("inference quota sentinel = (%q, %v), want unchanged", value, err)
	}

	storeA, testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr := NewRedisProjectionStore(RedisProjectionStoreOptions{Client: replicaA, KeyPrefix: prefix})
	if testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr != nil {
		t.Fatal(testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr)
	}
	storeB, testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr := NewRedisProjectionStore(RedisProjectionStoreOptions{Client: replicaB, KeyPrefix: prefix})
	if testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr != nil {
		t.Fatal(testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr)
	}
	projection := Projection{
		Schema: ProjectionSchema, NamespaceID: caller.NamespaceID, Revision: 1,
		Entries: []ProjectionEntry{{
			RecipeID: "recipe-balanced", RecipeName: "Balanced", RecipeRevision: 4,
			DecisionID: "complex", DecisionName: "Complex", DecisionTier: 3,
			ModelID: "model/served", ModelName: "served-model", ModelRevision: 9,
			GoodFitCount: 1,
		}},
	}
	payload, digest, testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr := projection.Canonical()
	if testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr != nil {
		t.Fatal(testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr)
	}
	if err := storeA.Publish(ctx, projection, payload, digest); err != nil {
		t.Fatal(err)
	}
	read, testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr := storeB.Read(ctx, caller.NamespaceID)
	if testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr != nil || read.Revision != 1 || len(read.Entries) != 1 || read.Entries[0].GoodFitCount != 1 {
		t.Fatalf("cross-replica Read() = (%+v, %v)", read, testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr)
	}
	if err := storeB.Publish(ctx, projection, payload, digest); err != nil {
		t.Fatalf("idempotent projection publish: %v", err)
	}

	conflicting := projection
	conflicting.Entries = append([]ProjectionEntry(nil), projection.Entries...)
	conflicting.Entries[0].FailedCount = 1
	conflictingPayload, conflictingDigest, testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr := conflicting.Canonical()
	if testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr != nil {
		t.Fatal(testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr)
	}
	if err := storeB.Publish(ctx, conflicting, conflictingPayload, conflictingDigest); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("conflicting projection error = %v, want ErrUnavailable", err)
	}

	newer := projection
	newer.Revision = 2
	newer.Entries = append([]ProjectionEntry(nil), projection.Entries...)
	newer.Entries[0].GoodFitCount = 2
	newerPayload, newerDigest, testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr := newer.Canonical()
	if testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr != nil {
		t.Fatal(testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr)
	}
	if err := storeB.Publish(ctx, newer, newerPayload, newerDigest); err != nil {
		t.Fatal(err)
	}
	if err := storeA.Publish(ctx, projection, payload, digest); err != nil {
		t.Fatalf("stale publish should be an idempotent no-op: %v", err)
	}

	// A fresh client after simulated process restart observes the latest global
	// revision; no process-local projection is needed for recovery.
	restartedClient := redis.NewClient(options)
	t.Cleanup(func() { _ = restartedClient.Close() })
	restartedStore, testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr := NewRedisProjectionStore(RedisProjectionStoreOptions{Client: restartedClient, KeyPrefix: prefix})
	if testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr != nil {
		t.Fatal(testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr)
	}
	restarted, testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr := restartedStore.Read(ctx, caller.NamespaceID)
	if testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr != nil || restarted.Revision != 2 || restarted.Entries[0].GoodFitCount != 2 {
		t.Fatalf("restart Read() = (%+v, %v)", restarted, testRedisOutcomeAbuseLimitAndProjectionAreGlobalAcrossReplicasErr)
	}
}

func deleteOutcomeRedisPrefix(client *redis.Client, prefix string) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	iterator := client.Scan(ctx, 0, prefix+":*", 100).Iterator()
	keys := make([]string, 0)
	for iterator.Next(ctx) {
		keys = append(keys, iterator.Val())
	}
	if len(keys) > 0 {
		_ = client.Del(ctx, keys...).Err()
	}
}
