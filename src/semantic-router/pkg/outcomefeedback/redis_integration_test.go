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

	caller := assertRedisAbuseLimitAcrossReplicas(t, ctx, replicaA, replicaB, prefix)
	assertRedisProjectionAcrossReplicas(t, ctx, options, replicaA, replicaB, prefix, caller)
}

func assertRedisAbuseLimitAcrossReplicas(
	t *testing.T,
	ctx context.Context,
	replicaA, replicaB *redis.Client,
	prefix string,
) Caller {
	t.Helper()
	limiterA, err := NewRedisAbuseLimiter(RedisAbuseLimiterOptions{
		Client: replicaA, KeyPrefix: prefix, Limit: 2, Window: time.Minute,
	})
	if err != nil {
		t.Fatal(err)
	}
	limiterB, err := NewRedisAbuseLimiter(RedisAbuseLimiterOptions{
		Client: replicaB, KeyPrefix: prefix, Limit: 2, Window: time.Minute,
	})
	if err != nil {
		t.Fatal(err)
	}
	caller := validCaller()
	quotaSentinel := prefix + ":quota:sentinel"
	if setErr := replicaA.Set(ctx, quotaSentinel, "unchanged", 0).Err(); setErr != nil {
		t.Fatal(setErr)
	}
	first, err := limiterA.Allow(ctx, caller)
	if err != nil || !first.Allowed {
		t.Fatalf("first Allow() = (%+v, %v)", first, err)
	}
	second, err := limiterB.Allow(ctx, caller)
	if err != nil || !second.Allowed {
		t.Fatalf("second Allow() = (%+v, %v)", second, err)
	}
	denied, err := limiterA.Allow(ctx, caller)
	if err != nil || denied.Allowed || denied.RetryAfter <= 0 {
		t.Fatalf("global denied Allow() = (%+v, %v)", denied, err)
	}
	otherKey := caller
	otherKey.APIKeyID = "00000000-0000-4000-8000-000000000099"
	independent, err := limiterB.Allow(ctx, otherKey)
	if err != nil || !independent.Allowed {
		t.Fatalf("independent key Allow() = (%+v, %v)", independent, err)
	}
	if value, err := replicaB.Get(ctx, quotaSentinel).Result(); err != nil || value != "unchanged" {
		t.Fatalf("inference quota sentinel = (%q, %v), want unchanged", value, err)
	}
	return caller
}

func assertRedisProjectionAcrossReplicas(
	t *testing.T,
	ctx context.Context,
	options *redis.Options,
	replicaA, replicaB *redis.Client,
	prefix string,
	caller Caller,
) {
	t.Helper()
	storeA, err := NewRedisProjectionStore(RedisProjectionStoreOptions{Client: replicaA, KeyPrefix: prefix})
	if err != nil {
		t.Fatal(err)
	}
	storeB, err := NewRedisProjectionStore(RedisProjectionStoreOptions{Client: replicaB, KeyPrefix: prefix})
	if err != nil {
		t.Fatal(err)
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
	payload, digest, err := projection.Canonical()
	if err != nil {
		t.Fatal(err)
	}
	if projectionPublishError := storeA.Publish(ctx, projection, payload, digest); projectionPublishError != nil {
		t.Fatal(projectionPublishError)
	}
	read, err := storeB.Read(ctx, caller.NamespaceID)
	if err != nil || read.Revision != 1 || len(read.Entries) != 1 || read.Entries[0].GoodFitCount != 1 {
		t.Fatalf("cross-replica Read() = (%+v, %v)", read, err)
	}
	if projectionPublishError := storeB.Publish(ctx, projection, payload, digest); projectionPublishError != nil {
		t.Fatalf("idempotent projection publish: %v", projectionPublishError)
	}

	conflicting := projection
	conflicting.Entries = append([]ProjectionEntry(nil), projection.Entries...)
	conflicting.Entries[0].FailedCount = 1
	conflictingPayload, conflictingDigest, err := conflicting.Canonical()
	if err != nil {
		t.Fatal(err)
	}
	if projectionPublishError := storeB.Publish(ctx, conflicting, conflictingPayload, conflictingDigest); !errors.Is(projectionPublishError, ErrUnavailable) {
		t.Fatalf("conflicting projection error = %v, want ErrUnavailable", projectionPublishError)
	}

	newer := projection
	newer.Revision = 2
	newer.Entries = append([]ProjectionEntry(nil), projection.Entries...)
	newer.Entries[0].GoodFitCount = 2
	newerPayload, newerDigest, err := newer.Canonical()
	if err != nil {
		t.Fatal(err)
	}
	if projectionPublishError := storeB.Publish(ctx, newer, newerPayload, newerDigest); projectionPublishError != nil {
		t.Fatal(projectionPublishError)
	}
	if projectionPublishError := storeA.Publish(ctx, projection, payload, digest); projectionPublishError != nil {
		t.Fatalf("stale publish should be an idempotent no-op: %v", projectionPublishError)
	}

	// A fresh client after simulated process restart observes the latest global
	// revision; no process-local projection is needed for recovery.
	restartedClient := redis.NewClient(options)
	t.Cleanup(func() { _ = restartedClient.Close() })
	restartedStore, err := NewRedisProjectionStore(RedisProjectionStoreOptions{Client: restartedClient, KeyPrefix: prefix})
	if err != nil {
		t.Fatal(err)
	}
	restarted, err := restartedStore.Read(ctx, caller.NamespaceID)
	if err != nil || restarted.Revision != 2 || restarted.Entries[0].GoodFitCount != 2 {
		t.Fatalf("restart Read() = (%+v, %v)", restarted, err)
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
