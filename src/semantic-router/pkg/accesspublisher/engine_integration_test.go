package accesspublisher

import (
	"context"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"
)

func TestEnginePublishesCompletePostgresDesiredStateThroughRedisGate(t *testing.T) {
	db := postgresIntegrationDatabase(t)
	redisStore, client, prefix, redisContext := redisIntegrationStore(t)
	ctx, cancel := context.WithTimeout(redisContext, 45*time.Second)
	defer cancel()
	namespaceID, partition, keyID, _ := insertCompleteDesiredState(t, ctx, db)
	insertOutbox(t, ctx, db, uuid.NewString(), namespaceID, 1)
	postgresStore, testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr := NewPostgresStore(db, PostgresStoreOptions{Projector: "engine-integration"})
	if testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr != nil {
		t.Fatal(testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr)
	}
	desired, testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr := NewPostgresDesiredStateReader(db)
	if testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr != nil {
		t.Fatal(testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr)
	}
	engine, testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr := NewEngine(EngineOptions{
		Outbox: postgresStore, Desired: desired, Runtime: redisStore, WorkerID: "engine-worker",
		ClaimLease: 20 * time.Second, RetryDelay: time.Millisecond, CompactionBatch: 2,
	})
	if testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr != nil {
		t.Fatal(testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr)
	}
	first, testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr := engine.ProcessOnce(ctx)
	if testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr != nil || first.Disposition != ProcessApplied || first.Revision != 1 || first.PublicationID == "" {
		t.Fatalf("first ProcessOnce() = %+v, %v", first, testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr)
	}
	readiness, testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr := redisStore.Readiness(ctx, namespaceID, partition)
	if testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr != nil || !readiness.Ready || readiness.AppliedRevision != 1 {
		t.Fatalf("first readiness = %+v, %v", readiness, testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr)
	}
	keys, _ := NewKeyspace(prefix, namespaceID, partition)
	gate := client.HGetAll(ctx, keys.AccessGate()).Val()
	pointer := client.HGetAll(ctx, keys.AccessPointer(keyID)).Val()
	selected, state, testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr := SelectPointer(pointer, gate["publication_id"])
	if testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr != nil || state != PointerStateActive || selected["revision"] != "1" {
		t.Fatalf("first active pointer = %+v, %q, %v", selected, state, testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr)
	}
	if _, err := redisStore.RegisterReplica(ctx, namespaceID, partition, ReplicaRegistration{
		ReplicaID: "router-engine", RuntimeEpoch: 11,
		AccessPublication: first.PublicationID, RoutingPublication: first.PublicationID,
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.ExecContext(ctx, `UPDATE rate_limit_rules r SET limit_value = 50
FROM rate_limit_policies p WHERE p.id = r.policy_id AND p.namespace_id = $1`, namespaceID); err != nil {
		t.Fatal(err)
	}
	insertRevision(t, ctx, db, namespaceID, 2, 11)
	insertOutbox(t, ctx, db, uuid.NewString(), namespaceID, 2)
	waiting, testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr := engine.ProcessOnce(ctx)
	if testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr != nil || waiting.Disposition != ProcessWaitingForAcks ||
		len(waiting.MissingBarrierReplicas) != 1 || len(waiting.MissingRoutingReplicas) != 1 {
		t.Fatalf("restrictive ProcessOnce() = %+v, %v", waiting, testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr)
	}
	secondState, testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr := desired.LoadDesiredState(ctx, namespaceID, 2)
	if testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr != nil {
		t.Fatal(testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr)
	}
	secondPublication, testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr := Compile(secondState)
	if testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr != nil {
		t.Fatal(testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr)
	}
	if secondPublication.ID != waiting.PublicationID {
		t.Fatalf("waiting publication ID = %s, compiled = %s", waiting.PublicationID, secondPublication.ID)
	}
	if err := redisStore.AcknowledgeBarriers(ctx, namespaceID, partition, "router-engine", secondPublication.ID, secondPublication.Digest); err != nil {
		t.Fatal(err)
	}
	if err := redisStore.AcknowledgeRouting(ctx, namespaceID, partition, "router-engine", secondPublication.ID, secondPublication.Digest); err != nil {
		t.Fatal(err)
	}
	second, testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr := engine.ProcessOnce(ctx)
	if testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr != nil || second.Disposition != ProcessApplied || second.Revision != 2 {
		t.Fatalf("acknowledged ProcessOnce() = %+v, %v", second, testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr)
	}
	readiness, testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr = redisStore.Readiness(ctx, namespaceID, partition)
	if testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr != nil || !readiness.Ready || readiness.AppliedRevision != 2 {
		t.Fatalf("second readiness = %+v, %v", readiness, testEnginePublishesCompletePostgresDesiredStateThroughRedisGateErr)
	}
	denyKey := keys.Deny("api_key", keyID)
	if client.Exists(ctx, denyKey).Val() != 0 {
		t.Fatal("applied engine publication retained its deny barrier")
	}
	assertEngineCrashRecovery(t, ctx, engine, redisStore, client, keys, namespaceID, partition, keyID, second)
}

func assertEngineCrashRecovery(
	t *testing.T,
	ctx context.Context,
	engine *Engine,
	store *RedisStore,
	client *redis.Client,
	keys Keyspace,
	namespaceID, partition, keyID string,
	publication ProcessResult,
) {
	t.Helper()
	// Recreate the exact post-PostgreSQL/pre-Redis-finalization crash window.
	if err := client.HSet(ctx, keys.Publication(publication.PublicationID), "state", "compacted").Err(); err != nil {
		t.Fatal(err)
	}
	if err := client.Del(ctx, keys.AppliedRevision()).Err(); err != nil {
		t.Fatal(err)
	}
	denyKey := keys.Deny("api_key", keyID)
	if err := client.SAdd(ctx, denyKey, publication.PublicationID).Err(); err != nil {
		t.Fatal(err)
	}
	if err := client.SAdd(ctx, keys.PublicationBarriers(publication.PublicationID), denyKey).Err(); err != nil {
		t.Fatal(err)
	}
	if err := client.ZAdd(ctx, keys.OpenPublications(), redis.Z{Score: 2, Member: publication.PublicationID}).Err(); err != nil {
		t.Fatal(err)
	}
	if _, err := engine.ReconcileApplied(ctx, namespaceID); err != nil {
		t.Fatalf("ReconcileApplied() error = %v", err)
	}
	readiness, err := store.Readiness(ctx, namespaceID, partition)
	if err != nil || !readiness.Ready || client.Exists(ctx, denyKey).Val() != 0 {
		t.Fatalf("reconciled readiness = %+v, deny=%d, err=%v", readiness, client.Exists(ctx, denyKey).Val(), err)
	}
}
