package backendinvoker

import (
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestRedisResponseTerminalStoreCrossReplicaSingleConsume(t *testing.T) {
	options := responseTerminalRedisTestOptions(t)
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
	prefix := "terminal-it:" + strings.ReplaceAll(uuid.NewString(), "-", "")
	storeA := newResponseTerminalRedisTestStore(t, replicaA, prefix)
	storeB := newResponseTerminalRedisTestStore(t, replicaB, prefix)
	t.Cleanup(func() { deleteResponseTerminalTestKeys(replicaA, storeA.testKeyPrefix()) })
	plan, attempt := terminalFixture("cross-replica-request", "cross-replica-dispatch")
	reference := terminalReference(plan)
	if err := storeA.Finalize(ctx, plan, attempt, successfulTerminal()); err != nil {
		t.Fatalf("replica A Finalize: %v", err)
	}
	if err := storeB.Finalize(ctx, plan, attempt, successfulTerminal()); !errors.Is(err, ErrResponseTerminalDuplicate) {
		t.Fatalf("duplicate cross-replica Finalize error = %v", err)
	}

	canceled, cancelTake := context.WithCancel(ctx)
	cancelTake()
	record, found, err := storeB.Take(context.WithoutCancel(canceled), reference)
	if err != nil || !found {
		t.Fatalf("replica B Take = (%+v, %v, %v)", record, found, err)
	}
	if record.Reference != reference || record.Attempt.ID != attempt.ID ||
		record.Terminal.StopReason != llmprotocol.StopEndTurn {
		t.Fatalf("cross-replica terminal = %+v", record)
	}
	if _, found, err := storeA.Take(ctx, reference); err != nil || found {
		t.Fatalf("terminal replay = (found %v, error %v)", found, err)
	}
}

func TestRedisResponseTerminalStoreRejectsWrongIdentityWithoutConsuming(t *testing.T) {
	options := responseTerminalRedisTestOptions(t)
	client := redis.NewClient(options)
	t.Cleanup(func() { _ = client.Close() })
	prefix := "terminal-identity-it:" + strings.ReplaceAll(uuid.NewString(), "-", "")
	store := newResponseTerminalRedisTestStore(t, client, prefix)
	t.Cleanup(func() { deleteResponseTerminalTestKeys(client, store.testKeyPrefix()) })
	plan, attempt := terminalFixture("identity-request", "identity-dispatch")
	reference := terminalReference(plan)
	if err := store.Finalize(context.Background(), plan, attempt, successfulTerminal()); err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*ResponseTerminalReference){
		"admission": func(value *ResponseTerminalReference) { value.AdmissionID += "-other" },
		"request":   func(value *ResponseTerminalReference) { value.RequestID += "-other" },
		"dispatch":  func(value *ResponseTerminalReference) { value.DispatchID += "-other" },
		"model":     func(value *ResponseTerminalReference) { value.ModelID += "-other" },
	} {
		t.Run(name, func(t *testing.T) {
			wrong := reference
			mutate(&wrong)
			if _, found, err := store.Take(context.Background(), wrong); err != nil || found {
				t.Fatalf("wrong-identity Take = (found %v, error %v)", found, err)
			}
		})
	}
	if _, found, err := store.Take(context.Background(), reference); err != nil || !found {
		t.Fatalf("correct identity was not preserved = (found %v, error %v)", found, err)
	}
}

func TestRedisResponseTerminalStoreRejectsMalformedAndUnavailableEvidence(t *testing.T) {
	options := responseTerminalRedisTestOptions(t)
	client := redis.NewClient(options)
	t.Cleanup(func() { _ = client.Close() })
	prefix := "terminal-malformed-it:" + strings.ReplaceAll(uuid.NewString(), "-", "")
	store := newResponseTerminalRedisTestStore(t, client, prefix)
	t.Cleanup(func() { deleteResponseTerminalTestKeys(client, store.testKeyPrefix()) })
	plan, _ := terminalFixture("malformed-request", "malformed-dispatch")
	reference := terminalReference(plan)
	keys, err := store.keys(reference)
	if err != nil {
		t.Fatal(err)
	}
	if err := client.Set(context.Background(), keys.record, `{"schema":"unexpected"}`, time.Minute).Err(); err != nil {
		t.Fatal(err)
	}
	if _, found, err := store.Take(context.Background(), reference); found ||
		!errors.Is(err, ErrResponseTerminalUnavailable) {
		t.Fatalf("malformed Take = (found %v, error %v)", found, err)
	}
	if _, found, err := store.Take(context.Background(), reference); err != nil || found {
		t.Fatalf("malformed evidence was not consumed once = (found %v, error %v)", found, err)
	}

	unavailableClient := redis.NewClient(&redis.Options{
		Addr:        net.JoinHostPort("127.0.0.1", "1"),
		DialTimeout: 20 * time.Millisecond, ReadTimeout: 20 * time.Millisecond,
		WriteTimeout: 20 * time.Millisecond, MaxRetries: 0,
	})
	t.Cleanup(func() { _ = unavailableClient.Close() })
	unavailable, err := NewRedisResponseTerminalStore(RedisResponseTerminalStoreOptions{
		Client: unavailableClient, KeyPrefix: prefix + ":unavailable",
		OperationTimeout: 50 * time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, found, err := unavailable.Take(context.Background(), reference); found ||
		!errors.Is(err, ErrResponseTerminalUnavailable) {
		t.Fatalf("unavailable Take = (found %v, error %v)", found, err)
	}
}

func TestRedisResponseTerminalStoreShardCapacityIsHardAndRecoverable(t *testing.T) {
	options := responseTerminalRedisTestOptions(t)
	client := redis.NewClient(options)
	t.Cleanup(func() { _ = client.Close() })
	prefix := "terminal-capacity-it:" + strings.ReplaceAll(uuid.NewString(), "-", "")
	store, err := NewRedisResponseTerminalStore(RedisResponseTerminalStoreOptions{
		Client: client, KeyPrefix: prefix, TTL: time.Minute, Capacity: managedTerminalShardCount,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { deleteResponseTerminalTestKeys(client, store.testKeyPrefix()) })
	firstPlan, firstAttempt := terminalFixture("capacity-request-0", "capacity-dispatch-0")
	firstReference := terminalReference(firstPlan)
	firstKeys, err := store.keys(firstReference)
	if err != nil {
		t.Fatal(err)
	}
	var secondPlan Plan
	var secondAttempt AttemptResult
	var secondReference ResponseTerminalReference
	for index := 1; index < 10_000; index++ {
		candidatePlan, candidateAttempt := terminalFixture(
			fmt.Sprintf("capacity-request-%d", index),
			fmt.Sprintf("capacity-dispatch-%d", index),
		)
		candidateReference := terminalReference(candidatePlan)
		candidateKeys, keyErr := store.keys(candidateReference)
		if keyErr != nil {
			t.Fatal(keyErr)
		}
		if candidateKeys.expiry == firstKeys.expiry {
			secondPlan, secondAttempt, secondReference = candidatePlan, candidateAttempt, candidateReference
			break
		}
	}
	if secondReference.RequestID == "" {
		t.Fatal("could not find two references in one terminal shard")
	}
	ctx := context.Background()
	if err := store.Finalize(ctx, firstPlan, firstAttempt, successfulTerminal()); err != nil {
		t.Fatal(err)
	}
	if err := store.Finalize(ctx, secondPlan, secondAttempt, successfulTerminal()); !errors.Is(err, ErrResponseTerminalCapacity) {
		t.Fatalf("full shard Finalize error = %v", err)
	}
	if _, found, err := store.Take(ctx, firstReference); err != nil || !found {
		t.Fatalf("release first terminal = (found %v, error %v)", found, err)
	}
	if err := store.Finalize(ctx, secondPlan, secondAttempt, successfulTerminal()); err != nil {
		t.Fatalf("Finalize after consume: %v", err)
	}
}

func (store *RedisResponseTerminalStore) testKeyPrefix() string {
	return "vsr:response-terminal:" + store.keyNamespace
}

func responseTerminalRedisTestOptions(t *testing.T) *redis.Options {
	t.Helper()
	raw := os.Getenv("VLLM_SR_RESPONSE_TERMINAL_TEST_REDIS_URL")
	if raw == "" {
		raw = os.Getenv("VLLM_SR_USAGE_LEDGER_TEST_REDIS_URL")
	}
	if raw == "" {
		t.Skip("response terminal Redis integration store is not configured")
	}
	options, err := redis.ParseURL(raw)
	if err != nil {
		t.Fatal(err)
	}
	return options
}

func newResponseTerminalRedisTestStore(
	t *testing.T,
	client redis.UniversalClient,
	prefix string,
) *RedisResponseTerminalStore {
	t.Helper()
	store, err := NewRedisResponseTerminalStore(RedisResponseTerminalStoreOptions{
		Client: client, KeyPrefix: prefix, TTL: time.Minute,
	})
	if err != nil {
		t.Fatal(err)
	}
	return store
}

func deleteResponseTerminalTestKeys(client *redis.Client, prefix string) {
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
