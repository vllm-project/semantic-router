package sessiontelemetry

import (
	"context"
	"fmt"
	"net"
	"os"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
)

// StorageIntegration: redis
func TestRedisSessionStoreMergeRoundTrip(t *testing.T) {
	storagetest.Require(t, "redis")
	host, port := os.Getenv("REDIS_HOST"), os.Getenv("REDIS_PORT")
	if host == "" {
		host = "127.0.0.1"
	}
	if port == "" {
		port = "6379"
	}
	config := RedisRouterSessionStoreConfig{
		Address:   net.JoinHostPort(host, port),
		Password:  os.Getenv("REDIS_PASSWORD"),
		Timeout:   5 * time.Second,
		KeyPrefix: fmt.Sprintf("vsr:test:session-merge:%d:", time.Now().UnixNano()),
	}
	firstStore, err := NewRedisRouterSessionStateStore(config)
	if err != nil {
		t.Fatal(err)
	}
	first := firstStore.(*redisRouterSessionStore)
	secondStore, err := NewRedisRouterSessionStateStore(config)
	if err != nil {
		_ = first.Close()
		t.Fatal(err)
	}
	second := secondStore.(*redisRouterSessionStore)
	const (
		sessionID = "shared-session"
		freshID   = "fresh-session"
		expiryID  = "expiring-session"
		foreignID = "foreign-encoding"
	)
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = first.client.Del(ctx, config.KeyPrefix+sessionID, config.KeyPrefix+freshID, config.KeyPrefix+expiryID, config.KeyPrefix+foreignID).Err()
		_ = first.Close()
		_ = second.Close()
	})
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if pingErr := first.client.Ping(ctx).Err(); pingErr != nil {
		storagetest.Unavailable(t, "redis", pingErr)
	}

	// Keep synthetic event times in the past: the outcome window deliberately
	// drops future timestamps, and the two replica offsets below are positive.
	now := time.Now().UTC().Add(-5 * time.Second)
	// Production persists through Merge even when the key does not exist yet.
	if mergeErr := first.Merge(RouterSessionSnapshot{SessionID: freshID, CurrentModel: "first-model", LastSeen: now}, time.Minute); mergeErr != nil {
		t.Fatalf("Merge fresh session: %v", mergeErr)
	}
	if loaded, found, loadErr := second.Load(freshID); loadErr != nil || !found || loaded.CurrentModel != "first-model" {
		t.Fatalf("Load fresh merged session: found=%t err=%v snapshot=%+v", found, loadErr, loaded)
	}

	seed := RouterSessionSnapshot{
		SessionID:               sessionID,
		CurrentModel:            "remote-model",
		LastSeen:                now.Add(-time.Minute),
		OutcomeWindowSize:       16,
		OutcomeWindowTTLSeconds: 3600,
		RecentOutcomes:          []TurnOutcome{mergeTestOutcome("seed", now.Add(-time.Minute))},
	}
	const sessionTTL = 30 * time.Second
	if saveErr := first.Save(seed, sessionTTL); saveErr != nil {
		t.Fatalf("Save seed: %v", saveErr)
	}
	if loaded, found, loadErr := second.Load(sessionID); loadErr != nil || !found || loaded.CurrentModel != seed.CurrentModel {
		t.Fatalf("Load before merge: found=%t err=%v snapshot=%+v", found, loadErr, loaded)
	}

	local := seed
	local.CurrentModel = "local-model"
	local.LastSeen = now
	local.RecentOutcomes = []TurnOutcome{mergeTestOutcome("local", now)}
	if mergeErr := second.Merge(local, sessionTTL); mergeErr != nil {
		t.Fatalf("Merge from second replica: %v", mergeErr)
	}
	merged, found, err := first.Load(sessionID)
	if err != nil || !found || merged.CurrentModel != local.CurrentModel {
		t.Fatalf("Load after merge: found=%t err=%v snapshot=%+v", found, err, merged)
	}
	assertSessionOutcomes(t, merged, "seed", "local")

	// Two replicas read the same base, then add different facts concurrently.
	left, found, err := first.Load(sessionID)
	if err != nil || !found {
		t.Fatalf("Load left replica: found=%t err=%v", found, err)
	}
	right, found, err := second.Load(sessionID)
	if err != nil || !found {
		t.Fatalf("Load right replica: found=%t err=%v", found, err)
	}
	left.LastSeen = now.Add(time.Second)
	left.RecentOutcomes = append(left.RecentOutcomes, mergeTestOutcome("left", left.LastSeen))
	right.LastSeen = now.Add(2 * time.Second)
	right.RecentOutcomes = append(right.RecentOutcomes, mergeTestOutcome("right", right.LastSeen))
	start := make(chan struct{})
	results := make(chan error, 2)
	go func() { <-start; results <- first.Merge(left, sessionTTL) }()
	go func() { <-start; results <- second.Merge(right, sessionTTL) }()
	close(start)
	for i := 0; i < 2; i++ {
		if mergeErr := <-results; mergeErr != nil {
			t.Fatalf("concurrent Merge: %v", mergeErr)
		}
	}
	final, found, err := second.Load(sessionID)
	if err != nil || !found {
		t.Fatalf("Load after concurrent merges: found=%t err=%v", found, err)
	}
	assertSessionOutcomes(t, final, "seed", "local", "left", "right")
	if mergeErr := second.Merge(right, sessionTTL); mergeErr != nil {
		t.Fatalf("replayed Merge: %v", mergeErr)
	}
	final, found, err = first.Load(sessionID)
	if err != nil || !found {
		t.Fatalf("Load after replay: found=%t err=%v", found, err)
	}
	assertSessionOutcomes(t, final, "seed", "local", "left", "right")

	key := config.KeyPrefix + sessionID
	stored, err := first.client.Get(ctx, key).Bytes()
	if err != nil {
		t.Fatalf("Get merged bytes: %v", err)
	}
	if _, readable, decodeErr := decodeRedisRouterSessionSnapshot(stored, sessionID); decodeErr != nil || !readable {
		t.Fatalf("merged bytes are not in the Load codec: readable=%t err=%v", readable, decodeErr)
	}
	if remainingTTL, ttlErr := first.client.PTTL(ctx, key).Result(); ttlErr != nil || remainingTTL <= 0 || remainingTTL > sessionTTL {
		t.Fatalf("Merge did not preserve requested TTL: ttl=%v err=%v", remainingTTL, ttlErr)
	}

	// A legacy or foreign value must survive a rejected merge unchanged.
	foreignKey := config.KeyPrefix + foreignID
	foreignPayload := []byte(`{"SessionID":"foreign-encoding"}`)
	if setErr := first.client.Set(ctx, foreignKey, foreignPayload, time.Minute).Err(); setErr != nil {
		t.Fatalf("seed foreign payload: %v", setErr)
	}
	if mergeErr := second.Merge(RouterSessionSnapshot{SessionID: foreignID, LastSeen: now}, sessionTTL); mergeErr == nil {
		t.Fatal("Merge overwrote a payload outside the store codec")
	}
	if got, getErr := first.client.Get(ctx, foreignKey).Bytes(); getErr != nil || string(got) != string(foreignPayload) {
		t.Fatalf("rejected merge changed stored payload: got=%q err=%v", got, getErr)
	}

	// Verify Redis expiry, as well as the envelope shape, on the real store.
	if saveErr := first.Save(RouterSessionSnapshot{SessionID: expiryID, LastSeen: now}, 300*time.Millisecond); saveErr != nil {
		t.Fatalf("Save expiring snapshot: %v", saveErr)
	}
	deadline := time.Now().Add(3 * time.Second)
	for {
		_, found, loadErr := second.Load(expiryID)
		if loadErr != nil {
			t.Fatalf("Load expiring snapshot: %v", loadErr)
		}
		if !found {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("Redis snapshot did not expire")
		}
		time.Sleep(25 * time.Millisecond)
	}
}

func assertSessionOutcomes(t *testing.T, snapshot RouterSessionSnapshot, ids ...string) {
	t.Helper()
	got := make(map[string]int, len(snapshot.RecentOutcomes))
	for _, outcome := range snapshot.RecentOutcomes {
		got[outcome.RequestID]++
	}
	if len(got) != len(ids) || len(snapshot.RecentOutcomes) != len(ids) {
		t.Fatalf("outcome count=%d, want %d: %+v", len(snapshot.RecentOutcomes), len(ids), snapshot.RecentOutcomes)
	}
	for _, id := range ids {
		if got[id] != 1 {
			t.Fatalf("outcome %q appears %d times: %+v", id, got[id], snapshot.RecentOutcomes)
		}
	}
}
