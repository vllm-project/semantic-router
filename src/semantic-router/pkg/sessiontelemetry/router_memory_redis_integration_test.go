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
	if err := first.client.Ping(ctx).Err(); err != nil {
		storagetest.Unavailable(t, "redis", err)
	}

	now := time.Now().UTC()
	// Production persists through Merge even when the key does not exist yet.
	if err := first.Merge(RouterSessionSnapshot{SessionID: freshID, CurrentModel: "first-model", LastSeen: now}, time.Minute); err != nil {
		t.Fatalf("Merge fresh session: %v", err)
	}
	if loaded, found, err := second.Load(freshID); err != nil || !found || loaded.CurrentModel != "first-model" {
		t.Fatalf("Load fresh merged session: found=%t err=%v snapshot=%+v", found, err, loaded)
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
	if err := first.Save(seed, sessionTTL); err != nil {
		t.Fatalf("Save seed: %v", err)
	}
	if loaded, found, err := second.Load(sessionID); err != nil || !found || loaded.CurrentModel != seed.CurrentModel {
		t.Fatalf("Load before merge: found=%t err=%v snapshot=%+v", found, err, loaded)
	}

	local := seed
	local.CurrentModel = "local-model"
	local.LastSeen = now
	local.RecentOutcomes = []TurnOutcome{mergeTestOutcome("local", now)}
	if err := second.Merge(local, sessionTTL); err != nil {
		t.Fatalf("Merge from second replica: %v", err)
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
		if err := <-results; err != nil {
			t.Fatalf("concurrent Merge: %v", err)
		}
	}
	final, found, err := second.Load(sessionID)
	if err != nil || !found {
		t.Fatalf("Load after concurrent merges: found=%t err=%v", found, err)
	}
	assertSessionOutcomes(t, final, "seed", "local", "left", "right")
	if err := second.Merge(right, sessionTTL); err != nil {
		t.Fatalf("replayed Merge: %v", err)
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
	if _, readable, err := decodeRedisRouterSessionSnapshot(stored, sessionID); err != nil || !readable {
		t.Fatalf("merged bytes are not in the Load codec: readable=%t err=%v", readable, err)
	}
	if ttl, err := first.client.PTTL(ctx, key).Result(); err != nil || ttl <= 0 || ttl > sessionTTL {
		t.Fatalf("Merge did not preserve requested TTL: ttl=%v err=%v", ttl, err)
	}

	// A legacy or foreign value must survive a rejected merge unchanged.
	foreignKey := config.KeyPrefix + foreignID
	foreignPayload := []byte(`{"SessionID":"foreign-encoding"}`)
	if err := first.client.Set(ctx, foreignKey, foreignPayload, time.Minute).Err(); err != nil {
		t.Fatalf("seed foreign payload: %v", err)
	}
	if err := second.Merge(RouterSessionSnapshot{SessionID: foreignID, LastSeen: now}, sessionTTL); err == nil {
		t.Fatal("Merge overwrote a payload outside the store codec")
	}
	if got, err := first.client.Get(ctx, foreignKey).Bytes(); err != nil || string(got) != string(foreignPayload) {
		t.Fatalf("rejected merge changed stored payload: got=%q err=%v", got, err)
	}

	// Verify Redis expiry, as well as the envelope shape, on the real store.
	if err := first.Save(RouterSessionSnapshot{SessionID: expiryID, LastSeen: now}, 300*time.Millisecond); err != nil {
		t.Fatalf("Save expiring snapshot: %v", err)
	}
	deadline := time.Now().Add(3 * time.Second)
	for {
		_, found, err := second.Load(expiryID)
		if err != nil {
			t.Fatalf("Load expiring snapshot: %v", err)
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
