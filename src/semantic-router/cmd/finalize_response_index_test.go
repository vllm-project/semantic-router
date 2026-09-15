package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responsestore"
)

const finalizeTestRedisAddr = "localhost:6379"

// TestExitIfFinalizeResponseIndexIsNoOpWhenDisabled covers the flag's default
// behavior: with the flag unset, the router's normal startup path must fall
// straight through without touching the config at all. A nil config would
// panic the moment anything looked at it, so surviving this call is the
// assertion.
func TestExitIfFinalizeResponseIndexIsNoOpWhenDisabled(t *testing.T) {
	exitIfFinalizeResponseIndex(false, nil)
}

// TestRunResponseIndexFinalizationRejectsNonRedisBackend covers the guard
// that keeps this command from reporting success against a deployment whose
// Response API has no conversation index to finalize at all.
func TestRunResponseIndexFinalizationRejectsNonRedisBackend(t *testing.T) {
	for _, backend := range []string{"memory", "file", ""} {
		cfg := &config.RouterConfig{}
		cfg.ResponseAPI.StoreBackend = backend

		_, err := runResponseIndexFinalization(context.Background(), cfg)
		if !errors.Is(err, errFinalizeRequiresRedis) {
			t.Fatalf("backend %q: error = %v, want errFinalizeRequiresRedis", backend, err)
		}
	}
}

// TestRunResponseIndexFinalizationIndexesLegacyResponses is the end-to-end
// command path — what `--finalize-response-index` actually does once the
// flag branch is taken. Legacy payloads that no index-aware write ever
// covered become discoverable through the conversation index, the reported
// stats match what was swept, and a second run is an idempotent no-op.
func TestRunResponseIndexFinalizationIndexesLegacyResponses(t *testing.T) {
	keyPrefix := fmt.Sprintf("srtestcmd:%d:", time.Now().UnixNano())
	cfg := &config.RouterConfig{}
	cfg.ResponseAPI.StoreBackend = "redis"
	cfg.ResponseAPI.TTLSeconds = 300
	cfg.ResponseAPI.Redis.Address = finalizeTestRedisAddr
	cfg.ResponseAPI.Redis.KeyPrefix = keyPrefix

	raw := redis.NewClient(&redis.Options{Addr: finalizeTestRedisAddr})
	ctx := context.Background()
	if err := raw.Ping(ctx).Err(); err != nil {
		t.Skipf("Redis not available: %v", err)
	}
	t.Cleanup(func() {
		iter := raw.Scan(ctx, 0, keyPrefix+"*", 0).Iterator()
		for iter.Next(ctx) {
			raw.Del(ctx, iter.Val())
		}
		_ = raw.Close()
	})

	now := time.Now().Unix()
	for _, response := range []*responseapi.StoredResponse{
		{ID: "resp_cmd_a1", ConversationID: "conv_cmd_a", Status: "completed", CreatedAt: now},
		{ID: "resp_cmd_a2", ConversationID: "conv_cmd_a", Status: "completed", CreatedAt: now + 1},
		{ID: "resp_cmd_b1", ConversationID: "conv_cmd_b", Status: "completed", CreatedAt: now},
	} {
		writeUnindexedResponse(t, raw, keyPrefix, response)
	}

	stats, err := runResponseIndexFinalization(ctx, cfg)
	if err != nil {
		t.Fatalf("runResponseIndexFinalization() error = %v", err)
	}
	if stats.ResponsesScanned != 3 || stats.ResponsesIndexed != 3 {
		t.Fatalf("stats = %+v, want scanned=3 indexed=3", stats)
	}

	assertConversationLists(t, cfg, "conv_cmd_a", "resp_cmd_a1", "resp_cmd_a2")
	assertFinalizationIsIdempotent(t, cfg)
}

// assertConversationLists checks that a conversation's responses are
// reachable through the ordinary read path, not just present in the raw
// index the sweep wrote.
func assertConversationLists(t *testing.T, cfg *config.RouterConfig, conversationID string, wantIDs ...string) {
	t.Helper()

	reader, err := newResponseIndexStore(cfg)
	if err != nil {
		t.Fatalf("newResponseIndexStore() error = %v", err)
	}
	t.Cleanup(func() { _ = reader.Close() })

	responses, err := reader.ListResponsesByConversation(
		context.Background(), conversationID, responsestore.ListOptions{Order: "asc"})
	if err != nil {
		t.Fatalf("ListResponsesByConversation() error = %v", err)
	}

	gotIDs := make([]string, len(responses))
	for i, response := range responses {
		gotIDs[i] = response.ID
	}
	if !slices.Equal(gotIDs, wantIDs) {
		t.Fatalf("listed responses = %v, want %v", gotIDs, wantIDs)
	}
}

// assertFinalizationIsIdempotent checks that a second run sees the store
// already finalized and sweeps nothing, rather than repeating the whole
// keyspace scan.
func assertFinalizationIsIdempotent(t *testing.T, cfg *config.RouterConfig) {
	t.Helper()

	again, err := runResponseIndexFinalization(context.Background(), cfg)
	if err != nil {
		t.Fatalf("second runResponseIndexFinalization() error = %v", err)
	}
	if again.ResponsesScanned != 0 || again.ResponsesIndexed != 0 {
		t.Fatalf("second run stats = %+v, want zero (already finalized)", again)
	}
}

// writeUnindexedResponse writes a response payload straight to Redis with no
// conversation index entry — the shape of data written before the secondary
// index existed, and the only thing the finalization sweep can discover that
// an ordinary read could not.
func writeUnindexedResponse(t *testing.T, client *redis.Client, keyPrefix string, response *responseapi.StoredResponse) {
	t.Helper()

	payload, err := json.Marshal(response)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	key := keyPrefix + responsestore.ResponseKeyPrefix + response.ID
	if err := client.Set(context.Background(), key, payload, time.Hour).Err(); err != nil {
		t.Fatalf("Set(%s) error = %v", key, err)
	}
}
