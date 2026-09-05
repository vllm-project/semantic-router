package responsestore

import (
	"context"
	"encoding/json"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

type pipelineFailureHook struct {
	name string
	key  string
	err  error
	used atomic.Bool
}

func (h *pipelineFailureHook) DialHook(next redis.DialHook) redis.DialHook          { return next }
func (h *pipelineFailureHook) ProcessHook(next redis.ProcessHook) redis.ProcessHook { return next }
func (h *pipelineFailureHook) ProcessPipelineHook(next redis.ProcessPipelineHook) redis.ProcessPipelineHook {
	return func(ctx context.Context, cmds []redis.Cmder) error {
		for _, cmd := range cmds {
			args := cmd.Args()
			if cmd.Name() == h.name && len(args) > 1 && args[1] == h.key && h.used.CompareAndSwap(false, true) {
				for _, pipelineCmd := range cmds {
					pipelineCmd.SetErr(h.err)
				}
				return h.err
			}
		}
		return next(ctx, cmds)
	}
}

type beforeCommandHook struct {
	name   string
	once   bool
	before func()
	used   atomic.Bool
}

func (h *beforeCommandHook) DialHook(next redis.DialHook) redis.DialHook { return next }
func (h *beforeCommandHook) ProcessHook(next redis.ProcessHook) redis.ProcessHook {
	return func(ctx context.Context, cmd redis.Cmder) error {
		if cmd.Name() == h.name && (!h.once || h.used.CompareAndSwap(false, true)) {
			h.before()
		}
		return next(ctx, cmd)
	}
}
func (h *beforeCommandHook) ProcessPipelineHook(next redis.ProcessPipelineHook) redis.ProcessPipelineHook {
	return next
}

// conversationIndexMembers reads the index directly, so tests can assert on it
// and not only on what a listing happens to return.
func conversationIndexMembers(t *testing.T, store *RedisStore, conversationID string) []string {
	t.Helper()

	members, err := store.client.ZRange(context.Background(), store.conversationIndexKey(conversationID), 0, -1).Result()
	require.NoError(t, err)
	return members
}

// newConversationIndexStore scopes the store to a key prefix unique to this run,
// so it cannot collide with the other suites sharing DB 0. Skips without Redis.
func newConversationIndexStore(t *testing.T) *RedisStore {
	t.Helper()
	return newConversationIndexStoreWithTTLSeconds(t, 300)
}

// newConversationIndexStoreWithTTLSeconds is newConversationIndexStore with
// an explicit data-retention TTL, for tests asserting behavior that depends
// on how the configured TTL compares to a fixed internal bound (e.g. the
// empty-marker TTL cap).
func newConversationIndexStoreWithTTLSeconds(t *testing.T, ttlSeconds int) *RedisStore {
	t.Helper()

	cfg := StoreConfig{
		Enabled:     true,
		TTLSeconds:  ttlSeconds,
		BackendType: RedisStoreType,
		Redis: RedisStoreConfig{
			Address:   "localhost:6379",
			DB:        0,
			KeyPrefix: fmt.Sprintf("srtest:%d:", time.Now().UnixNano()),
		},
	}

	store, err := NewRedisStore(cfg)
	if err != nil {
		t.Skipf("Redis not available: %v", err)
	}

	t.Cleanup(func() {
		ctx := context.Background()
		iter := store.client.Scan(ctx, 0, store.buildKey("*"), 0).Iterator()
		for iter.Next(ctx) {
			store.client.Del(ctx, iter.Val())
		}
		_ = store.Close()
	})

	return store
}

// directSetResponsePayload writes a response payload straight to Redis,
// bypassing StoreResponse and its indexing entirely — the same shape as data
// written before this index existed, or as an in-flight writer that landed
// its payload but has not indexed it yet.
func directSetResponsePayload(t *testing.T, store *RedisStore, response *responseapi.StoredResponse) {
	t.Helper()

	data, err := json.Marshal(response)
	require.NoError(t, err)

	key := store.buildKey(ResponseKeyPrefix + response.ID)
	require.NoError(t, store.client.Set(context.Background(), key, data, store.ttl).Err())
}

// exists reports whether a raw Redis key exists, for asserting on marker/
// index presence directly rather than only through store methods.
func exists(t *testing.T, store *RedisStore, key string) int64 {
	t.Helper()
	n, err := store.client.Exists(context.Background(), key).Result()
	require.NoError(t, err)
	return n
}

// seedPageResponses stores count responses in convID with strictly
// increasing CreatedAt (now+i) and IDs "resp_page_<i>", so ascending index
// order is resp_page_0..resp_page_(count-1) and descending is the reverse.
func seedPageResponses(t *testing.T, store *RedisStore, convID string, count int) []string {
	t.Helper()
	ctx := context.Background()
	now := time.Now().Unix()

	ids := make([]string, count)
	for i := 0; i < count; i++ {
		id := fmt.Sprintf("resp_page_%d", i)
		ids[i] = id
		require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
			ID:             id,
			ConversationID: convID,
			Status:         "completed",
			CreatedAt:      now + int64(i),
		}))
	}
	return ids
}

// clusterTestAddr is the seed address of a single-node Redis Cluster
// started alongside this suite's standalone sr-test-redis instance,
// specifically for tests that must prove no command they issue is ever
// cross-slot: Redis enforces per-command slot-locality independent of node
// count, so even one node in cluster mode genuinely rejects a multi-key
// command (e.g. MGET) spanning different slots with CROSSSLOT, making it a
// faithful (if minimal) Cluster-safety testbed.
const clusterTestAddr = "127.0.0.1:7000"

// newConversationIndexClusterStore builds a RedisStore backed by a real
// Redis Cluster client (see clusterTestAddr). Skips (not fails) if that
// cluster isn't reachable, since most environments running this package's
// tests only provide the standalone sr-test-redis container.
func newConversationIndexClusterStore(t *testing.T) *RedisStore {
	t.Helper()

	cfg := StoreConfig{
		Enabled:     true,
		TTLSeconds:  300,
		BackendType: RedisStoreType,
		Redis: RedisStoreConfig{
			ClusterMode:      true,
			ClusterAddresses: []string{clusterTestAddr},
			KeyPrefix:        fmt.Sprintf("srtestcluster:%d:", time.Now().UnixNano()),
		},
	}

	store, err := NewRedisStore(cfg)
	if err != nil {
		t.Skipf("Redis Cluster not available at %s: %v", clusterTestAddr, err)
	}

	t.Cleanup(func() {
		ctx := context.Background()
		clusterClient, ok := store.client.(interface {
			ForEachMaster(ctx context.Context, fn func(context.Context, *redis.Client) error) error
		})
		if ok {
			_ = clusterClient.ForEachMaster(ctx, func(ctx context.Context, master *redis.Client) error {
				iter := master.Scan(ctx, 0, store.buildKey("*"), 0).Iterator()
				for iter.Next(ctx) {
					master.Del(ctx, iter.Val())
				}
				return nil
			})
		}
		_ = store.Close()
	})

	return store
}

// mustMarshalResponse JSON-marshals a StoredResponse for tests writing a
// payload's raw bytes directly (e.g. repairing a deliberately corrupted
// payload, or constructing a raw value to compare-delete/compare-restore
// against), failing the test immediately on a marshal error rather than
// letting a malformed fixture masquerade as a real one.
func mustMarshalResponse(t *testing.T, response *responseapi.StoredResponse) []byte {
	t.Helper()
	data, err := json.Marshal(response)
	require.NoError(t, err)
	return data
}

func responseIDsOf(responses []*responseapi.StoredResponse) []string {
	ids := make([]string, len(responses))
	for i, r := range responses {
		ids[i] = r.ID
	}
	return ids
}
