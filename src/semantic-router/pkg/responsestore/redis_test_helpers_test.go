package responsestore

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

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
func responseIDsOf(responses []*responseapi.StoredResponse) []string {
	ids := make([]string, len(responses))
	for i, r := range responses {
		ids[i] = r.ID
	}
	return ids
}
