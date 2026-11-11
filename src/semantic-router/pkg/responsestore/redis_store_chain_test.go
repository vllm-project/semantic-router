package responsestore

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

func newMiniredisStore(t *testing.T) *RedisStore {
	t.Helper()

	server := miniredis.RunT(t)
	client := redis.NewClient(&redis.Options{Addr: server.Addr()})
	t.Cleanup(func() { _ = client.Close() })

	return &RedisStore{client: client, keyPrefix: "sr:", ttl: time.Minute, enabled: true}
}

func storeChain(t *testing.T, store *RedisStore, ids ...string) {
	t.Helper()

	ctx := context.Background()
	for index, id := range ids {
		response := &responseapi.StoredResponse{ID: id}
		if index > 0 {
			response.PreviousResponseID = ids[index-1]
		}
		if err := store.StoreResponse(ctx, response); err != nil {
			t.Fatalf("StoreResponse(%s): %v", id, err)
		}
	}
}

func TestRedisStoreGetConversationChainReadsEveryLinkOnce(t *testing.T) {
	store := newMiniredisStore(t)
	storeChain(t, store, "r1", "r2", "r3")

	chain, err := store.GetConversationChain(context.Background(), "r3")
	if err != nil {
		t.Fatalf("GetConversationChain(r3): %v", err)
	}
	if len(chain) != 3 {
		t.Fatalf("chain length = %d, want 3", len(chain))
	}
	for index, want := range []string{"r1", "r2", "r3"} {
		if chain[index].ID != want {
			t.Errorf("chain[%d].ID = %q, want %q", index, chain[index].ID, want)
		}
	}
}

func TestRedisStoreGetConversationChainGapFailsClosed(t *testing.T) {
	store := newMiniredisStore(t)
	storeChain(t, store, "r1", "r2", "r3")

	ctx := context.Background()
	if err := store.client.Del(ctx, store.buildKey(ResponseKeyPrefix+"r1")).Err(); err != nil {
		t.Fatalf("delete r1: %v", err)
	}

	chain, err := store.GetConversationChain(ctx, "r3")
	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetConversationChain(r3) = %d entries, err = %v; want ErrNotFound", len(chain), err)
	}
}
