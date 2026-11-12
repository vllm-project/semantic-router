package responsestore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
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

func TestRedisStoreGetConversationChainRejectsACycle(t *testing.T) {
	store := newMiniredisStore(t)
	storeChain(t, store, "r1", "r2")

	// r1 -> r2 -> r1: the walk revisits r1 instead of reaching a root. Stored
	// responses never change their link, so the cycle is written directly.
	ctx := context.Background()
	payload, err := json.Marshal(&responseapi.StoredResponse{ID: "r1", PreviousResponseID: "r2"})
	if err != nil {
		t.Fatalf("marshal r1: %v", err)
	}
	if setErr := store.client.Set(ctx, store.buildKey(ResponseKeyPrefix+"r1"), payload, time.Minute).Err(); setErr != nil {
		t.Fatalf("rewrite r1: %v", setErr)
	}

	chain, err := store.GetConversationChain(ctx, "r1")
	if err == nil {
		t.Fatalf("GetConversationChain(r1) = %d entries, err = nil; want an error", len(chain))
	}
}

func TestRedisStoreGetConversationChainRejectsAChainPastTheCap(t *testing.T) {
	store := newMiniredisStore(t)
	ctx := context.Background()

	previous := ""
	leaf := ""
	for index := 0; index <= 1000; index++ {
		id := fmt.Sprintf("r%d", index)
		response := &responseapi.StoredResponse{ID: id}
		if previous != "" {
			response.PreviousResponseID = previous
		}
		if err := store.StoreResponse(ctx, response); err != nil {
			t.Fatalf("StoreResponse(%s): %v", id, err)
		}
		previous, leaf = id, id
	}

	chain, err := store.GetConversationChain(ctx, leaf)
	if err == nil {
		t.Fatalf("a chain past the cap = %d entries, err = nil; want an error", len(chain))
	}
}
