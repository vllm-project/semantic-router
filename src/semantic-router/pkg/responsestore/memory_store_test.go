package responsestore

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

func newChainTestStore(t *testing.T) *MemoryStore {
	t.Helper()

	store, err := NewMemoryStore(StoreConfig{Enabled: true, TTLSeconds: 60})
	if err != nil {
		t.Fatalf("NewMemoryStore() error = %v", err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Errorf("MemoryStore.Close() error = %v", err)
		}
	})
	return store
}

func mustStoreResponse(t *testing.T, store *MemoryStore, response *responseapi.StoredResponse) {
	t.Helper()

	if err := store.StoreResponse(context.Background(), response); err != nil {
		t.Fatalf("StoreResponse(%s) error = %v", response.ID, err)
	}
}

// getConversationChainWithTimeout fails the test instead of hanging it if the
// traversal never terminates.
func getConversationChainWithTimeout(t *testing.T, store *MemoryStore, responseID string) ([]*responseapi.StoredResponse, error) {
	t.Helper()

	type result struct {
		chain []*responseapi.StoredResponse
		err   error
	}
	done := make(chan result, 1)
	go func() {
		chain, err := store.GetConversationChain(context.Background(), responseID)
		done <- result{chain: chain, err: err}
	}()

	select {
	case res := <-done:
		return res.chain, res.err
	case <-time.After(5 * time.Second):
		t.Fatalf("GetConversationChain(%s) hung - circular reference protection failed", responseID)
		return nil, nil
	}
}

func TestMemoryStoreGetConversationChain(t *testing.T) {
	store := newChainTestStore(t)

	mustStoreResponse(t, store, &responseapi.StoredResponse{ID: "resp_1"})
	mustStoreResponse(t, store, &responseapi.StoredResponse{ID: "resp_2", PreviousResponseID: "resp_1"})
	mustStoreResponse(t, store, &responseapi.StoredResponse{ID: "resp_3", PreviousResponseID: "resp_2"})

	chain, err := getConversationChainWithTimeout(t, store, "resp_3")
	if err != nil {
		t.Fatalf("GetConversationChain(resp_3) error = %v", err)
	}
	if len(chain) != 3 {
		t.Fatalf("chain length = %d, want 3", len(chain))
	}
	for i, wantID := range []string{"resp_1", "resp_2", "resp_3"} {
		if chain[i].ID != wantID {
			t.Errorf("chain[%d].ID = %q, want %q", i, chain[i].ID, wantID)
		}
	}
}

func TestMemoryStoreGetConversationChainCircularReference(t *testing.T) {
	store := newChainTestStore(t)

	mustStoreResponse(t, store, &responseapi.StoredResponse{ID: "resp_circular_1", PreviousResponseID: "resp_circular_2"})
	mustStoreResponse(t, store, &responseapi.StoredResponse{ID: "resp_circular_2", PreviousResponseID: "resp_circular_1"})

	chain, err := getConversationChainWithTimeout(t, store, "resp_circular_1")
	if err == nil {
		t.Fatalf("GetConversationChain(resp_circular_1) = %d entries, err = nil; want an error", len(chain))
	}

	// The store must stay writable after walking a circular chain.
	mustStoreResponse(t, store, &responseapi.StoredResponse{ID: "resp_after_cycle"})
}

func TestMemoryStoreGetConversationChainSelfReference(t *testing.T) {
	store := newChainTestStore(t)

	mustStoreResponse(t, store, &responseapi.StoredResponse{ID: "resp_self", PreviousResponseID: "resp_self"})

	chain, err := getConversationChainWithTimeout(t, store, "resp_self")
	if err == nil {
		t.Fatalf("GetConversationChain(resp_self) = %d entries, err = nil; want an error", len(chain))
	}
}

// An ancestor that expired before its descendants leaves the chain incomplete:
// continuing with the remainder answers a shorter conversation as if it were
// the whole one.
func TestMemoryStoreGetConversationChainExpiredAncestor(t *testing.T) {
	store := newChainTestStore(t)

	mustStoreResponse(t, store, &responseapi.StoredResponse{ID: "resp_root_expired", TTL: time.Now().Add(-time.Minute)})
	mustStoreResponse(t, store, &responseapi.StoredResponse{ID: "resp_mid", PreviousResponseID: "resp_root_expired"})
	mustStoreResponse(t, store, &responseapi.StoredResponse{ID: "resp_leaf", PreviousResponseID: "resp_mid"})

	chain, err := store.GetConversationChain(context.Background(), "resp_leaf")
	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetConversationChain(resp_leaf) = %d entries, err = %v; want ErrNotFound", len(chain), err)
	}
}

// The cap is a loop guard, not a truncation policy: a walk that stops with
// history still ahead of it cannot answer as if it had reached the start.
func TestMemoryStoreGetConversationChainRejectsAChainPastTheCap(t *testing.T) {
	store := newChainTestStore(t)
	storeChain := func(length int) string {
		previous := ""
		leaf := ""
		for index := 0; index < length; index++ {
			id := fmt.Sprintf("resp_cap_%d_%d", length, index)
			mustStoreResponse(t, store, &responseapi.StoredResponse{ID: id, PreviousResponseID: previous})
			previous, leaf = id, id
		}
		return leaf
	}

	chain, err := store.GetConversationChain(context.Background(), storeChain(1000))
	if err != nil || len(chain) != 1000 {
		t.Fatalf("a chain of exactly the cap = %d entries, err = %v; want 1000 and no error", len(chain), err)
	}

	chain, err = store.GetConversationChain(context.Background(), storeChain(1001))
	if err == nil {
		t.Fatalf("a chain past the cap = %d entries, err = nil; want an error", len(chain))
	}
}
