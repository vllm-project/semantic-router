package responsestore

import (
	"context"
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
func getConversationChainWithTimeout(t *testing.T, store *MemoryStore, responseID string) []*responseapi.StoredResponse {
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
		if res.err != nil {
			t.Fatalf("GetConversationChain(%s) error = %v", responseID, res.err)
		}
		return res.chain
	case <-time.After(5 * time.Second):
		t.Fatalf("GetConversationChain(%s) hung - circular reference protection failed", responseID)
		return nil
	}
}

func TestMemoryStoreGetConversationChain(t *testing.T) {
	store := newChainTestStore(t)

	mustStoreResponse(t, store, &responseapi.StoredResponse{ID: "resp_1"})
	mustStoreResponse(t, store, &responseapi.StoredResponse{ID: "resp_2", PreviousResponseID: "resp_1"})
	mustStoreResponse(t, store, &responseapi.StoredResponse{ID: "resp_3", PreviousResponseID: "resp_2"})

	chain := getConversationChainWithTimeout(t, store, "resp_3")
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

	chain := getConversationChainWithTimeout(t, store, "resp_circular_1")
	if len(chain) != 2 {
		t.Fatalf("chain length = %d, want 2 (bounded partial chain)", len(chain))
	}

	// The store must stay writable after walking a circular chain.
	mustStoreResponse(t, store, &responseapi.StoredResponse{ID: "resp_after_cycle"})
}

func TestMemoryStoreGetConversationChainSelfReference(t *testing.T) {
	store := newChainTestStore(t)

	mustStoreResponse(t, store, &responseapi.StoredResponse{ID: "resp_self", PreviousResponseID: "resp_self"})

	chain := getConversationChainWithTimeout(t, store, "resp_self")
	if len(chain) != 1 {
		t.Fatalf("chain length = %d, want 1", len(chain))
	}
	if chain[0].ID != "resp_self" {
		t.Errorf("chain[0].ID = %q, want %q", chain[0].ID, "resp_self")
	}
}
