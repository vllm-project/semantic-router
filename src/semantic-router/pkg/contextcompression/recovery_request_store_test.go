package contextcompression

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

// countingRecoveryStore counts the writes that reach the backing store.
type countingRecoveryStore struct {
	testRecoveryStore
	puts int
}

func (store *countingRecoveryStore) Put(ctx context.Context, entry RecoveryEntry, ttl time.Duration) error {
	store.puts++
	return store.testRecoveryStore.Put(ctx, entry, ttl)
}

// Two writes that each fit but together exceed the request's limit: the
// second is refused before it reaches the backing store.
func TestRequestRecoveryStoreRejectsAWriteBeyondTheSharedBudget(t *testing.T) {
	backing := &countingRecoveryStore{}
	store := NewRequestRecoveryStore(backing, 10)
	ctx := context.Background()

	if _, err := StoreRecoveryContent(ctx, store, "scope", strings.Repeat("a", 6), time.Minute); err != nil {
		t.Fatalf("first write within budget failed: %v", err)
	}
	_, err := StoreRecoveryContent(ctx, store, "scope", strings.Repeat("b", 6), time.Minute)
	if !errors.Is(err, ErrRecoveryRequestLimitExceeded) {
		t.Fatalf("second write past the budget returned %v", err)
	}
	if backing.puts != 1 {
		t.Fatalf("the backing store received %d writes, want only the one within budget", backing.puts)
	}
	if _, err := StoreRecoveryContent(ctx, store, "scope", strings.Repeat("c", 4), time.Minute); err != nil {
		t.Fatalf("a write that fits the remainder failed: %v", err)
	}
}

// Budget is spent only by writes the backing store accepted.
func TestRequestRecoveryStoreChargesOnlySuccessfulWrites(t *testing.T) {
	backing := &countingRecoveryStore{testRecoveryStore: testRecoveryStore{err: errors.New("backend down")}}
	store := NewRequestRecoveryStore(backing, 10)
	ctx := context.Background()

	if _, err := StoreRecoveryContent(ctx, store, "scope", strings.Repeat("a", 8), time.Minute); err == nil {
		t.Fatal("a failed backing write was reported as stored")
	}
	backing.err = nil
	if _, err := StoreRecoveryContent(ctx, store, "scope", strings.Repeat("b", 8), time.Minute); err != nil {
		t.Fatalf("a failed write consumed budget: %v", err)
	}
}

// The shared write derives the key from scope and content and stores an entry
// whose expiry matches its TTL. A TTL that is not positive is a caller error:
// this layer adds no default of its own.
func TestStoreRecoveryContentBuildsTheEntry(t *testing.T) {
	backing := &testRecoveryStore{}
	key, err := StoreRecoveryContent(context.Background(), backing, "scope", "content", time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if key != recoveryKey("scope", "content") {
		t.Fatal("the key is not derived from scope and content")
	}
	entry, err := backing.Get(context.Background(), "scope", key)
	if err != nil || entry.Content != "content" || entry.ExpiresAt.Sub(entry.CreatedAt) != time.Minute {
		t.Fatalf("stored entry %+v, %v", entry, err)
	}
	if _, err := StoreRecoveryContent(context.Background(), backing, "scope", "content", 0); err == nil {
		t.Fatal("a non-positive TTL was accepted")
	}
}
