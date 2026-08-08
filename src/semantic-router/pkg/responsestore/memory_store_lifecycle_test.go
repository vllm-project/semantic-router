package responsestore

import (
	"runtime"
	"testing"
	"time"
)

// TestMemoryStoreCloseStopsExpirySweep asserts Close ends the background expiry
// goroutine. Every store starts one, and a router reload builds a replacement
// store and drops the old one, so a sweep that outlives Close accumulates one
// live goroutine per reload for the lifetime of the process — regardless of
// which backend the operator configured.
func TestMemoryStoreCloseStopsExpirySweep(t *testing.T) {
	baseline := settledGoroutineCount(t)

	const iterations = 20
	for i := 0; i < iterations; i++ {
		store, err := NewMemoryStore(StoreConfig{Enabled: true, TTLSeconds: 60})
		if err != nil {
			t.Fatalf("NewMemoryStore() error = %v", err)
		}
		if err := store.Close(); err != nil {
			t.Fatalf("MemoryStore.Close() error = %v", err)
		}
	}

	if got := settledGoroutineCount(t); got > baseline {
		t.Fatalf("goroutine count grew from %d to %d across %d store lifecycles;"+
			" the expiry sweep outlives Close", baseline, got, iterations)
	}
}

// TestMemoryStoreCloseIsIdempotent guards the closeOnce: the store is reachable
// from both the router's field-by-field teardown and its generation, and closing
// an already-closed channel panics the process.
func TestMemoryStoreCloseIsIdempotent(t *testing.T) {
	store, err := NewMemoryStore(StoreConfig{Enabled: true})
	if err != nil {
		t.Fatalf("NewMemoryStore() error = %v", err)
	}

	for i := 0; i < 3; i++ {
		if err := store.Close(); err != nil {
			t.Fatalf("MemoryStore.Close() call %d error = %v", i+1, err)
		}
	}
}

// settledGoroutineCount polls until the goroutine count stops moving, so a
// goroutine still winding down from an earlier store is not counted as a leak.
func settledGoroutineCount(t *testing.T) int {
	t.Helper()

	deadline := time.Now().Add(10 * time.Second)
	last := -1
	consecutive := 0
	for time.Now().Before(deadline) {
		runtime.GC()
		current := runtime.NumGoroutine()
		if current == last {
			consecutive++
			if consecutive >= 3 {
				return last
			}
		} else {
			consecutive = 0
			last = current
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("goroutine count never settled (last = %d)", last)
	return last
}
