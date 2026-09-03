package responsestore

import (
	"runtime"
	"testing"
	"time"
)

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
