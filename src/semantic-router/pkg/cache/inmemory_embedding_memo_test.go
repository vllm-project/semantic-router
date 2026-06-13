package cache

import (
	"errors"
	"sync"
	"testing"
)

// A cache-miss request embeds the same query twice today: once in
// FindSimilarWithThreshold (lookup) and once in AddPendingRequest (pending
// write). The memo must compute a given query's embedding only once and serve
// the rest from memory.
func TestEmbeddingMemoComputesOncePerKey(t *testing.T) {
	m := newEmbeddingMemo(8)
	calls := 0
	compute := func(text string) ([]float32, error) {
		calls++
		return []float32{float32(len(text))}, nil
	}

	for i := 0; i < 3; i++ {
		v, err := m.getOrCompute("hello", compute)
		if err != nil {
			t.Fatalf("getOrCompute error: %v", err)
		}
		if len(v) != 1 || v[0] != 5 {
			t.Fatalf("unexpected embedding %v", v)
		}
	}
	if calls != 1 {
		t.Fatalf("expected compute called once, got %d", calls)
	}
}

// Distinct queries are computed independently.
func TestEmbeddingMemoDistinctKeys(t *testing.T) {
	m := newEmbeddingMemo(8)
	calls := 0
	compute := func(string) ([]float32, error) { calls++; return []float32{1}, nil }
	_, _ = m.getOrCompute("a", compute)
	_, _ = m.getOrCompute("b", compute)
	if calls != 2 {
		t.Fatalf("expected 2 computes for 2 distinct keys, got %d", calls)
	}
}

// The memo is bounded: past maxSize, the oldest key is evicted (and recomputed
// on next access), so memory stays capped.
func TestEmbeddingMemoEvictsOldestPastMax(t *testing.T) {
	m := newEmbeddingMemo(2)
	compute := func(string) ([]float32, error) { return []float32{1}, nil }
	_, _ = m.getOrCompute("a", compute)
	_, _ = m.getOrCompute("b", compute)
	_, _ = m.getOrCompute("c", compute) // evicts "a"

	if _, ok := m.get("a"); ok {
		t.Fatal("expected oldest key 'a' to be evicted")
	}
	if _, ok := m.get("c"); !ok {
		t.Fatal("expected newest key 'c' to be present")
	}
}

// A compute error is propagated and NOT memoized (next call retries).
func TestEmbeddingMemoDoesNotMemoizeErrors(t *testing.T) {
	m := newEmbeddingMemo(4)
	boom := errors.New("boom")
	calls := 0
	compute := func(string) ([]float32, error) {
		calls++
		if calls == 1 {
			return nil, boom
		}
		return []float32{2}, nil
	}

	if _, err := m.getOrCompute("x", compute); !errors.Is(err, boom) {
		t.Fatalf("expected boom error, got %v", err)
	}
	v, err := m.getOrCompute("x", compute) // must retry, not serve a cached error
	if err != nil || len(v) != 1 || v[0] != 2 {
		t.Fatalf("expected retry to succeed, got v=%v err=%v", v, err)
	}
}

// Concurrent access is safe (run with -race).
func TestEmbeddingMemoConcurrentSafe(t *testing.T) {
	m := newEmbeddingMemo(64)
	compute := func(text string) ([]float32, error) { return []float32{float32(len(text))}, nil }
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				_, _ = m.getOrCompute("shared-query", compute)
			}
		}()
	}
	wg.Wait()
	if _, ok := m.get("shared-query"); !ok {
		t.Fatal("expected shared-query to be memoized")
	}
}
