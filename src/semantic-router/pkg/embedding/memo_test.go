package embedding

import (
	"errors"
	"fmt"
	"sync"
	"testing"
)

func TestMemoPutExistingKeyIsNoOp(t *testing.T) {
	memo := NewMemo(2)
	memo.Put("a", []float32{1})
	memo.Put("a", []float32{9})

	got, ok := memo.Get("a")
	if !ok || got[0] != 1 {
		t.Fatalf("Get(a) = (%v, %v), want the first value {1}", got, ok)
	}

	// The duplicate Put must not have consumed a FIFO slot either: with maxSize 2
	// one more key fits without evicting "a".
	memo.Put("b", []float32{2})
	if _, ok := memo.Get("a"); !ok {
		t.Fatalf("a was evicted, so the duplicate Put consumed a FIFO slot")
	}
	if _, ok := memo.Get("b"); !ok {
		t.Fatalf("b missing after Put")
	}
}

func TestMemoEvictsOldestAtCapacity(t *testing.T) {
	memo := NewMemo(3)
	memo.Put("a", []float32{1})
	memo.Put("b", []float32{2})
	memo.Put("c", []float32{3})
	memo.Put("d", []float32{4}) // evicts "a"

	if _, ok := memo.Get("a"); ok {
		t.Fatalf("oldest entry a survived eviction at capacity")
	}
	for _, key := range []string{"b", "c", "d"} {
		if _, ok := memo.Get(key); !ok {
			t.Fatalf("entry %q missing after eviction of the oldest entry", key)
		}
	}

	memo.Put("e", []float32{5}) // evicts "b"
	if _, ok := memo.Get("b"); ok {
		t.Fatalf("eviction is not FIFO: b survived while newer entries were added")
	}
}

// TestMemoConcurrentAccess is a smoke test for -race: many goroutines read and
// write overlapping keys while eviction churns the memo.
func TestMemoConcurrentAccess(t *testing.T) {
	memo := NewMemo(32)

	var wg sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			for i := 0; i < 200; i++ {
				key := fmt.Sprintf("key-%d", i%64)
				if v, ok := memo.Get(key); ok && len(v) == 0 {
					t.Errorf("memo returned an empty vector for %q", key)
					return
				}
				memo.Put(key, []float32{float32(worker), float32(i)})
			}
		}(worker)
	}
	wg.Wait()
}

// A cache-miss request can embed the same text twice (e.g. a semantic cache
// lookup followed by its pending write). GetOrCompute must compute a given key
// only once and serve the rest from memory.
func TestMemoGetOrComputeComputesOncePerKey(t *testing.T) {
	m := NewMemo(8)
	calls := 0
	compute := func() ([]float32, error) {
		calls++
		return []float32{5}, nil
	}

	for i := 0; i < 3; i++ {
		v, err := m.GetOrCompute("hello", compute)
		if err != nil {
			t.Fatalf("GetOrCompute error: %v", err)
		}
		if len(v) != 1 || v[0] != 5 {
			t.Fatalf("unexpected embedding %v", v)
		}
	}
	if calls != 1 {
		t.Fatalf("expected compute called once, got %d", calls)
	}
}

// Distinct keys are computed independently.
func TestMemoGetOrComputeDistinctKeys(t *testing.T) {
	m := NewMemo(8)
	calls := 0
	compute := func() ([]float32, error) { calls++; return []float32{1}, nil }
	_, _ = m.GetOrCompute("a", compute)
	_, _ = m.GetOrCompute("b", compute)
	if calls != 2 {
		t.Fatalf("expected 2 computes for 2 distinct keys, got %d", calls)
	}
}

// A compute error is propagated and NOT memoized (next call retries).
func TestMemoGetOrComputeDoesNotMemoizeErrors(t *testing.T) {
	m := NewMemo(4)
	boom := errors.New("boom")
	calls := 0
	compute := func() ([]float32, error) {
		calls++
		if calls == 1 {
			return nil, boom
		}
		return []float32{2}, nil
	}

	if _, err := m.GetOrCompute("x", compute); !errors.Is(err, boom) {
		t.Fatalf("expected boom error, got %v", err)
	}
	v, err := m.GetOrCompute("x", compute) // must retry, not serve a cached error
	if err != nil || len(v) != 1 || v[0] != 2 {
		t.Fatalf("expected retry to succeed, got v=%v err=%v", v, err)
	}
}

// Concurrent GetOrCompute is safe (run with -race).
func TestMemoGetOrComputeConcurrentSafe(t *testing.T) {
	m := NewMemo(64)
	compute := func() ([]float32, error) { return []float32{1}, nil }
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				_, _ = m.GetOrCompute("shared-query", compute)
			}
		}()
	}
	wg.Wait()
	if _, ok := m.Get("shared-query"); !ok {
		t.Fatal("expected shared-query to be memoized")
	}
}
