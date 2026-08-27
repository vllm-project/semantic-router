package embedding

import (
	"fmt"
	"sync"
	"testing"
)

func TestToolEmbeddingMemoPutExistingKeyIsNoOp(t *testing.T) {
	memo := NewToolEmbeddingMemo(2)
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

func TestToolEmbeddingMemoEvictsOldestAtCapacity(t *testing.T) {
	memo := NewToolEmbeddingMemo(3)
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

// TestToolEmbeddingMemoConcurrentAccess is a smoke test for -race: many
// goroutines read and write overlapping keys while eviction churns the memo.
func TestToolEmbeddingMemoConcurrentAccess(t *testing.T) {
	memo := NewToolEmbeddingMemo(32)

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
