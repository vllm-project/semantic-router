package embedding

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func textKey(text string) VectorKey {
	return NewVectorKey("model-a", Options{}, InputText, []byte(text))
}

func TestVectorKeySeparatesRepresentationViewAndModality(t *testing.T) {
	base := NewVectorKey("model-a", Options{}, InputText, []byte("x"))
	for _, other := range []VectorKey{
		NewVectorKey("model-b", Options{}, InputText, []byte("x")),
		NewVectorKey("model-a", Options{Dimension: 256}, InputText, []byte("x")),
		NewVectorKey("model-a", Options{Layer: 6}, InputText, []byte("x")),
		NewVectorKey("model-a", Options{}, InputImage, []byte("x")),
		NewVectorKey("model-a", Options{}, InputText, []byte("y")),
		NewVectorKey("model-", Options{}, InputText, []byte("ax")),
	} {
		if other == base {
			t.Fatal("distinct inputs share a key")
		}
	}
}

func TestVectorCacheComputesMissesOnceInOneCall(t *testing.T) {
	cache := NewVectorCache(1 << 20)
	var calls atomic.Int32
	var embedded atomic.Int64
	compute := func(_ context.Context, missing []int) ([][]float32, error) {
		calls.Add(1)
		embedded.Add(int64(len(missing)))
		out := make([][]float32, len(missing))
		for i, index := range missing {
			out[i] = []float32{float32(index)}
		}
		return out, nil
	}
	keys := []VectorKey{textKey("a"), textKey("b"), textKey("a")}
	vectors, err := cache.Resolve(context.Background(), keys, compute)
	if err != nil || calls.Load() != 1 || embedded.Load() != 2 {
		t.Fatalf("first resolve: %v, %d calls, %d embedded", err, calls.Load(), embedded.Load())
	}
	if vectors[0][0] != 0 || vectors[1][0] != 1 || vectors[2][0] != 0 {
		t.Fatalf("vectors = %v", vectors)
	}
	vectors[0][0] = 99 // callers own their copies
	again, err := cache.Resolve(context.Background(), keys, compute)
	if err != nil || calls.Load() != 1 || again[0][0] != 0 {
		t.Fatalf("cached resolve: %v, %d calls, %v", err, calls.Load(), again)
	}
}

func TestVectorCacheCoalescesConcurrentRequests(t *testing.T) {
	cache := NewVectorCache(1 << 20)
	var calls atomic.Int32
	release := make(chan struct{})
	compute := func(_ context.Context, missing []int) ([][]float32, error) {
		calls.Add(1)
		<-release
		return [][]float32{{1, 2}}, nil
	}
	var wg sync.WaitGroup
	results := make([][][]float32, 8)
	for i := range results {
		wg.Add(1)
		go func() {
			defer wg.Done()
			results[i], _ = cache.Resolve(context.Background(), []VectorKey{textKey("same")}, compute)
		}()
	}
	time.Sleep(20 * time.Millisecond)
	close(release)
	wg.Wait()
	if calls.Load() != 1 {
		t.Fatalf("%d model calls for one key, want 1", calls.Load())
	}
	for _, result := range results {
		if len(result) != 1 || len(result[0]) != 2 {
			t.Fatalf("result %v", result)
		}
	}
}

func TestVectorCacheRetriesAfterAnotherCallerFails(t *testing.T) {
	cache := NewVectorCache(1 << 20)
	started := make(chan struct{})
	fail := make(chan struct{})
	go func() {
		_, _ = cache.Resolve(context.Background(), []VectorKey{textKey("k")}, func(context.Context, []int) ([][]float32, error) {
			close(started)
			<-fail
			return nil, errors.New("owner canceled")
		})
	}()
	<-started
	done := make(chan [][]float32)
	go func() {
		vectors, _ := cache.Resolve(context.Background(), []VectorKey{textKey("k")}, func(context.Context, []int) ([][]float32, error) {
			return [][]float32{{7}}, nil
		})
		done <- vectors
	}()
	time.Sleep(10 * time.Millisecond)
	close(fail)
	if vectors := <-done; len(vectors) != 1 || vectors[0][0] != 7 {
		t.Fatalf("waiter did not recompute: %v", vectors)
	}
}

func TestVectorCacheEvictsLeastRecentlyUsed(t *testing.T) {
	cache := NewVectorCache(vectorCacheShards * (4*4 + entryOverhead) * 2)
	compute := func(_ context.Context, missing []int) ([][]float32, error) {
		out := make([][]float32, len(missing))
		for i := range out {
			out[i] = make([]float32, 4)
		}
		return out, nil
	}
	var keys []VectorKey
	for i := 0; i < 2000; i++ {
		key := textKey(string(rune('a'+i%26)) + string(rune(i)))
		keys = append(keys, key)
		if _, err := cache.Resolve(context.Background(), []VectorKey{key}, compute); err != nil {
			t.Fatal(err)
		}
	}
	held := 0
	for i := range cache.shards {
		shard := &cache.shards[i]
		if shard.bytes > shard.maxBytes {
			t.Fatalf("shard %d holds %d bytes over its %d budget", i, shard.bytes, shard.maxBytes)
		}
		held += len(shard.entries)
	}
	if held == 0 || held >= len(keys) {
		t.Fatalf("cache holds %d of %d entries", held, len(keys))
	}
}

func TestVectorCacheWithoutBudgetOnlyCoalesces(t *testing.T) {
	cache := NewVectorCache(0)
	var calls atomic.Int32
	compute := func(_ context.Context, missing []int) ([][]float32, error) {
		calls.Add(1)
		return [][]float32{{1}}, nil
	}
	for i := 0; i < 2; i++ {
		if _, err := cache.Resolve(context.Background(), []VectorKey{textKey("x")}, compute); err != nil {
			t.Fatal(err)
		}
	}
	if calls.Load() != 2 {
		t.Fatalf("a disabled cache stored a vector: %d calls", calls.Load())
	}
}
