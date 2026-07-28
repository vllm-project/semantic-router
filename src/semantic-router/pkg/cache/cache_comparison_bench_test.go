//go:build !windows && cgo

package cache

import (
	"fmt"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

// newInMemoryBench builds a local in-memory HNSW cache for benchmarking. Unlike
// the remote stores it needs no server, so it never fails closed; it only needs
// the BERT model, which it shares with the other backends.
func newInMemoryBench(b *testing.B, size int) CacheBackend {
	b.Helper()
	if err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true); err != nil {
		b.Fatalf("failed to initialize BERT model: %v", err)
	}
	maxEntries := size * 2
	if maxEntries < 1000 {
		maxEntries = 1000
	}
	return NewInMemoryCache(InMemoryCacheOptions{
		SimilarityThreshold: 0.8,
		MaxEntries:          maxEntries,
		TTLSeconds:          300,
		Enabled:             true,
		UseHNSW:             true,
		HNSWM:               16,
		HNSWEfConstruction:  64,
		HNSWEfSearch:        50,
		EmbeddingModel:      "bert",
	})
}

// runCacheFindSimilarBench pre-populates a backend with size entries, then times
// the FindSimilar hot path. The timer is reset after populate so the reported
// ns/op reflects only lookups, and the hit rate is reported for context.
func runCacheFindSimilarBench(b *testing.B, cache CacheBackend, model string, size int) {
	b.Helper()
	for i := 0; i < size; i++ {
		q := fmt.Sprintf("benchmark query number %d about topic %d", i, i%97)
		if err := cache.AddEntry(
			fmt.Sprintf("req-%d", i), model, q,
			[]byte(q), []byte(fmt.Sprintf("response-%d", i)), 300,
		); err != nil {
			b.Fatalf("populate AddEntry failed at %d: %v", i, err)
		}
	}

	b.ResetTimer()
	b.ReportAllocs()

	var hits int
	for i := 0; i < b.N; i++ {
		var q string
		if i%10 < 7 {
			q = fmt.Sprintf("benchmark query number %d about topic %d", i%size, (i%size)%97)
		} else {
			q = fmt.Sprintf("entirely novel unseen request %d", i)
		}
		_, found, err := cache.FindSimilar(model, q)
		if err != nil {
			b.Fatalf("FindSimilar failed: %v", err)
		}
		if found {
			hits++
		}
	}
	b.ReportMetric(float64(hits)/float64(b.N)*100, "hit_rate_%")
}

// BenchmarkCacheComparison measures the FindSimilar hot path across the
// interchangeable semantic-cache backends — the local in-memory HNSW cache and
// the two remote KNN stores (Redis, Valkey) — so their lookup latencies can be
// compared apples-to-apples at the same cache size. It is referenced by
// `make benchmark-cache-comparison`, which had no matching benchmark until now
// (the target silently passed because `go test -bench=<nonexistent>` exits 0).
//
// Milvus is intentionally excluded: it requires heavy standalone infrastructure
// (etcd + object store) and already has dedicated coverage in
// BenchmarkHybridVsMilvus.
func BenchmarkCacheComparison(b *testing.B) {
	const model = "bench-model"

	backends := []struct {
		name string
		make func(b *testing.B, size int) CacheBackend
	}{
		{"InMemory", newInMemoryBench},
		{"Redis", func(b *testing.B, _ int) CacheBackend { return setupRedisCacheBench(b) }},
		{"Valkey", func(b *testing.B, _ int) CacheBackend { return setupValkeyCacheBench(b) }},
	}

	for _, be := range backends {
		for _, size := range benchCacheSizes() {
			b.Run(fmt.Sprintf("%s/CacheSize_%d", be.name, size), func(b *testing.B) {
				cache := be.make(b, size)
				defer func() { _ = cache.Close() }()
				runCacheFindSimilarBench(b, cache, model, size)
			})
		}
	}
}
