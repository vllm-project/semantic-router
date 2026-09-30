//go:build !windows && cgo

package benchmarks

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
)

// An op is a full batch even when Go calibrates b.N=1. Every configured worker
// therefore performs real lookups in the concurrency scenarios.
const (
	cacheBatchRequests = 100
	cacheHNSWSamples   = 5
)

type cacheScenario struct {
	entries, workers, repeatPercent int
	hnsw                            bool
}

func (scenario cacheScenario) samples() int {
	if scenario.hnsw {
		return cacheHNSWSamples
	}
	return 1
}

func (scenario cacheScenario) requestsPerOp() int {
	return scenario.samples() * cacheBatchRequests
}

type benchmarkCache interface {
	AddEntry(context.Context, string, string, string, []byte, []byte, int) error
	LookupSimilarWithThreshold(context.Context, string, string, float32) (cache.LookupResult, error)
	GetStats() cache.CacheStats
}

// Keep the original query topics and composed-query shape, but freeze their
// order independently of Go's iteration calibration and process-global RNG.
func cacheQueryCorpus(count int, repeatPercent int, seed int64) []string {
	base := []string{
		"What is the capital of France?",
		"How do I reverse a string in Python?",
		"Explain quantum computing",
		"What are the benefits of meditation?",
		"How does photosynthesis work?",
		"What is machine learning?",
		"How to make chocolate chip cookies?",
		"Explain the theory of relativity",
		"What is the meaning of life?",
		"How to start a business?",
	}
	rng := rand.New(rand.NewSource(seed))
	queries := make([]string, count)
	for i := range queries {
		if i%100 < repeatPercent {
			queries[i] = base[rng.Intn(len(base))]
		} else {
			queries[i] = base[rng.Intn(len(base))] + " Also, " + base[rng.Intn(len(base))]
		}
	}
	return queries
}

type cacheWorkerResult struct {
	requests, hits int
	err            error
}

type cacheWorkerPool struct {
	starts  []chan struct{}
	results []cacheWorkerResult
	batch   sync.WaitGroup
	stopped sync.WaitGroup
}

func newCacheWorkerPool(store benchmarkCache, queries []string, workers int) *cacheWorkerPool {
	pool := &cacheWorkerPool{
		starts:  make([]chan struct{}, workers),
		results: make([]cacheWorkerResult, workers),
	}
	var ready sync.WaitGroup
	ready.Add(workers)
	pool.stopped.Add(workers)
	for worker := range workers {
		pool.starts[worker] = make(chan struct{})
		go func() {
			defer pool.stopped.Done()
			ready.Done()
			for range pool.starts[worker] {
				result := &pool.results[worker]
				for i := worker; i < len(queries); i += workers {
					lookup, err := store.LookupSimilarWithThreshold(context.Background(), "test-model", queries[i], 0.85)
					if err != nil {
						result.err = err
						break
					}
					result.requests++
					if lookup.Found {
						result.hits++
					}
				}
				pool.batch.Done()
			}
		}()
	}
	ready.Wait()
	return pool
}

func (pool *cacheWorkerPool) runBatch() error {
	pool.batch.Add(len(pool.starts))
	for _, start := range pool.starts {
		start <- struct{}{}
	}
	pool.batch.Wait()
	for _, result := range pool.results {
		if result.err != nil {
			return result.err
		}
	}
	return nil
}

func (pool *cacheWorkerPool) close() {
	for _, start := range pool.starts {
		close(start)
	}
	pool.stopped.Wait()
}

func prepareCacheLookup(store benchmarkCache, scenario cacheScenario) (*cacheWorkerPool, error) {
	if scenario.entries <= 0 || scenario.workers <= 0 || scenario.workers > cacheBatchRequests ||
		scenario.repeatPercent < 0 || scenario.repeatPercent > 100 {
		return nil, fmt.Errorf("invalid cache scenario: %+v", scenario)
	}
	for i, query := range cacheQueryCorpus(scenario.entries, 10, 20260919) {
		if err := store.AddEntry(context.Background(), fmt.Sprintf("req-%d", i), "test-model", query,
			[]byte(query), []byte("Response for: "+query), -1); err != nil {
			return nil, fmt.Errorf("populate cache entry %d: %w", i, err)
		}
	}
	if got := store.GetStats().TotalEntries; got != scenario.entries {
		return nil, fmt.Errorf("cache population: got %d stored entries, want %d", got, scenario.entries)
	}
	queries := cacheQueryCorpus(cacheBatchRequests, scenario.repeatPercent, 20260920)
	pool := newCacheWorkerPool(store, queries, scenario.workers)
	// Warm the actual lookup path, embedding memo, metrics and worker pool before
	// accounting starts. No precomputed/fake vectors are used by real benchmarks.
	if err := pool.runBatch(); err != nil {
		pool.close()
		return nil, fmt.Errorf("warm cache lookups: %w", err)
	}
	clear(pool.results)
	return pool, nil
}

type cacheMeasurementTimer interface {
	StartTimer()
	Loop() bool
	StopTimer()
}

func measureCacheLookups(b cacheMeasurementTimer, pools []*cacheWorkerPool) error {
	// B.Loop resets counters on its first call and runs the prepared fixture only
	// once, instead of rebuilding different graphs during b.N calibration.
	b.StartTimer()
	defer b.StopTimer()
	for b.Loop() {
		for _, pool := range pools {
			if err := pool.runBatch(); err != nil {
				return err
			}
		}
	}
	return nil
}

func runCacheLookupBenchmark(b *testing.B, scenario cacheScenario) {
	b.Helper()
	b.StopTimer()
	provider := initCacheEmbeddingModels(b)
	recordCacheProtocol(b, scenario)
	var stores []*cache.InMemoryCache
	var pools []*cacheWorkerPool
	defer func() {
		for _, pool := range pools {
			pool.close()
		}
		for _, store := range stores {
			if err := store.Close(); err != nil {
				b.Errorf("close benchmark cache: %v", err)
			}
		}
	}()
	for range scenario.samples() {
		store := cache.NewInMemoryCache(cache.InMemoryCacheOptions{
			SimilarityThreshold: 0.85, MaxEntries: scenario.entries * 2,
			TTLSeconds: 0, Enabled: true, EvictionPolicy: cache.LRUEvictionPolicyType,
			UseHNSW: scenario.hnsw, HNSWM: 16, HNSWEfConstruction: 200, HNSWEfSearch: 50,
			EmbeddingModel: cacheEmbeddingModelType, EmbeddingProvider: provider,
		})
		stores = append(stores, store)
		pool, err := prepareCacheLookup(store, scenario)
		if err != nil {
			b.Fatal(err)
		}
		pools = append(pools, pool)
	}
	b.ReportAllocs()
	if err := measureCacheLookups(b, pools); err != nil {
		b.Fatal(err)
	}
	var requests, hits int
	for _, pool := range pools {
		for _, result := range pool.results {
			requests += result.requests
			hits += result.hits
		}
	}
	if requests != b.N*scenario.requestsPerOp() {
		b.Fatalf("completed %d requests, want %d", requests, b.N*scenario.requestsPerOp())
	}
	b.ReportMetric(float64(scenario.requestsPerOp()), "requests/op")
	b.ReportMetric(float64(scenario.samples()), "graphs/op")
	b.ReportMetric(float64(requests)/b.Elapsed().Seconds(), "qps")
	b.ReportMetric(float64(hits)*100/float64(requests), "hit_rate_%")
}
