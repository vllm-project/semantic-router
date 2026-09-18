//go:build !windows && cgo

package benchmarks

import (
	"context"
	"errors"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
)

type fixtureCache struct {
	entries    int
	stored     int
	addErr     error
	lookup     func() error
	lookups    atomic.Int64
	population []string
}

func (f *fixtureCache) AddEntry(_ context.Context, id, model, query string, request, response []byte, ttl int) error {
	if f.addErr != nil {
		return f.addErr
	}
	f.entries++
	f.population = append(f.population, id+"|"+model+"|"+query)
	return nil
}

func (f *fixtureCache) LookupSimilarWithThreshold(_ context.Context, model, query string, threshold float32) (cache.LookupResult, error) {
	f.lookups.Add(1)
	if model != "test-model" || query == "" || threshold != 0.85 {
		return cache.LookupResult{}, errors.New("lookup contract changed")
	}
	if f.lookup != nil {
		if err := f.lookup(); err != nil {
			return cache.LookupResult{}, err
		}
	}
	return cache.LookupResult{Found: true}, nil
}

func (f *fixtureCache) GetStats() cache.CacheStats {
	if f.stored != 0 {
		return cache.CacheStats{TotalEntries: f.stored}
	}
	return cache.CacheStats{TotalEntries: f.entries}
}

type fixtureTimer struct {
	running   atomic.Bool
	events    []string
	remaining int
	started   bool
}

func (t *fixtureTimer) StartTimer() { t.events = append(t.events, "start"); t.running.Store(true) }
func (t *fixtureTimer) Loop() bool {
	if !t.started {
		t.events = append(t.events, "reset")
		t.started = true
	}
	if t.remaining == 0 {
		return false
	}
	t.remaining--
	return true
}
func (t *fixtureTimer) StopTimer() { t.running.Store(false); t.events = append(t.events, "stop") }

func TestCacheCorpusDeterministicAndIndependentOfIterationCount(t *testing.T) {
	for _, repeat := range []int{10, 70, 90} {
		first := cacheQueryCorpus(100, repeat, 20260919)
		_ = cacheQueryCorpus(17, 40, 7)
		longer := cacheQueryCorpus(10000, repeat, 20260919)
		if !reflect.DeepEqual(first, longer[:100]) {
			t.Fatal("corpus depends on previous calls or requested iteration count")
		}
		composed := 0
		for _, query := range first {
			if strings.Contains(query, " Also, ") {
				composed++
			}
		}
		if composed != 100-repeat {
			t.Fatalf("got %d composed queries, want %d", composed, 100-repeat)
		}
	}
}

func TestCachePreparationRejectsMissingPopulationAndInferenceErrors(t *testing.T) {
	sentinel := errors.New("inference unavailable")
	for name, store := range map[string]*fixtureCache{
		"population failure": {addErr: sentinel},
		"collapsed entries":  {stored: 1},
		"warmup failure":     {lookup: func() error { return sentinel }},
	} {
		t.Run(name, func(t *testing.T) {
			pool, err := prepareCacheLookup(store, cacheScenario{entries: 10, workers: 1, repeatPercent: 70})
			if err == nil || pool != nil {
				t.Fatalf("invalid preparation accepted: pool=%v err=%v", pool, err)
			}
		})
	}
}

func TestCacheBatchExercisesEveryGraphAndWorkerOutsideSetup(t *testing.T) {
	for _, workers := range []int{1, 10, 50} {
		t.Run(strconv.Itoa(workers), func(t *testing.T) {
			scenario := cacheScenario{entries: 1000, workers: workers, repeatPercent: 70, hnsw: true}
			if scenario.samples() != 5 || scenario.requestsPerOp() != 500 {
				t.Fatal("HNSW protocol must retain all five graph samples and 500 requests/op")
			}
			var stores []*fixtureCache
			var pools []*cacheWorkerPool
			timer := &fixtureTimer{remaining: 2}
			for range scenario.samples() {
				store := &fixtureCache{}
				pool, err := prepareCacheLookup(store, scenario)
				if err != nil {
					t.Fatal(err)
				}
				defer pool.close()
				stores = append(stores, store)
				pools = append(pools, pool)
				if store.entries != 1000 || store.lookups.Load() != cacheBatchRequests {
					t.Fatalf("wrong untimed setup: %d entries, %d lookups", store.entries, store.lookups.Load())
				}
				var arrived atomic.Int64
				var release sync.Once
				barrier := make(chan struct{})
				store.lookup = func() error {
					if !timer.running.Load() {
						return errors.New("measured lookup outside timer")
					}
					if arrived.Add(1) == int64(workers) {
						release.Do(func() { close(barrier) })
					}
					select {
					case <-barrier:
						return nil
					case <-time.After(5 * time.Second):
						return errors.New("configured workers did not execute concurrently")
					}
				}
			}
			if err := measureCacheLookups(timer, pools); err != nil {
				t.Fatal(err)
			}
			if timer.running.Load() || !reflect.DeepEqual(timer.events, []string{"start", "reset", "stop"}) {
				t.Fatalf("wrong measurement boundary: %+v", timer.events)
			}
			for graph, store := range stores {
				if store.lookups.Load() != 3*cacheBatchRequests {
					t.Fatalf("graph %d: two ops must execute 200 requests after 100 warmups; got %d", graph, store.lookups.Load())
				}
				for worker, result := range pools[graph].results {
					if result.requests != 2*cacheBatchRequests/workers || result.hits != result.requests {
						t.Fatalf("graph %d worker %d had incorrect measured inventory: %+v", graph, worker, result)
					}
				}
			}
		})
	}
	linear := cacheScenario{hnsw: false}
	if linear.samples() != 1 || linear.requestsPerOp() != 100 {
		t.Fatal("linear protocol must retain one graph and 100 requests/op")
	}
}

func TestCacheMeasuredFailureStopsTimer(t *testing.T) {
	var stores []*fixtureCache
	var pools []*cacheWorkerPool
	for range cacheHNSWSamples {
		store := &fixtureCache{}
		pool, err := prepareCacheLookup(store, cacheScenario{entries: 10, workers: 10, repeatPercent: 70})
		if err != nil {
			t.Fatal(err)
		}
		defer pool.close()
		stores = append(stores, store)
		pools = append(pools, pool)
	}
	sentinel := errors.New("measured inference failed")
	stores[3].lookup = func() error { return sentinel }
	timer := &fixtureTimer{remaining: 3}
	if err := measureCacheLookups(timer, pools); !errors.Is(err, sentinel) {
		t.Fatalf("later graph failure was swallowed: %v", err)
	}
	if timer.running.Load() || !reflect.DeepEqual(timer.events, []string{"start", "reset", "stop"}) {
		t.Fatalf("failure left timer running: %+v", timer.events)
	}
	if stores[4].lookups.Load() != cacheBatchRequests {
		t.Fatal("failed graph must abort, not continue selecting other successful samples")
	}
}
