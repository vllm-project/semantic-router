package cache

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type serviceTestStore struct {
	mu          sync.Mutex
	exact       map[string][]byte
	exactLookup atomic.Int64
}

func newServiceTestStore() *serviceTestStore {
	return &serviceTestStore{exact: make(map[string][]byte)}
}

func (s *serviceTestStore) LookupExact(
	_ context.Context,
	lookup ExactLookup,
) (CacheResult, error) {
	s.exactLookup.Add(1)
	key := exactCacheStorageKey(lookup.Identity.Partition.Key(), lookup.Identity.ExactFingerprint)
	s.mu.Lock()
	defer s.mu.Unlock()
	body, ok := s.exact[key]
	return CacheResult{
		ResponseBody: append([]byte(nil), body...),
		Found:        ok,
		HitKind:      HitKindExact,
		Source:       CacheSourceL2,
		AgeKnown:     true,
	}, nil
}

func (s *serviceTestStore) StoreExact(_ context.Context, write CacheWrite) error {
	key := exactCacheStorageKey(write.Identity.Partition.Key(), write.Identity.ExactFingerprint)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.exact[key] = append([]byte(nil), write.ResponseBody...)
	return nil
}

func (s *serviceTestStore) LookupSemantic(
	_ context.Context,
	_ SemanticLookup,
) (CacheResult, error) {
	return CacheResult{HitKind: HitKindMiss}, nil
}

func (s *serviceTestStore) StoreSemantic(_ context.Context, _ CacheWrite) error {
	return nil
}

func (s *serviceTestStore) Health(context.Context) error { return nil }
func (s *serviceTestStore) Close() error                 { return nil }
func (s *serviceTestStore) Stats(context.Context) (CacheStats, error) {
	return CacheStats{}, nil
}

func (s *serviceTestStore) Capabilities() BackendCapabilities {
	return CapabilitiesForBackend(InMemoryCacheType)
}

func TestResponseCacheServiceUsesBoundedL1(t *testing.T) {
	store := newServiceTestStore()
	service := NewResponseCacheService(store, ResponseCacheServiceOptions{
		L1MaxEntries: 2,
		L1TTL:        time.Minute,
	})
	identity := serviceTestIdentity("a")
	if err := service.StoreExact(context.Background(), CacheWrite{
		Identity:     identity,
		ResponseBody: []byte("cached"),
		TTL:          TTL(time.Minute),
	}); err != nil {
		t.Fatalf("StoreExact() error = %v", err)
	}
	result, err := service.LookupExact(context.Background(), ExactLookup{Identity: identity})
	if err != nil {
		t.Fatalf("LookupExact() error = %v", err)
	}
	if !result.Found || result.Source != CacheSourceL1 {
		t.Fatalf("LookupExact() = %#v, want L1 hit", result)
	}
	if got := store.exactLookup.Load(); got != 0 {
		t.Fatalf("backend lookups = %d, want 0", got)
	}
}

func TestResponseCacheServiceSingleflightCoalescesConcurrentMisses(t *testing.T) {
	store := newServiceTestStore()
	service := NewResponseCacheService(store, ResponseCacheServiceOptions{
		L1MaxEntries:      10,
		L1TTL:             time.Minute,
		SingleflightWait:  time.Second,
		SingleflightLease: time.Second,
	})
	identity := serviceTestIdentity("shared")
	var upstreamCalls atomic.Int64
	var wait sync.WaitGroup
	start := make(chan struct{})
	for range 100 {
		wait.Add(1)
		go func() {
			defer wait.Done()
			<-start
			result, lease, err := service.LookupExactCoalesced(
				context.Background(),
				ExactLookup{Identity: identity},
			)
			if err != nil {
				t.Errorf("LookupExactCoalesced() error = %v", err)
				return
			}
			if result.Found {
				return
			}
			if !lease.Leader {
				t.Errorf("waiter returned a miss without becoming leader")
				return
			}
			upstreamCalls.Add(1)
			time.Sleep(20 * time.Millisecond)
			if err := service.StoreExact(context.Background(), CacheWrite{
				Identity:     lease.Identity,
				ResponseBody: []byte("cached"),
				TTL:          TTL(time.Minute),
			}); err != nil {
				t.Errorf("StoreExact() error = %v", err)
			}
		}()
	}
	close(start)
	wait.Wait()
	if got := upstreamCalls.Load(); got != 1 {
		t.Fatalf("upstream calls = %d, want 1", got)
	}
}

func TestResponseCacheServiceFlushAdvancesEpoch(t *testing.T) {
	store := newServiceTestStore()
	service := NewResponseCacheService(store, DefaultResponseCacheServiceOptions())
	identity := serviceTestIdentity("epoch")
	if err := service.StoreExact(context.Background(), CacheWrite{
		Identity:     identity,
		ResponseBody: []byte("old"),
		TTL:          TTL(time.Minute),
	}); err != nil {
		t.Fatalf("StoreExact() error = %v", err)
	}
	if _, err := service.Flush(context.Background(), CacheSelector{}); err != nil {
		t.Fatalf("Flush() error = %v", err)
	}
	result, err := service.LookupExact(context.Background(), ExactLookup{Identity: identity})
	if err != nil {
		t.Fatalf("LookupExact() error = %v", err)
	}
	if result.Found {
		t.Fatalf("LookupExact() found stale epoch entry")
	}
}

func serviceTestIdentity(fingerprint string) CacheIdentity {
	return CacheIdentity{
		Partition: CachePartition{
			Recipe:        "default",
			Decision:      "route",
			RequestModel:  "auto",
			SelectedModel: "model",
			Protocol:      "openai:body",
			Namespace:     "user",
		},
		ExactFingerprint: fingerprint,
	}
}

func BenchmarkResponseCacheServiceL1Lookup(b *testing.B) {
	service := NewResponseCacheService(
		newServiceTestStore(),
		ResponseCacheServiceOptions{L1MaxEntries: 100, L1TTL: time.Minute},
	)
	identity := serviceTestIdentity("benchmark")
	if err := service.StoreExact(context.Background(), CacheWrite{
		Identity:     identity,
		ResponseBody: []byte(`{"ok":true}`),
		TTL:          TTL(time.Minute),
	}); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		result, err := service.LookupExact(context.Background(), ExactLookup{Identity: identity})
		if err != nil || !result.Found {
			b.Fatalf("lookup = %#v, %v", result, err)
		}
	}
}
