package cache

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

type ResponseCacheServiceOptions struct {
	L1MaxEntries      int
	L1TTL             time.Duration
	SingleflightWait  time.Duration
	SingleflightLease time.Duration
}

func DefaultResponseCacheServiceOptions() ResponseCacheServiceOptions {
	return ResponseCacheServiceOptions{
		L1MaxEntries:      1024,
		L1TTL:             5 * time.Minute,
		SingleflightWait:  250 * time.Millisecond,
		SingleflightLease: 30 * time.Second,
	}
}

type inflightExact struct {
	done      chan struct{}
	expiresAt time.Time
}

type SingleflightLease struct {
	Identity CacheIdentity
	Key      string
	Leader   bool
}

type ResponseCacheService struct {
	store   TypedCacheStore
	l1      *exactL1
	options ResponseCacheServiceOptions

	mu              sync.Mutex
	globalEpoch     uint64
	partitionEpochs map[string]uint64
	inflight        map[string]*inflightExact

	exactHits           atomic.Int64
	semanticHits        atomic.Int64
	l1Hits              atomic.Int64
	l2Hits              atomic.Int64
	misses              atomic.Int64
	singleflightWaiters atomic.Int64
	invalidations       atomic.Int64
	staleMisses         atomic.Int64
	failOpen            atomic.Int64
}

func NewResponseCacheService(
	store TypedCacheStore,
	options ResponseCacheServiceOptions,
) *ResponseCacheService {
	defaults := DefaultResponseCacheServiceOptions()
	if options.L1MaxEntries == 0 {
		options.L1MaxEntries = defaults.L1MaxEntries
	}
	if options.L1TTL == 0 {
		options.L1TTL = defaults.L1TTL
	}
	if options.SingleflightWait == 0 {
		options.SingleflightWait = defaults.SingleflightWait
	}
	if options.SingleflightLease == 0 {
		options.SingleflightLease = defaults.SingleflightLease
	}
	return &ResponseCacheService{
		store:           store,
		l1:              newExactL1(options.L1MaxEntries, options.L1TTL),
		options:         options,
		partitionEpochs: make(map[string]uint64),
		inflight:        make(map[string]*inflightExact),
	}
}

func (s *ResponseCacheService) ResolveIdentity(identity CacheIdentity) CacheIdentity {
	if identity.Partition.epochResolved {
		return identity
	}
	partitionKey := identity.Partition.Key()
	s.mu.Lock()
	globalEpoch := s.globalEpoch
	partitionEpoch := s.partitionEpochs[partitionKey]
	s.mu.Unlock()
	configuredEpoch := identity.Partition.Epoch
	identity.Partition.Epoch = fmt.Sprintf(
		"%s:g%d:p%d",
		configuredEpoch,
		globalEpoch,
		partitionEpoch,
	)
	identity.Partition.epochResolved = true
	return identity
}

func (s *ResponseCacheService) LookupExact(
	ctx context.Context,
	lookup ExactLookup,
) (result CacheResult, err error) {
	start := time.Now()
	defer func() {
		s.recordOperation("lookup_exact", lookup.Identity, result, err, start)
	}()
	if s == nil || s.store == nil {
		return CacheResult{HitKind: HitKindMiss}, nil
	}
	lookup.Identity = s.ResolveIdentity(lookup.Identity)
	key := exactCacheStorageKey(
		lookup.Identity.Partition.Key(),
		lookup.Identity.ExactFingerprint,
	)
	if l1Result, ok := s.l1.get(key, lookup.MaxAge); ok {
		s.exactHits.Add(1)
		s.l1Hits.Add(1)
		return l1Result, nil
	}
	result, err = s.store.LookupExact(ctx, lookup)
	if errors.Is(err, ErrUnsupported) {
		return CacheResult{HitKind: HitKindMiss}, nil
	}
	if err != nil {
		s.failOpen.Add(1)
		return CacheResult{}, err
	}
	if result.Found && lookup.MaxAge != nil &&
		(!result.AgeKnown || result.Age > *lookup.MaxAge) {
		s.staleMisses.Add(1)
		result = CacheResult{HitKind: HitKindMiss}
	}
	if !result.Found {
		s.misses.Add(1)
		result.HitKind = HitKindMiss
		return result, nil
	}
	result.HitKind = HitKindExact
	result.Source = CacheSourceL2
	s.exactHits.Add(1)
	s.l2Hits.Add(1)
	ttl := DefaultTTL()
	if !result.ExpiresAt.IsZero() {
		ttl = TTL(time.Until(result.ExpiresAt))
	}
	s.l1.put(key, result.ResponseBody, ttl)
	return result, nil
}

func (s *ResponseCacheService) LookupExactCoalesced(
	ctx context.Context,
	lookup ExactLookup,
) (CacheResult, SingleflightLease, error) {
	lookup.Identity = s.ResolveIdentity(lookup.Identity)
	result, err := s.LookupExact(ctx, lookup)
	if err != nil || result.Found {
		return result, SingleflightLease{Identity: lookup.Identity}, err
	}
	key := exactCacheStorageKey(
		lookup.Identity.Partition.Key(),
		lookup.Identity.ExactFingerprint,
	)
	leader, done := s.acquireSingleflight(key)
	lease := SingleflightLease{
		Identity: lookup.Identity,
		Key:      key,
		Leader:   leader,
	}
	if leader {
		return result, lease, nil
	}
	s.singleflightWaiters.Add(1)
	backend := string(s.Capabilities().Backend)
	timer := time.NewTimer(s.options.SingleflightWait)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		s.failOpen.Add(1)
		metrics.RecordResponseCacheSingleflightWaiter(backend, "canceled")
		return result, lease, nil
	case <-timer.C:
		s.failOpen.Add(1)
		metrics.RecordResponseCacheSingleflightWaiter(backend, "timeout")
		return result, lease, nil
	case <-done:
		metrics.RecordResponseCacheSingleflightWaiter(backend, "completed")
		retry, retryErr := s.LookupExact(ctx, lookup)
		return retry, lease, retryErr
	}
}

func (s *ResponseCacheService) StoreExact(ctx context.Context, write CacheWrite) error {
	if s == nil || s.store == nil {
		return nil
	}
	write.Identity = s.ResolveIdentity(write.Identity)
	key := exactCacheStorageKey(
		write.Identity.Partition.Key(),
		write.Identity.ExactFingerprint,
	)
	defer s.completeSingleflight(key)
	if write.TTL.NoStore {
		return nil
	}
	start := time.Now()
	if err := s.store.StoreExact(ctx, write); err != nil && !errors.Is(err, ErrUnsupported) {
		s.failOpen.Add(1)
		s.recordOperation("store_exact", write.Identity, CacheResult{}, err, start)
		return err
	}
	s.l1.put(key, write.ResponseBody, write.TTL)
	s.recordOperation("store_exact", write.Identity, CacheResult{}, nil, start)
	return nil
}

func (s *ResponseCacheService) LookupSemantic(
	ctx context.Context,
	lookup SemanticLookup,
) (result CacheResult, err error) {
	start := time.Now()
	defer func() {
		s.recordOperation("lookup_semantic", lookup.Identity, result, err, start)
	}()
	if s == nil || s.store == nil {
		return CacheResult{HitKind: HitKindMiss}, nil
	}
	lookup.Identity = s.ResolveIdentity(lookup.Identity)
	result, err = s.store.LookupSemantic(ctx, lookup)
	if err != nil {
		s.failOpen.Add(1)
		return CacheResult{}, err
	}
	if result.Found {
		result.HitKind = HitKindSemantic
		result.Source = CacheSourceL2
		s.semanticHits.Add(1)
		s.l2Hits.Add(1)
	} else {
		result.HitKind = HitKindMiss
		s.misses.Add(1)
	}
	return result, nil
}

func (s *ResponseCacheService) StoreSemantic(ctx context.Context, write CacheWrite) error {
	if s == nil || s.store == nil || write.TTL.NoStore {
		return nil
	}
	write.Identity = s.ResolveIdentity(write.Identity)
	start := time.Now()
	if err := s.store.StoreSemantic(ctx, write); err != nil {
		s.failOpen.Add(1)
		s.recordOperation("store_semantic", write.Identity, CacheResult{}, err, start)
		return err
	}
	s.recordOperation("store_semantic", write.Identity, CacheResult{}, nil, start)
	return nil
}

func (s *ResponseCacheService) Health(ctx context.Context) error {
	if s == nil || s.store == nil {
		return nil
	}
	return s.store.Health(ctx)
}

func (s *ResponseCacheService) Capabilities() BackendCapabilities {
	if s == nil || s.store == nil {
		return BackendCapabilities{}
	}
	capabilities := s.store.Capabilities()
	if s.l1 != nil && s.l1.maxEntries > 0 {
		capabilities.Exact = true
	}
	return capabilities
}

func (s *ResponseCacheService) Stats(ctx context.Context) (CacheStats, error) {
	if s == nil || s.store == nil {
		return CacheStats{}, nil
	}
	stats, err := s.store.Stats(ctx)
	if err != nil {
		return CacheStats{}, err
	}
	stats.ExactHitCount = s.exactHits.Load()
	stats.SemanticHitCount = s.semanticHits.Load()
	stats.L1HitCount = s.l1Hits.Load()
	stats.L2HitCount = s.l2Hits.Load()
	stats.MissCount = s.misses.Load()
	stats.SingleflightWaiters = s.singleflightWaiters.Load()
	stats.InvalidationCount = s.invalidations.Load()
	stats.StaleMissCount = s.staleMisses.Load()
	stats.FailOpenCount = s.failOpen.Load()
	stats.L1Entries = s.l1.len()
	return stats, nil
}

func (s *ResponseCacheService) Invalidate(
	_ context.Context,
	selector CacheSelector,
	dryRun bool,
) (InvalidationResult, error) {
	if selector.Partition == nil && selector.Epoch == "" {
		return InvalidationResult{}, fmt.Errorf("invalidation requires a partition or epoch selector")
	}
	result := InvalidationResult{DryRun: dryRun}
	if dryRun {
		metrics.RecordResponseCacheInvalidation(string(s.Capabilities().Backend), "invalidate", true)
		return result, nil
	}
	s.mu.Lock()
	if selector.Partition != nil {
		key := selector.Partition.Key()
		s.partitionEpochs[key]++
	} else {
		s.globalEpoch++
	}
	s.mu.Unlock()
	s.l1.clear()
	s.invalidations.Add(1)
	metrics.RecordResponseCacheInvalidation(string(s.Capabilities().Backend), "invalidate", false)
	return result, nil
}

func (s *ResponseCacheService) Flush(
	_ context.Context,
	selector CacheSelector,
) (InvalidationResult, error) {
	s.mu.Lock()
	if selector.Partition == nil {
		s.globalEpoch++
	} else {
		s.partitionEpochs[selector.Partition.Key()]++
	}
	s.mu.Unlock()
	s.l1.clear()
	s.invalidations.Add(1)
	metrics.RecordResponseCacheInvalidation(string(s.Capabilities().Backend), "flush", false)
	return InvalidationResult{}, nil
}

func (s *ResponseCacheService) CurrentEpoch() string {
	if s == nil {
		return "0"
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return strconv.FormatUint(s.globalEpoch, 10)
}

func (s *ResponseCacheService) recordOperation(
	operation string,
	identity CacheIdentity,
	result CacheResult,
	err error,
	start time.Time,
) {
	if s == nil {
		return
	}
	status := "success"
	if err != nil {
		status = "error"
	} else if operation == "lookup_exact" || operation == "lookup_semantic" {
		if result.Found {
			status = "hit"
		} else {
			status = "miss"
		}
	}
	backend := string(s.Capabilities().Backend)
	metrics.RecordResponseCacheOperation(
		backend,
		operation,
		identity.Partition.Protocol,
		string(result.HitKind),
		string(result.Source),
		status,
		time.Since(start).Seconds(),
	)
	if result.Found && result.AgeKnown {
		metrics.RecordResponseCacheEntryAge(
			backend,
			string(result.HitKind),
			string(result.Source),
			result.Age.Seconds(),
		)
	}
}

func (s *ResponseCacheService) acquireSingleflight(key string) (bool, <-chan struct{}) {
	now := time.Now()
	s.mu.Lock()
	defer s.mu.Unlock()
	if active, ok := s.inflight[key]; ok {
		if now.Before(active.expiresAt) {
			return false, active.done
		}
		close(active.done)
		delete(s.inflight, key)
	}
	active := &inflightExact{
		done:      make(chan struct{}),
		expiresAt: now.Add(s.options.SingleflightLease),
	}
	s.inflight[key] = active
	return true, active.done
}

func (s *ResponseCacheService) completeSingleflight(key string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	active, ok := s.inflight[key]
	if !ok {
		return
	}
	close(active.done)
	delete(s.inflight, key)
}
