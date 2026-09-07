package cache

import (
	"context"
	"time"
)

// LegacyBackendAdapter confines the old backend API to one migration boundary.
// Request paths and management APIs depend on TypedCacheStore instead.
type LegacyBackendAdapter struct {
	backend      CacheBackend
	capabilities BackendCapabilities
}

func NewLegacyBackendAdapter(
	backend CacheBackend,
	backendType CacheBackendType,
) *LegacyBackendAdapter {
	capabilities := CapabilitiesForBackend(backendType)
	_, capabilities.Exact = backend.(ExactCacheBackend)
	return &LegacyBackendAdapter{
		backend:      backend,
		capabilities: capabilities,
	}
}

func (a *LegacyBackendAdapter) LookupExact(
	ctx context.Context,
	lookup ExactLookup,
) (CacheResult, error) {
	exact, ok := a.backend.(ExactCacheBackend)
	if !ok {
		return CacheResult{}, ErrUnsupported
	}
	result, err := exact.FindExact(
		ctx,
		lookup.Identity.Partition.Key(),
		lookup.Identity.ExactFingerprint,
	)
	if err != nil {
		return CacheResult{}, err
	}
	age, ageKnown := resultAge(result)
	return CacheResult{
		ResponseBody: result.ResponseBody,
		Found:        result.Found,
		HitKind:      HitKindExact,
		Source:       CacheSourceL2,
		Similarity:   result.Similarity,
		Age:          age,
		AgeKnown:     ageKnown,
		ExpiresAt:    result.ExpiresAt,
	}, nil
}

func (a *LegacyBackendAdapter) StoreExact(ctx context.Context, write CacheWrite) error {
	exact, ok := a.backend.(ExactCacheBackend)
	if !ok {
		return ErrUnsupported
	}
	if write.TTL.NoStore {
		return nil
	}
	return exact.AddExact(
		ctx,
		write.Identity.Partition.Key(),
		write.Identity.ExactFingerprint,
		write.ResponseBody,
		write.TTL.LegacySeconds(),
	)
}

func (a *LegacyBackendAdapter) LookupSemantic(
	ctx context.Context,
	lookup SemanticLookup,
) (CacheResult, error) {
	result, err := a.backend.LookupSimilarWithThreshold(
		ctx,
		lookup.Identity.Partition.Key(),
		lookup.Identity.SemanticQuery,
		lookup.Threshold,
	)
	if err != nil {
		return CacheResult{}, err
	}
	age, ageKnown := resultAge(result)
	return CacheResult{
		ResponseBody: result.ResponseBody,
		Found:        result.Found,
		HitKind:      HitKindSemantic,
		Source:       CacheSourceL2,
		Similarity:   result.Similarity,
		Age:          age,
		AgeKnown:     ageKnown,
		ExpiresAt:    result.ExpiresAt,
	}, nil
}

func resultAge(result LookupResult) (time.Duration, bool) {
	if !result.StoredAt.IsZero() {
		return time.Since(result.StoredAt), true
	}
	return result.Age, result.AgeKnown
}

func (a *LegacyBackendAdapter) StoreSemantic(ctx context.Context, write CacheWrite) error {
	if write.TTL.NoStore {
		return nil
	}
	return a.backend.AddEntry(
		ctx,
		write.RequestID,
		write.Identity.Partition.Key(),
		write.Identity.SemanticQuery,
		write.RequestBody,
		write.ResponseBody,
		write.TTL.LegacySeconds(),
	)
}

func (a *LegacyBackendAdapter) Health(ctx context.Context) error {
	return a.backend.CheckConnection(ctx)
}

func (a *LegacyBackendAdapter) Close() error {
	return a.backend.Close()
}

func (a *LegacyBackendAdapter) Stats(_ context.Context) (CacheStats, error) {
	return a.backend.GetStats(), nil
}

func (a *LegacyBackendAdapter) Capabilities() BackendCapabilities {
	return a.capabilities
}
