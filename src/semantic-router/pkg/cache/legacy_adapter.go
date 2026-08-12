package cache

import "context"

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
	_ context.Context,
	lookup ExactLookup,
) (CacheResult, error) {
	exact, ok := a.backend.(ExactCacheBackend)
	if !ok {
		return CacheResult{}, ErrUnsupported
	}
	result, err := exact.FindExact(
		lookup.Identity.Partition.Key(),
		lookup.Identity.ExactFingerprint,
	)
	if err != nil {
		return CacheResult{}, err
	}
	return CacheResult{
		ResponseBody: result.ResponseBody,
		Found:        result.Found,
		HitKind:      HitKindExact,
		Source:       CacheSourceL2,
		Similarity:   result.Similarity,
	}, nil
}

func (a *LegacyBackendAdapter) StoreExact(_ context.Context, write CacheWrite) error {
	exact, ok := a.backend.(ExactCacheBackend)
	if !ok {
		return ErrUnsupported
	}
	if write.TTL.NoStore {
		return nil
	}
	return exact.AddExact(
		write.Identity.Partition.Key(),
		write.Identity.ExactFingerprint,
		write.ResponseBody,
		write.TTL.LegacySeconds(),
	)
}

func (a *LegacyBackendAdapter) LookupSemantic(
	_ context.Context,
	lookup SemanticLookup,
) (CacheResult, error) {
	result, err := a.backend.LookupSimilarWithThreshold(
		lookup.Identity.Partition.Key(),
		lookup.Identity.SemanticQuery,
		lookup.Threshold,
	)
	if err != nil {
		return CacheResult{}, err
	}
	return CacheResult{
		ResponseBody: result.ResponseBody,
		Found:        result.Found,
		HitKind:      HitKindSemantic,
		Source:       CacheSourceL2,
		Similarity:   result.Similarity,
	}, nil
}

func (a *LegacyBackendAdapter) StoreSemantic(_ context.Context, write CacheWrite) error {
	if write.TTL.NoStore {
		return nil
	}
	return a.backend.AddEntry(
		write.RequestID,
		write.Identity.Partition.Key(),
		write.Identity.SemanticQuery,
		write.RequestBody,
		write.ResponseBody,
		write.TTL.LegacySeconds(),
	)
}

func (a *LegacyBackendAdapter) Health(_ context.Context) error {
	return a.backend.CheckConnection()
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
