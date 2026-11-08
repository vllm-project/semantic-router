package cache

import (
	"context"
	"errors"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

// LegacyBackendAdapter confines the old backend API to one migration boundary.
// Request paths and management APIs depend on TypedCacheStore instead.
type LegacyBackendAdapter struct {
	backend             CacheBackend
	capabilities        BackendCapabilities
	embeddingDeployment string
	embeddingProvider   embedding.Provider
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

func (a *LegacyBackendAdapter) WithEmbeddingDeployment(name string) *LegacyBackendAdapter {
	a.embeddingDeployment = name
	return a
}

func (a *LegacyBackendAdapter) WithEmbeddingProvider(provider embedding.Provider) *LegacyBackendAdapter {
	a.embeddingProvider = provider
	return a
}

// semanticEmbedder is a backend that embeds semantic queries; it returns the
// output view it embeds them with.
type semanticEmbedder interface {
	semanticEmbeddingProvider() embedding.Provider
}

// truncatesSemanticQuery reports whether the embedding model would read only
// a prefix of query, which could then match another prompt sharing that
// prefix. It asks the backend's own view, so the answer comes from the call
// that embeds the query next. A provider that cannot tell is trusted.
func (a *LegacyBackendAdapter) truncatesSemanticQuery(ctx context.Context, query string) bool {
	provider := a.embeddingProvider
	if backend, ok := a.backend.(semanticEmbedder); ok && backend.semanticEmbeddingProvider() != nil {
		provider = backend.semanticEmbeddingProvider()
	}
	checker, ok := provider.(embedding.InputChecker)
	if !ok {
		return false
	}
	if ctx == nil {
		ctx = context.Background()
	}
	fits, err := checker.FitsInput(ctx, query)
	if errors.Is(err, binding.ErrCapability) {
		return false
	}
	// A failed check cannot establish that a query fits. Exact cache
	// operations remain independent of this conservative semantic guard.
	return err != nil || !fits
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
) (SemanticRecall, error) {
	if a.truncatesSemanticQuery(ctx, lookup.Identity.SemanticQuery) {
		return SemanticRecall{}, nil
	}
	entries, err := a.backend.LookupSimilar(
		ctx,
		lookup.Identity.SemanticPartitionKey(),
		lookup.Identity.SemanticQuery,
		lookup.Bounds,
	)
	if err != nil {
		return SemanticRecall{}, err
	}
	recall := SemanticRecall{Closest: entries.Closest}
	for _, hit := range entries.Hits {
		age, ageKnown := resultAge(hit)
		recall.Candidates = append(recall.Candidates, CacheResult{
			ResponseBody:  hit.ResponseBody,
			Found:         true,
			HitKind:       HitKindSemantic,
			Source:        CacheSourceL2,
			Similarity:    hit.Similarity,
			Query:         hit.Query,
			NegationGuard: hit.NegationGuard,
			Age:           age,
			AgeKnown:      ageKnown,
			ExpiresAt:     hit.ExpiresAt,
		})
	}
	return recall, nil
}

func resultAge(result LookupResult) (time.Duration, bool) {
	if !result.StoredAt.IsZero() {
		return time.Since(result.StoredAt), true
	}
	return result.Age, result.AgeKnown
}

func (a *LegacyBackendAdapter) StoreSemantic(ctx context.Context, write CacheWrite) error {
	if write.TTL.NoStore || a.truncatesSemanticQuery(ctx, write.Identity.SemanticQuery) {
		return nil
	}
	return a.backend.AddEntry(
		ctx,
		write.RequestID,
		write.Identity.SemanticPartitionKey(),
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
