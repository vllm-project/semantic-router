package memory

import (
	"fmt"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

// MemoryVectorDimensionMismatchError identifies an existing vector store that
// cannot be used with the prepared memory embedding representation.
type MemoryVectorDimensionMismatchError struct {
	Backend           string
	CollectionName    string
	StoredDimension   int
	ExpectedDimension int
}

func (e *MemoryVectorDimensionMismatchError) Error() string {
	return fmt.Sprintf("%s memory collection %s vector dimension mismatch: stored=%d expected=%d",
		e.Backend, e.CollectionName, e.StoredDimension, e.ExpectedDimension)
}

// resolveMemoryEmbeddingDimension resolves the width used by a memory
// backend. A prepared provider is authoritative; deterministic mode is the
// explicit no-model test path; the Candle lookup is retained only for legacy
// callers that construct a store without a prepared provider.
func resolveMemoryEmbeddingDimension(embeddingCfg EmbeddingConfig, configured int) (int, error) {
	if deterministicEmbeddingsEnabled() {
		dimension := embeddingCfg.Dimension
		if dimension <= 0 {
			dimension = configured
		}
		if dimension <= 0 {
			dimension = deterministicEmbeddingDimension(embeddingCfg)
		}
		if dimension <= 0 {
			return 0, fmt.Errorf("deterministic memory embedding dimension must be positive: %d", dimension)
		}
		return dimension, nil
	}

	if embeddingCfg.Provider != nil {
		contractProvider, ok := embeddingCfg.Provider.(embedding.DimensionContractProvider)
		if !ok {
			return 0, fmt.Errorf("prepared embedding provider does not expose a dimension contract")
		}
		contract, err := contractProvider.EmbeddingDimensionContract()
		if err != nil {
			return 0, fmt.Errorf("failed to get embedding dimension contract: %w", err)
		}
		return contract.Resolve(embeddingCfg.Dimension)
	}

	requested := embeddingCfg.Dimension
	if requested <= 0 {
		requested = configured
	}
	return candle_binding.ResolveEmbeddingDimension(string(embeddingCfg.Model), requested)
}

// resolvePreparedEmbeddingDimension applies a prepared provider's contract
// before a memory operation invokes the provider. Providers created before the
// contract API may still be used by unit tests and custom integrations; those
// providers retain their existing option behavior.
func resolvePreparedEmbeddingDimension(provider embedding.Provider, requested int) (int, bool, error) {
	if provider == nil {
		return requested, false, nil
	}
	contractProvider, ok := provider.(embedding.DimensionContractProvider)
	if !ok {
		return requested, false, nil
	}
	contract, err := contractProvider.EmbeddingDimensionContract()
	if err != nil {
		return 0, true, fmt.Errorf("failed to get embedding dimension contract: %w", err)
	}
	dimension, err := contract.Resolve(requested)
	if err != nil {
		return 0, true, err
	}
	return dimension, true, nil
}
