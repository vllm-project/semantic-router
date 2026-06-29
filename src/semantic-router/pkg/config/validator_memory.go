package config

import "fmt"

// validateMemoryContracts validates the long-term-memory similarity threshold
// wherever it can be configured: the global memory block
// (default_similarity_threshold) and each decision's memory plugin
// (similarity_threshold).
//
// The threshold is a cosine similarity, which is bounded by [0.0, 1.0]. A value
// outside that range is silently accepted today and reaches vector-store
// retrieval unchanged (the Milvus/Qdrant/Valkey stores fall back to and compare
// against it without clamping). A threshold > 1.0 can never be reached by any
// similarity, so memory retrieval never matches despite enabled: true —
// long-term memory is silently disabled. This mirrors the bound the semantic
// cache and RAG plugins already enforce (validateSemanticCacheContracts /
// validateRAGSimilarityThreshold).
func validateMemoryContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}

	if err := validateMemorySimilarityThreshold(
		cfg.Memory.DefaultSimilarityThreshold,
		"global memory default_similarity_threshold",
	); err != nil {
		return err
	}

	for i := range cfg.Decisions {
		decision := &cfg.Decisions[i]
		pluginCfg := decision.GetMemoryConfig()
		if pluginCfg == nil || pluginCfg.SimilarityThreshold == nil {
			continue
		}
		scope := fmt.Sprintf("decision %q memory plugin similarity_threshold", decision.Name)
		if err := validateMemorySimilarityThreshold(*pluginCfg.SimilarityThreshold, scope); err != nil {
			return err
		}
	}
	return nil
}

// validateMemorySimilarityThreshold enforces that a configured memory similarity
// threshold is a valid cosine similarity in [0.0, 1.0]. The global field is a
// plain float32 whose zero value means "unset" (the runtime falls back to a
// default); zero is within range, so unset configs pass without special-casing.
func validateMemorySimilarityThreshold(threshold float32, scope string) error {
	if threshold < 0.0 || threshold > 1.0 {
		return fmt.Errorf("%s must be between 0.0 and 1.0, got %.2f", scope, threshold)
	}
	return nil
}
