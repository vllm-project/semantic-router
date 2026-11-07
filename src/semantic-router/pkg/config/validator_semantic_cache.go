package config

import (
	"fmt"
	"strings"
)

// validateGlobalSemanticCacheContracts validates the store's similarity
// threshold, which bounds recall for requests that match no decision. A
// decision's response_cache plugin bounds its own recall in semantic.match.
//
// The threshold is a cosine similarity, which is bounded by [0.0, 1.0]. A
// threshold > 1.0 can never be reached by any similarity, so the cache would
// never hit despite enabled: true.
func validateGlobalSemanticCacheContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	return validateCacheThreshold(cfg.SemanticCache.SimilarityThreshold, "global.stores.response_cache")
}

func validateCacheMode(mode string, scope string) error {
	switch strings.TrimSpace(mode) {
	case "", ResponseCacheModeSemantic, ResponseCacheModeExact, ResponseCacheModeExactThenSemantic:
		return nil
	default:
		return fmt.Errorf(
			"%s mode must be one of %q, %q, or %q, got %q",
			scope,
			ResponseCacheModeSemantic,
			ResponseCacheModeExact,
			ResponseCacheModeExactThenSemantic,
			mode,
		)
	}
}

// validateCacheThreshold enforces that a configured cache similarity threshold
// is a valid cosine similarity in [0.0, 1.0]. A nil pointer means "unset" and is
// valid (the runtime falls back to a default).
func validateCacheThreshold(threshold *float32, scope string) error {
	if threshold == nil {
		return nil
	}
	if *threshold < 0.0 || *threshold > 1.0 {
		return fmt.Errorf("%s similarity_threshold must be between 0.0 and 1.0, got %.2f", scope, *threshold)
	}
	return nil
}
