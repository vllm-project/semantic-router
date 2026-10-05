package config

import (
	"fmt"
	"math"
	"strings"
	"time"
)

const (
	// MaxMemoryPersistenceConcurrency limits worker allocation at startup and reload.
	MaxMemoryPersistenceConcurrency = 64
	// MaxMemoryPersistenceQueue limits queued snapshots at startup and reload.
	MaxMemoryPersistenceQueue = 1024
	// MaxMemoryPersistenceDurationSeconds is the largest whole-second value that
	// can be converted to time.Duration without overflow.
	MaxMemoryPersistenceDurationSeconds = math.MaxInt64 / int64(time.Second)
)

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
	if err := validateGlobalMemoryContracts(cfg); err != nil {
		return err
	}
	return validateDecisionMemoryContracts(cfg)
}

func validateGlobalMemoryContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	if err := validateMemorySimilarityThreshold(
		cfg.Memory.DefaultSimilarityThreshold,
		"global memory default_similarity_threshold",
	); err != nil {
		return err
	}
	if err := validateMemoryRetrievalLimit(cfg.Memory.DefaultRetrievalLimit, false, "global memory default_retrieval_limit"); err != nil {
		return err
	}
	if err := validateMemoryHybridMode(cfg.Memory.HybridMode, "global.stores.memory.hybrid_mode"); err != nil {
		return err
	}
	if err := validateMemoryReflectionContracts(cfg.Memory.Reflection, "global memory reflection"); err != nil {
		return err
	}
	return validateMemoryPersistence(cfg.Memory.Persistence)
}

// Zero values use runtime defaults. Reject invalid resource bounds before any
// persistence workers or queue entries are allocated.
func validateMemoryPersistence(cfg MemoryPersistenceConfig) error {
	if cfg.TimeoutSeconds < 0 || int64(cfg.TimeoutSeconds) > MaxMemoryPersistenceDurationSeconds {
		return fmt.Errorf(
			"global memory persistence timeout_seconds must be between 0 and %d, got %d",
			MaxMemoryPersistenceDurationSeconds,
			cfg.TimeoutSeconds,
		)
	}
	if cfg.Concurrency < 0 || cfg.Concurrency > MaxMemoryPersistenceConcurrency {
		return fmt.Errorf("global memory persistence concurrency must be between 0 and %d, got %d", MaxMemoryPersistenceConcurrency, cfg.Concurrency)
	}
	if cfg.Queue < 0 || cfg.Queue > MaxMemoryPersistenceQueue {
		return fmt.Errorf("global memory persistence queue must be between 0 and %d, got %d", MaxMemoryPersistenceQueue, cfg.Queue)
	}
	if cfg.ShutdownGraceSeconds < 0 || int64(cfg.ShutdownGraceSeconds) > MaxMemoryPersistenceDurationSeconds {
		return fmt.Errorf(
			"global memory persistence shutdown_grace_seconds must be between 0 and %d, got %d",
			MaxMemoryPersistenceDurationSeconds,
			cfg.ShutdownGraceSeconds,
		)
	}
	return nil
}

func validateDecisionMemoryContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	decisions := cfg.AllRoutingDecisions()
	for i := range decisions {
		decision := &decisions[i]
		pluginCfg := decision.GetMemoryConfig()
		if pluginCfg == nil {
			continue
		}
		if pluginCfg.SimilarityThreshold != nil {
			scope := fmt.Sprintf("decision %q memory plugin similarity_threshold", decision.Name)
			if err := validateMemorySimilarityThreshold(*pluginCfg.SimilarityThreshold, scope); err != nil {
				return err
			}
		}
		if pluginCfg.RetrievalLimit != nil {
			scope := fmt.Sprintf("decision %q memory plugin retrieval_limit", decision.Name)
			if err := validateMemoryRetrievalLimit(*pluginCfg.RetrievalLimit, true, scope); err != nil {
				return err
			}
		}
		field := fmt.Sprintf("routing.decisions[%s].plugins[memory].hybrid_mode", decision.Name)
		if err := validateMemoryHybridMode(pluginCfg.HybridMode, field); err != nil {
			return err
		}
		if pluginCfg.Reflection != nil {
			scope := fmt.Sprintf("decision %q memory plugin reflection", decision.Name)
			if err := validateMemoryReflectionContracts(*pluginCfg.Reflection, scope); err != nil {
				return err
			}
		}
	}
	return nil
}

// memoryHybridModes are the fusion methods the hybrid scorer implements. Empty
// selects the default (weighted). Matching is exact because the scorer compares
// the raw string.
var memoryHybridModes = []string{"weighted", "rrf"}

// legacyMemoryHybridModes maps retired values to the mode they always ran as.
var legacyMemoryHybridModes = map[string]string{"rerank": "weighted"}

// validateMemoryHybridMode rejects fusion methods the scorer does not
// implement. Any value other than "rrf" used to run as weighted without a
// warning, so a typo silently changed retrieval scoring.
func validateMemoryHybridMode(mode, field string) error {
	if mode == "" {
		return nil
	}
	for _, accepted := range memoryHybridModes {
		if mode == accepted {
			return nil
		}
	}
	accepted := `"", "` + strings.Join(memoryHybridModes, `", "`) + `"`
	if replacement, ok := legacyMemoryHybridModes[mode]; ok {
		return fmt.Errorf("%s %q is not supported (accepted: %s); use %q, which is how %q has always run",
			field, mode, accepted, replacement, mode)
	}
	return fmt.Errorf("%s %q is not supported (accepted: %s)", field, mode, accepted)
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

// validateMemoryRetrievalLimit rejects non-positive retrieval limits. The
// global field is a plain int whose zero value means "unset" (runtime
// default), while the decision plugin uses a pointer, where an explicit
// zero or negative would reach the vector stores as an unbounded retrieval
// ("a non-positive topK means unlimited results").
func validateMemoryRetrievalLimit(limit int, explicit bool, scope string) error {
	if limit == 0 && !explicit {
		return nil
	}
	if limit <= 0 {
		return fmt.Errorf("%s must be greater than 0, got %d", scope, limit)
	}
	return nil
}

func validateMemoryReflectionContracts(reflection MemoryReflectionConfig, scope string) error {
	if reflection.DedupThreshold < 0.0 || reflection.DedupThreshold > 1.0 {
		return fmt.Errorf("%s dedup_threshold must be between 0.0 and 1.0, got %.2f", scope, reflection.DedupThreshold)
	}
	if reflection.MaxInjectTokens < 0 {
		return fmt.Errorf("%s max_inject_tokens must not be negative, got %d", scope, reflection.MaxInjectTokens)
	}
	if reflection.RecencyDecayDays < 0 {
		return fmt.Errorf("%s recency_decay_days must not be negative, got %d", scope, reflection.RecencyDecayDays)
	}
	return nil
}
