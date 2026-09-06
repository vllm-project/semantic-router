package config

import (
	"fmt"
	"strings"
)

// validateSemanticCacheContracts validates the semantic-cache similarity
// threshold wherever it can be configured: the global semantic_cache block and
// each decision's semantic-cache plugin.
//
// The threshold is a cosine similarity, which is bounded by [0.0, 1.0]. A value
// outside that range is silently accepted today and reaches the runtime
// unchanged (GetCacheSimilarityThreshold / GetCacheSimilarityThresholdForDecision
// do not clamp it). A threshold > 1.0 can never be reached by any similarity, so
// the cache never hits despite enabled: true — caching is silently disabled.
// This mirrors the bound the RAG plugin already enforces
// (validateRAGSimilarityThreshold).
func validateSemanticCacheContracts(cfg *RouterConfig) error {
	if err := validateGlobalSemanticCacheContracts(cfg); err != nil {
		return err
	}
	return validateDecisionSemanticCacheContracts(cfg)
}

func validateGlobalSemanticCacheContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	if err := validateCacheThreshold(cfg.SemanticCache.SimilarityThreshold, "global semantic_cache"); err != nil {
		return err
	}
	return validatePolarityGuard(cfg)
}

// validatePolarityGuard enforces the polarity guard contract at config load: a
// known mode, an NLI contradiction threshold in [0.0, 1.0], and — when an NLI
// mode is selected on an enabled cache — a configured hallucination explainer,
// because the NLI tier binds that shared model. Failing here is deliberate:
// the issue contract is "validate early", not "discover at the first lookup".
func validatePolarityGuard(cfg *RouterConfig) error {
	guard := cfg.SemanticCache.PolarityGuard
	if guard == nil {
		return nil
	}
	const scope = "global semantic_cache polarity_guard"
	mode := guard.NormalizedMode()
	if !polarityGuardModeSupported(mode) {
		return fmt.Errorf(
			"%s mode must be one of %q, %q, or %q, got %q",
			scope,
			PolarityGuardModeLexical,
			PolarityGuardModeNLI,
			PolarityGuardModeLexicalNLI,
			guard.Mode,
		)
	}
	if t := guard.NLI.ContradictionThreshold; t != nil && (*t < 0.0 || *t > 1.0) {
		return fmt.Errorf("%s nli.contradiction_threshold must be between 0.0 and 1.0, got %.2f", scope, *t)
	}
	if cfg.SemanticCache.Enabled && guard.UsesNLI() &&
		strings.TrimSpace(cfg.HallucinationMitigation.NLIModel.ModelID) == "" {
		return fmt.Errorf(
			"%s mode %q requires an NLI model: configure global.model_catalog.modules.hallucination_mitigation.explainer (model_ref or model_id)",
			scope, mode,
		)
	}
	return nil
}

func validateDecisionSemanticCacheContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	decisions := cfg.Decisions
	for i := range decisions {
		decision := &decisions[i]
		pluginCfg := decision.GetResponseCacheConfig()
		if pluginCfg == nil {
			continue
		}
		scope := fmt.Sprintf("decision %q response_cache plugin", decision.Name)
		if err := validateCacheMode(pluginCfg.Mode, scope); err != nil {
			return err
		}
		if err := validateCacheThreshold(pluginCfg.EffectiveSimilarityThreshold(), scope); err != nil {
			return err
		}
	}
	return nil
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
