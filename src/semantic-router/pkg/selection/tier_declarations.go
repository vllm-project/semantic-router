package selection

import "fmt"

// This file centralizes Tier() and ExternalDependencies() implementations
// for all selection algorithms. Keeping them in one place avoids adding lines
// to algorithm files that are already at or over the 800-line hard limit.

// --- Supported-tier algorithms ---

// Tier returns the production readiness tier
func (s *StaticSelector) Tier() AlgorithmTier {
	return TierSupported
}

// ExternalDependencies returns external dependencies (none for static)
func (s *StaticSelector) ExternalDependencies() []Dependency {
	return []Dependency{}
}

// Tier returns the production readiness tier
func (e *EloSelector) Tier() AlgorithmTier {
	return TierSupported
}

// ExternalDependencies returns external dependencies (none for Elo)
func (e *EloSelector) ExternalDependencies() []Dependency {
	return []Dependency{}
}

// Tier returns the production readiness tier
func (r *RouterDCSelector) Tier() AlgorithmTier {
	return TierSupported
}

// ExternalDependencies returns external dependencies (none for RouterDC)
func (r *RouterDCSelector) ExternalDependencies() []Dependency {
	return []Dependency{}
}

// Tier returns the production readiness tier
func (l *LatencyAwareSelector) Tier() AlgorithmTier {
	return TierSupported
}

// ExternalDependencies returns external dependencies (none for latency-aware)
func (l *LatencyAwareSelector) ExternalDependencies() []Dependency {
	return []Dependency{}
}

// Tier returns the production readiness tier
func (m *MultiFactorSelector) Tier() AlgorithmTier {
	return TierSupported
}

// ExternalDependencies returns external dependencies (none for multi_factor)
func (m *MultiFactorSelector) ExternalDependencies() []Dependency {
	return []Dependency{}
}

// Tier returns the production readiness tier
func (h *HybridSelector) Tier() AlgorithmTier {
	return TierSupported
}

// ExternalDependencies returns external dependencies (none for hybrid)
func (h *HybridSelector) ExternalDependencies() []Dependency {
	return []Dependency{}
}

// --- Experimental-tier algorithms ---

// Tier returns the production readiness tier
func (a *AutoMixSelector) Tier() AlgorithmTier {
	return TierExperimental
}

// ExternalDependencies returns external dependencies for AutoMix
func (a *AutoMixSelector) ExternalDependencies() []Dependency {
	deps := []Dependency{}
	if a.config.EnableSelfVerification && a.config.VerifierServerURL != "" {
		deps = append(deps, Dependency{
			Name:        "AutoMix Verifier Server",
			Type:        DependencyExternalService,
			Description: "LLM-based entailment verification for self-verification (arXiv:2310.12963)",
			HealthURL:   a.config.VerifierServerURL + "/health",
			Required:    true,
		})
	}
	return deps
}

// Tier returns the production readiness tier
func (a *MLSelectorAdapter) Tier() AlgorithmTier {
	return TierExperimental
}

// ExternalDependencies returns external dependencies for ML-based selectors
func (a *MLSelectorAdapter) ExternalDependencies() []Dependency {
	return []Dependency{
		{
			Name:        fmt.Sprintf("Pre-trained %s model", a.method),
			Type:        DependencyPretrainedModel,
			Description: "ML model weights for query-to-model classification",
			Required:    true,
		},
	}
}
