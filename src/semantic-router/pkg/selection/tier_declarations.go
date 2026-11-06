package selection

// This file centralizes Tier() and ExternalDependencies() implementations
// for the selectors that keep no declarations of their own.

// Tier returns the production readiness tier
func (s *StaticSelector) Tier() AlgorithmTier {
	return TierSupported
}

// ExternalDependencies returns external dependencies (none for static)
func (s *StaticSelector) ExternalDependencies() []Dependency {
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
