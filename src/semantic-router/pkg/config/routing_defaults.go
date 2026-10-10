package config

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"

// RoutingDefaults are shared policy defaults from global.router. They are
// separate from IntelligentRouting, which describes the default recipe.
type RoutingDefaults struct {
	Strategy RoutingStrategy
	Fallback *fallback.FallbackPolicy
}

func (d RoutingDefaults) resolveStrategy(local RoutingStrategy) RoutingStrategy {
	if local != "" {
		return local
	}
	// Preserve omission in authoring/export. The decision engine applies its
	// built-in priority strategy when neither scope configures one.
	return d.Strategy
}

func (d RoutingDefaults) resolveFallback(local *fallback.FallbackPolicy) *fallback.FallbackPolicy {
	if local == nil && d.Fallback == nil {
		return nil
	}
	base := fallback.DefaultPolicy()
	if d.Fallback != nil {
		base = d.Fallback.WithDefaults()
	}
	if local != nil {
		base = local.Inherit(base)
	}
	return base.Clone()
}
