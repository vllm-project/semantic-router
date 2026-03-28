package extproc

// semanticCacheEnabledForScope resolves whether semantic cache should be active
// for the current request/response path.
//
// When routing decisions are configured, semantic-cache is route-local and must
// be enabled explicitly on the matched decision. If no decision matched, we do
// not fall back to the global semantic cache toggle.
//
// When no decisions are configured, preserve the legacy/global semantic cache
// behavior driven by RouterConfig.SemanticCache.Enabled.
func (r *OpenAIRouter) semanticCacheEnabledForScope(decisionName string) bool {
	if r == nil || r.Config == nil {
		return true
	}
	if decisionName != "" {
		return r.Config.IsCacheEnabledForDecision(decisionName)
	}
	if len(r.Config.Decisions) > 0 {
		return false
	}
	return r.Config.Enabled
}
