package config

// ResponseCacheDemand separates storage consumers from semantic model
// consumers. Global defaults alone must not provision an unused embedding.
// Exact-only routes use the store without a vector representation.
func (c *RouterConfig) ResponseCacheDemand() (store, semantic bool) {
	if c == nil || !c.SemanticCache.Enabled {
		return false, false
	}
	if !c.HasRoutingDecisions() {
		// Preserve the request path's global-cache behavior without decisions.
		return true, true
	}
	for _, ref := range c.RoutingDecisionRefs() {
		if !c.IsRecipeReachableForRouting(ref.Recipe) {
			continue
		}
		plugin := ref.Decision.GetResponseCacheConfig()
		if plugin == nil || !plugin.Enabled {
			continue
		}
		store = true
		if plugin.Mode != ResponseCacheModeExact {
			semantic = true
		}
	}
	return store, semantic
}

func (c *RouterConfig) NeedsSemanticResponseCache() bool {
	_, semantic := c.ResponseCacheDemand()
	return semantic
}
