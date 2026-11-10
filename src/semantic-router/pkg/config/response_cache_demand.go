package config

// ResponseCacheDemand separates storage consumers from semantic model
// consumers among the response_cache plugins that reachable requests run.
// Exact-only plugins use the store without a vector representation. The
// global services view keeps no recipes or decisions: its SemanticCache.InUse
// is the semantic demand of the configuration it was taken from.
func (c *RouterConfig) ResponseCacheDemand() (store, semantic bool) {
	if c == nil || !c.SemanticCache.InUse {
		return false, false
	}
	if c.RoutingScope == GlobalModelScope {
		return true, true
	}
	return c.responseCacheDemand()
}

func (c *RouterConfig) responseCacheDemand() (store, semantic bool) {
	c.visitReachablePlugins(func(plugins []DecisionPlugin) {
		plugin := ResponseCachePluginOf(plugins)
		if plugin == nil || !plugin.Enabled {
			return
		}
		store = true
		if plugin.Mode != ResponseCacheModeExact {
			semantic = true
		}
	})
	return store, semantic
}

func (c *RouterConfig) NeedsSemanticResponseCache() bool {
	_, semantic := c.ResponseCacheDemand()
	return semantic
}

// ResponseCacheConfig is the response_cache plugin a request runs
// (RequestPlugins), nil without one.
func (c *RouterConfig) ResponseCacheConfig(decision *Decision) *ResponseCachePluginConfig {
	return ResponseCachePluginOf(c.RequestPlugins(decision))
}

// ResponseCachePluginOf is the response_cache plugin among a request's
// plugins, nil without one.
func ResponseCachePluginOf(plugins []DecisionPlugin) *ResponseCachePluginConfig {
	return decodePlugin(plugins, DecisionPluginResponseCache, &ResponseCachePluginConfig{})
}
