package config

// ConfigForGlobalModelServices removes recipe execution overrides while
// retaining only demanded, globally owned embedding consumers. It is a
// preparation view and must never replace the canonical authoring document.
func (c *RouterConfig) ConfigForGlobalModelServices() *RouterConfig {
	if c == nil {
		return nil
	}
	scoped := c.ConfigForRecipe(&RoutingRecipe{Name: GlobalModelScope})
	scoped.ModelSelection.Enabled = false
	scoped.SemanticCache.InUse = c.NeedsSemanticResponseCache()
	scoped.Memory.InUse = c.usesReachableMemory()
	return scoped
}
