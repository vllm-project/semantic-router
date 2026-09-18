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
	scoped.SemanticCache.Enabled = c.NeedsSemanticResponseCache()
	for _, ref := range c.RoutingDecisionRefs() {
		if !c.IsRecipeReachableForRouting(ref.Recipe) {
			continue
		}
		if plugin := ref.Decision.GetToolSelectionConfig(); plugin != nil && plugin.Enabled {
			scoped.Tools.Enabled = true
		}
		if plugin := ref.Decision.GetMemoryConfig(); plugin != nil && plugin.Enabled {
			scoped.Memory.Enabled = true
		}
	}
	return scoped
}
