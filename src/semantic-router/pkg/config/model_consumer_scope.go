package config

// ModelConsumerScope is a preparation-only view of actual model consumers.
// Default public APIs remain available without a default routing entrypoint.
// Catalog declarations remain in the original config for contract validation;
// this view must not be exported as a replacement canonical configuration.
func (c *RouterConfig) ModelConsumerScope() *RouterConfig {
	scoped := c
	if scoped.RoutingScope == "" {
		scoped = c.ConfigForRecipe(c.DefaultRecipe())
	}
	if scoped.RoutingScope != DefaultRecipeName || c.IsRecipeReachableForRouting(DefaultRecipeName) {
		return scoped
	}
	view := *scoped
	view.Signals, view.Projections, view.Decisions, view.DefaultPlugins = Signals{}, Projections{}, nil, nil
	return &view
}
