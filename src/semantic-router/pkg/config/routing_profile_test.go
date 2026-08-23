package config

// scopedRoutingProfileForTest marks programmatically assembled flat routing
// values as one isolated Recipe document. Production root configs must carry
// explicit Recipes; tests of individual routing validators use this helper so
// they do not rely on the removed implicit-default runtime path.
func scopedRoutingProfileForTest(cfg *RouterConfig) *RouterConfig {
	if cfg == nil || cfg.RoutingScope != "" || len(cfg.Recipes) > 0 || len(cfg.Entrypoints) > 0 {
		return cfg
	}
	scoped := *cfg
	scoped.RoutingScope = "test-recipe"
	return &scoped
}
