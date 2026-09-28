package dsl

func applyRouteOptions(route *RouteDecl, options []*RouteOpt) {
	for _, option := range options {
		if option.Value == nil || option.Value.Str == nil {
			continue
		}
		switch option.Key {
		case "description":
			route.Description = unquote(*option.Value.Str)
		case "on_unknown":
			route.OnUnknown = unquote(*option.Value.Str)
		}
	}
}
