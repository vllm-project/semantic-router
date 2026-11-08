package config

// registeredModel is the built-in model a path, alias or Hub repository
// names, nil for any other model.
func registeredModel(model string) *ModelSpec {
	if spec := GetModelByPath(model); spec != nil {
		return spec
	}
	for i := range DefaultModelRegistry {
		if DefaultModelRegistry[i].RepoID == model {
			return &DefaultModelRegistry[i]
		}
	}
	return nil
}
