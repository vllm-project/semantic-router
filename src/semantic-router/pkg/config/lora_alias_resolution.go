package config

import "sort"

func canonicalModelOrLoRAExists(
	modelsByName map[string]RoutingModel,
	modelCards []RoutingModel,
	name string,
) bool {
	if _, ok := modelsByName[name]; ok {
		return true
	}
	for _, model := range modelCards {
		if routingModelHasLoRA(model, name) {
			return true
		}
	}
	return false
}

func (c *RouterConfig) resolveLoRABaseModel(alias string) (string, ModelParams, bool) {
	if c == nil || c.ModelConfig == nil || alias == "" {
		return "", ModelParams{}, false
	}

	modelNames := make([]string, 0, len(c.ModelConfig))
	for modelName := range c.ModelConfig {
		if modelName == alias {
			continue
		}
		modelNames = append(modelNames, modelName)
	}
	sort.Strings(modelNames)

	for _, modelName := range modelNames {
		params := c.ModelConfig[modelName]
		if modelParamsHasLoRA(params, alias) {
			return modelName, params, true
		}
	}

	return "", ModelParams{}, false
}

func modelParamsHasLoRA(params ModelParams, alias string) bool {
	for _, adapter := range params.LoRAs {
		if adapter.Name == alias {
			return true
		}
	}
	return false
}

func (c *RouterConfig) collectPreferredEndpoints(endpointNames []string) []VLLMEndpoint {
	if len(endpointNames) == 0 {
		return nil
	}

	endpoints := make([]VLLMEndpoint, 0, len(endpointNames))
	for _, endpointName := range endpointNames {
		if endpoint, found := c.GetEndpointByName(endpointName); found {
			endpoints = append(endpoints, *endpoint)
		}
	}
	return endpoints
}
