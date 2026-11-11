package config

import (
	"sort"
	"strings"
)

func (c *RouterConfig) resolveLoRABaseModel(alias string) (string, ModelParams, bool) {
	bases := c.loraBaseModels(alias)
	if len(bases) == 0 {
		return "", ModelParams{}, false
	}
	return bases[0], c.ModelConfig[bases[0]], true
}

// loraBaseModels returns, sorted, the models other than alias that declare a
// LoRA adapter named alias.
func (c *RouterConfig) loraBaseModels(alias string) []string {
	if c == nil || alias == "" {
		return nil
	}
	var bases []string
	for modelName, params := range c.ModelConfig {
		if modelName != alias && modelParamsHasLoRA(params, alias) {
			bases = append(bases, modelName)
		}
	}
	sort.Strings(bases)
	return bases
}

// RoutesModelToBackend reports whether a request naming model reaches one
// backend the Router configures: a model with a backend, or a LoRA adapter
// that exactly one model declares. A name several models declare as a LoRA
// is selected only through a decision's model and lora_name, unless a model
// of that name has its own backend.
func (c *RouterConfig) RoutesModelToBackend(model string) bool {
	if c == nil {
		return false
	}
	if len(c.ModelConfig[model].PreferredEndpoints) == 0 && len(c.loraBaseModels(model)) > 1 {
		return false
	}
	return len(c.GetEndpointsForModel(model)) > 0
}

// UsesExternalGatewayDispatch reports whether a concrete model is declared as
// protocol metadata only. In this mode the external gateway owns the route,
// backend, and credentials; Semantic Router owns only protocol conversion and
// Responses object state.
func (c *RouterConfig) UsesExternalGatewayDispatch(model string) bool {
	if c == nil || c.ModelSelection.Enabled || len(c.Listeners) != 0 {
		return false
	}
	model = strings.TrimSpace(model)
	if model == "" || len(c.GetEndpointsForModel(model)) != 0 {
		return false
	}
	_, configured := c.ModelConfig[model]
	return configured
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
