package config

import (
	"errors"
	"fmt"
	"sort"
)

func (c *RouterConfig) selectResolvableEndpointForModel(modelName string) (VLLMEndpoint, string, bool, error) {
	endpoints := c.GetEndpointsForModel(modelName)
	if len(endpoints) == 0 {
		return VLLMEndpoint{}, "", false, nil
	}

	var resolutionErrs []error
	for _, endpoint := range endpointsByPriority(endpoints) {
		addr, err := endpoint.ResolveAddress(c.ProviderProfiles)
		if err == nil {
			return endpoint, addr, true, nil
		}
		resolutionErrs = append(resolutionErrs, fmt.Errorf("%s: %w", endpoint.Name, err))
	}

	return VLLMEndpoint{}, "", false, fmt.Errorf(
		"no resolvable endpoint for model %q: %w",
		modelName,
		errors.Join(resolutionErrs...),
	)
}

func endpointsByPriority(endpoints []VLLMEndpoint) []VLLMEndpoint {
	ordered := append([]VLLMEndpoint(nil), endpoints...)
	sort.SliceStable(ordered, func(i, j int) bool {
		return ordered[i].Weight > ordered[j].Weight
	})
	return ordered
}
