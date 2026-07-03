package config

// ModelForwardsAuthorization reports whether any backend for the given model
// opts into forward_authorization_header. Used to enforce the per-request
// Authorization requirement consistently, including before looper re-dispatch.
func (c *RouterConfig) ModelForwardsAuthorization(modelName string) bool {
	if c == nil {
		return false
	}
	params, ok := c.ModelConfig[modelName]
	if !ok {
		return false
	}
	for _, endpointName := range params.PreferredEndpoints {
		profile, err := c.GetProviderProfileForEndpoint(endpointName)
		if err != nil || profile == nil {
			continue
		}
		if profile.ForwardAuthorizationHeader {
			return true
		}
	}
	return false
}
