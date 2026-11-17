package config

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apiregistry"
)

// validateUpstreamSchema checks a provider model's schema against the API
// registry: one of the schemas of the API its api_format names (Chat
// Completions without one), on a model its backend_refs reach. A model on a
// deployment speaks that deployment's own API.
func validateUpstreamSchema(model CanonicalProviderModel, modelIndex int) error {
	if model.Schema == "" {
		return nil
	}
	path := fmt.Sprintf("providers.models[%d].schema", modelIndex)
	if model.Deployment != "" {
		return fmt.Errorf("%s: a model on a deployment speaks that deployment's API; schema applies to a model its backend_refs reach", path)
	}
	api, err := apiregistry.Resolve(effectiveAPIFormat(model.APIFormat))
	if err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	if _, err := api.UpstreamSchema(model.Schema); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}

func effectiveAPIFormat(apiFormat string) string {
	if apiFormat == "" {
		return APIFormatChatCompletions
	}
	return CanonicalAPIFormat(apiFormat)
}

// UpstreamSchema is the schema a provider model's backends speak its API in:
// the schema it names, else the API's upstream default (systemone/0.2 for
// systemone). It is empty for a model the configuration does not declare.
func (c *RouterConfig) UpstreamSchema(model string) string {
	if c == nil {
		return ""
	}
	params, ok := c.ModelConfig[model]
	if !ok {
		return ""
	}
	if params.Schema != "" {
		return params.Schema
	}
	api, ok := apiregistry.Lookup(effectiveAPIFormat(params.APIFormat))
	if !ok {
		return ""
	}
	return api.UpstreamDefault
}
