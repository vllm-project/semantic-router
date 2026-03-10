package authz

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// StaticConfigProvider reads LLM API keys from the router's static YAML config.
// Each model entry can have an access_key field; this provider returns that key
// for the requested model regardless of the LLM provider enum (since the static
// config is model-scoped, not provider-scoped).
//
// This is the fallback provider — used when no auth backend injects
// per-user keys, or for models that don't require per-user auth.
type StaticConfigProvider struct {
	config *config.RouterConfig
}

// NewStaticConfigProvider creates a provider that reads keys from the router config.
func NewStaticConfigProvider(cfg *config.RouterConfig) *StaticConfigProvider {
	return &StaticConfigProvider{config: cfg}
}

func (p *StaticConfigProvider) Name() string {
	return "static-config"
}

// GetKey returns the access_key from the router config for the given model.
// The provider parameter is ignored because static config keys are model-scoped:
// each model entry has a single access_key regardless of whether it's OpenAI or Anthropic.
func (p *StaticConfigProvider) GetKey(_ LLMProvider, model string, _ map[string]string) string {
	if p.config == nil {
		return ""
	}
	return p.config.GetModelAccessKey(model)
}

// HeadersToStrip returns nil — static config doesn't inject any headers.
func (p *StaticConfigProvider) HeadersToStrip() []string {
	return nil
}
