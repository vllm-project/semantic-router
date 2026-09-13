package authz

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// StaticConfigProvider reads LLM API keys from the router's static YAML config.
// Credentials are keyed by both model alias and provider catalog identity.
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

// GetKey returns the credential attached to the selected provider instance.
func (p *StaticConfigProvider) GetKey(provider LLMProvider, model string, _ map[string]string) string {
	if p.config == nil {
		return ""
	}
	return p.config.GetModelAccessKeyForProvider(model, string(provider))
}

// HeadersToStrip returns nil — static config doesn't inject any headers.
func (p *StaticConfigProvider) HeadersToStrip() []string {
	return nil
}
