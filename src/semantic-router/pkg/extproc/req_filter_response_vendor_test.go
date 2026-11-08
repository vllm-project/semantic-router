package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Snowflake Cortex and Cloudflare Workers AI are resolved by provider id, not
// by host: their account-scoped hosts also front APIs whose bodies are not
// these providers' failure envelopes.
func TestResolveResponseVendorByProviderID(t *testing.T) {
	tests := []struct {
		name         string
		providerType string
		baseURL      string
		vendor       llmprotocol.ResponseVendor
	}{
		{name: "canonical snowflake profile", providerType: "snowflake-cortex", vendor: llmprotocol.ResponseVendorSnowflake},
		{name: "snowflake host without the provider id", providerType: "openai", baseURL: "https://my-account.snowflakecomputing.com/api/v2/cortex/v1"},
		{name: "canonical cloudflare profile", providerType: "cloudflare-workers-ai", vendor: llmprotocol.ResponseVendorCloudflare},
		{name: "cloudflare host without the provider id", providerType: "openai", baseURL: "https://api.cloudflare.com/client/v4/accounts/a/ai/v1"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			profile := &config.ProviderProfile{Type: tt.providerType, BaseURL: tt.baseURL}

			// The failure envelope is independent of request shaping, so
			// resolving the vendor must leave the profile's reasoning transport
			// exactly as the catalog resolved it. Asserting the invariant rather
			// than a literal transport keeps this test off catalog data, which
			// differs per provider type and changes over time.
			before := resolveProviderReasoningTransport(profile)

			assert.Equal(t, tt.vendor, resolveResponseVendor(profile))

			assert.Equal(t, before, resolveProviderReasoningTransport(profile))
		})
	}
}

// Every other backend decodes the canonical failure envelope. Members a wire
// contract does not declare need no vendor: Azure decorations, gateway cost
// fields and aggregator metadata are dropped and reported for every backend.
func TestResolveResponseVendorDefaultsToTheCanonicalEnvelope(t *testing.T) {
	tests := []struct {
		name    string
		profile *config.ProviderProfile
	}{
		{name: "legacy endpoint without profile", profile: nil},
		{name: "official openai", profile: &config.ProviderProfile{Type: "openai", BaseURL: "https://api.openai.com/v1"}},
		{name: "azure openai", profile: &config.ProviderProfile{Type: "azure-openai", BaseURL: "https://my-resource.openai.azure.com/openai/v1"}},
		{name: "azure host", profile: &config.ProviderProfile{Type: "openai", BaseURL: "https://my-resource.services.ai.azure.com/openai/v1"}},
		{name: "openrouter", profile: &config.ProviderProfile{Type: "openai", BaseURL: "https://openrouter.ai/api/v1"}},
		{name: "vllm relay", profile: &config.ProviderProfile{Type: "vllm", BaseURL: "https://relay.example.com/v1"}},
		{
			name: "anthropic-format gateway",
			profile: &config.ProviderProfile{
				Type: "anthropic-compatible", Protocol: "anthropic/messages@1", BaseURL: "https://gateway.example.com/anthropic",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Empty(t, resolveResponseVendor(tt.profile))
		})
	}
}

// OpenRouter is resolved by provider id only, and the resolution must not
// disturb the profile's reasoning transport.
func TestResolveResponseVendorOpenRouter(t *testing.T) {
	profile := &config.ProviderProfile{Type: "openrouter", BaseURL: "https://openrouter.ai/api/v1"}
	before := resolveProviderReasoningTransport(profile)

	assert.Equal(t, llmprotocol.ResponseVendorOpenRouter, resolveResponseVendor(profile))

	assert.Equal(t, before, resolveProviderReasoningTransport(profile))
}
