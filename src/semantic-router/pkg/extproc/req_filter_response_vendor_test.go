package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestResolveResponseVendorAzure(t *testing.T) {
	tests := []struct {
		name         string
		providerType string
		baseURL      string
	}{
		{name: "canonical azure profile", providerType: "azure-openai"},
		{name: "azure openai host", providerType: "openai", baseURL: "https://my-resource.openai.azure.com/openai/v1"},
		{name: "azure ai foundry host", providerType: "openai", baseURL: "https://my-resource.services.ai.azure.com/openai/v1"},
		{name: "cognitive services host", providerType: "openai", baseURL: "https://my-resource.cognitiveservices.azure.com/openai/v1"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			profile := &config.ProviderProfile{Type: tt.providerType, BaseURL: tt.baseURL}

			// Response decoration is independent of request shaping, so
			// resolving the vendor must leave the profile's reasoning transport
			// exactly as the catalog resolved it. Asserting the invariant rather
			// than a literal transport keeps this test off catalog data, which
			// differs per provider type and changes over time.
			before := resolveProviderReasoningTransport(profile)

			assert.Equal(t, llmprotocol.ResponseVendorAzure, resolveResponseVendor(profile))

			assert.Equal(t, before, resolveProviderReasoningTransport(profile))
		})
	}
}

// Snowflake Cortex is resolved by provider id, not by host: its account-scoped
// host also fronts the Snowflake SQL API, whose bodies are not this provider's
// failure envelope, so a host match would classify those as a vendor envelope.
func TestResolveResponseVendorSnowflake(t *testing.T) {
	tests := []struct {
		name         string
		providerType string
		baseURL      string
		vendor       llmprotocol.ResponseVendor
	}{
		{name: "canonical snowflake profile", providerType: "snowflake-cortex", vendor: llmprotocol.ResponseVendorSnowflake},
		{name: "snowflake host without the provider id", providerType: "openai", baseURL: "https://my-account.snowflakecomputing.com/api/v2/cortex/v1"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			profile := &config.ProviderProfile{Type: tt.providerType, BaseURL: tt.baseURL}

			// See TestResolveResponseVendorAzure: resolving the vendor must not
			// disturb the profile's reasoning transport.
			before := resolveProviderReasoningTransport(profile)

			assert.Equal(t, tt.vendor, resolveResponseVendor(profile))

			assert.Equal(t, before, resolveProviderReasoningTransport(profile))
		})
	}
}

// Every other backend keeps the strict contract: no vendor allowance at all.
func TestResolveResponseVendorGrantsNoAllowanceByDefault(t *testing.T) {
	tests := []struct {
		name    string
		profile *config.ProviderProfile
	}{
		{name: "legacy endpoint without profile", profile: nil},
		{name: "official openai", profile: &config.ProviderProfile{Type: "openai", BaseURL: "https://api.openai.com/v1"}},
		{name: "official deepseek", profile: &config.ProviderProfile{Type: "openai", BaseURL: "https://api.deepseek.com/v1"}},
		{name: "openrouter", profile: &config.ProviderProfile{Type: "openai", BaseURL: "https://openrouter.ai/api/v1"}},
		{name: "generic openai compatible", profile: &config.ProviderProfile{Type: "openai", BaseURL: "https://llm.example.com/v1"}},
		// A host that merely mentions azure is not an Azure endpoint.
		{name: "azure lookalike host", profile: &config.ProviderProfile{Type: "openai", BaseURL: "https://azure.example.com/v1"}},
		// Suffix matching must not fire on a bare label match either.
		{name: "azure lookalike suffix", profile: &config.ProviderProfile{Type: "openai", BaseURL: "https://notopenai.azure.com.evil.test/v1"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Empty(t, resolveResponseVendor(tt.profile))
		})
	}
}
