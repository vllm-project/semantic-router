package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
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

// The gateway allowance is opt-in per profile: a backend earns it only by
// declaring the relay, and its wire format selects which relay contract
// applies. A plain backend keeps the strict contract.
func TestResolveResponseVendorCompatGateway(t *testing.T) {
	tests := []struct {
		name    string
		profile *config.ProviderProfile
		vendor  llmprotocol.ResponseVendor
	}{
		{
			name:    "vllm profile with the gateway opt-in",
			profile: &config.ProviderProfile{Type: "vllm", BaseURL: "https://relay.example.com/v1", CompatGateway: true},
			vendor:  llmprotocol.ResponseVendorOpenAICompatGateway,
		},
		{
			name: "openai-compatible Responses relay",
			profile: &config.ProviderProfile{
				Type: "openai-compatible", Protocol: "openai/responses@1", BaseURL: "https://relay.example.com/v1", CompatGateway: true,
			},
			vendor: llmprotocol.ResponseVendorOpenAICompatGateway,
		},
		{
			name: "anthropic profile pointed at a Messages gateway",
			profile: &config.ProviderProfile{
				Type: "anthropic", Protocol: "anthropic/messages@1", BaseURL: "https://gateway.example.com/anthropic", CompatGateway: true,
			},
			vendor: llmprotocol.ResponseVendorAnthropicCompatGateway,
		},
		{
			name: "anthropic-compatible gateway",
			profile: &config.ProviderProfile{
				Type: "anthropic-compatible", Protocol: "anthropic/messages@1", BaseURL: "https://gateway.example.com/anthropic", CompatGateway: true,
			},
			vendor: llmprotocol.ResponseVendorAnthropicCompatGateway,
		},
		{
			name: "vllm serving Messages through a gateway",
			profile: &config.ProviderProfile{
				Type: "vllm", Protocol: "anthropic/messages@1", BaseURL: "https://relay.example.com", CompatGateway: true,
			},
			vendor: llmprotocol.ResponseVendorAnthropicCompatGateway,
		},
		{
			name:    "plain vllm profile without the opt-in",
			profile: &config.ProviderProfile{Type: "vllm", BaseURL: "https://vllm.example.com/v1"},
		},
		{
			name:    "plain anthropic profile without the opt-in",
			profile: &config.ProviderProfile{Type: "anthropic", Protocol: "anthropic/messages@1", BaseURL: "https://api.anthropic.com"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.vendor, resolveResponseVendor(tt.profile))
		})
	}
}

// Config load rejects compat_gateway exactly where it would have no effect, so
// every provider type and protocol it accepts must reach a relay contract, and
// every one it rejects must keep its own.
func TestCompatGatewayValidationMatchesVendorResolution(t *testing.T) {
	registry, err := modelcatalog.BuiltIn()
	if err != nil {
		t.Fatal(err)
	}
	for _, provider := range registry.Providers() {
		for _, protocol := range provider.Protocols {
			profile := &config.ProviderProfile{Type: provider.ID, Protocol: protocol, CompatGateway: true}
			vendor := resolveResponseVendor(profile)
			relay := vendor == llmprotocol.ResponseVendorOpenAICompatGateway ||
				vendor == llmprotocol.ResponseVendorAnthropicCompatGateway
			accepted := config.ValidateCompatGateway(provider.ID, protocol) == nil
			if accepted != relay {
				t.Fatalf("%s %s: config accepts compat_gateway=%v but the response vendor is %q", provider.ID, protocol, accepted, vendor)
			}
		}
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
		// A vLLM profile without the gateway opt-in keeps the strict contract.
		{name: "vllm without the gateway opt-in", profile: &config.ProviderProfile{Type: "vllm", BaseURL: "https://vllm.example.com/v1"}},
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
