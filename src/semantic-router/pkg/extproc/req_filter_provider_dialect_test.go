package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestResolveOpenAIBackendDialect(t *testing.T) {
	tests := []struct {
		name                      string
		profile                   *config.ProviderProfile
		wantKind                  openAIBackendDialectKind
		wantTopLevelEffort        bool
		wantTopLevelDeepSeekThink bool
	}{
		{
			name:               "legacy endpoint without profile is vllm",
			wantKind:           openAIBackendDialectVLLM,
			wantTopLevelEffort: false,
		},
		{
			name:               "local openai-compatible provider is generic",
			profile:            &config.ProviderProfile{Type: "openai", BaseURL: "http://localhost:8000/v1"},
			wantKind:           openAIBackendDialectGenericOpenAICompat,
			wantTopLevelEffort: false,
		},
		{
			name:               "official openai uses top-level reasoning effort",
			profile:            &config.ProviderProfile{Type: "openai", BaseURL: "https://api.openai.com/v1"},
			wantKind:           openAIBackendDialectOfficialOpenAI,
			wantTopLevelEffort: true,
		},
		{
			name:                      "official deepseek uses top-level 'thinking' and effort",
			profile:                   &config.ProviderProfile{Type: "openai", BaseURL: "https://api.deepseek.com"},
			wantKind:                  openAIBackendDialectOfficialDeepSeek,
			wantTopLevelEffort:        true,
			wantTopLevelDeepSeekThink: true,
		},
		{
			name:               "openrouter uses top-level reasoning effort",
			profile:            &config.ProviderProfile{Type: "openai", BaseURL: "https://openrouter.ai/api/v1"},
			wantKind:           openAIBackendDialectOpenRouter,
			wantTopLevelEffort: true,
		},
		{
			name:               "unknown openai-compatible provider is generic",
			profile:            &config.ProviderProfile{Type: "openai", BaseURL: "https://proxy.example.com/v1"},
			wantKind:           openAIBackendDialectGenericOpenAICompat,
			wantTopLevelEffort: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dialect := resolveOpenAIBackendDialect(tt.profile)
			assert.Equal(t, tt.wantKind, dialect.kind)
			assert.Equal(t, tt.wantTopLevelEffort, dialect.usesTopLevelReasoningEffort())
			assert.Equal(t, tt.wantTopLevelDeepSeekThink, dialect.usesDeepSeekOfficialReasoning())
		})
	}
}

func TestResolveOpenAIBackendDialectAzureHosts(t *testing.T) {
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
			dialect := resolveOpenAIBackendDialect(&config.ProviderProfile{Type: tt.providerType, BaseURL: tt.baseURL})

			assert.Equal(t, openAIBackendDialectAzureOpenAI, dialect.kind)
			assert.Equal(t, llmprotocol.ResponseVendorAzure, dialect.vendorExtensionProvider())
			// Azure request shaping is unchanged from the generic dialect.
			assert.False(t, dialect.usesTopLevelReasoningEffort())
			assert.False(t, dialect.usesDeepSeekOfficialReasoning())
		})
	}
}

// Every other backend keeps the strict contract: no vendor allowance at all.
func TestResolveOpenAIBackendDialectGrantsNoVendorAllowanceByDefault(t *testing.T) {
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
			assert.Empty(t, resolveOpenAIBackendDialect(tt.profile).vendorExtensionProvider())
		})
	}
}
