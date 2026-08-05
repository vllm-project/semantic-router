package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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
