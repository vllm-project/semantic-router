package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestResolveProviderReasoningTransport(t *testing.T) {
	tests := []struct {
		name      string
		profile   *config.ProviderProfile
		want      modelcatalog.ReasoningTransport
		wantTop   bool
		wantThink bool
		wantDeep  bool
	}{
		{
			name: "endpoint without profile uses template kwargs",
			want: modelcatalog.ReasoningTransportChatTemplate,
		},
		{
			name:    "vllm uses template kwargs",
			profile: &config.ProviderProfile{Type: "vllm", BaseURL: "http://localhost:8000/v1"},
			want:    modelcatalog.ReasoningTransportChatTemplate,
		},
		{
			name:    "openai uses top-level reasoning effort",
			profile: &config.ProviderProfile{Type: "openai", BaseURL: "https://proxy.example/v1"},
			want:    modelcatalog.ReasoningTransportTopLevelEffort,
			wantTop: true,
		},
		{
			name:      "deepseek uses thinking object and effort",
			profile:   &config.ProviderProfile{Type: "deepseek", BaseURL: "https://private.example/v1"},
			want:      modelcatalog.ReasoningTransportDeepSeekThinking,
			wantTop:   true,
			wantThink: true,
			wantDeep:  true,
		},
		{
			name:      "zai uses thinking object without effort",
			profile:   &config.ProviderProfile{Type: "zai", BaseURL: "https://private.example/v1"},
			want:      modelcatalog.ReasoningTransportThinkingObject,
			wantThink: true,
		},
		{
			name:    "openrouter uses top-level reasoning effort",
			profile: &config.ProviderProfile{Type: "openrouter"},
			want:    modelcatalog.ReasoningTransportTopLevelEffort,
			wantTop: true,
		},
		{
			name:    "generic compatible provider uses template kwargs regardless of hostname",
			profile: &config.ProviderProfile{Type: "openai-compatible", BaseURL: "https://api.openai.com/v1"},
			want:    modelcatalog.ReasoningTransportChatTemplate,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			transport := resolveProviderReasoningTransport(tt.profile)
			assert.Equal(t, tt.want, transport)
			assert.Equal(t, tt.wantTop, usesTopLevelReasoningEffort(transport))
			assert.Equal(t, tt.wantThink, usesThinkingObjectTransport(transport))
			assert.Equal(t, tt.wantDeep, isDeepSeekThinkingTransport(transport))
		})
	}
}
