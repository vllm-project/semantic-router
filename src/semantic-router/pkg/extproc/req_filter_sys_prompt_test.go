package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestAddSystemPromptUsesRouterConfigBeforeGlobal(t *testing.T) {
	restoreGlobalConfig := replaceExtProcGlobalConfigForTest(&config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				decisionWithSystemPrompt("support", "global prompt"),
			},
		},
	})
	defer restoreGlobalConfig()

	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Decisions: []config.Decision{
					decisionWithSystemPrompt("support", "router prompt"),
				},
			},
		},
	}

	body, err := router.addSystemPromptIfConfigured(
		[]byte(`{"messages":[{"role":"user","content":"hello"}]}`),
		"support",
		"test-model",
		&RequestContext{},
	)

	require.NoError(t, err)
	assert.Contains(t, string(body), "router prompt")
	assert.NotContains(t, string(body), "global prompt")
}

func TestAddSystemPromptDoesNotUseGlobalWhenRouterConfigMissesDecision(t *testing.T) {
	restoreGlobalConfig := replaceExtProcGlobalConfigForTest(&config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				decisionWithSystemPrompt("support", "global prompt"),
			},
		},
	})
	defer restoreGlobalConfig()

	router := &OpenAIRouter{
		Config: &config.RouterConfig{},
	}
	originalBody := []byte(`{"messages":[{"role":"user","content":"hello"}]}`)

	body, err := router.addSystemPromptIfConfigured(
		originalBody,
		"support",
		"test-model",
		&RequestContext{},
	)

	require.NoError(t, err)
	assert.JSONEq(t, string(originalBody), string(body))
}

func TestAddSystemPromptMissingDecisionDoesNotPanic(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{},
	}
	originalBody := []byte(`{"messages":[{"role":"user","content":"hello"}]}`)

	body, err := router.addSystemPromptIfConfigured(
		originalBody,
		"missing",
		"test-model",
		&RequestContext{},
	)

	require.NoError(t, err)
	assert.JSONEq(t, string(originalBody), string(body))
}

func decisionWithSystemPrompt(name string, prompt string) config.Decision {
	return config.Decision{
		Name: name,
		Plugins: []config.DecisionPlugin{
			{
				Type: config.DecisionPluginSystemPrompt,
				Configuration: config.MustStructuredPayload(map[string]interface{}{
					"enabled":       true,
					"system_prompt": prompt,
					"mode":          "insert",
				}),
			},
		},
	}
}
