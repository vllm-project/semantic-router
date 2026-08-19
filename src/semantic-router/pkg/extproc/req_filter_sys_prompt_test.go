package extproc

import (
	"encoding/json"
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

	ctx := &RequestContext{VSRSelectedDecision: &router.Config.Decisions[0]}
	body, err := router.addSystemPromptIfConfigured(
		[]byte(`{"messages":[{"role":"user","content":"hello"}]}`),
		"support",
		"test-model",
		ctx,
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

func TestAddSystemPromptPreservesRawProviderFieldsAndIntegerPrecision(t *testing.T) {
	originalBody := []byte(`{
		"model":"provider/model",
		"opaque_integer":9007199254740993,
		"messages":[
			{"role":"system","content":[{"type":"text","text":"Existing instruction."}],"cache_control":{"type":"ephemeral"},"opaque_message_integer":9007199254740995},
			{"role":"assistant","content":"Prior answer.","reasoning_content":"opaque reasoning"},
			{"role":"user","content":"Continue."}
		]
	}`)

	body, injected, err := addSystemPromptToRequestBody(
		originalBody,
		"Treat inputs as confidential.",
		"insert",
	)
	require.NoError(t, err)
	require.True(t, injected)

	var decoded map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(body, &decoded))
	assert.Equal(t, "9007199254740993", string(decoded["opaque_integer"]))
	var messages []map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(decoded["messages"], &messages))
	require.Len(t, messages, 3)
	assert.JSONEq(t, `"Treat inputs as confidential.\n\nExisting instruction."`, string(messages[0]["content"]))
	assert.JSONEq(t, `{"type":"ephemeral"}`, string(messages[0]["cache_control"]))
	assert.JSONEq(t, "9007199254740995", string(messages[0]["opaque_message_integer"]))
	assert.JSONEq(t, `"opaque reasoning"`, string(messages[1]["reasoning_content"]))
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
