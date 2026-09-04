package extproc

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestThinkingObjectReasoningTransport(t *testing.T) {
	router := newThinkingObjectReasoningRouter()

	t.Run("hosted provider uses top-level thinking object", func(t *testing.T) {
		request := setReasoningModeForProvider(
			t,
			router,
			"glm-model",
			nil,
			true,
			"test",
			&config.ProviderProfile{Type: "zai"},
		)
		assertThinkingObjectReasoningRequest(t, request, "enabled")
	})

	t.Run("hosted provider explicitly disables thinking", func(t *testing.T) {
		request := setReasoningModeForProvider(
			t,
			router,
			"glm-model",
			nil,
			false,
			"test",
			&config.ProviderProfile{Type: "zai"},
		)
		assertThinkingObjectReasoningRequest(t, request, "disabled")
	})

	t.Run("local vllm keeps model chat-template syntax", func(t *testing.T) {
		request := setReasoningModeForProvider(
			t,
			router,
			"glm-model",
			nil,
			true,
			"test",
			localVLLMProviderProfile(),
		)
		assertChatTemplateReasoningField(t, request, "enable_thinking", true)
		_, hasThinking := request["thinking"]
		assert.False(t, hasThinking)
	})

	t.Run("hosted shape removes incompatible local template fields", func(t *testing.T) {
		requestBody, err := json.Marshal(map[string]interface{}{
			"model":    "glm-model",
			"messages": []map[string]string{{"role": "user", "content": "hello"}},
			"chat_template_kwargs": map[string]interface{}{
				"enable_thinking": true,
				"other":           "local-only",
			},
			"reasoning_effort": "high",
		})
		require.NoError(t, err)

		modifiedBody, err := router.setReasoningModeToRequestBodyForProvider(
			requestBody,
			true,
			router.Config.GetDecisionByName("test"),
			&config.ProviderProfile{Type: "zai"},
		)
		require.NoError(t, err)

		request := unmarshalReasoningRequest(t, modifiedBody)
		assertThinkingObjectReasoningRequest(t, request, "enabled")
	})
}

func newThinkingObjectReasoningRouter() *OpenAIRouter {
	return newReasoningRouter(
		config.ReasoningConfig{
			DefaultReasoningEffort: "medium",
			ReasoningFamilies: map[string]config.ReasoningFamilyConfig{
				"glm": {Type: "chat_template_kwargs", Parameter: "enable_thinking"},
			},
		},
		[]config.Decision{reasoningDecision("test", "", 0, "glm-model", boolPtr(true), "high")},
		map[string]config.ModelParams{
			"glm-model": {ReasoningFamily: "glm"},
		},
	)
}

func assertThinkingObjectReasoningRequest(t *testing.T, request map[string]interface{}, thinkingType string) {
	t.Helper()
	thinking, ok := request["thinking"].(map[string]interface{})
	require.True(t, ok, "thinking should be an object")
	assert.Equal(t, thinkingType, thinking["type"])
	_, hasEffort := request["reasoning_effort"]
	assert.False(t, hasEffort)
	_, hasTemplateKwargs := request["chat_template_kwargs"]
	assert.False(t, hasTemplateKwargs)
}
