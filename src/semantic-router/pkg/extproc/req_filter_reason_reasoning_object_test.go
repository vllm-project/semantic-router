package extproc

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestReasoningObjectTransport(t *testing.T) {
	router := newReasoningObjectRouter()
	profile := &config.ProviderProfile{Type: "openrouter"}

	t.Run("effort family becomes reasoning effort", func(t *testing.T) {
		request := setReasoningModeForProvider(
			t, router, "effort-model", nil, true, "effort", profile,
		)
		assertReasoningObject(t, request, map[string]interface{}{"effort": "high"})
	})

	t.Run("boolean family becomes reasoning enabled", func(t *testing.T) {
		request := setReasoningModeForProvider(
			t, router, "boolean-model", nil, true, "boolean", profile,
		)
		assertReasoningObject(t, request, map[string]interface{}{"enabled": true})
	})

	t.Run("disabled reasoning removes conflicting controls", func(t *testing.T) {
		body, err := json.Marshal(map[string]interface{}{
			"model":            "effort-model",
			"messages":         []map[string]string{{"role": "user", "content": "hello"}},
			"reasoning_effort": "low",
			"reasoning": map[string]interface{}{
				"effort": "max", "max_tokens": 4096, "exclude": true,
			},
			"chat_template_kwargs": map[string]interface{}{"reasoning_effort": "medium"},
		})
		require.NoError(t, err)

		modified, err := router.setReasoningModeToRequestBodyForProvider(
			body, false, router.Config.GetDecisionByName("effort"), profile,
		)
		require.NoError(t, err)
		request := unmarshalReasoningRequest(t, modified)
		assertReasoningObject(
			t, request, map[string]interface{}{"enabled": false, "exclude": true},
		)
	})
}

func TestTopLevelBooleanReasoningTransport(t *testing.T) {
	router := newReasoningObjectRouter()
	profile := &config.ProviderProfile{
		Type:               "dashscope",
		ReasoningTransport: modelcatalog.ReasoningTransportTopLevelBoolean,
	}

	enabled := setReasoningModeForProvider(
		t, router, "boolean-model", nil, true, "boolean", profile,
	)
	assert.Equal(t, true, enabled["enable_thinking"])
	_, hasTemplate := enabled["chat_template_kwargs"]
	assert.False(t, hasTemplate)

	disabled := setReasoningModeForProvider(
		t, router, "boolean-model", nil, false, "boolean", profile,
	)
	assert.Equal(t, false, disabled["enable_thinking"])
	_, hasTemplate = disabled["chat_template_kwargs"]
	assert.False(t, hasTemplate)
}

func TestReasoningEffortWithIndependentActivationControl(t *testing.T) {
	router := newReasoningRouter(
		config.ReasoningConfig{
			ReasoningFamilies: map[string]config.ReasoningFamilyConfig{
				"qwen3.8": {
					Type: config.ReasoningFamilyTypeReasoningEffort, Parameter: "reasoning_effort",
					ActivationParameter: "enable_thinking",
					Levels:              []string{"low", "medium", "xhigh"}, Default: "xhigh",
				},
			},
		},
		[]config.Decision{reasoningDecision("qwen", "", 0, "qwen-model", boolPtr(true), "xhigh")},
		map[string]config.ModelParams{"qwen-model": {ReasoningFamily: "qwen3.8"}},
	)

	local := setReasoningModeForProvider(
		t, router, "qwen-model", nil, true, "qwen", localVLLMProviderProfile(),
	)
	assertChatTemplateReasoningField(t, local, "reasoning_effort", "xhigh")
	assertChatTemplateReasoningField(t, local, "enable_thinking", true)

	disabled := setReasoningModeForProvider(
		t, router, "qwen-model", nil, false, "qwen", localVLLMProviderProfile(),
	)
	assertChatTemplateReasoningField(t, disabled, "enable_thinking", false)
	kwargs := disabled["chat_template_kwargs"].(map[string]interface{})
	_, hasEffort := kwargs["reasoning_effort"]
	assert.False(t, hasEffort)

	hosted := setReasoningModeForProvider(
		t,
		router,
		"qwen-model",
		nil,
		true,
		"qwen",
		&config.ProviderProfile{ReasoningTransport: modelcatalog.ReasoningTransportTopLevelEffort},
	)
	assert.Equal(t, "xhigh", hosted["reasoning_effort"])
	assert.Equal(t, true, hosted["enable_thinking"])
}

func newReasoningObjectRouter() *OpenAIRouter {
	return newReasoningRouter(
		config.ReasoningConfig{
			ReasoningFamilies: map[string]config.ReasoningFamilyConfig{
				"effort": {
					Type: "reasoning_effort", Parameter: "reasoning_effort",
					Levels: []string{"none", "low", "high"}, Default: "high", Disabled: "none",
				},
				"boolean": {
					Type: "chat_template_kwargs", Parameter: "enable_thinking",
					Levels: []string{"disabled", "enabled"}, Default: "enabled", Disabled: "disabled",
				},
			},
		},
		[]config.Decision{
			reasoningDecision("effort", "", 0, "effort-model", boolPtr(true), "high"),
			reasoningDecision("boolean", "", 0, "boolean-model", boolPtr(true), ""),
		},
		map[string]config.ModelParams{
			"effort-model":  {ReasoningFamily: "effort"},
			"boolean-model": {ReasoningFamily: "boolean"},
		},
	)
}

func assertReasoningObject(
	t *testing.T,
	request map[string]interface{},
	want map[string]interface{},
) {
	t.Helper()
	assert.Equal(t, want, request["reasoning"])
	_, hasEffort := request["reasoning_effort"]
	assert.False(t, hasEffort)
	_, hasTemplate := request["chat_template_kwargs"]
	assert.False(t, hasTemplate)
	_, hasThinking := request["thinking"]
	assert.False(t, hasThinking)
}
