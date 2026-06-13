package extproc

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestDeepSeekOfficialBuildReasoningRequestFields(t *testing.T) {
	router := newDeepSeekOfficialReasoningRouter()

	t.Run("official API uses top-level thinking fields", func(t *testing.T) {
		fields, effort := router.buildReasoningRequestFieldsForProvider(
			"deepseek-v4-pro",
			true,
			"test",
			deepSeekOfficialProviderProfile(),
		)
		assertDeepSeekOfficialReasoningFields(t, fields, "enabled", "max")
		assert.Equal(t, "max", effort)
	})

	t.Run("local vLLM keeps configured chat template syntax", func(t *testing.T) {
		fields, effort := router.buildReasoningRequestFieldsForProvider(
			"deepseek-v4-pro",
			true,
			"test",
			localVLLMProviderProfile(),
		)
		assertReasoningRequestField(t, fields, "thinking", true)
		assert.Empty(t, effort)
		assertNoTopLevelDeepSeekThinking(t, fields)
	})
}

func TestDeepSeekOfficialReasoningMode(t *testing.T) {
	router := newDeepSeekOfficialReasoningRouter()

	tests := []struct {
		name         string
		model        string
		categoryName string
		profile      *config.ProviderProfile
		enabled      bool
		local        bool
		thinkingType string
		effort       string
	}{
		{"official low effort", "deepseek-v4-pro", "low-effort-task", deepSeekOfficialProviderProfile(), true, false, "enabled", "low"},
		{"official alias effort", "deepseek-v4-alias", "alias-effort-task", deepSeekOfficialProviderProfile(), true, false, "enabled", "max"},
		{"official disabled", "deepseek-v4-pro", "low-effort-task", deepSeekOfficialProviderProfile(), false, false, "disabled", ""},
		{"local vLLM", "deepseek-v4-pro", "low-effort-task", localVLLMProviderProfile(), true, true, "", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			modifiedRequest := setReasoningModeForProvider(
				t,
				router,
				tt.model,
				nil,
				tt.enabled,
				tt.categoryName,
				tt.profile,
			)
			if tt.local {
				assertChatTemplateReasoningField(t, modifiedRequest, "thinking", true)
				assertNoTopLevelDeepSeekThinking(t, modifiedRequest)
				return
			}
			assertDeepSeekOfficialReasoningRequest(t, modifiedRequest, tt.thinkingType, tt.effort)
		})
	}
}

func TestDeepSeekOfficialRemovesChatTemplateKwargs(t *testing.T) {
	router := newDeepSeekOfficialReasoningRouter()
	requestBytes, err := json.Marshal(map[string]interface{}{
		"model": "deepseek-v4-pro",
		"messages": []map[string]string{
			{"role": "user", "content": "test message"},
		},
		"chat_template_kwargs": map[string]interface{}{
			"think":           true,
			"thinking":        true,
			"enable_thinking": true,
			"other":           "preserved",
		},
	})
	require.NoError(t, err)

	modifiedBytes, err := router.setReasoningModeToRequestBodyForProvider(
		requestBytes,
		false,
		"low-effort-task",
		deepSeekOfficialProviderProfile(),
	)
	require.NoError(t, err)

	modifiedRequest := unmarshalReasoningRequest(t, modifiedBytes)
	assertDeepSeekOfficialReasoningRequest(t, modifiedRequest, "disabled", "")
	_, ok := modifiedRequest["chat_template_kwargs"]
	assert.False(t, ok)
}

func newDeepSeekOfficialReasoningRouter() *OpenAIRouter {
	return newReasoningRouter(
		config.ReasoningConfig{
			DefaultReasoningEffort: "medium",
			ReasoningFamilies: map[string]config.ReasoningFamilyConfig{
				"deepseek": {Type: "chat_template_kwargs", Parameter: "thinking"},
			},
		},
		[]config.Decision{
			{
				Name: "test",
				ModelRefs: []config.ModelRef{
					{
						Model: "deepseek-v4-pro",
						ModelReasoningControl: config.ModelReasoningControl{
							ReasoningEffort: "max",
						},
					},
				},
			},
			reasoningDecision("low-effort-task", "", 0, "deepseek-v4-pro", boolPtr(true), "low"),
			reasoningDecision("alias-effort-task", "", 0, "deepseek-v4-alias", boolPtr(true), "max"),
		},
		map[string]config.ModelParams{
			"deepseek-v4-pro": {ReasoningFamily: "deepseek"},
			"deepseek-v4-alias": {
				ReasoningFamily: "deepseek",
				ExternalModelIDs: map[string]string{
					"deepseek": "deepseek-v4-pro",
				},
			},
		},
	)
}

func deepSeekOfficialProviderProfile() *config.ProviderProfile {
	return &config.ProviderProfile{Type: "openai", BaseURL: "https://api.deepseek.com"}
}

func localVLLMProviderProfile() *config.ProviderProfile {
	return &config.ProviderProfile{Type: "openai", BaseURL: "http://localhost:8000/v1"}
}

func assertDeepSeekOfficialReasoningRequest(t *testing.T, request map[string]interface{}, thinkingType string, effort string) {
	t.Helper()
	assertDeepSeekOfficialReasoningFields(t, request, thinkingType, effort)
	if effort == "" {
		assertReasoningEffortAbsent(t, request)
	}
	if kwargs, ok := request["chat_template_kwargs"].(map[string]interface{}); ok {
		assertNoDeepSeekTemplateReasoningArgs(t, kwargs)
	}
}

func assertDeepSeekOfficialReasoningFields(t *testing.T, fields map[string]interface{}, thinkingType string, effort string) {
	t.Helper()
	require.NotNil(t, fields)
	thinking, ok := fields["thinking"].(map[string]interface{})
	require.True(t, ok, "thinking should be an object")
	assert.Equal(t, thinkingType, thinking["type"])
	if effort != "" {
		assert.Equal(t, effort, fields["reasoning_effort"])
	}
}

func assertNoTopLevelDeepSeekThinking(t *testing.T, fields map[string]interface{}) {
	t.Helper()
	_, hasThinking := fields["thinking"]
	assert.False(t, hasThinking)
	_, hasEffort := fields["reasoning_effort"]
	assert.False(t, hasEffort)
}

func assertNoDeepSeekTemplateReasoningArgs(t *testing.T, kwargs map[string]interface{}) {
	t.Helper()
	for _, key := range []string{"think", "thinking", "enable_thinking"} {
		_, ok := kwargs[key]
		assert.False(t, ok, "chat_template_kwargs[%s] should be absent", key)
	}
}
