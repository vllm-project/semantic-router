package extproc

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestReasoningModeComprehensive provides comprehensive test coverage for reasoning mode functionality.
func TestReasoningModeComprehensive(t *testing.T) {
	router := newComprehensiveReasoningRouter()

	for _, tt := range comprehensiveReasoningModeCases() {
		t.Run(tt.name, func(t *testing.T) {
			modifiedRequest := setReasoningModeForCase(t, router, tt)
			assertReasoningModeCase(t, modifiedRequest, tt)
		})
	}
}

func comprehensiveReasoningModeCases() []reasoningModeCase {
	return []reasoningModeCase{
		{
			name:                      "DeepSeek - reasoning enabled",
			model:                     "deepseek-v3",
			categoryName:              "math",
			enableReasoning:           true,
			expectChatTemplateKwargs:  true,
			expectedChatTemplateParam: "thinking",
			expectedChatTemplateValue: true,
			expectReasoningEffortKey:  false,
		},
		{
			name:                      "DeepSeek - reasoning disabled",
			model:                     "deepseek-v3",
			categoryName:              "math",
			enableReasoning:           false,
			initialReasoningEffort:    "low",
			expectChatTemplateKwargs:  true,
			expectedChatTemplateParam: "thinking",
			expectedChatTemplateValue: false,
		},
		{
			name:                      "Qwen3 - reasoning enabled",
			model:                     "qwen3-model",
			categoryName:              "code",
			enableReasoning:           true,
			expectChatTemplateKwargs:  true,
			expectedChatTemplateParam: "enable_thinking",
			expectedChatTemplateValue: true,
			expectReasoningEffortKey:  false,
		},
		{
			name:                      "Qwen3 - reasoning disabled",
			model:                     "qwen3-model",
			categoryName:              "code",
			enableReasoning:           false,
			initialReasoningEffort:    "medium",
			expectChatTemplateKwargs:  true,
			expectedChatTemplateParam: "enable_thinking",
			expectedChatTemplateValue: false,
		},
		{
			name:                      "GPT-OSS - reasoning enabled with high effort",
			model:                     "gpt-oss-model",
			categoryName:              "math",
			enableReasoning:           true,
			expectChatTemplateKwargs:  true,
			expectedChatTemplateParam: "reasoning_effort",
			expectedChatTemplateValue: "medium",
			expectReasoningEffortKey:  false,
		},
		{
			name:                      "GPT-OSS - reasoning disabled preserves effort",
			model:                     "gpt-oss-model",
			categoryName:              "creative",
			enableReasoning:           false,
			initialReasoningEffort:    "low",
			expectChatTemplateKwargs:  true,
			expectedChatTemplateParam: "reasoning_effort",
			expectedChatTemplateValue: "low",
			expectReasoningEffortKey:  false,
		},
		{
			name:                      "Claude - reasoning enabled",
			model:                     "claude-opus",
			categoryName:              "creative",
			enableReasoning:           true,
			expectChatTemplateKwargs:  true,
			expectedChatTemplateParam: "thinking",
			expectedChatTemplateValue: true,
		},
		{
			name:                      "Claude - reasoning disabled",
			model:                     "claude-opus",
			categoryName:              "creative",
			enableReasoning:           false,
			expectChatTemplateKwargs:  true,
			expectedChatTemplateParam: "thinking",
			expectedChatTemplateValue: false,
		},
		{
			name:                   "Phi4 - no reasoning family, enabled",
			model:                  "phi4",
			categoryName:           "math",
			enableReasoning:        true,
			expectBothFieldsAbsent: true,
		},
		{
			name:                   "Phi4 - no reasoning family, disabled",
			model:                  "phi4",
			categoryName:           "code",
			enableReasoning:        false,
			initialReasoningEffort: "low",
			expectBothFieldsAbsent: true,
		},
	}
}

func TestChatTemplateKwargsPreservedWhenTogglingReasoning(t *testing.T) {
	router := newQwen3ReasoningRouter()

	makeBody := func() []byte {
		b, _ := json.Marshal(map[string]interface{}{
			"model": "qwen3-model",
			"messages": []map[string]string{
				{"role": "user", "content": "test"},
			},
			"chat_template_kwargs": map[string]interface{}{
				"foo":             "bar",
				"enable_thinking": true,
			},
		})
		return b
	}

	t.Run("disable reasoning overrides enable_thinking but preserves other keys", func(t *testing.T) {
		modified, err := router.setReasoningModeToRequestBody(makeBody(), false, nil)
		require.NoError(t, err)

		out := unmarshalReasoningRequest(t, modified)
		ctk, ok := out["chat_template_kwargs"].(map[string]interface{})
		require.True(t, ok, "expected chat_template_kwargs to be a map")
		assert.Equal(t, "bar", ctk["foo"])
		assert.Equal(t, false, ctk["enable_thinking"])
	})

	t.Run("enable reasoning sets enable_thinking true and preserves other keys", func(t *testing.T) {
		modified, err := router.setReasoningModeToRequestBody(makeBody(), true, nil)
		require.NoError(t, err)

		out := unmarshalReasoningRequest(t, modified)
		ctk, ok := out["chat_template_kwargs"].(map[string]interface{})
		require.True(t, ok, "expected chat_template_kwargs to be a map")
		assert.Equal(t, "bar", ctk["foo"])
		assert.Equal(t, true, ctk["enable_thinking"])
	})
}

func TestReasoningEffortPreservesOpaqueChatTemplateFields(t *testing.T) {
	router := newReasoningRouter(
		config.ReasoningConfig{
			ReasoningFamilies: map[string]config.ReasoningFamilyConfig{
				"glm": {
					Type:      config.ReasoningFamilyTypeReasoningEffort,
					Parameter: "reasoning_effort",
				},
			},
		},
		[]config.Decision{
			reasoningDecision("fusion", "", 0, "glm-5.2", boolPtr(true), "high"),
		},
		map[string]config.ModelParams{"glm-5.2": {ReasoningFamily: "glm"}},
	)
	body := []byte(`{
		"model":"glm-5.2",
		"messages":[{"role":"user","content":"synthesize"}],
		"chat_template_kwargs":{
			"enable_thinking":false,
			"vendor_limit":123456789012345678901234567890
		}
	}`)

	modified, err := router.setReasoningModeToRequestBody(
		body,
		true,
		router.Config.GetDecisionByName("fusion"),
	)
	require.NoError(t, err)
	assert.Contains(t, string(modified), `"vendor_limit":123456789012345678901234567890`)

	request := unmarshalReasoningRequest(t, modified)
	kwargs, ok := request["chat_template_kwargs"].(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, false, kwargs["enable_thinking"])
	assert.Equal(t, "high", kwargs["reasoning_effort"])
}

// TestReasoningEffortLevels tests all reasoning effort levels.
func TestReasoningEffortLevels(t *testing.T) {
	router := newReasoningEffortLevelsRouter()
	efforts := []struct {
		categoryName   string
		expectedEffort string
	}{
		{"low-effort-task", "low"},
		{"medium-effort-task", "medium"},
		{"high-effort-task", "high"},
	}

	for _, tt := range efforts {
		t.Run("Effort_"+tt.expectedEffort, func(t *testing.T) {
			modifiedRequest := setReasoningMode(
				t,
				router,
				"gpt-oss-model",
				nil,
				true,
				tt.categoryName,
			)
			assertChatTemplateReasoningField(
				t,
				modifiedRequest,
				"reasoning_effort",
				tt.expectedEffort,
			)
		})
	}
}

// TestGetReasoningEffort tests the getReasoningEffort method.
func TestGetReasoningEffort(t *testing.T) {
	router := newReasoningEffortLookupRouter()
	tests := []struct {
		name           string
		categoryName   string
		modelName      string
		expectedEffort string
	}{
		{"Model-specific high effort", "math", "model-a", "high"},
		{"Model-specific low effort", "math", "model-b", "low"},
		{"Provider model ID resolves model-specific effort", "math", "gpt-5-mini", "high"},
		{"Falls back to default", "code", "model-c", "medium"},
		{"Unknown category falls back to default", "unknown", "model-a", "medium"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decision := router.Config.GetDecisionByName(tt.categoryName)
			effort := router.getReasoningEffort(decision, tt.modelName)
			assert.Equal(t, tt.expectedEffort, effort)
		})
	}
}

// TestGetModelReasoningFamily tests the getModelReasoningFamily method.
func TestGetModelReasoningFamily(t *testing.T) {
	router := newModelReasoningFamilyRouter()
	tests := []struct {
		name          string
		model         string
		expectNil     bool
		expectedType  string
		expectedParam string
	}{
		{"DeepSeek family", "deepseek-v3", false, "chat_template_kwargs", "thinking"},
		{"Qwen3 family", "qwen3-7b", false, "chat_template_kwargs", "enable_thinking"},
		{"GPT-OSS family", "gpt-oss-model", false, "reasoning_effort", "reasoning_effort"},
		{name: "No reasoning family", model: "phi4", expectNil: true},
		{name: "Unknown model", model: "unknown", expectNil: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			family := router.getModelReasoningFamily(tt.model)
			if tt.expectNil {
				assert.Nil(t, family)
				return
			}
			require.NotNil(t, family)
			assert.Equal(t, tt.expectedType, family.Type)
			assert.Equal(t, tt.expectedParam, family.Parameter)
		})
	}
}

func TestGetReasoningEffortUsesNamedRecipeDecision(t *testing.T) {
	namedDecision := reasoningDecision(
		"frontier-route",
		"",
		0,
		"model-a",
		boolPtr(true),
		"high",
	)
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				ReasoningConfig: config.ReasoningConfig{
					DefaultReasoningEffort: "medium",
				},
				Decisions: []config.Decision{
					reasoningDecision("frontier-route", "", 0, "model-a", boolPtr(true), "low"),
				},
			},
			Recipes: []config.RoutingRecipe{
				{
					Name:    "accuracy-first",
					Profile: config.RoutingProfile{Decisions: []config.Decision{namedDecision}},
				},
			},
		},
	}

	if got := router.getReasoningEffort(&namedDecision, "model-a"); got != "high" {
		t.Fatalf("named-recipe reasoning effort = %q, want high", got)
	}
}

// TestBuildReasoningRequestFields tests reasoning fields at the provider boundary.
func TestBuildReasoningRequestFields(t *testing.T) {
	router := newBuildReasoningRequestFieldsRouter()
	tests := []struct {
		name               string
		model              string
		useReasoning       bool
		categoryName       string
		expectNil          bool
		expectEffortReturn string
		profile            *config.ProviderProfile
		verifyFunc         func(t *testing.T, fields map[string]interface{})
	}{
		{
			name:         "DeepSeek with reasoning enabled",
			model:        "deepseek-v3",
			useReasoning: true,
			categoryName: "test",
			verifyFunc: func(t *testing.T, fields map[string]interface{}) {
				assertReasoningRequestField(t, fields, "thinking", true)
			},
		},
		{
			name:               "GPT-OSS with reasoning enabled",
			model:              "gpt-oss-model",
			useReasoning:       true,
			categoryName:       "test",
			expectEffortReturn: "low",
			verifyFunc: func(t *testing.T, fields map[string]interface{}) {
				assertReasoningRequestField(t, fields, "reasoning_effort", "low")
			},
		},
		{
			name:               "OpenAI provider model ID uses modelRef effort",
			model:              "gpt-5-mini",
			useReasoning:       true,
			categoryName:       "test",
			expectEffortReturn: "high",
			profile:            &config.ProviderProfile{Type: "openai", BaseURL: "https://api.openai.com/v1"},
			verifyFunc: func(t *testing.T, fields map[string]interface{}) {
				require.NotNil(t, fields)
				reasoningEffort, exists := fields["reasoning_effort"]
				require.True(t, exists)
				assert.Equal(t, "high", reasoningEffort)
				_, hasChatTemplate := fields["chat_template_kwargs"]
				assert.False(t, hasChatTemplate)
			},
		},
		{
			name:               "OpenRouter provider model ID uses top-level effort",
			model:              "gpt-5-mini",
			useReasoning:       true,
			categoryName:       "test",
			expectEffortReturn: "high",
			profile:            &config.ProviderProfile{Type: "openai", BaseURL: "https://openrouter.ai/api/v1"},
			verifyFunc: func(t *testing.T, fields map[string]interface{}) {
				require.NotNil(t, fields)
				reasoningEffort, exists := fields["reasoning_effort"]
				require.True(t, exists)
				assert.Equal(t, "high", reasoningEffort)
				_, hasChatTemplate := fields["chat_template_kwargs"]
				assert.False(t, hasChatTemplate)
			},
		},
		{name: "Reasoning disabled", model: "deepseek-v3", categoryName: "test", expectNil: true},
		{name: "No reasoning family", model: "phi4", useReasoning: true, categoryName: "test", expectNil: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decision := router.Config.GetDecisionByName(tt.categoryName)
			fields, effort := router.buildReasoningFieldsForTest(
				tt.model,
				tt.useReasoning,
				decision,
				tt.profile,
			)
			assertBuiltReasoningRequestFields(t, fields, effort, tt.expectNil, tt.expectEffortReturn, tt.verifyFunc)
		})
	}
}

// TestReasoningModeEdgeCases tests edge cases and error conditions.
func TestReasoningModeEdgeCases(t *testing.T) {
	router := newDeepSeekReasoningRouter()

	t.Run("Empty request body", func(t *testing.T) {
		_, err := router.setReasoningModeToRequestBody([]byte("{}"), true, nil)
		assert.NoError(t, err)
	})

	t.Run("Invalid JSON", func(t *testing.T) {
		_, err := router.setReasoningModeToRequestBody([]byte("invalid json"), true, nil)
		assert.Error(t, err)
	})

	t.Run("Large request body", func(t *testing.T) {
		requestBytes, _ := json.Marshal(largeReasoningRequest())
		modifiedBytes, err := router.setReasoningModeToRequestBody(requestBytes, true, nil)
		assert.NoError(t, err)
		assert.NotNil(t, modifiedBytes)
	})

	t.Run("Nil config", func(t *testing.T) {
		nilRouter := &OpenAIRouter{Config: nil}
		assert.Equal(t, "medium", nilRouter.getReasoningEffort(nil, "model"))
		assert.Nil(t, nilRouter.getModelReasoningFamily("model"))
	})
}
