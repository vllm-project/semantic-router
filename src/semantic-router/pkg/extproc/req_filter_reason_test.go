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
		modified, err := router.setReasoningModeToRequestBody(makeBody(), false, "any")
		require.NoError(t, err)

		out := unmarshalReasoningRequest(t, modified)
		ctk, ok := out["chat_template_kwargs"].(map[string]interface{})
		require.True(t, ok, "expected chat_template_kwargs to be a map")
		assert.Equal(t, "bar", ctk["foo"])
		assert.Equal(t, false, ctk["enable_thinking"])
	})

	t.Run("enable reasoning sets enable_thinking true and preserves other keys", func(t *testing.T) {
		modified, err := router.setReasoningModeToRequestBody(makeBody(), true, "any")
		require.NoError(t, err)

		out := unmarshalReasoningRequest(t, modified)
		ctk, ok := out["chat_template_kwargs"].(map[string]interface{})
		require.True(t, ok, "expected chat_template_kwargs to be a map")
		assert.Equal(t, "bar", ctk["foo"])
		assert.Equal(t, true, ctk["enable_thinking"])
	})
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
			effort := router.getReasoningEffort(tt.categoryName, tt.modelName)
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

// TestBuildReasoningRequestFields tests the buildReasoningRequestFieldsForProvider method.
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
			fields, effort := router.buildReasoningRequestFieldsForProvider(
				tt.model,
				tt.useReasoning,
				tt.categoryName,
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
		_, err := router.setReasoningModeToRequestBody([]byte("{}"), true, "test")
		assert.NoError(t, err)
	})

	t.Run("Invalid JSON", func(t *testing.T) {
		_, err := router.setReasoningModeToRequestBody([]byte("invalid json"), true, "test")
		assert.Error(t, err)
	})

	t.Run("Large request body", func(t *testing.T) {
		requestBytes, _ := json.Marshal(largeReasoningRequest())
		modifiedBytes, err := router.setReasoningModeToRequestBody(requestBytes, true, "test")
		assert.NoError(t, err)
		assert.NotNil(t, modifiedBytes)
	})

	t.Run("Nil config", func(t *testing.T) {
		nilRouter := &OpenAIRouter{Config: nil}
		assert.Equal(t, "medium", nilRouter.getReasoningEffort("test", "model"))
		assert.Nil(t, nilRouter.getModelReasoningFamily("model"))
	})
}

type reasoningModeCase struct {
	name                          string
	model                         string
	categoryName                  string
	enableReasoning               bool
	initialReasoningEffort        interface{}
	expectChatTemplateKwargs      bool
	expectedChatTemplateParam     string
	expectedChatTemplateValue     interface{}
	expectReasoningEffortKey      bool
	expectedReasoningEffort       string
	expectBothFieldsAbsent        bool
	expectOriginalEffortPreserved bool
}

func newReasoningRouter(
	reasoningConfig config.ReasoningConfig,
	decisions []config.Decision,
	modelConfig map[string]config.ModelParams,
) *OpenAIRouter {
	return &OpenAIRouter{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				ReasoningConfig: reasoningConfig,
				Decisions:       decisions,
			},
			BackendModels: config.BackendModels{ModelConfig: modelConfig},
		},
	}
}

func newComprehensiveReasoningRouter() *OpenAIRouter {
	return newReasoningRouter(
		config.ReasoningConfig{
			DefaultReasoningEffort: "medium",
			ReasoningFamilies: map[string]config.ReasoningFamilyConfig{
				"deepseek": {Type: "chat_template_kwargs", Parameter: "thinking"},
				"qwen3":    {Type: "chat_template_kwargs", Parameter: "enable_thinking"},
				"gpt-oss":  {Type: "reasoning_effort", Parameter: "reasoning_effort"},
				"gpt":      {Type: "reasoning_effort", Parameter: "reasoning_effort"},
				"claude":   {Type: "chat_template_kwargs", Parameter: "thinking"},
			},
		},
		[]config.Decision{
			reasoningDecision("math", "Math problems", 100, "deepseek-v3", boolPtr(true), "high"),
			reasoningDecision("code", "Coding tasks", 90, "qwen3-model", boolPtr(true), "medium"),
			reasoningDecision("creative", "Creative writing", 80, "claude-opus", boolPtr(false), ""),
		},
		map[string]config.ModelParams{
			"deepseek-v3":   {ReasoningFamily: "deepseek"},
			"qwen3-model":   {ReasoningFamily: "qwen3"},
			"gpt-oss-model": {ReasoningFamily: "gpt-oss"},
			"claude-opus":   {ReasoningFamily: "claude"},
			"phi4":          {},
		},
	)
}

func newQwen3ReasoningRouter() *OpenAIRouter {
	return newReasoningRouter(
		config.ReasoningConfig{
			ReasoningFamilies: map[string]config.ReasoningFamilyConfig{
				"qwen3": {Type: "chat_template_kwargs", Parameter: "enable_thinking"},
			},
		},
		nil,
		map[string]config.ModelParams{"qwen3-model": {ReasoningFamily: "qwen3"}},
	)
}

func newReasoningEffortLevelsRouter() *OpenAIRouter {
	return newReasoningRouter(
		config.ReasoningConfig{
			DefaultReasoningEffort: "medium",
			ReasoningFamilies: map[string]config.ReasoningFamilyConfig{
				"gpt-oss": {Type: "reasoning_effort", Parameter: "reasoning_effort"},
			},
		},
		[]config.Decision{
			reasoningDecision("low-effort-task", "", 0, "gpt-oss-model", boolPtr(true), "low"),
			reasoningDecision("medium-effort-task", "", 0, "gpt-oss-model", boolPtr(true), "medium"),
			reasoningDecision("high-effort-task", "", 0, "gpt-oss-model", boolPtr(true), "high"),
		},
		map[string]config.ModelParams{"gpt-oss-model": {ReasoningFamily: "gpt-oss"}},
	)
}

func newReasoningEffortLookupRouter() *OpenAIRouter {
	return newReasoningRouter(
		config.ReasoningConfig{DefaultReasoningEffort: "medium"},
		[]config.Decision{
			{
				Name: "math",
				ModelRefs: []config.ModelRef{
					{Model: "model-a", ModelReasoningControl: config.ModelReasoningControl{ReasoningEffort: "high"}},
					{Model: "model-b", ModelReasoningControl: config.ModelReasoningControl{ReasoningEffort: "low"}},
					{Model: "model-openai", ModelReasoningControl: config.ModelReasoningControl{ReasoningEffort: "high"}},
				},
			},
			{Name: "code", ModelRefs: []config.ModelRef{{Model: "model-c"}}},
		},
		map[string]config.ModelParams{
			"model-openai": {
				ExternalModelIDs: map[string]string{"openai": "gpt-5-mini"},
			},
		},
	)
}

func newModelReasoningFamilyRouter() *OpenAIRouter {
	return newReasoningRouter(
		config.ReasoningConfig{
			ReasoningFamilies: map[string]config.ReasoningFamilyConfig{
				"deepseek": {Type: "chat_template_kwargs", Parameter: "thinking"},
				"qwen3":    {Type: "chat_template_kwargs", Parameter: "enable_thinking"},
				"gpt-oss":  {Type: "reasoning_effort", Parameter: "reasoning_effort"},
			},
		},
		nil,
		map[string]config.ModelParams{
			"deepseek-v3":   {ReasoningFamily: "deepseek"},
			"qwen3-7b":      {ReasoningFamily: "qwen3"},
			"gpt-oss-model": {ReasoningFamily: "gpt-oss"},
			"phi4":          {},
		},
	)
}

func newBuildReasoningRequestFieldsRouter() *OpenAIRouter {
	return newReasoningRouter(
		config.ReasoningConfig{
			DefaultReasoningEffort: "medium",
			ReasoningFamilies: map[string]config.ReasoningFamilyConfig{
				"deepseek": {Type: "chat_template_kwargs", Parameter: "thinking"},
				"gpt-oss":  {Type: "reasoning_effort", Parameter: "reasoning_effort"},
			},
		},
		[]config.Decision{{
			Name: "test",
			ModelRefs: []config.ModelRef{
				{Model: "deepseek-v3", ModelReasoningControl: config.ModelReasoningControl{ReasoningEffort: "high"}},
				{Model: "gpt-oss-model", ModelReasoningControl: config.ModelReasoningControl{ReasoningEffort: "low"}},
				{Model: "openai-alias", ModelReasoningControl: config.ModelReasoningControl{ReasoningEffort: "high"}},
			},
		}},
		map[string]config.ModelParams{
			"deepseek-v3":   {ReasoningFamily: "deepseek"},
			"gpt-oss-model": {ReasoningFamily: "gpt-oss"},
			"openai-alias": {
				ReasoningFamily: "gpt-oss",
				ExternalModelIDs: map[string]string{
					"openai": "gpt-5-mini",
				},
			},
			"phi4": {},
		},
	)
}

func newDeepSeekReasoningRouter() *OpenAIRouter {
	return newReasoningRouter(
		config.ReasoningConfig{
			DefaultReasoningEffort: "medium",
			ReasoningFamilies: map[string]config.ReasoningFamilyConfig{
				"deepseek": {Type: "chat_template_kwargs", Parameter: "thinking"},
			},
		},
		nil,
		map[string]config.ModelParams{"deepseek-v3": {ReasoningFamily: "deepseek"}},
	)
}

func reasoningDecision(name string, description string, priority int, model string, useReasoning *bool, effort string) config.Decision {
	return config.Decision{
		Name:        name,
		Description: description,
		Priority:    priority,
		ModelRefs: []config.ModelRef{{
			Model: model,
			ModelReasoningControl: config.ModelReasoningControl{
				UseReasoning:    useReasoning,
				ReasoningEffort: effort,
			},
		}},
	}
}

func setReasoningModeForCase(t *testing.T, router *OpenAIRouter, tt reasoningModeCase) map[string]interface{} {
	t.Helper()
	return setReasoningMode(t, router, tt.model, tt.initialReasoningEffort, tt.enableReasoning, tt.categoryName)
}

func setReasoningMode(
	t *testing.T,
	router *OpenAIRouter,
	model string,
	initialReasoningEffort interface{},
	enableReasoning bool,
	categoryName string,
) map[string]interface{} {
	t.Helper()
	requestBytes := marshalReasoningRequest(t, model, initialReasoningEffort)
	modifiedBytes, err := router.setReasoningModeToRequestBody(requestBytes, enableReasoning, categoryName)
	require.NoError(t, err)
	return unmarshalReasoningRequest(t, modifiedBytes)
}

func marshalReasoningRequest(t *testing.T, model string, initialReasoningEffort interface{}) []byte {
	t.Helper()
	requestBody := map[string]interface{}{
		"model": model,
		"messages": []map[string]string{
			{"role": "user", "content": "test message"},
		},
	}
	if initialReasoningEffort != nil {
		requestBody["reasoning_effort"] = initialReasoningEffort
	}
	requestBytes, err := json.Marshal(requestBody)
	require.NoError(t, err)
	return requestBytes
}

func unmarshalReasoningRequest(t *testing.T, requestBytes []byte) map[string]interface{} {
	t.Helper()
	var request map[string]interface{}
	require.NoError(t, json.Unmarshal(requestBytes, &request))
	return request
}

func assertReasoningModeCase(t *testing.T, modifiedRequest map[string]interface{}, tt reasoningModeCase) {
	t.Helper()
	if tt.expectBothFieldsAbsent {
		assertNoReasoningFields(t, modifiedRequest)
	}
	if tt.expectChatTemplateKwargs {
		assertChatTemplateReasoningField(
			t,
			modifiedRequest,
			tt.expectedChatTemplateParam,
			tt.expectedChatTemplateValue,
		)
		assertReasoningEffortAbsent(t, modifiedRequest)
	}
	if tt.expectReasoningEffortKey {
		assertReasoningEffortField(t, modifiedRequest, tt)
		assertChatTemplateAbsent(t, modifiedRequest)
	}
}

func assertNoReasoningFields(t *testing.T, request map[string]interface{}) {
	t.Helper()
	assertChatTemplateAbsent(t, request)
	assertReasoningEffortAbsent(t, request)
}

func assertChatTemplateReasoningField(t *testing.T, request map[string]interface{}, param string, value interface{}) {
	t.Helper()
	chatTemplateKwargs, exists := request["chat_template_kwargs"]
	require.True(t, exists, "chat_template_kwargs should exist")

	kwargs, ok := chatTemplateKwargs.(map[string]interface{})
	require.True(t, ok, "chat_template_kwargs should be a map")

	actualValue, paramExists := kwargs[param]
	require.True(t, paramExists, "Expected parameter %s should exist", param)
	assert.Equal(t, value, actualValue, "chat_template_kwargs[%s] value mismatch", param)
}

func assertReasoningEffortField(t *testing.T, request map[string]interface{}, tt reasoningModeCase) {
	t.Helper()
	reasoningEffort, exists := request["reasoning_effort"]
	require.True(t, exists, "reasoning_effort should exist")
	if tt.expectOriginalEffortPreserved {
		assert.Equal(t, tt.initialReasoningEffort, reasoningEffort, "Original reasoning_effort should be preserved")
		return
	}
	assert.Equal(t, tt.expectedReasoningEffort, reasoningEffort, "reasoning_effort value mismatch")
}

func assertChatTemplateAbsent(t *testing.T, request map[string]interface{}) {
	t.Helper()
	_, hasChatTemplate := request["chat_template_kwargs"]
	assert.False(t, hasChatTemplate, "chat_template_kwargs should be absent")
}

func assertReasoningEffortAbsent(t *testing.T, request map[string]interface{}) {
	t.Helper()
	_, hasReasoningEffort := request["reasoning_effort"]
	assert.False(t, hasReasoningEffort, "reasoning_effort should be absent")
}

func assertReasoningRequestField(t *testing.T, fields map[string]interface{}, key string, value interface{}) {
	t.Helper()
	require.NotNil(t, fields)
	chatTemplate, exists := fields["chat_template_kwargs"]
	require.True(t, exists)
	kwargs := chatTemplate.(map[string]interface{})
	assert.Equal(t, value, kwargs[key])
}

func assertBuiltReasoningRequestFields(
	t *testing.T,
	fields map[string]interface{},
	effort string,
	expectNil bool,
	expectedEffort string,
	verifyFunc func(t *testing.T, fields map[string]interface{}),
) {
	t.Helper()
	if expectNil {
		assert.Nil(t, fields)
		assert.Empty(t, effort)
		return
	}
	if verifyFunc != nil {
		verifyFunc(t, fields)
	}
	if expectedEffort != "" {
		assert.Equal(t, expectedEffort, effort)
	}
}

func largeReasoningRequest() map[string]interface{} {
	largeRequest := map[string]interface{}{
		"model":    "deepseek-v3",
		"messages": make([]map[string]string, 1000),
	}
	for i := 0; i < 1000; i++ {
		largeRequest["messages"].([]map[string]string)[i] = map[string]string{
			"role":    "user",
			"content": "test message",
		}
	}
	return largeRequest
}

func boolPtr(b bool) *bool {
	return &b
}
