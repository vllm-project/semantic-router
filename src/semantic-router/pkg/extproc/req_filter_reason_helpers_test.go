package extproc

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

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
				{Model: "deepseek-v4-pro", ModelReasoningControl: config.ModelReasoningControl{ReasoningEffort: "max"}},
			},
		}},
		map[string]config.ModelParams{
			"deepseek-v3":     {ReasoningFamily: "deepseek"},
			"deepseek-v4-pro": {ReasoningFamily: "deepseek"},
			"gpt-oss-model":   {ReasoningFamily: "gpt-oss"},
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

// buildReasoningFieldsForTest keeps field-oriented assertions test-only.
// Production dispatch always exercises the complete
// neutral-request -> codec -> provider-adapter path.
func (r *OpenAIRouter) buildReasoningFieldsForTest(
	model string,
	useReasoning bool,
	decision *config.Decision,
	profile *config.ProviderProfile,
) (map[string]interface{}, string) {
	if !useReasoning || r.getModelReasoningFamily(model) == nil {
		return nil, ""
	}
	body, err := json.Marshal(map[string]interface{}{
		"model": model,
		"messages": []map[string]string{{
			"role": "user", "content": "test message",
		}},
	})
	if err != nil {
		return nil, ""
	}
	encoded, err := r.setReasoningModeToRequestBodyForModelAndProvider(
		body, model, true, decision, profile,
	)
	if err != nil {
		return nil, ""
	}
	var fields map[string]interface{}
	if json.Unmarshal(encoded, &fields) != nil {
		return nil, ""
	}
	delete(fields, "model")
	delete(fields, "messages")
	effort, _ := fields["reasoning_effort"].(string)
	if kwargs, ok := fields["chat_template_kwargs"].(map[string]interface{}); ok {
		if nested, ok := kwargs["reasoning_effort"].(string); ok {
			effort = nested
		}
	}
	return fields, effort
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
	return setReasoningModeForProvider(t, router, model, initialReasoningEffort, enableReasoning, categoryName, nil)
}

func setReasoningModeForProvider(
	t *testing.T,
	router *OpenAIRouter,
	model string,
	initialReasoningEffort interface{},
	enableReasoning bool,
	categoryName string,
	profile *config.ProviderProfile,
) map[string]interface{} {
	t.Helper()
	requestBytes := marshalReasoningRequest(t, model, initialReasoningEffort)
	decision := router.Config.GetDecisionByName(categoryName)
	modifiedBytes, err := router.setReasoningModeToRequestBodyForProvider(
		requestBytes,
		enableReasoning,
		decision,
		profile,
	)
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
