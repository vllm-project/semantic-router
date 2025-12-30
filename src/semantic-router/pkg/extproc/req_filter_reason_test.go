package extproc

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestReasoningModeComprehensive provides comprehensive test coverage for reasoning mode functionality
func TestReasoningModeComprehensive(t *testing.T) {
	// Create a router with all reasoning families configured
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				ReasoningConfig: config.ReasoningConfig{
					DefaultReasoningEffort: "medium",
					ReasoningFamilies: map[string]config.ReasoningFamilyConfig{
						"deepseek": {
							Type:      "chat_template_kwargs",
							Parameter: "thinking",
						},
						"qwen3": {
							Type:      "chat_template_kwargs",
							Parameter: "enable_thinking",
						},
						"gpt-oss": {
							Type:      "reasoning_effort",
							Parameter: "reasoning_effort",
						},
						"gpt": {
							Type:      "reasoning_effort",
							Parameter: "reasoning_effort",
						},
						"claude": {
							Type:      "chat_template_kwargs",
							Parameter: "thinking",
						},
					},
				},
				Decisions: []config.Decision{
					{
						Name:        "math",
						Description: "Math problems",
						Priority:    100,
						ModelRefs: []config.ModelRef{
							{
								Model: "deepseek-v3",
								ModelReasoningControl: config.ModelReasoningControl{
									UseReasoning:    boolPtr(true),
									ReasoningEffort: "high",
								},
							},
						},
					},
					{
						Name:        "code",
						Description: "Coding tasks",
						Priority:    90,
						ModelRefs: []config.ModelRef{
							{
								Model: "qwen3-model",
								ModelReasoningControl: config.ModelReasoningControl{
									UseReasoning:    boolPtr(true),
									ReasoningEffort: "medium",
								},
							},
						},
					},
					{
						Name:        "creative",
						Description: "Creative writing",
						Priority:    80,
						ModelRefs: []config.ModelRef{
							{
								Model: "claude-opus",
								ModelReasoningControl: config.ModelReasoningControl{
									UseReasoning: boolPtr(false),
								},
							},
						},
					},
				},
			},
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"deepseek-v3": {
						ReasoningFamily: "deepseek",
					},
					"qwen3-model": {
						ReasoningFamily: "qwen3",
					},
					"gpt-oss-model": {
						ReasoningFamily: "gpt-oss",
					},
					"claude-opus": {
						ReasoningFamily: "claude",
					},
					"phi4": {
						// No reasoning family
					},
				},
			},
		},
	}

	tests := []struct {
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
	}{
		// Test 1: DeepSeek with reasoning enabled - should use chat_template_kwargs
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
		// Test 2: DeepSeek with reasoning disabled - should clear all fields
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
		// Test 3: Qwen3 with reasoning enabled - should use enable_thinking
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
		// Test 4: Qwen3 with reasoning disabled - should clear all fields
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
		// Test 5: GPT-OSS with reasoning enabled - should use reasoning_effort with HIGH
		{
			name:                     "GPT-OSS - reasoning enabled with high effort",
			model:                    "gpt-oss-model",
			categoryName:             "math",
			enableReasoning:          true,
			expectReasoningEffortKey: true,
			expectedReasoningEffort:  "medium", // Falls back to default
		},
		// Test 6: GPT-OSS with reasoning disabled - should preserve reasoning_effort
		{
			name:                          "GPT-OSS - reasoning disabled preserves effort",
			model:                         "gpt-oss-model",
			categoryName:                  "creative",
			enableReasoning:               false,
			initialReasoningEffort:        "low",
			expectReasoningEffortKey:      true,
			expectOriginalEffortPreserved: true,
		},
		// Test 7: Claude with reasoning enabled
		{
			name:                      "Claude - reasoning enabled",
			model:                     "claude-opus",
			categoryName:              "creative",
			enableReasoning:           true,
			expectChatTemplateKwargs:  true,
			expectedChatTemplateParam: "thinking",
			expectedChatTemplateValue: true,
		},
		// Test 8: Claude with reasoning disabled
		{
			name:                      "Claude - reasoning disabled",
			model:                     "claude-opus",
			categoryName:              "creative",
			enableReasoning:           false,
			expectChatTemplateKwargs:  true,
			expectedChatTemplateParam: "thinking",
			expectedChatTemplateValue: false,
		},
		// Test 9: Phi4 (no reasoning family) - should not add any fields
		{
			name:                   "Phi4 - no reasoning family, enabled",
			model:                  "phi4",
			categoryName:           "math",
			enableReasoning:        true,
			expectBothFieldsAbsent: true,
		},
		// Test 10: Phi4 with reasoning disabled - should clear everything
		{
			name:                   "Phi4 - no reasoning family, disabled",
			model:                  "phi4",
			categoryName:           "code",
			enableReasoning:        false,
			initialReasoningEffort: "low",
			expectBothFieldsAbsent: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Prepare request body
			requestBody := map[string]interface{}{
				"model": tt.model,
				"messages": []map[string]string{
					{"role": "user", "content": "test message"},
				},
			}
			if tt.initialReasoningEffort != nil {
				requestBody["reasoning_effort"] = tt.initialReasoningEffort
			}

			requestBytes, err := json.Marshal(requestBody)
			require.NoError(t, err)

			// Call setReasoningModeToRequestBody
			modifiedBytes, err := router.setReasoningModeToRequestBody(requestBytes, tt.enableReasoning, tt.categoryName)
			require.NoError(t, err)

			// Parse modified request
			var modifiedRequest map[string]interface{}
			err = json.Unmarshal(modifiedBytes, &modifiedRequest)
			require.NoError(t, err)

			// Verify expectations
			if tt.expectBothFieldsAbsent {
				_, hasChatTemplate := modifiedRequest["chat_template_kwargs"]
				_, hasReasoningEffort := modifiedRequest["reasoning_effort"]
				assert.False(t, hasChatTemplate, "chat_template_kwargs should be absent")
				assert.False(t, hasReasoningEffort, "reasoning_effort should be absent")
			}

			if tt.expectChatTemplateKwargs {
				chatTemplateKwargs, exists := modifiedRequest["chat_template_kwargs"]
				require.True(t, exists, "chat_template_kwargs should exist")

				kwargs, ok := chatTemplateKwargs.(map[string]interface{})
				require.True(t, ok, "chat_template_kwargs should be a map")

				value, paramExists := kwargs[tt.expectedChatTemplateParam]
				require.True(t, paramExists, "Expected parameter %s should exist", tt.expectedChatTemplateParam)
				assert.Equal(t, tt.expectedChatTemplateValue, value, "chat_template_kwargs[%s] value mismatch", tt.expectedChatTemplateParam)

				// When chat_template_kwargs is set, reasoning_effort should NOT be present
				_, hasReasoningEffort := modifiedRequest["reasoning_effort"]
				assert.False(t, hasReasoningEffort, "reasoning_effort should not exist when chat_template_kwargs is used")
			}

			if tt.expectReasoningEffortKey {
				reasoningEffort, exists := modifiedRequest["reasoning_effort"]
				require.True(t, exists, "reasoning_effort should exist")

				if tt.expectOriginalEffortPreserved {
					assert.Equal(t, tt.initialReasoningEffort, reasoningEffort, "Original reasoning_effort should be preserved")
				} else {
					assert.Equal(t, tt.expectedReasoningEffort, reasoningEffort, "reasoning_effort value mismatch")
				}

				// When reasoning_effort is set, chat_template_kwargs should NOT be present
				_, hasChatTemplate := modifiedRequest["chat_template_kwargs"]
				assert.False(t, hasChatTemplate, "chat_template_kwargs should not exist when reasoning_effort is used")
			}
		})
	}
}

func TestChatTemplateKwargsPreservedWhenTogglingReasoning(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				ReasoningConfig: config.ReasoningConfig{
					ReasoningFamilies: map[string]config.ReasoningFamilyConfig{
						"qwen3": {
							Type:      "chat_template_kwargs",
							Parameter: "enable_thinking",
						},
					},
				},
			},
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"qwen3-model": {
						ReasoningFamily: "qwen3",
					},
				},
			},
		},
	}

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

		var out map[string]interface{}
		require.NoError(t, json.Unmarshal(modified, &out))

		ctk, ok := out["chat_template_kwargs"].(map[string]interface{})
		require.True(t, ok, "expected chat_template_kwargs to be a map")
		assert.Equal(t, "bar", ctk["foo"])
		assert.Equal(t, false, ctk["enable_thinking"])
	})

	t.Run("enable reasoning sets enable_thinking true and preserves other keys", func(t *testing.T) {
		modified, err := router.setReasoningModeToRequestBody(makeBody(), true, "any")
		require.NoError(t, err)

		var out map[string]interface{}
		require.NoError(t, json.Unmarshal(modified, &out))

		ctk, ok := out["chat_template_kwargs"].(map[string]interface{})
		require.True(t, ok, "expected chat_template_kwargs to be a map")
		assert.Equal(t, "bar", ctk["foo"])
		assert.Equal(t, true, ctk["enable_thinking"])
	})
}

// TestReasoningEffortLevels tests all reasoning effort levels
func TestReasoningEffortLevels(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				ReasoningConfig: config.ReasoningConfig{
					DefaultReasoningEffort: "medium",
					ReasoningFamilies: map[string]config.ReasoningFamilyConfig{
						"gpt-oss": {
							Type:      "reasoning_effort",
							Parameter: "reasoning_effort",
						},
					},
				},
				Decisions: []config.Decision{
					{
						Name: "low-effort-task",
						ModelRefs: []config.ModelRef{
							{
								Model: "gpt-oss-model",
								ModelReasoningControl: config.ModelReasoningControl{
									UseReasoning:    boolPtr(true),
									ReasoningEffort: "low",
								},
							},
						},
					},
					{
						Name: "medium-effort-task",
						ModelRefs: []config.ModelRef{
							{
								Model: "gpt-oss-model",
								ModelReasoningControl: config.ModelReasoningControl{
									UseReasoning:    boolPtr(true),
									ReasoningEffort: "medium",
								},
							},
						},
					},
					{
						Name: "high-effort-task",
						ModelRefs: []config.ModelRef{
							{
								Model: "gpt-oss-model",
								ModelReasoningControl: config.ModelReasoningControl{
									UseReasoning:    boolPtr(true),
									ReasoningEffort: "high",
								},
							},
						},
					},
				},
			},
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"gpt-oss-model": {
						ReasoningFamily: "gpt-oss",
					},
				},
			},
		},
	}

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
			requestBody := map[string]interface{}{
				"model": "gpt-oss-model",
				"messages": []map[string]string{
					{"role": "user", "content": "test"},
				},
			}

			requestBytes, err := json.Marshal(requestBody)
			require.NoError(t, err)

			modifiedBytes, err := router.setReasoningModeToRequestBody(requestBytes, true, tt.categoryName)
			require.NoError(t, err)

			var modifiedRequest map[string]interface{}
			err = json.Unmarshal(modifiedBytes, &modifiedRequest)
			require.NoError(t, err)

			reasoningEffort, exists := modifiedRequest["reasoning_effort"]
			require.True(t, exists, "reasoning_effort should exist")
			assert.Equal(t, tt.expectedEffort, reasoningEffort)
		})
	}
}

// TestGetReasoningEffort tests the getReasoningEffort method
func TestGetReasoningEffort(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				ReasoningConfig: config.ReasoningConfig{
					DefaultReasoningEffort: "medium",
				},
				Decisions: []config.Decision{
					{
						Name: "math",
						ModelRefs: []config.ModelRef{
							{
								Model: "model-a",
								ModelReasoningControl: config.ModelReasoningControl{
									ReasoningEffort: "high",
								},
							},
							{
								Model: "model-b",
								ModelReasoningControl: config.ModelReasoningControl{
									ReasoningEffort: "low",
								},
							},
						},
					},
					{
						Name: "code",
						ModelRefs: []config.ModelRef{
							{
								Model: "model-c",
								// No reasoning effort specified
							},
						},
					},
				},
			},
		},
	}

	tests := []struct {
		name           string
		categoryName   string
		modelName      string
		expectedEffort string
	}{
		{
			name:           "Model-specific high effort",
			categoryName:   "math",
			modelName:      "model-a",
			expectedEffort: "high",
		},
		{
			name:           "Model-specific low effort",
			categoryName:   "math",
			modelName:      "model-b",
			expectedEffort: "low",
		},
		{
			name:           "Falls back to default",
			categoryName:   "code",
			modelName:      "model-c",
			expectedEffort: "medium",
		},
		{
			name:           "Unknown category falls back to default",
			categoryName:   "unknown",
			modelName:      "model-a",
			expectedEffort: "medium",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			effort := router.getReasoningEffort(tt.categoryName, tt.modelName)
			assert.Equal(t, tt.expectedEffort, effort)
		})
	}
}

// TestGetModelReasoningFamily tests the getModelReasoningFamily method
func TestGetModelReasoningFamily(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				ReasoningConfig: config.ReasoningConfig{
					ReasoningFamilies: map[string]config.ReasoningFamilyConfig{
						"deepseek": {
							Type:      "chat_template_kwargs",
							Parameter: "thinking",
						},
						"qwen3": {
							Type:      "chat_template_kwargs",
							Parameter: "enable_thinking",
						},
						"gpt-oss": {
							Type:      "reasoning_effort",
							Parameter: "reasoning_effort",
						},
					},
				},
			},
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"deepseek-v3": {
						ReasoningFamily: "deepseek",
					},
					"qwen3-7b": {
						ReasoningFamily: "qwen3",
					},
					"gpt-oss-model": {
						ReasoningFamily: "gpt-oss",
					},
					"phi4": {
						// No reasoning family
					},
				},
			},
		},
	}

	tests := []struct {
		name          string
		model         string
		expectNil     bool
		expectedType  string
		expectedParam string
	}{
		{
			name:          "DeepSeek family",
			model:         "deepseek-v3",
			expectNil:     false,
			expectedType:  "chat_template_kwargs",
			expectedParam: "thinking",
		},
		{
			name:          "Qwen3 family",
			model:         "qwen3-7b",
			expectNil:     false,
			expectedType:  "chat_template_kwargs",
			expectedParam: "enable_thinking",
		},
		{
			name:          "GPT-OSS family",
			model:         "gpt-oss-model",
			expectNil:     false,
			expectedType:  "reasoning_effort",
			expectedParam: "reasoning_effort",
		},
		{
			name:      "No reasoning family",
			model:     "phi4",
			expectNil: true,
		},
		{
			name:      "Unknown model",
			model:     "unknown",
			expectNil: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			family := router.getModelReasoningFamily(tt.model)

			if tt.expectNil {
				assert.Nil(t, family)
			} else {
				require.NotNil(t, family)
				assert.Equal(t, tt.expectedType, family.Type)
				assert.Equal(t, tt.expectedParam, family.Parameter)
			}
		})
	}
}

// TestBuildReasoningRequestFields tests the buildReasoningRequestFields method
func TestBuildReasoningRequestFields(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				ReasoningConfig: config.ReasoningConfig{
					DefaultReasoningEffort: "medium",
					ReasoningFamilies: map[string]config.ReasoningFamilyConfig{
						"deepseek": {
							Type:      "chat_template_kwargs",
							Parameter: "thinking",
						},
						"gpt-oss": {
							Type:      "reasoning_effort",
							Parameter: "reasoning_effort",
						},
					},
				},
				Decisions: []config.Decision{
					{
						Name: "test",
						ModelRefs: []config.ModelRef{
							{
								Model: "deepseek-v3",
								ModelReasoningControl: config.ModelReasoningControl{
									ReasoningEffort: "high",
								},
							},
							{
								Model: "gpt-oss-model",
								ModelReasoningControl: config.ModelReasoningControl{
									ReasoningEffort: "low",
								},
							},
						},
					},
				},
			},
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"deepseek-v3": {
						ReasoningFamily: "deepseek",
					},
					"gpt-oss-model": {
						ReasoningFamily: "gpt-oss",
					},
					"phi4": {
						// No reasoning family
					},
				},
			},
		},
	}

	tests := []struct {
		name               string
		model              string
		useReasoning       bool
		categoryName       string
		expectNil          bool
		expectEffortReturn string
		verifyFunc         func(t *testing.T, fields map[string]interface{})
	}{
		{
			name:         "DeepSeek with reasoning enabled",
			model:        "deepseek-v3",
			useReasoning: true,
			categoryName: "test",
			expectNil:    false,
			verifyFunc: func(t *testing.T, fields map[string]interface{}) {
				require.NotNil(t, fields)
				chatTemplate, exists := fields["chat_template_kwargs"]
				require.True(t, exists)
				kwargs := chatTemplate.(map[string]interface{})
				assert.Equal(t, true, kwargs["thinking"])
			},
		},
		{
			name:               "GPT-OSS with reasoning enabled",
			model:              "gpt-oss-model",
			useReasoning:       true,
			categoryName:       "test",
			expectNil:          false,
			expectEffortReturn: "low",
			verifyFunc: func(t *testing.T, fields map[string]interface{}) {
				require.NotNil(t, fields)
				effort, exists := fields["reasoning_effort"]
				require.True(t, exists)
				assert.Equal(t, "low", effort)
			},
		},
		{
			name:         "Reasoning disabled",
			model:        "deepseek-v3",
			useReasoning: false,
			categoryName: "test",
			expectNil:    true,
		},
		{
			name:         "No reasoning family",
			model:        "phi4",
			useReasoning: true,
			categoryName: "test",
			expectNil:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fields, effort := router.buildReasoningRequestFields(tt.model, tt.useReasoning, tt.categoryName)

			if tt.expectNil {
				assert.Nil(t, fields)
				assert.Empty(t, effort)
			} else {
				if tt.verifyFunc != nil {
					tt.verifyFunc(t, fields)
				}
				if tt.expectEffortReturn != "" {
					assert.Equal(t, tt.expectEffortReturn, effort)
				}
			}
		})
	}
}

// TestReasoningModeEdgeCases tests edge cases and error conditions
func TestReasoningModeEdgeCases(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				ReasoningConfig: config.ReasoningConfig{
					DefaultReasoningEffort: "medium",
					ReasoningFamilies: map[string]config.ReasoningFamilyConfig{
						"deepseek": {
							Type:      "chat_template_kwargs",
							Parameter: "thinking",
						},
					},
				},
			},
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"deepseek-v3": {
						ReasoningFamily: "deepseek",
					},
				},
			},
		},
	}

	t.Run("Empty request body", func(t *testing.T) {
		_, err := router.setReasoningModeToRequestBody([]byte("{}"), true, "test")
		assert.NoError(t, err)
	})

	t.Run("Invalid JSON", func(t *testing.T) {
		_, err := router.setReasoningModeToRequestBody([]byte("invalid json"), true, "test")
		assert.Error(t, err)
	})

	t.Run("Large request body", func(t *testing.T) {
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
		requestBytes, _ := json.Marshal(largeRequest)
		modifiedBytes, err := router.setReasoningModeToRequestBody(requestBytes, true, "test")
		assert.NoError(t, err)
		assert.NotNil(t, modifiedBytes)
	})

	t.Run("Nil config", func(t *testing.T) {
		nilRouter := &OpenAIRouter{
			Config: nil,
		}
		effort := nilRouter.getReasoningEffort("test", "model")
		assert.Equal(t, "medium", effort)

		family := nilRouter.getModelReasoningFamily("model")
		assert.Nil(t, family)
	})
}

// Helper function to create bool pointer
func boolPtr(b bool) *bool {
	return &b
}
