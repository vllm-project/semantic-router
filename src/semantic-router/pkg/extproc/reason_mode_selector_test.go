package extproc

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/semantic-router/pkg/config"
)

// TestGetModelFamilyAndTemplateParam verifies model-family detection and template parameter mapping
// TestModelReasoningConfig tests the new config-driven approach
func TestModelReasoningConfig(t *testing.T) {
	// Create a router with sample model configurations
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			DefaultReasoningEffort: "medium",
			ModelReasoningConfigs: []config.ModelReasoningConfig{
				{
					Name:     "qwen3",
					Patterns: []string{"qwen3"},
					ReasoningSyntax: config.ModelReasoningSyntax{
						Type:      "chat_template_kwargs",
						Parameter: "enable_thinking",
					},
				},
				{
					Name:     "deepseek",
					Patterns: []string{"deepseek", "ds-", "ds_", "ds:", "ds "},
					ReasoningSyntax: config.ModelReasoningSyntax{
						Type:      "chat_template_kwargs",
						Parameter: "thinking",
					},
				},
				{
					Name:     "gpt-oss",
					Patterns: []string{"gpt-oss", "gpt_oss"},
					ReasoningSyntax: config.ModelReasoningSyntax{
						Type:      "reasoning_effort",
						Parameter: "reasoning_effort",
					},
				},
				{
					Name:     "gpt",
					Patterns: []string{"gpt"},
					ReasoningSyntax: config.ModelReasoningSyntax{
						Type:      "reasoning_effort",
						Parameter: "reasoning_effort",
					},
				},
				// No default config - unknown models should not get reasoning syntax
			},
		},
	}

	testCases := []struct {
		name              string
		model             string
		expectedConfig    string // expected config name or empty for no config
		expectedType      string
		expectedParameter string
		expectConfig      bool
	}{
		{
			name:              "Qwen3 family",
			model:             "Qwen3-7B",
			expectedConfig:    "qwen3",
			expectedType:      "chat_template_kwargs",
			expectedParameter: "enable_thinking",
			expectConfig:      true,
		},
		{
			name:              "DeepSeek family",
			model:             "deepseek-v31",
			expectedConfig:    "deepseek",
			expectedType:      "chat_template_kwargs",
			expectedParameter: "thinking",
			expectConfig:      true,
		},
		{
			name:              "DeepSeek alias ds (prefix)",
			model:             "DS-1.5B",
			expectedConfig:    "deepseek",
			expectedType:      "chat_template_kwargs",
			expectedParameter: "thinking",
			expectConfig:      true,
		},
		{
			name:              "GPT-OSS family",
			model:             "gpt-oss-20b",
			expectedConfig:    "gpt-oss",
			expectedType:      "reasoning_effort",
			expectedParameter: "reasoning_effort",
			expectConfig:      true,
		},
		{
			name:              "GPT generic family",
			model:             "gpt-4o-mini",
			expectedConfig:    "gpt",
			expectedType:      "reasoning_effort",
			expectedParameter: "reasoning_effort",
			expectConfig:      true,
		},
		{
			name:              "Unknown family (phi4) should have no config",
			model:             "phi4",
			expectedConfig:    "",
			expectedType:      "",
			expectedParameter: "",
			expectConfig:      false,
		},
		{
			name:              "Non-matching ds should have no config",
			model:             "mistral-ds-1b",
			expectedConfig:    "",
			expectedType:      "",
			expectedParameter: "",
			expectConfig:      false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			modelConfig := router.getModelReasoningConfig(tc.model)

			if !tc.expectConfig {
				// For unknown models, we expect no configuration
				if modelConfig != nil {
					t.Fatalf("Expected no model config for %q, got %+v", tc.model, modelConfig)
				}
				return
			}

			// For known models, we expect a valid configuration
			if modelConfig == nil {
				t.Fatalf("Expected model config for %q, got nil", tc.model)
			}
			if modelConfig.Name != tc.expectedConfig {
				t.Fatalf("Expected config name %q for model %q, got %q", tc.expectedConfig, tc.model, modelConfig.Name)
			}
			if modelConfig.ReasoningSyntax.Type != tc.expectedType {
				t.Fatalf("Expected type %q for model %q, got %q", tc.expectedType, tc.model, modelConfig.ReasoningSyntax.Type)
			}
			if modelConfig.ReasoningSyntax.Parameter != tc.expectedParameter {
				t.Fatalf("Expected parameter %q for model %q, got %q", tc.expectedParameter, tc.model, modelConfig.ReasoningSyntax.Parameter)
			}
		})
	}
}

// TestSetReasoningModeToRequestBody verifies that reasoning_effort is handled correctly for different model families
func TestSetReasoningModeToRequestBody(t *testing.T) {
	// Create a router with model reasoning configurations
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			DefaultReasoningEffort: "medium",
			ModelReasoningConfigs: []config.ModelReasoningConfig{
				{
					Name:     "deepseek",
					Patterns: []string{"deepseek", "ds-", "ds_", "ds:", "ds "},
					ReasoningSyntax: config.ModelReasoningSyntax{
						Type:      "chat_template_kwargs",
						Parameter: "thinking",
					},
				},
				{
					Name:     "qwen3",
					Patterns: []string{"qwen3"},
					ReasoningSyntax: config.ModelReasoningSyntax{
						Type:      "chat_template_kwargs",
						Parameter: "enable_thinking",
					},
				},
				{
					Name:     "gpt-oss",
					Patterns: []string{"gpt-oss", "gpt_oss"},
					ReasoningSyntax: config.ModelReasoningSyntax{
						Type:      "reasoning_effort",
						Parameter: "reasoning_effort",
					},
				},
				// No default config - unknown models should not get reasoning syntax
			},
		},
	}

	testCases := []struct {
		name                       string
		model                      string
		enabled                    bool
		initialReasoningEffort     interface{}
		expectReasoningEffortKey   bool
		expectedReasoningEffort    interface{}
		expectedChatTemplateKwargs bool
	}{
		{
			name:                       "GPT-OSS model with reasoning disabled - preserve reasoning_effort",
			model:                      "gpt-oss-20b",
			enabled:                    false,
			initialReasoningEffort:     "low",
			expectReasoningEffortKey:   true,
			expectedReasoningEffort:    "low",
			expectedChatTemplateKwargs: false,
		},
		{
			name:                       "Phi4 model with reasoning disabled - remove reasoning_effort",
			model:                      "phi4",
			enabled:                    false,
			initialReasoningEffort:     "low",
			expectReasoningEffortKey:   false,
			expectedReasoningEffort:    nil,
			expectedChatTemplateKwargs: false,
		},
		{
			name:                       "Phi4 model with reasoning enabled - no fields set (unknown model)",
			model:                      "phi4",
			enabled:                    true,
			initialReasoningEffort:     "low",
			expectReasoningEffortKey:   false,
			expectedReasoningEffort:    nil,
			expectedChatTemplateKwargs: false,
		},
		{
			name:                       "DeepSeek model with reasoning disabled - remove reasoning_effort",
			model:                      "deepseek-v31",
			enabled:                    false,
			initialReasoningEffort:     "low",
			expectReasoningEffortKey:   false,
			expectedReasoningEffort:    nil,
			expectedChatTemplateKwargs: false,
		},
		{
			name:                       "GPT-OSS model with reasoning enabled - set reasoning_effort",
			model:                      "gpt-oss-20b",
			enabled:                    true,
			initialReasoningEffort:     "low",
			expectReasoningEffortKey:   true,
			expectedReasoningEffort:    "medium",
			expectedChatTemplateKwargs: false,
		},
		{
			name:                       "DeepSeek model with reasoning enabled - set chat_template_kwargs",
			model:                      "deepseek-v31",
			enabled:                    true,
			initialReasoningEffort:     "low",
			expectReasoningEffortKey:   false,
			expectedReasoningEffort:    nil,
			expectedChatTemplateKwargs: true,
		},
		{
			name:                       "DeepSeek alias (ds-) with reasoning enabled - set chat_template_kwargs",
			model:                      "ds-1.5b",
			enabled:                    true,
			initialReasoningEffort:     "low",
			expectReasoningEffortKey:   false,
			expectedReasoningEffort:    nil,
			expectedChatTemplateKwargs: true,
		},
		{
			name:                       "DeepSeek alias (ds-) with reasoning disabled - no fields set",
			model:                      "ds-1.5b",
			enabled:                    false,
			initialReasoningEffort:     "low",
			expectReasoningEffortKey:   false,
			expectedReasoningEffort:    nil,
			expectedChatTemplateKwargs: false,
		},
		{
			name:                       "Non-matching ds model - no fields set (unknown model)",
			model:                      "mistral-ds-7b",
			enabled:                    true,
			initialReasoningEffort:     "low",
			expectReasoningEffortKey:   false,
			expectedReasoningEffort:    nil,
			expectedChatTemplateKwargs: false,
		},
		{
			name:                       "Qwen3 model with reasoning enabled - set chat_template_kwargs",
			model:                      "qwen3-7b",
			enabled:                    true,
			initialReasoningEffort:     "low",
			expectReasoningEffortKey:   false,
			expectedReasoningEffort:    nil,
			expectedChatTemplateKwargs: true,
		},
		{
			name:                       "Qwen3 model with reasoning disabled - no fields set",
			model:                      "qwen3-7b",
			enabled:                    false,
			initialReasoningEffort:     "low",
			expectReasoningEffortKey:   false,
			expectedReasoningEffort:    nil,
			expectedChatTemplateKwargs: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Prepare initial request body
			requestBody := map[string]interface{}{
				"model": tc.model,
				"messages": []map[string]string{
					{"role": "user", "content": "test message"},
				},
			}
			if tc.initialReasoningEffort != nil {
				requestBody["reasoning_effort"] = tc.initialReasoningEffort
			}

			requestBytes, err := json.Marshal(requestBody)
			if err != nil {
				t.Fatalf("Failed to marshal request body: %v", err)
			}

			// Call the function under test
			modifiedBytes, err := router.setReasoningModeToRequestBody(requestBytes, tc.enabled, "test-category")
			if err != nil {
				t.Fatalf("setReasoningModeToRequestBody failed: %v", err)
			}

			// Parse the modified request body
			var modifiedRequest map[string]interface{}
			if err := json.Unmarshal(modifiedBytes, &modifiedRequest); err != nil {
				t.Fatalf("Failed to unmarshal modified request body: %v", err)
			}

			// Check reasoning_effort handling
			reasoningEffort, hasReasoningEffort := modifiedRequest["reasoning_effort"]
			if tc.expectReasoningEffortKey != hasReasoningEffort {
				t.Fatalf("Expected reasoning_effort key presence: %v, got: %v", tc.expectReasoningEffortKey, hasReasoningEffort)
			}
			if tc.expectReasoningEffortKey && reasoningEffort != tc.expectedReasoningEffort {
				t.Fatalf("Expected reasoning_effort: %v, got: %v", tc.expectedReasoningEffort, reasoningEffort)
			}

			// Check chat_template_kwargs handling
			chatTemplateKwargs, hasChatTemplateKwargs := modifiedRequest["chat_template_kwargs"]
			if tc.expectedChatTemplateKwargs != hasChatTemplateKwargs {
				t.Fatalf("Expected chat_template_kwargs key presence: %v, got: %v", tc.expectedChatTemplateKwargs, hasChatTemplateKwargs)
			}
			if tc.expectedChatTemplateKwargs {
				kwargs, ok := chatTemplateKwargs.(map[string]interface{})
				if !ok {
					t.Fatalf("Expected chat_template_kwargs to be a map")
				}
				if len(kwargs) == 0 {
					t.Fatalf("Expected non-empty chat_template_kwargs")
				}

				// Validate the specific parameter based on model type
				if tc.model == "deepseek-v31" || tc.model == "ds-1.5b" {
					if thinkingValue, exists := kwargs["thinking"]; !exists {
						t.Fatalf("Expected 'thinking' parameter in chat_template_kwargs for DeepSeek model")
					} else if thinkingValue != true {
						t.Fatalf("Expected 'thinking' to be true, got %v", thinkingValue)
					}
				} else if tc.model == "qwen3-7b" {
					if thinkingValue, exists := kwargs["enable_thinking"]; !exists {
						t.Fatalf("Expected 'enable_thinking' parameter in chat_template_kwargs for Qwen3 model")
					} else if thinkingValue != true {
						t.Fatalf("Expected 'enable_thinking' to be true, got %v", thinkingValue)
					}
				}
			}
		})
	}
}
