package extproc

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestModelReasoningFamily tests the new family-based configuration approach
func TestModelReasoningFamily(t *testing.T) {
	// Create a router with sample model configurations
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			DefaultReasoningEffort: "medium",
			ReasoningFamilies: map[string]config.ReasoningFamilyConfig{
				"qwen3": {
					Type:      "chat_template_kwargs",
					Parameter: "enable_thinking",
				},
				"deepseek": {
					Type:      "chat_template_kwargs",
					Parameter: "thinking",
				},
				"gpt-oss": {
					Type:      "reasoning_effort",
					Parameter: "reasoning_effort",
				},
				"gpt": {
					Type:      "reasoning_effort",
					Parameter: "reasoning_effort",
				},
			},
			ModelConfig: map[string]config.ModelParams{
				"qwen3-model": {
					ReasoningFamily: "qwen3",
				},
				"ds-v31-custom": {
					ReasoningFamily: "deepseek",
				},
				"my-deepseek": {
					ReasoningFamily: "deepseek",
				},
				"gpt-oss-model": {
					ReasoningFamily: "gpt-oss",
				},
				"custom-gpt": {
					ReasoningFamily: "gpt",
				},
				"phi4": {
					// No reasoning family - doesn't support reasoning
				},
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
			name:              "qwen3-model with qwen3 family",
			model:             "qwen3-model",
			expectedConfig:    "qwen3",
			expectedType:      "chat_template_kwargs",
			expectedParameter: "enable_thinking",
			expectConfig:      true,
		},
		{
			name:              "ds-v31-custom with deepseek family",
			model:             "ds-v31-custom",
			expectedConfig:    "deepseek",
			expectedType:      "chat_template_kwargs",
			expectedParameter: "thinking",
			expectConfig:      true,
		},
		{
			name:              "my-deepseek with deepseek family",
			model:             "my-deepseek",
			expectedConfig:    "deepseek",
			expectedType:      "chat_template_kwargs",
			expectedParameter: "thinking",
			expectConfig:      true,
		},
		{
			name:              "gpt-oss-model with gpt-oss family",
			model:             "gpt-oss-model",
			expectedConfig:    "gpt-oss",
			expectedType:      "reasoning_effort",
			expectedParameter: "reasoning_effort",
			expectConfig:      true,
		},
		{
			name:              "custom-gpt with gpt family",
			model:             "custom-gpt",
			expectedConfig:    "gpt",
			expectedType:      "reasoning_effort",
			expectedParameter: "reasoning_effort",
			expectConfig:      true,
		},
		{
			name:              "phi4 - no reasoning family",
			model:             "phi4",
			expectedConfig:    "",
			expectedType:      "",
			expectedParameter: "",
			expectConfig:      false,
		},
		{
			name:              "unknown model - no config",
			model:             "unknown-model",
			expectedConfig:    "",
			expectedType:      "",
			expectedParameter: "",
			expectConfig:      false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			familyConfig := router.getModelReasoningFamily(tc.model)

			if !tc.expectConfig {
				// For unknown models, we expect no configuration
				if familyConfig != nil {
					t.Fatalf("Expected no family config for %q, got %+v", tc.model, familyConfig)
				}
				return
			}

			// For known models, we expect a valid configuration
			if familyConfig == nil {
				t.Fatalf("Expected family config for %q, got nil", tc.model)
			}
			if familyConfig.Type != tc.expectedType {
				t.Fatalf("Expected type %q for model %q, got %q", tc.expectedType, tc.model, familyConfig.Type)
			}
			if familyConfig.Parameter != tc.expectedParameter {
				t.Fatalf("Expected parameter %q for model %q, got %q", tc.expectedParameter, tc.model, familyConfig.Parameter)
			}
		})
	}
}

// TestSetReasoningModeToRequestBody verifies that reasoning_effort is handled correctly for different model families
func TestSetReasoningModeToRequestBody(t *testing.T) {
	// Create a router with family-based reasoning configurations
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
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
			},
			ModelConfig: map[string]config.ModelParams{
				"ds-v31-custom": {
					ReasoningFamily: "deepseek",
				},
				"qwen3-model": {
					ReasoningFamily: "qwen3",
				},
				"gpt-oss-model": {
					ReasoningFamily: "gpt-oss",
				},
				"phi4": {
					// No reasoning family - doesn't support reasoning
				},
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
			model:                      "gpt-oss-model",
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
			name:                       "Phi4 model with reasoning enabled - no fields set (no reasoning family)",
			model:                      "phi4",
			enabled:                    true,
			initialReasoningEffort:     "low",
			expectReasoningEffortKey:   false,
			expectedReasoningEffort:    nil,
			expectedChatTemplateKwargs: false,
		},
		{
			name:                       "DeepSeek model with reasoning disabled - remove reasoning_effort",
			model:                      "ds-v31-custom",
			enabled:                    false,
			initialReasoningEffort:     "low",
			expectReasoningEffortKey:   false,
			expectedReasoningEffort:    nil,
			expectedChatTemplateKwargs: false,
		},
		{
			name:                       "GPT-OSS model with reasoning enabled - set reasoning_effort",
			model:                      "gpt-oss-model",
			enabled:                    true,
			initialReasoningEffort:     "low",
			expectReasoningEffortKey:   true,
			expectedReasoningEffort:    "medium",
			expectedChatTemplateKwargs: false,
		},
		{
			name:                       "DeepSeek model with reasoning enabled - set chat_template_kwargs",
			model:                      "ds-v31-custom",
			enabled:                    true,
			initialReasoningEffort:     "low",
			expectReasoningEffortKey:   false,
			expectedReasoningEffort:    nil,
			expectedChatTemplateKwargs: true,
		},
		{
			name:                       "Unknown model - no fields set",
			model:                      "unknown-model",
			enabled:                    true,
			initialReasoningEffort:     "low",
			expectReasoningEffortKey:   false,
			expectedReasoningEffort:    nil,
			expectedChatTemplateKwargs: false,
		},
		{
			name:                       "Qwen3 model with reasoning enabled - set chat_template_kwargs",
			model:                      "qwen3-model",
			enabled:                    true,
			initialReasoningEffort:     "low",
			expectReasoningEffortKey:   false,
			expectedReasoningEffort:    nil,
			expectedChatTemplateKwargs: true,
		},
		{
			name:                       "Qwen3 model with reasoning disabled - no fields set",
			model:                      "qwen3-model",
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
				switch tc.model {
				case "deepseek-v31", "ds-1.5b":
					if thinkingValue, exists := kwargs["thinking"]; !exists {
						t.Fatalf("Expected 'thinking' parameter in chat_template_kwargs for DeepSeek model")
					} else if thinkingValue != true {
						t.Fatalf("Expected 'thinking' to be true, got %v", thinkingValue)
					}
				case "qwen3-7b":
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
