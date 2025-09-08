package extproc

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/semantic-router/pkg/config"
)

// TestGetModelFamilyAndTemplateParam verifies model-family detection and template parameter mapping
func TestGetModelFamilyAndTemplateParam(t *testing.T) {
	testCases := []struct {
		name           string
		model          string
		expectedFamily string
		expectedParam  string
	}{
		{
			name:           "Qwen3 family",
			model:          "Qwen3-7B",
			expectedFamily: "qwen3",
			expectedParam:  "enable_thinking",
		},
		{
			name:           "DeepSeek family",
			model:          "deepseek-v31",
			expectedFamily: "deepseek",
			expectedParam:  "thinking",
		},
		{
			name:           "DeepSeek alias ds (prefix)",
			model:          "DS-1.5B",
			expectedFamily: "deepseek",
			expectedParam:  "thinking",
		},
		{
			name:           "Non-leading ds should not match DeepSeek",
			model:          "mistral-ds-1b",
			expectedFamily: "unknown",
			expectedParam:  "",
		},
		{
			name:           "GPT-OSS family",
			model:          "gpt-oss-20b",
			expectedFamily: "gpt-oss",
			expectedParam:  "reasoning_effort",
		},
		{
			name:           "GPT generic family",
			model:          "gpt-4o-mini",
			expectedFamily: "gpt",
			expectedParam:  "reasoning_effort",
		},
		{
			name:           "GPT underscore variant",
			model:          "  GPT_OSS-foo  ",
			expectedFamily: "gpt-oss",
			expectedParam:  "reasoning_effort",
		},
		{
			name:           "Unknown family",
			model:          "phi4",
			expectedFamily: "unknown",
			expectedParam:  "",
		},
		{
			name:           "Empty model name",
			model:          "",
			expectedFamily: "unknown",
			expectedParam:  "",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			family, param := getModelFamilyAndTemplateParam(tc.model)
			if family != tc.expectedFamily || param != tc.expectedParam {
				t.Fatalf("for model %q got (family=%q, param=%q), want (family=%q, param=%q)", tc.model, family, param, tc.expectedFamily, tc.expectedParam)
			}
		})
	}
}

// TestSetReasoningModeToRequestBody verifies that reasoning_effort is handled correctly for different model families
func TestSetReasoningModeToRequestBody(t *testing.T) {
	// Create a minimal router for testing
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			DefaultReasoningEffort: "medium",
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
			}
		})
	}
}
