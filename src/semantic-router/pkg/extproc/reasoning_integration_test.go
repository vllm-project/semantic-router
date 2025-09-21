package extproc

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestReasoningModeIntegration tests the complete reasoning mode integration
func TestReasoningModeIntegration(t *testing.T) {
	// Create a mock router with reasoning configuration
	cfg := &config.RouterConfig{
		DefaultReasoningEffort: "medium",
		Categories: []config.Category{
			{
				Name:                 "math",
				ReasoningDescription: "Mathematical problems require step-by-step reasoning",
				ModelScores: []config.ModelScore{
					{Model: "deepseek-v31", Score: 0.9, UseReasoning: config.BoolPtr(true)},
					{Model: "phi4", Score: 0.7, UseReasoning: config.BoolPtr(false)},
				},
			},
			{
				Name:                 "business",
				ReasoningDescription: "Business content is typically conversational",
				ModelScores: []config.ModelScore{
					{Model: "phi4", Score: 0.8, UseReasoning: config.BoolPtr(false)},
					{Model: "deepseek-v31", Score: 0.6, UseReasoning: config.BoolPtr(false)},
				},
			},
		},
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
			"deepseek-v31": {
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
	}

	router := &OpenAIRouter{
		Config: cfg,
	}

	// Test case 1: Math query should enable reasoning (when classifier works)
	t.Run("Math query enables reasoning", func(t *testing.T) {
		mathQuery := "What is the derivative of x^2 + 3x + 1?"

		// Since we don't have the actual classifier, this will return false
		// But we can test the configuration logic directly
		useReasoning := router.shouldUseReasoningMode(mathQuery)

		// Without a working classifier, this should be false
		expectedReasoning := false

		if useReasoning != expectedReasoning {
			t.Errorf("Expected reasoning mode %v for math query without classifier, got %v", expectedReasoning, useReasoning)
		}

		// Test the configuration logic directly
		mathCategory := cfg.Categories[0] // math category
		if len(mathCategory.ModelScores) == 0 || mathCategory.ModelScores[0].UseReasoning == nil || !*mathCategory.ModelScores[0].UseReasoning {
			t.Error("Math category's best model should have UseReasoning set to true in configuration")
		}
	})

	// Test case 2: Business query should not enable reasoning
	t.Run("Business query disables reasoning", func(t *testing.T) {
		businessQuery := "Write a business plan for a coffee shop"

		useReasoning := router.shouldUseReasoningMode(businessQuery)

		// Should be false because classifier returns empty (no category found)
		if useReasoning != false {
			t.Errorf("Expected reasoning mode false for business query, got %v", useReasoning)
		}
	})

	// Test case 3: Test addReasoningModeToRequestBody function
	t.Run("addReasoningModeToRequestBody adds correct fields", func(t *testing.T) {
		// Test with DeepSeek model (which supports chat_template_kwargs)
		originalRequest := map[string]interface{}{
			"model": "deepseek-v31",
			"messages": []map[string]interface{}{
				{"role": "user", "content": "What is 2 + 2?"},
			},
			"stream": false,
		}

		originalBody, err := json.Marshal(originalRequest)
		if err != nil {
			t.Fatalf("Failed to marshal original request: %v", err)
		}

		modifiedBody, err := router.setReasoningModeToRequestBody(originalBody, true, "math")
		if err != nil {
			t.Fatalf("Failed to add reasoning mode: %v", err)
		}

		var modifiedRequest map[string]interface{}
		if err := json.Unmarshal(modifiedBody, &modifiedRequest); err != nil {
			t.Fatalf("Failed to unmarshal modified request: %v", err)
		}

		// Check if chat_template_kwargs was added for DeepSeek model
		chatTemplateKwargs, exists := modifiedRequest["chat_template_kwargs"]
		if !exists {
			t.Error("chat_template_kwargs not found in modified request for DeepSeek model")
		}

		// Check if thinking: true was set for DeepSeek model
		if kwargs, ok := chatTemplateKwargs.(map[string]interface{}); ok {
			if thinking, hasThinking := kwargs["thinking"]; hasThinking {
				if thinkingBool, isBool := thinking.(bool); !isBool || !thinkingBool {
					t.Errorf("Expected thinking: true for DeepSeek model, got %v", thinking)
				}
			} else {
				t.Error("thinking field not found in chat_template_kwargs for DeepSeek model")
			}
		} else {
			t.Errorf("chat_template_kwargs is not a map for DeepSeek model, got %T", chatTemplateKwargs)
		}

		// Verify original fields are preserved
		originalFields := []string{"model", "messages", "stream"}
		for _, field := range originalFields {
			if _, exists := modifiedRequest[field]; !exists {
				t.Errorf("Original field '%s' was lost", field)
			}
		}

		// Test with unsupported model (phi4) - should not add chat_template_kwargs
		originalRequestPhi4 := map[string]interface{}{
			"model": "phi4",
			"messages": []map[string]interface{}{
				{"role": "user", "content": "What is 2 + 2?"},
			},
			"stream": false,
		}

		originalBodyPhi4, err := json.Marshal(originalRequestPhi4)
		if err != nil {
			t.Fatalf("Failed to marshal phi4 request: %v", err)
		}

		modifiedBodyPhi4, err := router.setReasoningModeToRequestBody(originalBodyPhi4, true, "math")
		if err != nil {
			t.Fatalf("Failed to process phi4 request: %v", err)
		}

		var modifiedRequestPhi4 map[string]interface{}
		if err := json.Unmarshal(modifiedBodyPhi4, &modifiedRequestPhi4); err != nil {
			t.Fatalf("Failed to unmarshal phi4 request: %v", err)
		}

		// For phi4, no reasoning fields should be added (since it's an unknown model)
		if _, exists := modifiedRequestPhi4["chat_template_kwargs"]; exists {
			t.Error("chat_template_kwargs should not be added for unknown model phi4")
		}

		// reasoning_effort should also not be set for unknown models
		if reasoningEffort, exists := modifiedRequestPhi4["reasoning_effort"]; exists {
			t.Errorf("reasoning_effort should NOT be set for unknown model phi4, but got %v", reasoningEffort)
		}
	})

	// Test case 4: Test buildReasoningRequestFields function with config-driven approach
	t.Run("buildReasoningRequestFields returns correct values", func(t *testing.T) {
		// Create a router with sample configurations for testing
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
				},
				ModelConfig: map[string]config.ModelParams{
					"deepseek-v31": {
						ReasoningFamily: "deepseek",
					},
					"qwen3-model": {
						ReasoningFamily: "qwen3",
					},
					"phi4": {
						// No reasoning family - doesn't support reasoning
					},
				},
			},
		}

		// Test with DeepSeek model and reasoning enabled
		fields, _ := router.buildReasoningRequestFields("deepseek-v31", true, "test-category")
		if fields == nil {
			t.Error("Expected non-nil fields for DeepSeek model with reasoning enabled")
		}
		if chatKwargs, ok := fields["chat_template_kwargs"]; !ok {
			t.Error("Expected chat_template_kwargs for DeepSeek model")
		} else if kwargs, ok := chatKwargs.(map[string]interface{}); !ok {
			t.Error("Expected chat_template_kwargs to be a map")
		} else if thinking, ok := kwargs["thinking"]; !ok || thinking != true {
			t.Errorf("Expected thinking: true for DeepSeek model, got %v", thinking)
		}

		// Test with DeepSeek model and reasoning disabled
		fields, _ = router.buildReasoningRequestFields("deepseek-v31", false, "test-category")
		if fields != nil {
			t.Errorf("Expected nil fields for DeepSeek model with reasoning disabled, got %v", fields)
		}

		// Test with Qwen3 model and reasoning enabled
		fields, _ = router.buildReasoningRequestFields("qwen3-model", true, "test-category")
		if fields == nil {
			t.Error("Expected non-nil fields for Qwen3 model with reasoning enabled")
		}
		if chatKwargs, ok := fields["chat_template_kwargs"]; !ok {
			t.Error("Expected chat_template_kwargs for Qwen3 model")
		} else if kwargs, ok := chatKwargs.(map[string]interface{}); !ok {
			t.Error("Expected chat_template_kwargs to be a map")
		} else if enableThinking, ok := kwargs["enable_thinking"]; !ok || enableThinking != true {
			t.Errorf("Expected enable_thinking: true for Qwen3 model, got %v", enableThinking)
		}

		// Test with unknown model (should return no fields)
		fields, effort := router.buildReasoningRequestFields("unknown-model", true, "test-category")
		if fields != nil {
			t.Errorf("Expected nil fields for unknown model with reasoning enabled, got %v", fields)
		}
		if effort != "" {
			t.Errorf("Expected effort string: empty for unknown model, got %v", effort)
		}
	})

	// Test case 5: Test empty query handling
	t.Run("Empty query defaults to no reasoning", func(t *testing.T) {
		useReasoning := router.shouldUseReasoningMode("")
		if useReasoning != false {
			t.Errorf("Expected reasoning mode false for empty query, got %v", useReasoning)
		}
	})

	// Test case 6: Test unknown category handling
	t.Run("Unknown category defaults to no reasoning", func(t *testing.T) {
		unknownQuery := "This is some unknown category query"
		useReasoning := router.shouldUseReasoningMode(unknownQuery)
		if useReasoning != false {
			t.Errorf("Expected reasoning mode false for unknown category, got %v", useReasoning)
		}
	})
}

// TestReasoningModeConfigurationValidation tests the configuration validation
func TestReasoningModeConfigurationValidation(t *testing.T) {
	testCases := []struct {
		name     string
		category config.Category
		expected bool
	}{
		{
			name: "Math category with reasoning enabled",
			category: config.Category{
				Name:                 "math",
				ReasoningDescription: "Mathematical problems require step-by-step reasoning",
				ModelScores: []config.ModelScore{
					{Model: "deepseek-v31", Score: 0.9, UseReasoning: config.BoolPtr(true)},
				},
			},
			expected: true,
		},
		{
			name: "Business category with reasoning disabled",
			category: config.Category{
				Name:                 "business",
				ReasoningDescription: "Business content is typically conversational",
				ModelScores: []config.ModelScore{
					{Model: "phi4", Score: 0.8, UseReasoning: config.BoolPtr(false)},
				},
			},
			expected: false,
		},
		{
			name: "Science category with reasoning enabled",
			category: config.Category{
				Name:                 "science",
				ReasoningDescription: "Scientific concepts benefit from structured analysis",
				ModelScores: []config.ModelScore{
					{Model: "deepseek-v31", Score: 0.9, UseReasoning: config.BoolPtr(true)},
				},
			},
			expected: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Check the best model's reasoning capability
			bestModelReasoning := false
			if len(tc.category.ModelScores) > 0 && tc.category.ModelScores[0].UseReasoning != nil {
				bestModelReasoning = *tc.category.ModelScores[0].UseReasoning
			}

			if bestModelReasoning != tc.expected {
				t.Errorf("Expected best model UseReasoning %v for %s, got %v",
					tc.expected, tc.category.Name, bestModelReasoning)
			}

			// Verify description is not empty
			if tc.category.ReasoningDescription == "" {
				t.Errorf("ReasoningDescription should not be empty for category %s", tc.category.Name)
			}
		})
	}
}
