package extproc

import (
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/consts"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// setReasoningModeToRequestBody adds chat_template_kwargs to the JSON request body
func (r *OpenAIRouter) setReasoningModeToRequestBody(requestBody []byte, enabled bool, categoryName string) ([]byte, error) {
	// Parse the JSON request body
	var requestMap map[string]interface{}
	if err := json.Unmarshal(requestBody, &requestMap); err != nil {
		return nil, fmt.Errorf("failed to parse request body: %w", err)
	}

	// Determine model for kwargs and logging
	model := consts.UnknownLabel
	if modelValue, ok := requestMap["model"]; ok {
		if modelStr, ok := modelValue.(string); ok {
			model = modelStr
		}
	}

	// Get original reasoning effort for potential preservation
	originalReasoningEffort, hasOriginalEffort := requestMap["reasoning_effort"]
	if !hasOriginalEffort {
		originalReasoningEffort = "low" // Default for compatibility
	}

	// Clear both reasoning fields to start with a clean state
	delete(requestMap, "chat_template_kwargs")
	delete(requestMap, "reasoning_effort")

	appliedEffort := ""

	var reasoningApplied bool

	if enabled {
		// When reasoning is enabled, build the appropriate fields
		reasoningFields, effort := r.buildReasoningRequestFields(model, enabled, categoryName)
		if reasoningFields != nil {
			for key, value := range reasoningFields {
				requestMap[key] = value
			}
			appliedEffort = effort
			reasoningApplied = true
		} else {
			// Model has no reasoning family configured
			reasoningApplied = false
		}
	} else {
		// When reasoning is disabled, only preserve reasoning_effort for gpt-oss models
		familyConfig := r.getModelReasoningFamily(model)
		if familyConfig != nil && familyConfig.Type == "reasoning_effort" {
			requestMap["reasoning_effort"] = originalReasoningEffort
			if s, ok := originalReasoningEffort.(string); ok {
				appliedEffort = s
			}
		}
		reasoningApplied = false
		// For all other models, reasoning fields remain cleared
	}

	// Log based on what actually happened
	if enabled && !reasoningApplied {
		logging.Infof("No reasoning support for model: %s (no reasoning family configured)", model)
	} else if reasoningApplied {
		logging.Infof("Applied reasoning mode (enabled: %v) with effort (%s) to model: %s", enabled, appliedEffort, model)
	} else {
		logging.Infof("Reasoning mode disabled for model: %s", model)
	}

	// Record metrics for template usage and effort when enabled
	if enabled {
		familyConfig := r.getModelReasoningFamily(model)
		modelFamily := consts.UnknownLabel
		templateParam := "reasoning_effort" // default fallback

		if familyConfig != nil {
			// Use the model's actual reasoning family name from model_config
			if r.Config != nil && r.Config.ModelConfig != nil {
				if modelParams, exists := r.Config.ModelConfig[model]; exists && modelParams.ReasoningFamily != "" {
					modelFamily = modelParams.ReasoningFamily
				}
			}

			if familyConfig.Type == "chat_template_kwargs" {
				templateParam = familyConfig.Parameter
			} else {
				templateParam = "reasoning_effort"
			}
		}

		// Record template usage and effort
		metrics.RecordReasoningTemplateUsage(modelFamily, templateParam)
		if appliedEffort != "" {
			metrics.RecordReasoningEffortUsage(modelFamily, appliedEffort)
		}
	}

	// Serialize back to JSON
	modifiedBody, err := json.Marshal(requestMap)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize modified request: %w", err)
	}

	return modifiedBody, nil
}

// getReasoningEffort returns the reasoning effort level for a given category and model
func (r *OpenAIRouter) getReasoningEffort(categoryName string, modelName string) string {
	// Handle case where Config is nil (e.g., in tests)
	if r.Config == nil {
		return "medium"
	}

	// Find the category and model configuration
	for _, category := range r.Config.Categories {
		if category.Name == categoryName {
			// Find the specific model in the category's model scores
			for _, modelScore := range category.ModelScores {
				if modelScore.Model == modelName {
					// Use model-specific effort if configured
					if modelScore.ReasoningEffort != "" {
						return modelScore.ReasoningEffort
					}
					break
				}
			}
			break
		}
	}

	// Fall back to global default if configured
	if r.Config.DefaultReasoningEffort != "" {
		return r.Config.DefaultReasoningEffort
	}

	// Final fallback to "medium" as a reasonable default
	return "medium"
}

// getModelReasoningFamily finds the reasoning family configuration for a model using the config system
func (r *OpenAIRouter) getModelReasoningFamily(model string) *config.ReasoningFamilyConfig {
	if r.Config == nil {
		return nil
	}
	return r.Config.GetModelReasoningFamily(model)
}

// buildReasoningRequestFields returns the appropriate fields to add to the request based on model config
func (r *OpenAIRouter) buildReasoningRequestFields(model string, useReasoning bool, categoryName string) (map[string]interface{}, string) {
	familyConfig := r.getModelReasoningFamily(model)
	if familyConfig == nil {
		// No reasoning family configured for this model - don't apply any reasoning syntax
		// Models without reasoning_family don't support reasoning mode
		return nil, ""
	}

	if !useReasoning {
		// When reasoning is disabled, don't add any reasoning fields
		return nil, ""
	}

	// When reasoning is enabled, use the configured family syntax
	switch familyConfig.Type {
	case "chat_template_kwargs":
		kwargs := map[string]interface{}{
			familyConfig.Parameter: useReasoning,
		}
		return map[string]interface{}{"chat_template_kwargs": kwargs}, ""
	case "reasoning_effort":
		effort := r.getReasoningEffort(categoryName, model)
		return map[string]interface{}{"reasoning_effort": effort}, effort
	default:
		// Unknown reasoning syntax type - don't apply anything
		return nil, ""
	}
}
