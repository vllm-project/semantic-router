package extproc

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/consts"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// shouldUseReasoningMode determines if reasoning mode should be enabled based on the query category
func (r *OpenAIRouter) shouldUseReasoningMode(query string) bool {
	enabled, _ := r.getReasoningModeAndCategory(query)
	return enabled
}

// getReasoningModeAndCategory determines if reasoning mode should be enabled and returns the category name
func (r *OpenAIRouter) getReasoningModeAndCategory(query string) (bool, string) {
	// Get the category for this query using the existing classification system
	categoryName := r.findCategoryForClassification(query)

	// If no category was determined (empty string), default to no reasoning
	if categoryName == "" {
		observability.Infof("No category determined for query, defaulting to no reasoning mode")
		return false, ""
	}

	// Normalize category name for consistent lookup
	normalizedCategory := strings.ToLower(strings.TrimSpace(categoryName))

	// Look up the category in the configuration
	for _, category := range r.Config.Categories {
		if strings.EqualFold(category.Name, normalizedCategory) {
			reasoningStatus := "DISABLED"
			if category.UseReasoning {
				reasoningStatus = "ENABLED"
			}
			observability.Infof("Reasoning mode decision: Category '%s' → %s",
				categoryName, reasoningStatus)
			return category.UseReasoning, categoryName
		}
	}

	// If category not found in config, default to no reasoning
	observability.Infof("Category '%s' not found in configuration, defaulting to no reasoning mode", categoryName)
	return false, categoryName
}

// getEntropyBasedReasoningModeAndCategory uses entropy-based analysis for reasoning decisions
func (r *OpenAIRouter) getEntropyBasedReasoningModeAndCategory(query string) (bool, string, entropy.ReasoningDecision) {
	// Use the classifier with entropy analysis
	categoryName, confidence, reasoningDecision, err := r.Classifier.ClassifyCategoryWithEntropy(query)

	if err != nil {
		observability.Warnf("Entropy-based classification error: %v, falling back to traditional method", err)

		// Record fallback metrics
		metrics.RecordEntropyFallback("classification_error", "traditional_method")

		useReasoning, category := r.getReasoningModeAndCategory(query)

		// Record traditional classification confidence for comparison
		metrics.RecordClassificationConfidence(category, "traditional_fallback", 0.5)

		return useReasoning, category, entropy.ReasoningDecision{
			UseReasoning:     useReasoning,
			Confidence:       0.5,
			DecisionReason:   "fallback_traditional_classification",
			FallbackStrategy: "classification_error_fallback",
		}
	}

	// Log the entropy-based decision
	observability.Infof("Entropy-based reasoning decision: category='%s', confidence=%.3f, use_reasoning=%t, reason=%s, strategy=%s",
		categoryName, confidence, reasoningDecision.UseReasoning, reasoningDecision.DecisionReason, reasoningDecision.FallbackStrategy)

	// If we have top categories from entropy analysis, log them
	if len(reasoningDecision.TopCategories) > 0 {
		observability.Infof("Top predicted categories: %v", reasoningDecision.TopCategories)
	}

	return reasoningDecision.UseReasoning, categoryName, reasoningDecision
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
		effort := r.getReasoningEffort(categoryName)
		return map[string]interface{}{"reasoning_effort": effort}, effort
	default:
		// Unknown reasoning syntax type - don't apply anything
		return nil, ""
	}
}

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

	var appliedEffort string = ""

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
		observability.Infof("No reasoning support for model: %s (no reasoning family configured)", model)
	} else if reasoningApplied {
		observability.Infof("Applied reasoning mode (enabled: %v) with effort (%s) to model: %s", enabled, appliedEffort, model)
	} else {
		observability.Infof("Reasoning mode disabled for model: %s", model)
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

// logReasoningConfiguration logs the reasoning mode configuration for all categories during startup
func (r *OpenAIRouter) logReasoningConfiguration() {
	if len(r.Config.Categories) == 0 {
		observability.Infof("No categories configured for reasoning mode")
		return
	}

	reasoningEnabled := []string{}
	reasoningDisabled := []string{}

	for _, category := range r.Config.Categories {
		if category.UseReasoning {
			reasoningEnabled = append(reasoningEnabled, category.Name)
		} else {
			reasoningDisabled = append(reasoningDisabled, category.Name)
		}
	}

	observability.Infof("Reasoning configuration - Total categories: %d", len(r.Config.Categories))

	if len(reasoningEnabled) > 0 {
		observability.Infof("Reasoning ENABLED for categories (%d): %v", len(reasoningEnabled), reasoningEnabled)
	}

	if len(reasoningDisabled) > 0 {
		observability.Infof("Reasoning DISABLED for categories (%d): %v", len(reasoningDisabled), reasoningDisabled)
	}
}

// ClassifyAndDetermineReasoningMode performs category classification and returns both the best model and reasoning mode setting
func (r *OpenAIRouter) ClassifyAndDetermineReasoningMode(query string) (string, bool) {
	// Get the best model using existing logic
	bestModel := r.classifyAndSelectBestModel(query)

	// Determine if reasoning mode should be used
	useReasoning := r.shouldUseReasoningMode(query)

	reasoningStatus := "disabled"
	if useReasoning {
		reasoningStatus = "enabled"
	}
	observability.Infof("Model selection complete: model=%s, reasoning=%s", bestModel, reasoningStatus)

	return bestModel, useReasoning
}

// LogReasoningConfigurationSummary provides a compact summary of reasoning configuration
func (r *OpenAIRouter) LogReasoningConfigurationSummary() {
	if len(r.Config.Categories) == 0 {
		return
	}

	enabledCount := 0
	for _, category := range r.Config.Categories {
		if category.UseReasoning {
			enabledCount++
		}
	}

	observability.Infof("Reasoning mode summary: %d/%d categories have reasoning enabled", enabledCount, len(r.Config.Categories))
}

// getReasoningEffort returns the reasoning effort level for a given category
func (r *OpenAIRouter) getReasoningEffort(categoryName string) string {
	// Handle case where Config is nil (e.g., in tests)
	if r.Config == nil {
		return "medium"
	}

	// Find the category configuration
	for _, category := range r.Config.Categories {
		if category.Name == categoryName {
			// Use category-specific effort if configured
			if category.ReasoningEffort != "" {
				return category.ReasoningEffort
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
