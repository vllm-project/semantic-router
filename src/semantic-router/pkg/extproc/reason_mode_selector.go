package extproc

import (
	"encoding/json"
	"fmt"
	"log"
	"strings"

	"github.com/vllm-project/semantic-router/semantic-router/pkg/metrics"
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
		log.Printf("No category determined for query, defaulting to no reasoning mode")
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
			log.Printf("Reasoning mode decision: Category '%s' → %s",
				categoryName, reasoningStatus)
			return category.UseReasoning, categoryName
		}
	}

	// If category not found in config, default to no reasoning
	log.Printf("Category '%s' not found in configuration, defaulting to no reasoning mode", categoryName)
	return false, categoryName
}

// hasDeepSeekAlias returns true if the model uses a short alias for DeepSeek (e.g., "ds-*")
// Rules:
//   - Accept only when the model string starts with: "ds-", "ds_", "ds:", "ds " or exactly equals "ds"
//   - Do NOT match occurrences of "ds" in the middle of the model name (e.g., "foo-ds-1b")
func hasDeepSeekAlias(lower string) bool {
	lower = strings.TrimSpace(lower)
	if strings.HasPrefix(lower, "ds") {
		if len(lower) == 2 { // exactly "ds"
			return true
		}
		sep := lower[2]
		return sep == '-' || sep == '_' || sep == ':' || sep == ' '
	}
	return false
}

// getModelFamilyAndTemplateParam returns a normalized model family name and the template param to be used (if any)
func getModelFamilyAndTemplateParam(model string) (string, string) {
	lower := strings.ToLower(strings.TrimSpace(model))
	if strings.Contains(lower, "qwen3") {
		return "qwen3", "enable_thinking"
	}
	if strings.Contains(lower, "deepseek") || hasDeepSeekAlias(lower) {
		return "deepseek", "thinking"
	}
	// GPT-OSS family and generic GPT fall back to using reasoning_effort (OpenAI-compatible field)
	if strings.Contains(lower, "gpt-oss") || strings.Contains(lower, "gpt_oss") {
		return "gpt-oss", "reasoning_effort"
	}
	if strings.Contains(lower, "gpt") {
		return "gpt", "reasoning_effort"
	}
	return "unknown", ""
}

// getChatTemplateKwargs returns the appropriate chat template kwargs based on model and reasoning mode
func getChatTemplateKwargs(model string, useReasoning bool) map[string]interface{} {
	lower := strings.ToLower(strings.TrimSpace(model))

	// Qwen3: use enable_thinking true/false
	if strings.Contains(lower, "qwen3") {
		return map[string]interface{}{
			"enable_thinking": useReasoning,
		}
	}

	// DeepSeek v3 family: use thinking true/false
	if strings.Contains(lower, "deepseek") || strings.Contains(lower, "ds") {
		return map[string]interface{}{
			"thinking": useReasoning,
		}
	}

	// Default: no chat template kwargs for unknown models
	return nil
}

// setReasoningModeToRequestBody adds chat_template_kwargs to the JSON request body
func (r *OpenAIRouter) setReasoningModeToRequestBody(requestBody []byte, enabled bool, categoryName string) ([]byte, error) {
	// Parse the JSON request body
	var requestMap map[string]interface{}
	if err := json.Unmarshal(requestBody, &requestMap); err != nil {
		return nil, fmt.Errorf("failed to parse request body: %w", err)
	}

	// Determine model for kwargs and logging
	model := "unknown"
	if modelValue, ok := requestMap["model"]; ok {
		if modelStr, ok := modelValue.(string); ok {
			model = modelStr
		}
	}

	family, param := getModelFamilyAndTemplateParam(model)

	// Add chat_template_kwargs for reasoning mode
	kwargs := getChatTemplateKwargs(model, enabled)
	if kwargs != nil {
		requestMap["chat_template_kwargs"] = kwargs
	} else {
		delete(requestMap, "chat_template_kwargs")
	}
	// Also set Reasoning-Effort in openai request
	// This is a hack to get the reasoning mode for openai/gpt-oss-20b to work
	originalReasoningEffort, ok := requestMap["reasoning_effort"]
	if !ok {
		// This seems to be the default for openai/gpt-oss models
		originalReasoningEffort = "low"
	}
	var appliedEffort string
	if enabled {
		// Use configurable reasoning effort based on category
		effort := r.getReasoningEffort(categoryName)
		requestMap["reasoning_effort"] = effort
		appliedEffort = effort
	} else {
		requestMap["reasoning_effort"] = originalReasoningEffort
		if s, ok := originalReasoningEffort.(string); ok {
			appliedEffort = s
		}
	}

	log.Printf("Original reasoning effort: %s", originalReasoningEffort)
	log.Printf("Added reasoning mode (enabled: %v) and reasoning effort (%s) to request for model: %s", enabled, requestMap["reasoning_effort"], model)

	// Record metrics for template usage and effort when enabled
	if enabled {
		// If we applied a known template param, record its usage
		if kwargs != nil && param != "" {
			metrics.RecordReasoningTemplateUsage(family, param)
		} else if kwargs == nil && param == "reasoning_effort" {
			// For GPT/GPT-OSS, we only set reasoning_effort
			metrics.RecordReasoningTemplateUsage(family, param)
		}
		// Record which effort level was used for this family
		metrics.RecordReasoningEffortUsage(family, appliedEffort)
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
		log.Printf("No categories configured for reasoning mode")
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

	log.Printf("Reasoning configuration - Total categories: %d", len(r.Config.Categories))

	if len(reasoningEnabled) > 0 {
		log.Printf("Reasoning ENABLED for categories (%d): %v", len(reasoningEnabled), reasoningEnabled)
	}

	if len(reasoningDisabled) > 0 {
		log.Printf("Reasoning DISABLED for categories (%d): %v", len(reasoningDisabled), reasoningDisabled)
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
	log.Printf("Model selection complete: model=%s, reasoning=%s", bestModel, reasoningStatus)

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

	log.Printf("Reasoning mode summary: %d/%d categories have reasoning enabled", enabledCount, len(r.Config.Categories))
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
