package config

import (
	"fmt"
	"slices"
)

// GetModelReasoningFamily returns the reasoning family configuration for a given model name
func (rc *RouterConfig) GetModelReasoningFamily(modelName string) *ReasoningFamilyConfig {
	if rc == nil || rc.ModelConfig == nil || rc.ReasoningFamilies == nil {
		return nil
	}

	// Look up the model in model_config
	modelParams, exists := rc.ModelConfig[modelName]
	if !exists || modelParams.ReasoningFamily == "" {
		return nil
	}

	// Look up the reasoning family configuration
	familyConfig, exists := rc.ReasoningFamilies[modelParams.ReasoningFamily]
	if !exists {
		return nil
	}

	return &familyConfig
}

// GetEffectiveAutoModelName returns the effective auto model name for automatic model selection
// Returns the configured AutoModelName if set, otherwise defaults to "MoM"
// This is the primary model name that triggers automatic routing
func (c *RouterConfig) GetEffectiveAutoModelName() string {
	if c.AutoModelName != "" {
		return c.AutoModelName
	}
	return "MoM" // Default value
}

// IsAutoModelName checks if the given model name should trigger automatic model selection
// Returns true if the model name is either the configured AutoModelName or "auto" (for backward compatibility)
func (c *RouterConfig) IsAutoModelName(modelName string) bool {
	if modelName == "auto" {
		return true // Always support "auto" for backward compatibility
	}
	return modelName == c.GetEffectiveAutoModelName()
}

// GetCategoryDescriptions returns all category descriptions for similarity matching
func (c *RouterConfig) GetCategoryDescriptions() []string {
	var descriptions []string
	for _, category := range c.Categories {
		if category.Description != "" {
			descriptions = append(descriptions, category.Description)
		} else {
			// Use category name if no description is available
			descriptions = append(descriptions, category.Name)
		}
	}
	return descriptions
}

// GetModelForCategoryIndex returns the best LLM model name for the category at the given index
func (c *RouterConfig) GetModelForCategoryIndex(index int) string {
	if index < 0 || index >= len(c.Categories) {
		return c.DefaultModel
	}

	category := c.Categories[index]
	if len(category.ModelScores) > 0 {
		return category.ModelScores[0].Model
	}

	// Fall back to default model if category has no models
	return c.DefaultModel
}

// GetModelPricing returns pricing per 1M tokens and its currency for the given model.
// The currency indicates the unit of the returned rates (e.g., "USD").
func (c *RouterConfig) GetModelPricing(modelName string) (promptPer1M float64, completionPer1M float64, currency string, ok bool) {
	if modelConfig, okc := c.ModelConfig[modelName]; okc {
		p := modelConfig.Pricing
		if p.PromptPer1M != 0 || p.CompletionPer1M != 0 {
			cur := p.Currency
			if cur == "" {
				cur = "USD"
			}
			return p.PromptPer1M, p.CompletionPer1M, cur, true
		}
	}
	return 0, 0, "", false
}

// GetModelPIIPolicy returns the PII policy for a given model
// If the model is not found in the config, returns a default policy that allows all PII
func (c *RouterConfig) GetModelPIIPolicy(modelName string) PIIPolicy {
	if modelConfig, ok := c.ModelConfig[modelName]; ok {
		return modelConfig.PIIPolicy
	}
	// Default policy allows all PII
	return PIIPolicy{
		AllowByDefault: true,
		PIITypes:       []string{},
	}
}

// IsModelAllowedForPIIType checks if a model is allowed to process a specific PII type
func (c *RouterConfig) IsModelAllowedForPIIType(modelName string, piiType string) bool {
	policy := c.GetModelPIIPolicy(modelName)

	// If allow_by_default is true, all PII types are allowed unless explicitly denied
	if policy.AllowByDefault {
		return true
	}

	// If allow_by_default is false, only explicitly allowed PII types are permitted
	return slices.Contains(policy.PIITypes, piiType)
}

// IsModelAllowedForPIITypes checks if a model is allowed to process any of the given PII types
func (c *RouterConfig) IsModelAllowedForPIITypes(modelName string, piiTypes []string) bool {
	for _, piiType := range piiTypes {
		if !c.IsModelAllowedForPIIType(modelName, piiType) {
			return false
		}
	}
	return true
}

// IsPIIClassifierEnabled checks if PII classification is enabled
func (c *RouterConfig) IsPIIClassifierEnabled() bool {
	return c.PIIModel.ModelID != "" && c.PIIMappingPath != ""
}

// IsCategoryClassifierEnabled checks if category classification is enabled
func (c *RouterConfig) IsCategoryClassifierEnabled() bool {
	return c.CategoryModel.ModelID != "" && c.CategoryMappingPath != ""
}

// IsMCPCategoryClassifierEnabled checks if MCP-based category classification is enabled
func (c *RouterConfig) IsMCPCategoryClassifierEnabled() bool {
	return c.Enabled && c.ToolName != ""
}

// GetPromptGuardConfig returns the prompt guard configuration
func (c *RouterConfig) GetPromptGuardConfig() PromptGuardConfig {
	return c.PromptGuard
}

// IsPromptGuardEnabled checks if prompt guard jailbreak detection is enabled
func (c *RouterConfig) IsPromptGuardEnabled() bool {
	return c.PromptGuard.Enabled && c.PromptGuard.ModelID != "" && c.PromptGuard.JailbreakMappingPath != ""
}

// GetEndpointsForModel returns all endpoints that can serve the specified model
// Returns endpoints based on the model's preferred_endpoints configuration in model_config
func (c *RouterConfig) GetEndpointsForModel(modelName string) []VLLMEndpoint {
	var endpoints []VLLMEndpoint

	// Check if model has preferred endpoints configured
	if modelConfig, ok := c.ModelConfig[modelName]; ok && len(modelConfig.PreferredEndpoints) > 0 {
		// Return only the preferred endpoints
		for _, endpointName := range modelConfig.PreferredEndpoints {
			if endpoint, found := c.GetEndpointByName(endpointName); found {
				endpoints = append(endpoints, *endpoint)
			}
		}
	}

	return endpoints
}

// GetEndpointByName returns the endpoint with the specified name
func (c *RouterConfig) GetEndpointByName(name string) (*VLLMEndpoint, bool) {
	for _, endpoint := range c.VLLMEndpoints {
		if endpoint.Name == name {
			return &endpoint, true
		}
	}
	return nil, false
}

// GetAllModels returns a list of all models configured in model_config
func (c *RouterConfig) GetAllModels() []string {
	var models []string

	for modelName := range c.ModelConfig {
		models = append(models, modelName)
	}

	return models
}

// SelectBestEndpointForModel selects the best endpoint for a model based on weights and availability
// Returns the endpoint name and whether selection was successful
func (c *RouterConfig) SelectBestEndpointForModel(modelName string) (string, bool) {
	endpoints := c.GetEndpointsForModel(modelName)
	if len(endpoints) == 0 {
		return "", false
	}

	// If only one endpoint, return it
	if len(endpoints) == 1 {
		return endpoints[0].Name, true
	}

	// Select endpoint with highest weight
	bestEndpoint := endpoints[0]
	for _, endpoint := range endpoints[1:] {
		if endpoint.Weight > bestEndpoint.Weight {
			bestEndpoint = endpoint
		}
	}

	return bestEndpoint.Name, true
}

// SelectBestEndpointAddressForModel selects the best endpoint for a model and returns the address:port
// Returns the endpoint address:port string and whether selection was successful
func (c *RouterConfig) SelectBestEndpointAddressForModel(modelName string) (string, bool) {
	endpoints := c.GetEndpointsForModel(modelName)
	if len(endpoints) == 0 {
		return "", false
	}

	// If only one endpoint, return it
	if len(endpoints) == 1 {
		return fmt.Sprintf("%s:%d", endpoints[0].Address, endpoints[0].Port), true
	}

	// Select endpoint with highest weight
	bestEndpoint := endpoints[0]
	for _, endpoint := range endpoints[1:] {
		if endpoint.Weight > bestEndpoint.Weight {
			bestEndpoint = endpoint
		}
	}

	return fmt.Sprintf("%s:%d", bestEndpoint.Address, bestEndpoint.Port), true
}

// GetModelReasoningForCategory returns whether a specific model supports reasoning in a given category
func (c *RouterConfig) GetModelReasoningForCategory(categoryName string, modelName string) bool {
	for _, category := range c.Categories {
		if category.Name == categoryName {
			for _, modelScore := range category.ModelScores {
				if modelScore.Model == modelName {
					return modelScore.UseReasoning != nil && *modelScore.UseReasoning
				}
			}
		}
	}
	return false // Default to false if category or model not found
}

// GetBestModelForCategory returns the best scoring model for a given category
func (c *RouterConfig) GetBestModelForCategory(categoryName string) (string, bool) {
	for _, category := range c.Categories {
		if category.Name == categoryName {
			if len(category.ModelScores) > 0 {
				useReasoning := category.ModelScores[0].UseReasoning != nil && *category.ModelScores[0].UseReasoning
				return category.ModelScores[0].Model, useReasoning
			}
		}
	}
	return "", false // Return empty string and false if category not found or has no models
}

// ValidateEndpoints validates that all configured models have at least one endpoint
func (c *RouterConfig) ValidateEndpoints() error {
	// Get all models from categories
	allCategoryModels := make(map[string]bool)
	for _, category := range c.Categories {
		for _, modelScore := range category.ModelScores {
			allCategoryModels[modelScore.Model] = true
		}
	}

	// Add default model
	if c.DefaultModel != "" {
		allCategoryModels[c.DefaultModel] = true
	}

	// Check that each model has at least one endpoint
	for model := range allCategoryModels {
		endpoints := c.GetEndpointsForModel(model)
		if len(endpoints) == 0 {
			return fmt.Errorf("model '%s' has no available endpoints", model)
		}
	}

	return nil
}

// IsSystemPromptEnabled returns whether system prompt injection is enabled for a category
func (c *Category) IsSystemPromptEnabled() bool {
	// If SystemPromptEnabled is explicitly set, use that value
	if c.SystemPromptEnabled != nil {
		return *c.SystemPromptEnabled
	}
	// Default to true if SystemPrompt is not empty
	return c.SystemPrompt != ""
}

// GetSystemPromptMode returns the system prompt injection mode, defaulting to "replace"
func (c *Category) GetSystemPromptMode() string {
	if c.SystemPromptMode == "" {
		return "replace" // Default mode
	}
	return c.SystemPromptMode
}

// GetCategoryByName returns a category by name
func (c *RouterConfig) GetCategoryByName(name string) *Category {
	for i := range c.Categories {
		if c.Categories[i].Name == name {
			return &c.Categories[i]
		}
	}
	return nil
}

// IsCacheEnabledForCategory returns whether semantic caching is enabled for a specific category
// If the category has an explicit setting, it takes precedence; otherwise, uses global setting
func (c *RouterConfig) IsCacheEnabledForCategory(categoryName string) bool {
	category := c.GetCategoryByName(categoryName)
	if category != nil && category.SemanticCacheEnabled != nil {
		return *category.SemanticCacheEnabled
	}
	// Fall back to global setting
	return c.Enabled
}

// GetCacheSimilarityThresholdForCategory returns the effective cache similarity threshold for a category
// Priority: category-specific > global semantic_cache > bert_model threshold
func (c *RouterConfig) GetCacheSimilarityThresholdForCategory(categoryName string) float32 {
	category := c.GetCategoryByName(categoryName)
	if category != nil && category.SemanticCacheSimilarityThreshold != nil {
		return *category.SemanticCacheSimilarityThreshold
	}
	// Fall back to global cache threshold or bert threshold
	return c.GetCacheSimilarityThreshold()
}

// IsJailbreakEnabledForCategory returns whether jailbreak detection is enabled for a specific category
// If the category has an explicit setting, it takes precedence; otherwise, uses global setting
func (c *RouterConfig) IsJailbreakEnabledForCategory(categoryName string) bool {
	category := c.GetCategoryByName(categoryName)
	if category != nil && category.JailbreakEnabled != nil {
		return *category.JailbreakEnabled
	}
	// Fall back to global setting
	return c.PromptGuard.Enabled
}

// GetJailbreakThresholdForCategory returns the effective jailbreak detection threshold for a category
// Priority: category-specific > global prompt_guard threshold
func (c *RouterConfig) GetJailbreakThresholdForCategory(categoryName string) float32 {
	category := c.GetCategoryByName(categoryName)
	if category != nil && category.JailbreakThreshold != nil {
		return *category.JailbreakThreshold
	}
	// Fall back to global threshold
	return c.PromptGuard.Threshold
}

// IsPIIEnabledForCategory returns whether PII detection is enabled for a specific category
// If the category has an explicit setting, it takes precedence; otherwise, uses global setting
func (c *RouterConfig) IsPIIEnabledForCategory(categoryName string) bool {
	category := c.GetCategoryByName(categoryName)
	if category != nil && category.PIIEnabled != nil {
		return *category.PIIEnabled
	}
	// Fall back to global setting
	return c.IsPIIClassifierEnabled()
}

// GetPIIThresholdForCategory returns the effective PII detection threshold for a category
// Priority: category-specific > global classifier.pii_model threshold
func (c *RouterConfig) GetPIIThresholdForCategory(categoryName string) float32 {
	category := c.GetCategoryByName(categoryName)
	if category != nil && category.PIIThreshold != nil {
		return *category.PIIThreshold
	}
	// Fall back to global threshold
	return c.PIIModel.Threshold
}

// GetCacheSimilarityThreshold returns the effective threshold for the semantic cache
func (c *RouterConfig) GetCacheSimilarityThreshold() float32 {
	if c.SimilarityThreshold != nil {
		return *c.SimilarityThreshold
	}
	return c.Threshold
}
