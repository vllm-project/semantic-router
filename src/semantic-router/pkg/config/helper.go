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

// GetModelForDecisionIndex returns the best LLM model name for the decision at the given index
func (c *RouterConfig) GetModelForDecisionIndex(index int) string {
	if index < 0 || index >= len(c.Decisions) {
		return c.DefaultModel
	}

	decision := c.Decisions[index]
	if len(decision.ModelRefs) > 0 {
		return decision.ModelRefs[0].Model
	}

	// Fall back to default model if decision has no models
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

// GetDecisionPIIPolicy returns the PII policy for a given decision
// If the decision doesn't have a PII plugin or policy config, returns a default policy that allows all PII
func (d *Decision) GetDecisionPIIPolicy() PIIPolicy {
	piiConfig := d.GetPIIConfig()
	if piiConfig == nil {
		// Default policy allows all PII (no PII plugin configured)
		return PIIPolicy{
			AllowByDefault: true,
			PIITypes:       []string{},
		}
	}

	// When PII plugin is enabled, default behavior is to block all PII (AllowByDefault: false)
	// unless specific types are listed in PIITypesAllowed
	allowByDefault := !piiConfig.Enabled

	return PIIPolicy{
		AllowByDefault: allowByDefault,
		PIITypes:       piiConfig.PIITypesAllowed,
	}
}

// IsDecisionAllowedForPIIType checks if a decision is allowed to process a specific PII type
func (d *Decision) IsDecisionAllowedForPIIType(piiType string) bool {
	policy := d.GetDecisionPIIPolicy()

	// If allow_by_default is true, all PII types are allowed unless explicitly denied
	if policy.AllowByDefault {
		return true
	}

	// If allow_by_default is false, only explicitly allowed PII types are permitted
	return slices.Contains(policy.PIITypes, piiType)
}

// IsDecisionAllowedForPIITypes checks if a decision is allowed to process any of the given PII types
func (d *Decision) IsDecisionAllowedForPIITypes(piiTypes []string) bool {
	for _, piiType := range piiTypes {
		if !d.IsDecisionAllowedForPIIType(piiType) {
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
	if !c.PromptGuard.Enabled || c.PromptGuard.JailbreakMappingPath == "" {
		return false
	}

	// Check configuration based on whether using vLLM or Candle
	if c.PromptGuard.UseVLLM {
		// For vLLM: need external model with role="guardrail"
		externalCfg := c.FindExternalModelByRole(ModelRoleGuardrail)
		return externalCfg != nil &&
			externalCfg.ModelEndpoint.Address != "" &&
			externalCfg.ModelName != ""
	}

	// For Candle: need model ID
	return c.PromptGuard.ModelID != ""
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

// GetModelReasoningForDecision returns whether a specific model supports reasoning in a given decision
func (c *RouterConfig) GetModelReasoningForDecision(decisionName string, modelName string) bool {
	for _, decision := range c.Decisions {
		if decision.Name == decisionName {
			for _, modelRef := range decision.ModelRefs {
				if modelRef.Model == modelName {
					return modelRef.UseReasoning != nil && *modelRef.UseReasoning
				}
			}
		}
	}
	return false // Default to false if decision or model not found
}

// GetBestModelForDecision returns the best model for a given decision (first model in ModelRefs)
func (c *RouterConfig) GetBestModelForDecision(decisionName string) (string, bool) {
	for _, decision := range c.Decisions {
		if decision.Name == decisionName {
			if len(decision.ModelRefs) > 0 {
				useReasoning := decision.ModelRefs[0].UseReasoning != nil && *decision.ModelRefs[0].UseReasoning
				return decision.ModelRefs[0].Model, useReasoning
			}
		}
	}
	return "", false // Return empty string and false if decision not found or has no models
}

// ValidateEndpoints validates that all configured models have at least one endpoint
func (c *RouterConfig) ValidateEndpoints() error {
	// Get all models from decisions
	allCategoryModels := make(map[string]bool)
	for _, decision := range c.Decisions {
		for _, modelRef := range decision.ModelRefs {
			allCategoryModels[modelRef.Model] = true
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

// IsSystemPromptEnabled returns whether system prompt injection is enabled for a decision
func (d *Decision) IsSystemPromptEnabled() bool {
	config := d.GetSystemPromptConfig()
	if config == nil {
		return false
	}
	// If Enabled is explicitly set, use that value
	if config.Enabled != nil {
		return *config.Enabled
	}
	// Default to true if SystemPrompt is not empty
	return config.SystemPrompt != ""
}

// GetSystemPromptMode returns the system prompt injection mode, defaulting to "replace"
func (d *Decision) GetSystemPromptMode() string {
	config := d.GetSystemPromptConfig()
	if config == nil || config.Mode == "" {
		return "replace" // Default mode
	}
	return config.Mode
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

// GetDecisionByName returns a decision by name
func (c *RouterConfig) GetDecisionByName(name string) *Decision {
	for i := range c.Decisions {
		if c.Decisions[i].Name == name {
			return &c.Decisions[i]
		}
	}
	return nil
}

// IsCacheEnabledForDecision returns whether semantic caching is enabled for a specific decision
func (c *RouterConfig) IsCacheEnabledForDecision(decisionName string) bool {
	decision := c.GetDecisionByName(decisionName)
	if decision != nil {
		config := decision.GetSemanticCacheConfig()
		if config != nil {
			return config.Enabled
		}
	}
	// Fall back to global setting
	return c.Enabled
}

// GetCacheSimilarityThresholdForDecision returns the effective cache similarity threshold for a decision
func (c *RouterConfig) GetCacheSimilarityThresholdForDecision(decisionName string) float32 {
	decision := c.GetDecisionByName(decisionName)
	if decision != nil {
		config := decision.GetSemanticCacheConfig()
		if config != nil && config.SimilarityThreshold != nil {
			return *config.SimilarityThreshold
		}
	}
	// Fall back to global cache threshold or bert threshold
	return c.GetCacheSimilarityThreshold()
}

// GetCacheTTLSecondsForDecision returns the effective TTL for a decision
// Returns 0 if caching should be skipped for this decision
// Returns -1 to use the global default TTL when not specified at decision level
func (c *RouterConfig) GetCacheTTLSecondsForDecision(decisionName string) int {
	decision := c.GetDecisionByName(decisionName)
	if decision != nil {
		config := decision.GetSemanticCacheConfig()
		if config != nil && config.TTLSeconds != nil {
			return *config.TTLSeconds
		}
	}
	// Return -1 to indicate "use global default"
	return -1
}

// IsJailbreakEnabledForDecision returns whether jailbreak detection is enabled for a specific decision
func (c *RouterConfig) IsJailbreakEnabledForDecision(decisionName string) bool {
	decision := c.GetDecisionByName(decisionName)
	if decision != nil {
		config := decision.GetJailbreakConfig()
		if config != nil {
			return config.Enabled
		}
	}
	// Fall back to global setting
	return c.PromptGuard.Enabled
}

// GetJailbreakThresholdForDecision returns the effective jailbreak detection threshold for a decision
func (c *RouterConfig) GetJailbreakThresholdForDecision(decisionName string) float32 {
	decision := c.GetDecisionByName(decisionName)
	if decision != nil {
		config := decision.GetJailbreakConfig()
		if config != nil && config.Threshold != nil {
			return *config.Threshold
		}
	}
	// Fall back to global threshold
	return c.PromptGuard.Threshold
}

// IsPIIEnabledForDecision returns whether PII detection is enabled for a specific decision
func (c *RouterConfig) IsPIIEnabledForDecision(decisionName string) bool {
	decision := c.GetDecisionByName(decisionName)
	if decision != nil {
		config := decision.GetPIIConfig()
		if config != nil {
			return config.Enabled
		}
	}
	// Fall back to global setting
	return c.IsPIIClassifierEnabled()
}

// GetPIIThresholdForDecision returns the effective PII detection threshold for a decision
func (c *RouterConfig) GetPIIThresholdForDecision(decisionName string) float32 {
	decision := c.GetDecisionByName(decisionName)
	if decision != nil {
		config := decision.GetPIIConfig()
		if config != nil && config.Threshold != nil {
			return *config.Threshold
		}
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

// IsHallucinationMitigationEnabled checks if hallucination mitigation is enabled and properly configured
func (c *RouterConfig) IsHallucinationMitigationEnabled() bool {
	return c.HallucinationMitigation.Enabled
}

// IsFactCheckClassifierEnabled checks if the fact-check classifier is enabled and properly configured
// Enabled when fact_check_rules are configured, or legacy HallucinationMitigation is enabled
func (c *RouterConfig) IsFactCheckClassifierEnabled() bool {
	// Check new fact_check_rules config first
	if len(c.FactCheckRules) > 0 {
		// For new signal config, still need the model from HallucinationMitigation
		return c.HallucinationMitigation.FactCheckModel.ModelID != ""
	}

	// Fall back to legacy HallucinationMitigation config
	if !c.HallucinationMitigation.Enabled {
		return false
	}
	return c.HallucinationMitigation.FactCheckModel.ModelID != ""
}

// GetFactCheckRules returns all configured fact_check_rules
func (c *RouterConfig) GetFactCheckRules() []FactCheckRule {
	return c.FactCheckRules
}

// IsHallucinationModelEnabled checks if hallucination detection is enabled and properly configured
// Returns true if either:
// 1. hallucination_mitigation.enabled is true (legacy global config)
// 2. Any decision has a hallucination plugin enabled (new per-decision config)
// AND the hallucination model is properly configured
func (c *RouterConfig) IsHallucinationModelEnabled() bool {
	// Must have hallucination model configured
	if c.HallucinationMitigation.HallucinationModel.ModelID == "" {
		return false
	}

	// Check legacy global config
	if c.HallucinationMitigation.Enabled {
		return true
	}

	// Check if any decision has hallucination plugin enabled
	for _, decision := range c.Decisions {
		halConfig := decision.GetHallucinationConfig()
		if halConfig != nil && halConfig.Enabled {
			return true
		}
	}

	return false
}

// GetFactCheckThreshold returns the threshold for fact-check classification
// Returns default of 0.7 if not specified
func (c *RouterConfig) GetFactCheckThreshold() float32 {
	if c.HallucinationMitigation.FactCheckModel.Threshold > 0 {
		return c.HallucinationMitigation.FactCheckModel.Threshold
	}
	return 0.7 // Default threshold
}

// GetHallucinationModelThreshold returns the threshold for hallucination detection
// Returns default of 0.5 if not specified
func (c *RouterConfig) GetHallucinationModelThreshold() float32 {
	if c.HallucinationMitigation.HallucinationModel.Threshold > 0 {
		return c.HallucinationMitigation.HallucinationModel.Threshold
	}
	return 0.5 // Default threshold
}

// GetHallucinationAction returns the action to take when hallucination is detected
// Returns "warn" as default if not specified
func (c *RouterConfig) GetHallucinationAction() string {
	action := c.HallucinationMitigation.OnHallucinationDetected
	if action == "" {
		return "warn"
	}
	// Only "warn" is supported now
	return "warn"
}

// IsFeedbackDetectorEnabled checks if feedback detection is enabled
func (c *RouterConfig) IsFeedbackDetectorEnabled() bool {
	return c.InlineModels.FeedbackDetector.Enabled &&
		c.InlineModels.FeedbackDetector.ModelID != ""
}
