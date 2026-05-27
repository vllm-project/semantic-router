package config

import (
	"fmt"
	"os"
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
		_, baseParams, fallback := rc.resolveLoRABaseModel(modelName)
		if !fallback || baseParams.ReasoningFamily == "" {
			return nil
		}
		modelParams = baseParams
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

// resolveModelConfig looks up a model by name, falling back to a reverse
// lookup through ExternalModelIDs (provider_model_id) when the name is not
// a direct key in ModelConfig. This handles the case where the Envoy AI
// Gateway rewrites the model field to the provider model ID.
func (c *RouterConfig) resolveModelConfig(modelName string) (ModelParams, bool) {
	if params, ok := c.ModelConfig[modelName]; ok {
		return params, true
	}
	for _, params := range c.ModelConfig {
		for _, extID := range params.ExternalModelIDs {
			if extID == modelName {
				return params, true
			}
		}
	}
	return ModelParams{}, false
}

// ModelPricingResult holds the full pricing breakdown for a model.
type ModelPricingResult struct {
	PromptPer1M     float64
	CompletionPer1M float64
	CacheReadPer1M  float64
	CacheWritePer1M float64
	Currency        string
}

// GetModelPricing returns pricing per 1M tokens and its currency for the given model.
// The currency indicates the unit of the returned rates (e.g., "USD").
// Accepts both short names ("claude-haiku-4-5") and provider model IDs
// ("eu.anthropic.claude-haiku-4-5-20251001-v1:0").
func (c *RouterConfig) GetModelPricing(modelName string) (promptPer1M float64, completionPer1M float64, currency string, ok bool) {
	if modelConfig, okc := c.resolveModelConfig(modelName); okc {
		p := modelConfig.Pricing
		// Treat an explicit zero-price entry as configured pricing when a currency is
		// present so self-hosted/free models still produce cost=0 and savings data.
		if p.PromptPer1M != 0 || p.CompletionPer1M != 0 || p.Currency != "" {
			cur := p.Currency
			if cur == "" {
				cur = "USD"
			}
			return p.PromptPer1M, p.CompletionPer1M, cur, true
		}
	}
	return 0, 0, "", false
}

// GetModelPricingFull returns the complete pricing breakdown for a model,
// including cache read/write rates for Anthropic prompt caching.
// Accepts both short names and provider model IDs.
func (c *RouterConfig) GetModelPricingFull(modelName string) (ModelPricingResult, bool) {
	if modelConfig, okc := c.resolveModelConfig(modelName); okc {
		p := modelConfig.Pricing
		if p.PromptPer1M != 0 || p.CompletionPer1M != 0 || p.CacheReadPer1M != 0 || p.CacheWritePer1M != 0 || p.Currency != "" {
			cur := p.Currency
			if cur == "" {
				cur = "USD"
			}
			return ModelPricingResult{
				PromptPer1M:     p.PromptPer1M,
				CompletionPer1M: p.CompletionPer1M,
				CacheReadPer1M:  p.CacheReadPer1M,
				CacheWritePer1M: p.CacheWritePer1M,
				Currency:        cur,
			}, true
		}
	}
	return ModelPricingResult{}, false
}

// GetMostExpensivePricedModel returns the configured model with the highest combined
// prompt+completion rate among models that define pricing.
func (c *RouterConfig) GetMostExpensivePricedModel() (modelName string, promptPer1M float64, completionPer1M float64, currency string, ok bool) {
	if c == nil || c.ModelConfig == nil {
		return "", 0, 0, "", false
	}

	bestScore := 0.0
	for candidate := range c.ModelConfig {
		promptRate, completionRate, candidateCurrency, found := c.GetModelPricing(candidate)
		if !found {
			continue
		}

		score := promptRate + completionRate
		if !ok || score > bestScore {
			modelName = candidate
			promptPer1M = promptRate
			completionPer1M = completionRate
			currency = candidateCurrency
			bestScore = score
			ok = true
		}
	}

	return modelName, promptPer1M, completionPer1M, currency, ok
}

// GetModelAPIFormat returns the API format for the given model.
// Returns APIFormatAnthropic if configured, otherwise APIFormatOpenAI (default).
func (c *RouterConfig) GetModelAPIFormat(modelName string) string {
	if c == nil || c.ModelConfig == nil {
		return APIFormatOpenAI
	}
	if modelConfig, ok := c.ModelConfig[modelName]; ok && modelConfig.APIFormat != "" {
		return modelConfig.APIFormat
	}
	if _, baseConfig, ok := c.resolveLoRABaseModel(modelName); ok && baseConfig.APIFormat != "" {
		return baseConfig.APIFormat
	}
	return APIFormatOpenAI
}

// GetModelAccessKey returns the access key for the given model.
func (c *RouterConfig) GetModelAccessKey(modelName string) string {
	if c == nil || c.ModelConfig == nil {
		return ""
	}
	if modelConfig, ok := c.ModelConfig[modelName]; ok {
		rawKey := modelConfig.AccessKey
		if rawKey != "" {
			expandedKey := os.ExpandEnv(rawKey)
			return expandedKey
		}
	}
	if _, baseConfig, ok := c.resolveLoRABaseModel(modelName); ok && baseConfig.AccessKey != "" {
		return os.ExpandEnv(baseConfig.AccessKey)
	}
	return ""
}

// GetDecisionPIIPolicy returns the PII policy for a given decision by looking at
// the PIIRule signals referenced in the decision's rules tree.
// If the decision doesn't reference any PII signals, returns a default policy that allows all PII.
func (d *Decision) GetDecisionPIIPolicy(piiRules []PIIRule) PIIPolicy {
	// Collect PII signal names referenced in the decision's rules
	piiSignalNames := collectSignalNames(&d.Rules, "pii")
	if len(piiSignalNames) == 0 {
		// No PII signals → allow all PII
		return PIIPolicy{
			AllowByDefault: true,
			PIITypes:       []string{},
		}
	}

	// Build a lookup for PIIRules by name
	rulesByName := make(map[string]*PIIRule, len(piiRules))
	for i := range piiRules {
		rulesByName[piiRules[i].Name] = &piiRules[i]
	}

	// Aggregate PIITypesAllowed from all referenced PIIRules
	var allAllowed []string
	for _, name := range piiSignalNames {
		if rule, ok := rulesByName[name]; ok {
			allAllowed = append(allAllowed, rule.PIITypesAllowed...)
		}
	}

	return PIIPolicy{
		AllowByDefault: false,
		PIITypes:       allAllowed,
	}
}

// IsDecisionAllowedForPIIType checks if a decision is allowed to process a specific PII type
func (d *Decision) IsDecisionAllowedForPIIType(piiType string, piiRules []PIIRule) bool {
	policy := d.GetDecisionPIIPolicy(piiRules)

	// If allow_by_default is true, all PII types are allowed unless explicitly denied
	if policy.AllowByDefault {
		return true
	}

	// If allow_by_default is false, only explicitly allowed PII types are permitted
	return slices.Contains(policy.PIITypes, piiType)
}

// IsDecisionAllowedForPIITypes checks if a decision is allowed to process any of the given PII types
func (d *Decision) IsDecisionAllowedForPIITypes(piiTypes []string, piiRules []PIIRule) bool {
	for _, piiType := range piiTypes {
		if !d.IsDecisionAllowedForPIIType(piiType, piiRules) {
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
	if c == nil || c.ModelConfig == nil {
		return nil
	}

	if modelConfig, ok := c.ModelConfig[modelName]; ok && len(modelConfig.PreferredEndpoints) > 0 {
		return c.collectPreferredEndpoints(modelConfig.PreferredEndpoints)
	}
	if _, baseConfig, ok := c.resolveLoRABaseModel(modelName); ok && len(baseConfig.PreferredEndpoints) > 0 {
		return c.collectPreferredEndpoints(baseConfig.PreferredEndpoints)
	}
	return nil
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

// SelectBestEndpointAddressForModel selects the best endpoint for a model and returns the address:port.
// When the endpoint has a provider_profile with a base_url, the host:port is extracted from it.
// Returns ("", false, nil) when no endpoints match the model.
// Returns ("", false, err) when the selected endpoint has a broken provider_profile/base_url.
func (c *RouterConfig) SelectBestEndpointAddressForModel(modelName string) (string, bool, error) {
	endpoints := c.GetEndpointsForModel(modelName)
	if len(endpoints) == 0 {
		return "", false, nil
	}

	bestEndpoint := endpoints[0]
	for _, endpoint := range endpoints[1:] {
		if endpoint.Weight > bestEndpoint.Weight {
			bestEndpoint = endpoint
		}
	}

	addr, err := bestEndpoint.ResolveAddress(c.ProviderProfiles)
	if err != nil {
		return "", false, fmt.Errorf("endpoint %q for model %q: %w", bestEndpoint.Name, modelName, err)
	}
	return addr, true, nil
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
		return "insert" // Default mode
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
// Returns true only if the decision has an explicit semantic-cache plugin configured with enabled: true
// This ensures per-decision scoping - decisions without semantic-cache plugin won't execute caching
func (c *RouterConfig) IsCacheEnabledForDecision(decisionName string) bool {
	decision := c.GetDecisionByName(decisionName)
	if decision != nil {
		config := decision.GetSemanticCacheConfig()
		if config != nil {
			return config.Enabled
		}
	}
	// No explicit semantic-cache plugin configured for this decision
	// Return false to respect per-decision plugin scoping
	return false
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

// ResolveExternalModelID resolves the external model ID for a given model name and endpoint.
// When a model alias (e.g., "qwen14b-rack1") is configured with external_model_ids,
// this returns the real model name that the backend expects (e.g., "Qwen/Qwen2.5-14B-Instruct").
// The endpoint type (e.g., "vllm", "ollama") is looked up from the selected endpoint.
// Returns the original modelName if no mapping is found.
func (c *RouterConfig) ResolveExternalModelID(modelName string, endpointName string) string {
	if c == nil || c.ModelConfig == nil {
		return modelName
	}

	modelConfig, ok := c.ModelConfig[modelName]
	if !ok || len(modelConfig.ExternalModelIDs) == 0 {
		return modelName
	}

	// Get the endpoint type from the endpoint name
	endpointType := ""
	if endpoint, found := c.GetEndpointByName(endpointName); found && endpoint.Type != "" {
		endpointType = endpoint.Type
	} else {
		// Default endpoint type is "vllm"
		endpointType = "vllm"
	}

	// Look up the external model ID for this endpoint type
	if externalID, ok := modelConfig.ExternalModelIDs[endpointType]; ok && externalID != "" {
		return externalID
	}

	return modelName
}

// SelectBestEndpointWithDetailsForModel selects the best endpoint for a model and returns
// both the address:port and the endpoint name (needed for external_model_ids resolution).
// Returns (address, endpointName, found).
func (c *RouterConfig) SelectBestEndpointWithDetailsForModel(modelName string) (string, string, bool, error) {
	endpoints := c.GetEndpointsForModel(modelName)
	if len(endpoints) == 0 {
		return "", "", false, nil
	}

	bestEndpoint := endpoints[0]
	for _, endpoint := range endpoints[1:] {
		if endpoint.Weight > bestEndpoint.Weight {
			bestEndpoint = endpoint
		}
	}

	addr, err := bestEndpoint.ResolveAddress(c.ProviderProfiles)
	if err != nil {
		return "", "", false, fmt.Errorf("endpoint %q for model %q: %w", bestEndpoint.Name, modelName, err)
	}
	return addr, bestEndpoint.Name, true, nil
}
