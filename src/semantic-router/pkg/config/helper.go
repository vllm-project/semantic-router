package config

import (
	"fmt"
	"slices"
	"strings"
)

const (
	DefaultAutoModelName    = "MoM"
	LegacyAutoModelAlias    = "auto"
	DefaultVSRAutoModelName = "vllm-sr/auto"
)

// GetModelReasoningFamily returns the reasoning family configuration for a given model name
func (rc *RouterConfig) GetModelReasoningFamily(modelName string) *ReasoningFamilyConfig {
	familyName, ok := rc.GetModelReasoningFamilyName(modelName)
	if !ok {
		return nil
	}

	familyConfig, exists := rc.ReasoningFamilies[familyName]
	if !exists {
		return nil
	}

	return &familyConfig
}

func (rc *RouterConfig) GetModelReasoningFamilyName(modelName string) (string, bool) {
	if rc == nil || rc.ModelConfig == nil || rc.ReasoningFamilies == nil {
		return "", false
	}

	// Look up the model in model_config, including provider_model_id/external IDs
	// because request-body mutation may already have rewritten the model field.
	modelParams, exists := rc.resolveModelConfig(modelName)
	if !exists || modelParams.ReasoningFamily == "" {
		_, baseParams, fallback := rc.resolveLoRABaseModel(modelName)
		if !fallback || baseParams.ReasoningFamily == "" {
			return "", false
		}
		modelParams = baseParams
	}
	_, exists = rc.ReasoningFamilies[modelParams.ReasoningFamily]
	return modelParams.ReasoningFamily, exists
}

// GetEffectiveAutoModelName returns the configured primary automatic-routing
// name, or the stable v0.3 default when omitted.
func (c *RouterConfig) GetEffectiveAutoModelName() string {
	if c != nil && c.AutoModelName != "" {
		return c.AutoModelName
	}
	return DefaultAutoModelName
}

func DefaultAutoModelNames() []string {
	return []string{DefaultVSRAutoModelName, LegacyAutoModelAlias, DefaultAutoModelName}
}

// EffectiveAutoModelNames returns the existing v0.3 implicit Entrypoint names.
// A present empty auto_model_names list disables implicit names so explicit
// Entrypoints can own the request-facing surface without collisions.
func (c *RouterConfig) EffectiveAutoModelNames() []string {
	if c == nil {
		return DefaultAutoModelNames()
	}
	if c.AutoModelNames != nil {
		return normalizeAutoModelNames(c.AutoModelNames)
	}
	return normalizeAutoModelNames([]string{
		DefaultVSRAutoModelName,
		LegacyAutoModelAlias,
		c.GetEffectiveAutoModelName(),
	})
}

func normalizeAutoModelNames(names []string) []string {
	seen := make(map[string]struct{}, len(names))
	result := make([]string, 0, len(names))
	for _, name := range names {
		normalized := strings.TrimSpace(name)
		if normalized == "" {
			continue
		}
		if _, duplicate := seen[normalized]; duplicate {
			continue
		}
		seen[normalized] = struct{}{}
		result = append(result, normalized)
	}
	return result
}

func (rc *RouterConfig) ModelNameMatches(candidateName string, modelName string) bool {
	if candidateName == modelName {
		return true
	}
	if rc == nil || rc.ModelConfig == nil {
		return false
	}

	candidateKey, candidateExists := rc.resolveModelConfigKey(candidateName)
	modelKey, modelExists := rc.resolveModelConfigKey(modelName)
	return candidateExists && modelExists && candidateKey == modelKey
}

func (c *RouterConfig) resolveModelConfigKey(modelName string) (string, bool) {
	if _, ok := c.ModelConfig[modelName]; ok {
		return modelName, true
	}
	for configuredName, params := range c.ModelConfig {
		for _, extID := range params.ExternalModelIDs {
			if extID == modelName {
				return configuredName, true
			}
		}
	}
	return "", false
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

// GetModelPricing returns pricing per 1M tokens and its currency for the given model.
// The currency indicates the unit of the returned rates (e.g., "USD").
// Accepts both short names ("claude-haiku-4-5") and provider model IDs
// ("eu.anthropic.claude-haiku-4-5-20251001-v1:0").
func (c *RouterConfig) GetModelPricing(modelName string) (promptPer1M float64, completionPer1M float64, currency string, ok bool) {
	pricing, ok := c.GetFullModelPricing(modelName)
	if !ok {
		return 0, 0, "", false
	}
	return pricing.PromptPer1M, pricing.CompletionPer1M, pricing.Currency, true
}

// GetMostExpensivePricedModel returns prompt and completion rates for the model
// selected by GetMostExpensiveFullModelPricing.
func (c *RouterConfig) GetMostExpensivePricedModel() (modelName string, promptPer1M float64, completionPer1M float64, currency string, ok bool) {
	modelName, pricing, ok := c.GetMostExpensiveFullModelPricing()
	if !ok {
		return "", 0, 0, "", false
	}
	return modelName, pricing.PromptPer1M, pricing.CompletionPer1M, pricing.Currency, true
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
	return c.PIIModel.Active() && c.PIIModel.ModelID != "" && c.PIIMappingPath != ""
}

// IsCategoryClassifierEnabled checks if category classification is enabled
func (c *RouterConfig) IsCategoryClassifierEnabled() bool {
	return c.CategoryModel.Active() && c.CategoryModel.ModelID != "" && c.CategoryMappingPath != ""
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

	// Check configuration based on the selected backend
	if c.PromptGuard.Protocol != "" {
		// For remote backends: need external model with role="guardrail"
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

// GetDecisionByName returns a decision from an already-scoped Recipe view.
// Names are Recipe-local; request-time callers should retain the selected
// Decision object instead of performing a global bare-name lookup.
func (c *RouterConfig) GetDecisionByName(name string) *Decision {
	if c == nil || c.RoutingScope == "" {
		return nil
	}
	for i := range c.Decisions {
		if c.Decisions[i].Name == name {
			return &c.Decisions[i]
		}
	}
	return nil
}

// IsCacheEnabledForDecision returns whether response caching is enabled for a specific decision.
// Decisions without an explicit response_cache plugin never execute caching.
func (c *RouterConfig) IsCacheEnabledForDecision(decisionName string) bool {
	return c.IsCacheEnabledForDecisionObject(c.GetDecisionByName(decisionName))
}

// IsCacheEnabled resolves response_cache policy from a scoped decision.
func (c *RouterConfig) IsCacheEnabledForDecisionObject(decision *Decision) bool {
	if decision != nil {
		config := decision.GetResponseCacheConfig()
		if config != nil {
			return config.Enabled
		}
	}
	// No explicit response_cache plugin configured for this decision
	// Return false to respect per-decision plugin scoping
	return false
}

// GetCacheSimilarityThresholdForDecision returns the effective cache similarity threshold for a decision
func (c *RouterConfig) GetCacheSimilarityThresholdForDecision(decisionName string) float32 {
	return c.GetCacheSimilarityThresholdForDecisionObject(c.GetDecisionByName(decisionName))
}

// GetCacheSimilarityThreshold resolves the threshold from a scoped decision.
func (c *RouterConfig) GetCacheSimilarityThresholdForDecisionObject(decision *Decision) float32 {
	if decision != nil {
		config := decision.GetResponseCacheConfig()
		if config != nil && config.EffectiveSimilarityThreshold() != nil {
			return *config.EffectiveSimilarityThreshold()
		}
	}
	// Route policy does not cascade the backend's global threshold.
	return 0.8
}

// GetCacheTTLSecondsForDecision returns the effective TTL for a decision
// Returns 0 if caching should be skipped for this decision
// Returns -1 to use the global default TTL when not specified at decision level
func (c *RouterConfig) GetCacheTTLSecondsForDecision(decisionName string) int {
	return c.GetCacheTTLSecondsForDecisionObject(c.GetDecisionByName(decisionName))
}

// GetCacheTTLSeconds resolves TTL from a scoped decision.
func (c *RouterConfig) GetCacheTTLSecondsForDecisionObject(decision *Decision) int {
	if decision != nil {
		config := decision.GetResponseCacheConfig()
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
// The endpoint type (e.g., "vllm", "ollama") is looked up from the resolved backend metadata.
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
