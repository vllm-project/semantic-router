package config

import (
	"fmt"
	"net/url"
	"os"
	"slices"
	"strings"
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

// GetModelAPIFormat returns the API format for the given model.
// Returns APIFormatAnthropic if configured, otherwise APIFormatOpenAI (default).
func (c *RouterConfig) GetModelAPIFormat(modelName string) string {
	if c == nil || c.ModelConfig == nil {
		return APIFormatOpenAI
	}
	if modelConfig, ok := c.ModelConfig[modelName]; ok && modelConfig.APIFormat != "" {
		return modelConfig.APIFormat
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
		expandedKey := os.ExpandEnv(rawKey)
		return expandedKey
	}
	return ""
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

// IsJailbreakEnabledForDecision returns whether jailbreak detection is enabled for a specific decision
// Returns true only if the decision has an explicit jailbreak plugin configured with enabled: true
// This ensures per-decision scoping - decisions without jailbreak plugin won't execute jailbreak detection
func (c *RouterConfig) IsJailbreakEnabledForDecision(decisionName string) bool {
	decision := c.GetDecisionByName(decisionName)
	if decision != nil {
		config := decision.GetJailbreakConfig()
		if config != nil {
			return config.Enabled
		}
	}
	// No explicit jailbreak plugin configured for this decision
	// Return false to respect per-decision plugin scoping
	return false
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

// GetJailbreakIncludeHistoryForDecision returns whether to include conversation history in jailbreak detection
// Returns false by default (only check current user message)
func (c *RouterConfig) GetJailbreakIncludeHistoryForDecision(decisionName string) bool {
	decision := c.GetDecisionByName(decisionName)
	if decision != nil {
		config := decision.GetJailbreakConfig()
		if config != nil {
			return config.IncludeHistory
		}
	}
	// Default to false (only check current user message)
	return false
}

// IsPIIEnabledForDecision returns whether PII detection is enabled for a specific decision
// Returns true only if the decision has an explicit PII plugin configured with enabled: true
// This ensures per-decision scoping - decisions without PII plugin won't execute PII detection
func (c *RouterConfig) IsPIIEnabledForDecision(decisionName string) bool {
	decision := c.GetDecisionByName(decisionName)
	if decision != nil {
		config := decision.GetPIIConfig()
		if config != nil {
			return config.Enabled
		}
	}
	// No explicit PII plugin configured for this decision
	// Return false to respect per-decision plugin scoping
	return false
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

// GetPIIIncludeHistoryForDecision returns whether to include conversation history in PII detection
// Returns false by default (only check current user message)
func (c *RouterConfig) GetPIIIncludeHistoryForDecision(decisionName string) bool {
	decision := c.GetDecisionByName(decisionName)
	if decision != nil {
		config := decision.GetPIIConfig()
		if config != nil {
			return config.IncludeHistory
		}
	}
	// Default to false (only check current user message)
	return false
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

// IsFeedbackDetectorEnabled checks if feedback detection is enabled
func (c *RouterConfig) IsFeedbackDetectorEnabled() bool {
	return c.InlineModels.FeedbackDetector.Enabled &&
		c.InlineModels.FeedbackDetector.ModelID != ""
}

// ---------------------------------------------------------------------------
// Provider profile helpers
// ---------------------------------------------------------------------------

// providerTypeInfo holds the per-type defaults for a cloud provider.
// Every supported type MUST have an entry — no default/fallback branch.
type providerTypeInfo struct {
	AuthHeader string // HTTP header name for the API key
	AuthPrefix string // value prefix ("Bearer", "" etc.)
	ChatPath   string // path suffix appended after base_url path
}

// providerTypeRegistry is the single source of truth for type defaults.
// To add a new provider, add one entry here and a matching LLMProvider
// constant in pkg/authz/provider.go — nothing else needs a switch/default.
var providerTypeRegistry = map[string]providerTypeInfo{
	"openai":       {AuthHeader: "Authorization", AuthPrefix: "Bearer", ChatPath: "/chat/completions"},
	"anthropic":    {AuthHeader: "x-api-key", AuthPrefix: "", ChatPath: "/v1/messages"},
	"azure-openai": {AuthHeader: "api-key", AuthPrefix: "", ChatPath: "/chat/completions"},
	"bedrock":      {AuthHeader: "Authorization", AuthPrefix: "Bearer", ChatPath: "/chat/completions"},
	"gemini":       {AuthHeader: "Authorization", AuthPrefix: "Bearer", ChatPath: "/chat/completions"},
	"vertex-ai":    {AuthHeader: "Authorization", AuthPrefix: "Bearer", ChatPath: "/chat/completions"},
}

// ValidProviderTypes returns the set of recognised type strings (for error messages).
func ValidProviderTypes() []string {
	types := make([]string, 0, len(providerTypeRegistry))
	for t := range providerTypeRegistry {
		types = append(types, t)
	}
	return types
}

// GetProviderProfileForEndpoint resolves the ProviderProfile for a named endpoint.
//
// Returns (nil, nil) when the endpoint exists but has no provider_profile set
// (legacy address:port endpoint — this is not an error).
//
// Returns a non-nil error when:
//   - endpointName does not match any VLLMEndpoint
//   - the endpoint references a provider_profile name that does not exist in the map
func (c *RouterConfig) GetProviderProfileForEndpoint(endpointName string) (*ProviderProfile, error) {
	if endpointName == "" {
		return nil, nil // no endpoint selected (e.g., model has no preferred_endpoints)
	}
	ep, found := c.GetEndpointByName(endpointName)
	if !found {
		return nil, fmt.Errorf("endpoint %q not found in vllm_endpoints", endpointName)
	}
	if ep.ProviderProfileName == "" {
		return nil, nil // legacy endpoint, no profile — not an error
	}
	if c.ProviderProfiles == nil {
		return nil, fmt.Errorf("endpoint %q references provider_profile %q but no provider_profiles map is defined",
			endpointName, ep.ProviderProfileName)
	}
	profile, ok := c.ProviderProfiles[ep.ProviderProfileName]
	if !ok {
		return nil, fmt.Errorf("endpoint %q references provider_profile %q which does not exist in provider_profiles (have: %v)",
			endpointName, ep.ProviderProfileName, mapKeys(c.ProviderProfiles))
	}
	return &profile, nil
}

// mapKeys returns the keys of a map for diagnostic messages.
func mapKeys(m map[string]ProviderProfile) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

// ResolveAddress returns the host:port string for this endpoint.
//
// Two distinct modes — no silent fallback between them:
//   - provider_profile set → host:port is extracted from the profile's base_url.
//     Returns error if profile is missing, has no base_url, or base_url is unparseable.
//   - provider_profile NOT set → uses address:port fields directly.
func (ep *VLLMEndpoint) ResolveAddress(profiles map[string]ProviderProfile) (string, error) {
	if ep.ProviderProfileName == "" {
		// Legacy endpoint: address:port is the intended mode.
		return fmt.Sprintf("%s:%d", ep.Address, ep.Port), nil
	}

	// Profile-based endpoint: MUST resolve from base_url.
	if profiles == nil {
		return "", fmt.Errorf("endpoint %q has provider_profile %q but no provider_profiles map is defined",
			ep.Name, ep.ProviderProfileName)
	}
	profile, ok := profiles[ep.ProviderProfileName]
	if !ok {
		return "", fmt.Errorf("endpoint %q references provider_profile %q which does not exist",
			ep.Name, ep.ProviderProfileName)
	}
	if profile.BaseURL == "" {
		return "", fmt.Errorf("endpoint %q: provider_profile %q has no base_url",
			ep.Name, ep.ProviderProfileName)
	}

	u, err := url.Parse(profile.BaseURL)
	if err != nil {
		return "", fmt.Errorf("endpoint %q: cannot parse base_url %q: %w",
			ep.Name, profile.BaseURL, err)
	}
	if u.Host == "" {
		return "", fmt.Errorf("endpoint %q: base_url %q has no host",
			ep.Name, profile.BaseURL)
	}

	host := u.Host
	if !strings.Contains(host, ":") {
		switch u.Scheme {
		case "https":
			host += ":443"
		case "http":
			host += ":80"
		default:
			return "", fmt.Errorf("endpoint %q: base_url %q has unsupported scheme %q (expected http or https)",
				ep.Name, profile.BaseURL, u.Scheme)
		}
	}
	return host, nil
}

// ProviderType returns the provider type string, which matches authz.LLMProvider values.
// Returns an error if the type is empty or not in providerTypeRegistry.
func (p *ProviderProfile) ProviderType() (string, error) {
	if p == nil {
		return "", fmt.Errorf("provider profile is nil")
	}
	if p.Type == "" {
		return "", fmt.Errorf("provider profile has empty type")
	}
	if _, ok := providerTypeRegistry[p.Type]; !ok {
		return "", fmt.Errorf("unknown provider profile type %q (valid types: %v)", p.Type, ValidProviderTypes())
	}
	return p.Type, nil
}

// ResolveAuthHeader returns the (headerName, prefix) for the upstream auth header.
// Explicit AuthHeader/AuthPrefix fields override the type defaults.
// Returns error if the profile's type is not recognised.
func (p *ProviderProfile) ResolveAuthHeader() (string, string, error) {
	info, ok := providerTypeRegistry[p.Type]
	if !ok {
		return "", "", fmt.Errorf("unknown provider type %q — cannot determine auth header", p.Type)
	}
	headerName := info.AuthHeader
	prefix := info.AuthPrefix
	if p.AuthHeader != "" {
		headerName = p.AuthHeader
	}
	if p.AuthPrefix != "" {
		prefix = p.AuthPrefix
	}
	return headerName, prefix, nil
}

// ResolveChatPath returns the HTTP path for upstream requests.
//
// Resolution order (no silent fallback):
//  1. Explicit ChatPath field on the profile (used as-is, plus ?api-version for azure-openai).
//  2. base_url path + type-default suffix from providerTypeRegistry.
//  3. Type-default suffix alone if base_url has no path component.
//
// Returns error if the type is not recognised or base_url is unparseable.
func (p *ProviderProfile) ResolveChatPath() (string, error) {
	if p == nil {
		return "", fmt.Errorf("provider profile is nil")
	}

	info, ok := providerTypeRegistry[p.Type]
	if !ok {
		return "", fmt.Errorf("unknown provider type %q — cannot determine chat path", p.Type)
	}

	// Explicit override
	if p.ChatPath != "" {
		path := p.ChatPath
		if p.Type == "azure-openai" && p.APIVersion != "" {
			path += "?api-version=" + p.APIVersion
		}
		return path, nil
	}

	suffix := info.ChatPath
	if p.Type == "azure-openai" && p.APIVersion != "" {
		suffix += "?api-version=" + p.APIVersion
	}

	// Prepend base_url path component if present
	if p.BaseURL != "" {
		u, err := url.Parse(p.BaseURL)
		if err != nil {
			return "", fmt.Errorf("cannot parse base_url %q: %w", p.BaseURL, err)
		}
		if u.Path != "" && u.Path != "/" {
			return strings.TrimRight(u.Path, "/") + suffix, nil
		}
	}

	return suffix, nil
}
