/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package k8s

import (
	"bytes"
	"encoding/json"
	"fmt"

	yamlv3 "gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apis/vllm.ai/v1alpha1"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// CRDConverter converts Kubernetes CRDs to internal configuration structures
type CRDConverter struct{}

// NewCRDConverter creates a new CRD converter
func NewCRDConverter() *CRDConverter { return &CRDConverter{} }

// Convert builds a canonical v0.3 config from IntelligentPool/IntelligentRoute
// CRDs plus a static canonical base.
func (c *CRDConverter) Convert(
	pool *v1alpha1.IntelligentPool,
	route *v1alpha1.IntelligentRoute,
	staticBase *config.CanonicalConfig,
) (*config.CanonicalConfig, error) {
	if pool == nil {
		return nil, fmt.Errorf("pool cannot be nil")
	}
	if route == nil {
		return nil, fmt.Errorf("route cannot be nil")
	}

	canonical, err := cloneCanonicalConfig(staticBase)
	if err != nil {
		return nil, err
	}
	canonical.Version = "v0.3"

	canonical.Providers.Defaults.DefaultModel = pool.Spec.DefaultModel
	canonical.Routing.ModelCards = convertRoutingModelCards(pool.Spec.Models)
	canonical.Routing.Signals = convertSignals(route.Spec.Signals)
	canonical.Routing.Decisions = make([]config.Decision, 0, len(route.Spec.Decisions))
	for _, decision := range route.Spec.Decisions {
		configDecision, err := c.convertDecision(decision)
		if err != nil {
			return nil, fmt.Errorf("failed to convert decision %s: %w", decision.Name, err)
		}
		canonical.Routing.Decisions = append(canonical.Routing.Decisions, configDecision)
	}

	canonical.Providers.Models = mergeProviderMetadata(canonical.Providers.Models, convertProviderMetadata(pool.Spec.Models))
	return canonical, nil
}

// convertDecision converts a CRD Decision to config Decision
func (c *CRDConverter) convertDecision(decision v1alpha1.Decision) (config.Decision, error) {
	configDecision := config.Decision{
		Name:        decision.Name,
		Description: decision.Description,
		Priority:    int(decision.Priority),
		Rules: config.RuleCombination{
			Operator:   decision.Signals.Operator,
			Conditions: make([]config.RuleCondition, 0),
		},
		ModelRefs: make([]config.ModelRef, 0),
		Plugins:   make([]config.DecisionPlugin, 0),
	}

	// Convert signal conditions
	for _, condition := range decision.Signals.Conditions {
		configDecision.Rules.Conditions = append(configDecision.Rules.Conditions, config.RuleCondition{
			Type: condition.Type,
			Name: condition.Name,
		})
	}

	// Convert model refs
	for _, ms := range decision.ModelRefs {
		modelRef := config.ModelRef{
			Model:    ms.Model,
			LoRAName: ms.LoRAName,
			ModelReasoningControl: config.ModelReasoningControl{
				UseReasoning:         &ms.UseReasoning,
				ReasoningDescription: ms.ReasoningDescription,
				ReasoningEffort:      ms.ReasoningEffort,
			},
		}
		configDecision.ModelRefs = append(configDecision.ModelRefs, modelRef)
		break // Only take the first model
	}

	// Convert plugins
	for _, plugin := range decision.Plugins {
		var pluginConfig *config.StructuredPayload
		if plugin.Configuration != nil && plugin.Configuration.Raw != nil {
			// Validate plugin configuration format
			if err := validatePluginConfiguration(plugin.Type, plugin.Configuration.Raw); err != nil {
				return config.Decision{}, fmt.Errorf("invalid configuration for plugin %s in decision %s: %w", plugin.Type, decision.Name, err)
			}
			raw := make([]byte, len(plugin.Configuration.Raw))
			copy(raw, plugin.Configuration.Raw)
			pluginConfig = &config.StructuredPayload{Raw: raw}
		}
		configDecision.Plugins = append(configDecision.Plugins, config.DecisionPlugin{
			Type:          plugin.Type,
			Configuration: pluginConfig,
		})
	}

	return configDecision, nil
}

func convertRoutingModelCards(models []v1alpha1.ModelConfig) []config.RoutingModel {
	if len(models) == 0 {
		return nil
	}

	cards := make([]config.RoutingModel, 0, len(models))
	for _, model := range models {
		card := config.RoutingModel{
			Name: model.Name,
		}
		if len(model.LoRAs) > 0 {
			card.LoRAs = make([]config.LoRAAdapter, len(model.LoRAs))
			for i, lora := range model.LoRAs {
				card.LoRAs[i] = config.LoRAAdapter{
					Name:        lora.Name,
					Description: lora.Description,
				}
			}
		}
		cards = append(cards, card)
	}
	return cards
}

func convertSignals(signals v1alpha1.Signals) config.CanonicalSignals {
	converted := config.CanonicalSignals{
		Keywords:     make([]config.KeywordRule, 0, len(signals.Keywords)),
		Embeddings:   make([]config.EmbeddingRule, 0, len(signals.Embeddings)),
		Domains:      make([]config.Category, 0, len(signals.Domains)),
		Context:      make([]config.ContextRule, 0, len(signals.ContextRules)),
		Structure:    make([]config.StructureRule, 0, len(signals.Structure)),
		FactCheck:    make([]config.FactCheckRule, 0, len(signals.FactCheckRules)),
		Conversation: make([]config.ConversationRule, 0, len(signals.Conversation)),
	}

	for _, signal := range signals.Keywords {
		converted.Keywords = append(converted.Keywords, config.KeywordRule{
			Name:          signal.Name,
			Operator:      signal.Operator,
			Keywords:      signal.Keywords,
			CaseSensitive: signal.CaseSensitive,
		})
	}

	for _, signal := range signals.Embeddings {
		converted.Embeddings = append(converted.Embeddings, config.EmbeddingRule{
			Name:                      signal.Name,
			SimilarityThreshold:       signal.Threshold,
			Candidates:                signal.Candidates,
			AggregationMethodConfiged: config.AggregationMethod(signal.AggregationMethod),
		})
	}

	for _, domain := range signals.Domains {
		converted.Domains = append(converted.Domains, config.Category{
			CategoryMetadata: config.CategoryMetadata{
				Name:           domain.Name,
				Description:    domain.Description,
				MMLUCategories: []string{domain.Name},
			},
		})
	}

	for _, rule := range signals.ContextRules {
		converted.Context = append(converted.Context, config.ContextRule{
			Name:        rule.Name,
			MinTokens:   config.TokenCount(rule.MinTokens),
			MaxTokens:   config.TokenCount(rule.MaxTokens),
			Description: rule.Description,
		})
	}

	for _, rule := range signals.Structure {
		converted.Structure = append(converted.Structure, config.StructureRule{
			Name:        rule.Name,
			Description: rule.Description,
			Feature: config.StructureFeature{
				Type: rule.Feature.Type,
				Source: config.StructureSource{
					Type:          rule.Feature.Source.Type,
					Pattern:       rule.Feature.Source.Pattern,
					Keywords:      append([]string(nil), rule.Feature.Source.Keywords...),
					CaseSensitive: rule.Feature.Source.CaseSensitive,
					Sequences:     append([][]string(nil), rule.Feature.Source.Sequences...),
				},
			},
			Predicate: convertNumericPredicate(rule.Predicate),
		})
	}

	for _, rule := range signals.FactCheckRules {
		converted.FactCheck = append(converted.FactCheck, config.FactCheckRule{
			Name:        rule.Name,
			Description: rule.Description,
		})
	}

	for _, rule := range signals.Conversation {
		converted.Conversation = append(converted.Conversation, config.ConversationRule{
			Name:        rule.Name,
			Description: rule.Description,
			Feature: config.ConversationFeature{
				Type: rule.Feature.Type,
				Source: config.ConversationSource{
					Type: rule.Feature.Source.Type,
					Role: rule.Feature.Source.Role,
				},
			},
			Predicate: convertNumericPredicate(rule.Predicate),
		})
	}

	return converted
}

func convertNumericPredicate(predicate *v1alpha1.NumericPredicate) *config.NumericPredicate {
	if predicate == nil {
		return nil
	}
	return &config.NumericPredicate{
		GT:  predicate.GT,
		GTE: predicate.GTE,
		LT:  predicate.LT,
		LTE: predicate.LTE,
	}
}

func convertProviderMetadata(models []v1alpha1.ModelConfig) []config.CanonicalProviderModel {
	converted := make([]config.CanonicalProviderModel, 0, len(models))
	for _, model := range models {
		providerModel := config.CanonicalProviderModel{
			Name:            model.Name,
			ReasoningFamily: model.ReasoningFamily,
		}
		if model.Pricing == nil && providerModel.ReasoningFamily == "" {
			continue
		}
		if model.Pricing != nil {
			providerModel.Pricing = config.ModelPricing{
				PromptPer1M:     model.Pricing.InputTokenPrice * 1000000,
				CompletionPer1M: model.Pricing.OutputTokenPrice * 1000000,
			}
		}
		converted = append(converted, providerModel)
	}
	return converted
}

func mergeProviderMetadata(
	base []config.CanonicalProviderModel,
	overlay []config.CanonicalProviderModel,
) []config.CanonicalProviderModel {
	if len(overlay) == 0 {
		return base
	}

	merged := make([]config.CanonicalProviderModel, 0, len(base)+len(overlay))
	index := make(map[string]int, len(base)+len(overlay))
	for _, providerModel := range base {
		index[providerModel.Name] = len(merged)
		merged = append(merged, providerModel)
	}
	for _, providerModel := range overlay {
		existingIndex, ok := index[providerModel.Name]
		if !ok {
			index[providerModel.Name] = len(merged)
			merged = append(merged, providerModel)
			continue
		}
		merged[existingIndex] = mergeCanonicalProviderMetadata(
			merged[existingIndex],
			providerModel,
		)
	}
	return merged
}

func mergeCanonicalProviderMetadata(
	existing config.CanonicalProviderModel,
	overlay config.CanonicalProviderModel,
) config.CanonicalProviderModel {
	if overlay.Pricing != (config.ModelPricing{}) {
		existing.Pricing = overlay.Pricing
	}
	if overlay.ReasoningFamily != "" {
		existing.ReasoningFamily = overlay.ReasoningFamily
	}
	if overlay.ProviderModelID != "" {
		existing.ProviderModelID = overlay.ProviderModelID
	}
	if overlay.APIFormat != "" {
		existing.APIFormat = overlay.APIFormat
	}
	if len(overlay.ExternalModelIDs) > 0 {
		existing.ExternalModelIDs = overlay.ExternalModelIDs
	}
	return existing
}

func cloneCanonicalConfig(base *config.CanonicalConfig) (*config.CanonicalConfig, error) {
	if base == nil {
		return &config.CanonicalConfig{Version: "v0.3"}, nil
	}

	data, err := yamlv3.Marshal(base)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal canonical base config: %w", err)
	}
	var cloned config.CanonicalConfig
	if err := yamlv3.Unmarshal(data, &cloned); err != nil {
		return nil, fmt.Errorf("failed to clone canonical base config: %w", err)
	}
	return &cloned, nil
}

// validatePluginConfiguration validates that plugin configuration matches the expected schema
func validatePluginConfiguration(pluginType string, rawConfig []byte) error {
	if len(rawConfig) == 0 {
		return nil // Empty configuration is allowed
	}
	validator, ok := pluginConfigurationValidators[pluginType]
	if !ok {
		// Unknown plugin types are passed through without schema validation.
		// This allows extensibility — only well-known types are validated.
		return nil
	}
	return validator(rawConfig)
}

var pluginConfigurationValidators = map[string]func([]byte) error{
	"semantic-cache":  validateSemanticCachePluginConfig,
	"system_prompt":   validateSystemPromptPluginConfig,
	"header_mutation": validateHeaderMutationPluginConfig,
	"router_replay":   validateRouterReplayPluginConfig,
}

func decodePluginConfiguration(rawConfig []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(rawConfig))
	decoder.DisallowUnknownFields()
	return decoder.Decode(target)
}

func validateSemanticCachePluginConfig(rawConfig []byte) error {
	var cfg config.SemanticCachePluginConfig
	if err := decodePluginConfiguration(rawConfig, &cfg); err != nil {
		return fmt.Errorf("failed to unmarshal semantic-cache config: %w", err)
	}
	return nil
}

func validateSystemPromptPluginConfig(rawConfig []byte) error {
	var cfg config.SystemPromptPluginConfig
	if err := decodePluginConfiguration(rawConfig, &cfg); err != nil {
		return fmt.Errorf("failed to unmarshal system_prompt config: %w", err)
	}
	if cfg.Mode != "" && cfg.Mode != "replace" && cfg.Mode != "insert" {
		return fmt.Errorf("system_prompt mode must be 'replace' or 'insert', got: %s", cfg.Mode)
	}
	return nil
}

func validateHeaderMutationPluginConfig(rawConfig []byte) error {
	var cfg config.HeaderMutationPluginConfig
	if err := decodePluginConfiguration(rawConfig, &cfg); err != nil {
		return fmt.Errorf("failed to unmarshal header_mutation config: %w", err)
	}
	if len(cfg.Add) == 0 && len(cfg.Update) == 0 && len(cfg.Delete) == 0 {
		return fmt.Errorf("header_mutation plugin must specify at least one of: add, update, delete")
	}
	if err := validateHeaderMutationEntries("add", cfg.Add); err != nil {
		return err
	}
	return validateHeaderMutationEntries("update", cfg.Update)
}

func validateHeaderMutationEntries(operation string, headers []config.HeaderPair) error {
	for _, header := range headers {
		if header.Name == "" {
			return fmt.Errorf("header_mutation %s: header name cannot be empty", operation)
		}
	}
	return nil
}

func validateRouterReplayPluginConfig(rawConfig []byte) error {
	var cfg config.RouterReplayPluginConfig
	if err := decodePluginConfiguration(rawConfig, &cfg); err != nil {
		return fmt.Errorf("failed to unmarshal router_replay config: %w", err)
	}
	if cfg.MaxRecords < 0 {
		return fmt.Errorf("router_replay max_records cannot be negative")
	}
	if cfg.MaxBodyBytes < 0 {
		return fmt.Errorf("router_replay max_body_bytes cannot be negative")
	}
	return nil
}
