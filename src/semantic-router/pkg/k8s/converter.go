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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apis/vllm.ai/v1alpha1"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// CRDConverter converts Kubernetes CRDs to internal configuration structures
type CRDConverter struct{}

// NewCRDConverter creates a new CRD converter
func NewCRDConverter() *CRDConverter {
	return &CRDConverter{}
}

// ConvertIntelligentPool converts IntelligentPool CRD to BackendModels config
func (c *CRDConverter) ConvertIntelligentPool(pool *v1alpha1.IntelligentPool) (*config.BackendModels, error) {
	if pool == nil {
		return nil, fmt.Errorf("pool cannot be nil")
	}

	backendModels := &config.BackendModels{
		DefaultModel: pool.Spec.DefaultModel,
		ModelConfig:  make(map[string]config.ModelParams),
		// VLLMEndpoints is not managed by CRD, will be loaded from static config
		VLLMEndpoints: nil,
	}

	// Convert models
	for _, model := range pool.Spec.Models {
		modelParams := config.ModelParams{
			ReasoningFamily: model.ReasoningFamily,
		}

		// Convert pricing
		if model.Pricing != nil {
			modelParams.Pricing = config.ModelPricing{
				PromptPer1M:     model.Pricing.InputTokenPrice * 1000000,  // Convert per-token to per-1M
				CompletionPer1M: model.Pricing.OutputTokenPrice * 1000000, // Convert per-token to per-1M
			}
		}

		// Convert LoRAs
		if len(model.LoRAs) > 0 {
			modelParams.LoRAs = make([]config.LoRAAdapter, len(model.LoRAs))
			for i, lora := range model.LoRAs {
				modelParams.LoRAs[i] = config.LoRAAdapter{
					Name:        lora.Name,
					Description: lora.Description,
				}
			}
		}

		backendModels.ModelConfig[model.Name] = modelParams
	}

	return backendModels, nil
}

// ConvertIntelligentRoute converts IntelligentRoute CRD to IntelligentRouting config
func (c *CRDConverter) ConvertIntelligentRoute(route *v1alpha1.IntelligentRoute) (*config.IntelligentRouting, error) {
	if route == nil {
		return nil, fmt.Errorf("route cannot be nil")
	}

	intelligentRouting := &config.IntelligentRouting{
		Signals: config.Signals{
			KeywordRules:   make([]config.KeywordRule, 0),
			EmbeddingRules: make([]config.EmbeddingRule, 0),
			Categories:     make([]config.Category, 0),
		},
		Decisions: make([]config.Decision, 0),
		Strategy:  "priority", // Always use priority strategy
	}

	// Convert keyword signals
	for _, signal := range route.Spec.Signals.Keywords {
		intelligentRouting.KeywordRules = append(intelligentRouting.KeywordRules, config.KeywordRule{
			Name:          signal.Name,
			Operator:      signal.Operator,
			Keywords:      signal.Keywords,
			CaseSensitive: signal.CaseSensitive,
		})
	}

	// Convert embedding signals
	for _, signal := range route.Spec.Signals.Embeddings {
		embeddingRule := config.EmbeddingRule{
			Name:                      signal.Name,
			SimilarityThreshold:       signal.Threshold,
			Candidates:                signal.Candidates,
			AggregationMethodConfiged: config.AggregationMethod(signal.AggregationMethod),
		}
		intelligentRouting.EmbeddingRules = append(intelligentRouting.EmbeddingRules, embeddingRule)
	}

	// Convert domain signals to categories (only metadata)
	// Domains is now an array of DomainSignal with name and description
	for _, domain := range route.Spec.Signals.Domains {
		category := config.Category{
			CategoryMetadata: config.CategoryMetadata{
				Name:           domain.Name,
				Description:    domain.Description,
				MMLUCategories: []string{domain.Name}, // Single MMLU category
			},
		}
		intelligentRouting.Categories = append(intelligentRouting.Categories, category)
	}

	// Convert decisions
	for _, decision := range route.Spec.Decisions {
		configDecision, err := c.convertDecision(decision)
		if err != nil {
			return nil, fmt.Errorf("failed to convert decision %s: %w", decision.Name, err)
		}
		intelligentRouting.Decisions = append(intelligentRouting.Decisions, configDecision)
	}

	return intelligentRouting, nil
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
		var pluginConfig any
		if plugin.Configuration != nil && plugin.Configuration.Raw != nil {
			// Validate plugin configuration format
			if err := validatePluginConfiguration(plugin.Type, plugin.Configuration.Raw); err != nil {
				return config.Decision{}, fmt.Errorf("invalid configuration for plugin %s in decision %s: %w", plugin.Type, decision.Name, err)
			}
			// Store the raw bytes from RawExtension
			// The Get*Config methods will unmarshal this to the appropriate type
			pluginConfig = plugin.Configuration.Raw
		}
		configDecision.Plugins = append(configDecision.Plugins, config.DecisionPlugin{
			Type:          plugin.Type,
			Configuration: pluginConfig,
		})
	}

	return configDecision, nil
}

// validatePluginConfiguration validates that plugin configuration matches the expected schema
func validatePluginConfiguration(pluginType string, rawConfig []byte) error {
	if len(rawConfig) == 0 {
		return nil // Empty configuration is allowed
	}

	switch pluginType {
	case "semantic-cache":
		var cfg config.SemanticCachePluginConfig
		decoder := json.NewDecoder(bytes.NewReader(rawConfig))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&cfg); err != nil {
			return fmt.Errorf("failed to unmarshal semantic-cache config: %w", err)
		}

	case "jailbreak":
		var cfg config.JailbreakPluginConfig
		decoder := json.NewDecoder(bytes.NewReader(rawConfig))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&cfg); err != nil {
			return fmt.Errorf("failed to unmarshal jailbreak config: %w", err)
		}

	case "pii":
		var cfg config.PIIPluginConfig
		decoder := json.NewDecoder(bytes.NewReader(rawConfig))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&cfg); err != nil {
			return fmt.Errorf("failed to unmarshal pii config: %w", err)
		}

	case "system_prompt":
		var cfg config.SystemPromptPluginConfig
		decoder := json.NewDecoder(bytes.NewReader(rawConfig))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&cfg); err != nil {
			return fmt.Errorf("failed to unmarshal system_prompt config: %w", err)
		}
		// Validate mode if present
		if cfg.Mode != "" && cfg.Mode != "replace" && cfg.Mode != "insert" {
			return fmt.Errorf("system_prompt mode must be 'replace' or 'insert', got: %s", cfg.Mode)
		}

	case "header_mutation":
		var cfg config.HeaderMutationPluginConfig
		decoder := json.NewDecoder(bytes.NewReader(rawConfig))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&cfg); err != nil {
			return fmt.Errorf("failed to unmarshal header_mutation config: %w", err)
		}
		// Validate that at least one operation is specified
		if len(cfg.Add) == 0 && len(cfg.Update) == 0 && len(cfg.Delete) == 0 {
			return fmt.Errorf("header_mutation plugin must specify at least one of: add, update, delete")
		}
		// Validate header pairs
		for _, h := range cfg.Add {
			if h.Name == "" {
				return fmt.Errorf("header_mutation add: header name cannot be empty")
			}
		}
		for _, h := range cfg.Update {
			if h.Name == "" {
				return fmt.Errorf("header_mutation update: header name cannot be empty")
			}
		}

	case "router_replay":
		var cfg config.RouterReplayPluginConfig
		decoder := json.NewDecoder(bytes.NewReader(rawConfig))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&cfg); err != nil {
			return fmt.Errorf("failed to unmarshal router_replay config: %w", err)
		}
		if cfg.MaxRecords < 0 {
			return fmt.Errorf("router_replay max_records cannot be negative")
		}
		if cfg.MaxBodyBytes < 0 {
			return fmt.Errorf("router_replay max_body_bytes cannot be negative")
		}

	default:
		return fmt.Errorf("unknown plugin type: %s", pluginType)
	}

	return nil
}
