package config

import (
	"fmt"
	"reflect"

	"gopkg.in/yaml.v2"
)

// ParseRoutingYAMLBytes parses a routing fragment: a document holding only the
// DSL-owned surface, its recipes and entrypoints and the providers.models[]
// entries that state model facts. It intentionally skips provider and global
// cross-reference checks so DSL compile/decompile flows can round-trip
// fragments.
func ParseRoutingYAMLBytes(data []byte) (*RouterConfig, error) {
	return ParseRoutingYAMLBytesWithLimits(data, DecisionRuleLimits{})
}

// ParseRoutingYAMLBytesWithLimits applies an enclosing document's limits to a
// routing fragment; standalone fragments use ParseRoutingYAMLBytes defaults.
func ParseRoutingYAMLBytesWithLimits(data []byte, limits DecisionRuleLimits) (*RouterConfig, error) {
	raw, err := parseRawConfigMap(data)
	if err != nil {
		return nil, fmt.Errorf("failed to parse routing fragment: %w", err)
	}
	depth, nodes, err := limits.Effective()
	if err != nil {
		return nil, err
	}
	budgetDocument := map[string]interface{}{
		"recipes": raw["recipes"],
		"global": map[string]interface{}{"router": map[string]interface{}{
			"decision_rule_limits": map[string]interface{}{"max_depth": depth, "max_nodes": nodes},
		}},
	}
	if _, err := validateRawDecisionRuleLimits(budgetDocument); err != nil {
		return nil, err
	}

	data, formWarnings, err := applyConfigForms(data, configForms)
	if err != nil {
		return nil, err
	}
	raw, err = parseRawConfigMap(data)
	if err != nil {
		return nil, err
	}

	// Fragments share the canonical routing vocabulary while omitting only
	// infrastructure cross-reference validation.
	if err := validateKnownFields(raw, reflect.TypeOf(CanonicalConfig{})); err != nil {
		return nil, err
	}

	doc := &CanonicalConfig{}
	if err := yaml.Unmarshal(data, doc); err != nil {
		return nil, fmt.Errorf("failed to parse routing fragment: %w", err)
	}
	if err := validateCanonicalFallback(doc); err != nil {
		return nil, err
	}
	cfg := DefaultGlobalConfig()
	cfg.DecisionRuleLimits = limits
	cfg.RoutingFragmentOnly = true
	cfg.FormWarnings = formWarnings
	routing := &CanonicalConfig{Recipes: doc.Recipes, Entrypoints: doc.Entrypoints}
	if err := applyCanonicalRecipeState(&cfg, routing); err != nil {
		return nil, err
	}
	cfg.ModelConfig = make(map[string]ModelParams)
	for _, model := range doc.Providers.Models {
		if !model.ModelFacts.declared() {
			continue
		}
		params := ModelParams{
			ParamSize:         model.ParamSize,
			ContextWindowSize: model.ContextWindowSize,
			MaxOutputTokens:   model.MaxOutputTokens,
			Description:       model.Description,
			Capabilities:      append([]string(nil), model.Capabilities...),
			LoRAs:             copyLoRAAdapters(model.LoRAs),
			Tags:              append([]string(nil), model.Tags...),
		}
		if model.Modalities != nil {
			params.Modalities = cloneModalities(*model.Modalities)
		}
		cfg.ModelConfig[model.Name] = params
	}

	if cfg.VectorStore != nil {
		cfg.VectorStore.ApplyDefaults()
	}

	if err := validateDomainContracts(&cfg); err != nil {
		return nil, err
	}
	if err := validateProjectionContracts(&cfg); err != nil {
		return nil, err
	}
	if err := validateDecisionContracts(&cfg); err != nil {
		return nil, err
	}
	if err := validateApplicationSignalContracts(&cfg); err != nil {
		return nil, err
	}
	if err := visitRoutingProfileConfigs(&cfg, validateDecisionOutputContracts); err != nil {
		return nil, err
	}

	return &cfg, nil
}
