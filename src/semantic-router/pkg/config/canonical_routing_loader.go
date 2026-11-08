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
	data, formWarnings, err := applyConfigForms(data, configForms)
	if err != nil {
		return nil, err
	}
	raw, err := parseRawConfigMap(data)
	if err != nil {
		return nil, fmt.Errorf("failed to parse routing fragment: %w", err)
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
		cfg.ModelConfig[model.Name] = ModelParams{
			ParamSize:         model.ParamSize,
			ContextWindowSize: model.ContextWindowSize,
			MaxOutputTokens:   model.MaxOutputTokens,
			Description:       model.Description,
			Capabilities:      append([]string(nil), model.Capabilities...),
			LoRAs:             copyLoRAAdapters(model.LoRAs),
			Tags:              append([]string(nil), model.Tags...),
			Modality:          model.Modality,
		}
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
	if err := validateRoutingModalityContracts(&cfg); err != nil {
		return nil, err
	}

	return &cfg, nil
}
