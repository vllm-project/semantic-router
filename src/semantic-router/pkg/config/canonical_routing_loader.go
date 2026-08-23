package config

import (
	"fmt"
	"sort"

	"gopkg.in/yaml.v2"
)

type routingFragmentDocument struct {
	Document CanonicalRouting `yaml:"document"`
}

const routingFragmentScope RecipeName = "authoring-document"

// ParseRoutingYAMLBytes parses a model-free Recipe document fragment. The
// fragment is an authoring value and cannot be served until a Recipe and an
// Entrypoint bind it to immutable Models.
func ParseRoutingYAMLBytes(data []byte) (*RouterConfig, error) {
	raw, err := parseRawConfigMap(data)
	if err != nil {
		return nil, fmt.Errorf("failed to parse routing fragment: %w", err)
	}
	if _, found := raw["document"]; !found || len(raw) != 1 {
		fields := make([]string, 0, len(raw))
		for field := range raw {
			fields = append(fields, field)
		}
		sort.Strings(fields)
		return nil, fmt.Errorf("routing fragment must contain exactly one top-level document field, got %v", fields)
	}
	for _, validate := range []func(map[string]interface{}) error{
		rejectRemovedStructureFields,
		rejectRemovedTaxonomyLegacyFields,
		rejectRemovedDecisionToolFields,
		rejectDecisionLearningFields,
		rejectUnsupportedDecisionAdaptationFields,
	} {
		if rejectErr := validate(raw); rejectErr != nil {
			return nil, rejectErr
		}
	}

	doc := &routingFragmentDocument{}
	if err := yaml.UnmarshalStrict(data, doc); err != nil {
		return nil, fmt.Errorf("failed to parse routing fragment: %w", err)
	}

	cfg := DefaultGlobalConfig()
	// A routing fragment is one isolated Recipe document, not a root runtime
	// config. Marking the view explicitly lets shared validators read its flat
	// fields without reviving the removed implicit-default Recipe path.
	cfg.RoutingScope = routingFragmentScope
	cfg.Decisions = copyDecisions(doc.Document.Decisions)
	ensureModelRefDefaults(cfg.Decisions)
	cfg.Signals = normalizeSignals(doc.Document.Signals, cfg.Decisions)
	cfg.Projections = normalizeProjections(doc.Document.Projections)
	cfg.Strategy = doc.Document.Strategy
	cfg.ModelConfig = make(map[string]ModelParams)

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
	if err := validateModalityContracts(&cfg); err != nil {
		return nil, err
	}

	return &cfg, nil
}
