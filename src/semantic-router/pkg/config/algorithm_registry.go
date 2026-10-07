package config

import (
	"fmt"
	"slices"
	"strings"
)

// AlgorithmOptions configure an algorithm type whose payload is a *P.
type AlgorithmOptions[P any] struct {
	// Strict rejects block fields P does not declare.
	Strict bool
	// Defaults, when set, fills a decoded payload's unset fields.
	Defaults func(payload *P)
	// Validate, when set, checks a decoded payload after its defaults.
	Validate func(decision string, payload *P) error
}

// NewDecisionAlgorithmType is the algorithm type of catalog whose block
// decodes into a *P. The block is the one under the type's name in a
// decision's algorithm; the type runs as a selector.
func NewDecisionAlgorithmType[P any](catalog AlgorithmCatalogEntry, opts AlgorithmOptions[P]) DecisionAlgorithmType {
	catalog.ConfigField, catalog.Execution = catalog.Type, AlgorithmExecutionSelector
	typ := catalog.Type
	spec := DecisionAlgorithmType{
		Catalog:      catalog,
		IsConfigured: func(algorithm *AlgorithmConfig) bool { return algorithm.Extensions[typ] != nil },
		NewPayload:   func() interface{} { return new(P) },
		Strict:       opts.Strict,
	}
	if opts.Defaults != nil {
		spec.Defaults = func(payload interface{}) { opts.Defaults(payload.(*P)) }
	}
	if opts.Validate != nil {
		spec.Validate = func(decision string, payload interface{}) error { return opts.Validate(decision, payload.(*P)) }
	}
	return spec
}

// retiredAlgorithmTypes are types the Router refuses with a pointer to what
// replaced them, so no Router build may register them again.
var retiredAlgorithmTypes = []string{
	"session_aware", "elo", "rl_driven", "gmtrouter", "bandit", "personalization", "thompson", "router_r1",
}

// RegisterDecisionAlgorithm adds a decision algorithm type, typically from
// an init function. Its block lives in AlgorithmConfig.Extensions under its
// name, so it needs a payload; build the spec with NewDecisionAlgorithmType.
func RegisterDecisionAlgorithm(spec DecisionAlgorithmType) error {
	if spec.NewPayload == nil {
		return fmt.Errorf("decision algorithm %q: a payload type is required", spec.Catalog.Type)
	}
	if slices.Contains(retiredAlgorithmTypes, spec.Catalog.Type) {
		return fmt.Errorf("decision algorithm %q is a type the Router retired", spec.Catalog.Type)
	}
	if spec.Catalog.ConfigField != spec.Catalog.Type || spec.Catalog.Execution != AlgorithmExecutionSelector {
		return fmt.Errorf("decision algorithm %q: its block is named after its type and it runs as a selector", spec.Catalog.Type)
	}
	return decisionAlgorithms.Register(spec.Catalog.Type, spec)
}

// DecisionAlgorithmPayloadSamples returns a fresh payload for every algorithm
// type registered outside the Router, by type: the schema of its block.
func DecisionAlgorithmPayloadSamples() map[string]interface{} {
	samples := map[string]interface{}{}
	for _, entry := range decisionAlgorithms.Entries() {
		if entry.Spec.NewPayload != nil {
			samples[entry.Type] = entry.Spec.NewPayload()
		}
	}
	return samples
}

// extensionAlgorithmTypes are the registered algorithm types whose blocks
// live in AlgorithmConfig.Extensions.
func extensionAlgorithmTypes() []string {
	var types []string
	for _, entry := range decisionAlgorithms.Entries() {
		if entry.Spec.NewPayload != nil {
			types = append(types, entry.Type)
		}
	}
	return types
}

// DecodeDecisionAlgorithm decodes the block of algorithm's type when it is a
// type registered outside the Router, applies its defaults and validates it.
// It reports false for a built-in type.
func DecodeDecisionAlgorithm(decision string, algorithm *AlgorithmConfig) (interface{}, bool, error) {
	if algorithm == nil {
		return nil, false, nil
	}
	typ := strings.TrimSpace(algorithm.Type)
	spec, ok := decisionAlgorithms.Lookup(typ)
	if !ok || spec.NewPayload == nil {
		return nil, false, nil
	}
	payload := spec.NewPayload()
	if block := algorithm.Extensions[typ]; block != nil {
		decode := block.DecodeInto
		if spec.Strict {
			decode = block.DecodeIntoStrict
		}
		if err := decode(payload); err != nil {
			return nil, true, fmt.Errorf("decision '%s': algorithm.%s: %w", decision, typ, err)
		}
	}
	if spec.Defaults != nil {
		spec.Defaults(payload)
	}
	if spec.Validate != nil {
		if err := spec.Validate(decision, payload); err != nil {
			return nil, true, fmt.Errorf("decision '%s': algorithm.%s: %w", decision, typ, err)
		}
	}
	return payload, true, nil
}
