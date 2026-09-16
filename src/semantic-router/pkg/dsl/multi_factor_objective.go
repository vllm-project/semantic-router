package dsl

import (
	"bytes"
	"fmt"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Decode through the canonical type so new selector fields cannot silently
// disappear at the DSL boundary. Only shared algorithm fields are removed.
func decodeMultiFactorFields(fields map[string]Value) (*config.MultiFactorSelectionConfig, error) {
	values := fieldsToMap(fields)
	delete(values, "minimum_candidates")
	delete(values, "on_error")
	payload, err := yaml.Marshal(values)
	if err != nil {
		return nil, err
	}
	cfg := &config.MultiFactorSelectionConfig{}
	decoder := yaml.NewDecoder(bytes.NewReader(payload))
	decoder.KnownFields(true)
	if err := decoder.Decode(cfg); err != nil {
		return nil, fmt.Errorf("multi_factor: %w", err)
	}
	return cfg, nil
}

func multiFactorObjectiveValue(objective *config.MultiFactorObjectiveConfig) ObjectValue {
	fields := map[string]Value{}
	setStringValue(fields, "strategy", objective.Strategy)
	if objective.Priorities != nil {
		priorities := make([]Value, 0, len(objective.Priorities))
		for _, priority := range objective.Priorities {
			values := map[string]Value{"factor": StringValue{V: priority.Factor}}
			setFloatValue(values, "tolerance", priority.Tolerance)
			priorities = append(priorities, ObjectValue{Fields: values})
		}
		fields["priorities"] = ArrayValue{Items: priorities}
	}
	return ObjectValue{Fields: fields}
}
