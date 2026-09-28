package dsl

import (
	"fmt"

	"gopkg.in/yaml.v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func prototypeScoringFromSignal(s *SignalDecl) (*config.PrototypeScoringConfig, error) {
	value, present := s.Fields["prototype_scoring"]
	if !present {
		return nil, nil
	}
	object, ok := value.(ObjectValue)
	if !ok {
		return nil, fmt.Errorf("prototype_scoring must be an object")
	}
	raw, err := yaml.Marshal(fieldsToMap(object.Fields))
	if err != nil {
		return nil, err
	}
	var result config.PrototypeScoringConfig
	if err := yaml.UnmarshalStrict(raw, &result); err != nil {
		return nil, fmt.Errorf("prototype_scoring: %w", err)
	}
	return &result, nil
}

// Preserve the authored override, including an empty object and explicit false.
// Expanding defaults here would change whether family settings are inherited.
func prototypeScoringFields(cfg *config.PrototypeScoringConfig) map[string]Value {
	fields := make(map[string]Value)
	if cfg.Enabled != nil {
		fields["enabled"] = BoolValue{V: *cfg.Enabled}
	}
	setFloatValue(fields, "cluster_similarity_threshold", float64(cfg.ClusterSimilarityThreshold))
	setIntValue(fields, "max_prototypes", cfg.MaxPrototypes)
	setFloatValue(fields, "best_weight", float64(cfg.BestWeight))
	setIntValue(fields, "top_m", cfg.TopM)
	setFloatValue(fields, "margin_threshold", float64(cfg.MarginThreshold))
	return fields
}
