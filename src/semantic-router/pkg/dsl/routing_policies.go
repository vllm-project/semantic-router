package dsl

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func parseRoutingPolicy[T any](value Value) (*T, error) {
	object, ok := value.(ObjectValue)
	if !ok {
		return nil, fmt.Errorf("must be an object")
	}
	payload, err := json.Marshal(fieldsToMap(object.Fields))
	if err != nil {
		return nil, err
	}
	var policy T
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&policy); err != nil {
		return nil, err
	}
	return &policy, nil
}

func applyRoutingPolicies(prog *Program, fields map[string]Value) error {
	if value, exists := fields["candidate_requirements"]; exists {
		policy, err := parseRoutingPolicy[config.CandidateRequirements](value)
		if err != nil {
			return fmt.Errorf("candidate_requirements: %w", err)
		}
		if err := policy.Validate(); err != nil {
			return err
		}
		prog.CandidateRequirements = policy
	}
	if value, exists := fields["data_policy"]; exists {
		policy, err := parseRoutingPolicy[config.RoutingDataPolicy](value)
		if err != nil {
			return fmt.Errorf("data_policy: %w", err)
		}
		prog.DataPolicy = policy
	}
	return nil
}

func (d *decompiler) decompileRoutingPolicies() {
	if policy := d.cfg.CandidateRequirements; policy != nil {
		fields := map[string]interface{}{}
		if policy.Capabilities != "" {
			fields["capabilities"] = policy.Capabilities
		}
		if policy.Context != "" {
			fields["context"] = policy.Context
		}
		d.write("  candidate_requirements: %s\n", formatPluginConfigValue(fields))
	}
	if policy := d.cfg.DataPolicy; policy != nil {
		fields := map[string]interface{}{}
		if policy.Replay != nil {
			fields["replay"] = *policy.Replay
		}
		d.write("  data_policy: %s\n", formatPluginConfigValue(fields))
	}
}
