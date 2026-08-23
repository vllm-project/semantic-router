package config

import (
	"fmt"

	"gopkg.in/yaml.v2"
)

// mergeCanonicalGlobalDefaults applies a sparse canonical global override at
// YAML-field granularity. Unmarshalling a nested YAML object directly into a
// populated Go struct resets sibling fields in that object, which would make
// adding one service silently erase every other service default.
func mergeCanonicalGlobalDefaults(
	defaults CanonicalGlobal,
	override *CanonicalGlobal,
	rawOverride *StructuredPayload,
) (CanonicalGlobal, error) {
	defaultValues, err := canonicalGlobalMap(defaults)
	if err != nil {
		return CanonicalGlobal{}, fmt.Errorf("encode canonical global defaults: %w", err)
	}

	var overrideValues map[string]interface{}
	if rawOverride != nil {
		overrideValues, err = rawOverride.AsStringMap()
	} else {
		overrideValues, err = canonicalGlobalMap(override)
	}
	if err != nil {
		return CanonicalGlobal{}, fmt.Errorf("encode canonical global override: %w", err)
	}

	merged := mergeCanonicalMaps(defaultValues, overrideValues)
	encoded, err := yaml.Marshal(merged)
	if err != nil {
		return CanonicalGlobal{}, fmt.Errorf("marshal merged canonical global: %w", err)
	}
	var result CanonicalGlobal
	if err := yaml.Unmarshal(encoded, &result); err != nil {
		return CanonicalGlobal{}, fmt.Errorf("decode merged canonical global: %w", err)
	}
	return result, nil
}

func canonicalGlobalMap(value interface{}) (map[string]interface{}, error) {
	encoded, err := yaml.Marshal(value)
	if err != nil {
		return nil, err
	}
	var decoded interface{}
	if err := yaml.Unmarshal(encoded, &decoded); err != nil {
		return nil, err
	}
	normalized, ok := normalizeStructuredPayloadValue(decoded).(map[string]interface{})
	if !ok && decoded != nil {
		return nil, fmt.Errorf("canonical global must encode as an object")
	}
	if normalized == nil {
		normalized = make(map[string]interface{})
	}
	return normalized, nil
}

func mergeCanonicalMaps(base, override map[string]interface{}) map[string]interface{} {
	result := make(map[string]interface{}, len(base)+len(override))
	for key, value := range base {
		result[key] = cloneCanonicalValue(value)
	}
	for key, value := range override {
		baseMap, baseIsMap := result[key].(map[string]interface{})
		overrideMap, overrideIsMap := value.(map[string]interface{})
		if baseIsMap && overrideIsMap {
			result[key] = mergeCanonicalMaps(baseMap, overrideMap)
			continue
		}
		result[key] = cloneCanonicalValue(value)
	}
	return result
}

func cloneCanonicalValue(value interface{}) interface{} {
	switch typed := value.(type) {
	case map[string]interface{}:
		return mergeCanonicalMaps(nil, typed)
	case []interface{}:
		result := make([]interface{}, len(typed))
		for index := range typed {
			result[index] = cloneCanonicalValue(typed[index])
		}
		return result
	default:
		return value
	}
}
