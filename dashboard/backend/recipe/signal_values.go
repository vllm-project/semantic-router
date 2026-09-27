package recipe

import (
	"errors"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

// SignalValueBounds use a signal's native units, not an assumed probability.
type SignalValueBounds struct {
	GTE *float64 `json:"gte,omitempty"`
	LTE *float64 `json:"lte,omitempty"`
}

var signalValueKeyPattern = regexp.MustCompile(`^[a-z][a-z0-9_]*:\S(?:.*\S)?$`)

// A YAML node preserves omitted versus empty versus null, and rejects coercion
// of booleans/strings/null into numeric bounds before typed materialization.
func parseSignalValueExpectations(node yaml.Node) (map[string]SignalValueBounds, error) {
	if node.Kind == 0 {
		return nil, nil
	}
	if node.Kind != yaml.MappingNode {
		return nil, errors.New("expected_signal_values must be a mapping")
	}
	values := make(map[string]SignalValueBounds, len(node.Content)/2)
	for i := 0; i < len(node.Content); i += 2 {
		key := node.Content[i]
		if key.Tag != "!!str" || !signalValueKeyPattern.MatchString(key.Value) {
			return nil, errors.New("expected_signal_values requires a runtime type:name key")
		}
		if _, exists := values[key.Value]; exists {
			return nil, fmt.Errorf("expected_signal_values duplicate key %q", key.Value)
		}
		bounds, err := parseSignalValueBounds(node.Content[i+1])
		if err != nil {
			return nil, fmt.Errorf("expected_signal_values.%s: %w", key.Value, err)
		}
		values[key.Value] = bounds
	}
	return values, nil
}

func parseSignalValueBounds(node *yaml.Node) (SignalValueBounds, error) {
	result := SignalValueBounds{}
	if node.Kind != yaml.MappingNode || len(node.Content) == 0 {
		return result, errors.New("requires gte and/or lte")
	}
	for i := 0; i < len(node.Content); i += 2 {
		key, value := node.Content[i].Value, node.Content[i+1]
		var target **float64
		switch key {
		case "gte":
			target = &result.GTE
		case "lte":
			target = &result.LTE
		default:
			return result, fmt.Errorf("unknown bound %q", key)
		}
		if *target != nil || (value.Tag != "!!int" && value.Tag != "!!float") {
			return result, fmt.Errorf("%s must be a single finite number, not bool", key)
		}
		var number float64
		if err := value.Decode(&number); err != nil {
			return result, fmt.Errorf("decode %s: %w", key, err)
		}
		if math.IsNaN(number) || math.IsInf(number, 0) {
			return result, fmt.Errorf("%s must be finite", key)
		}
		*target = &number
	}
	if result.GTE != nil && result.LTE != nil && *result.GTE > *result.LTE {
		return result, errors.New("gte must not exceed lte")
	}
	return result, nil
}

func cloneSignalValueBounds(source map[string]SignalValueBounds) map[string]SignalValueBounds {
	if source == nil {
		return nil
	}
	result := make(map[string]SignalValueBounds, len(source))
	for key, bound := range source {
		if bound.GTE != nil {
			value := *bound.GTE
			bound.GTE = &value
		}
		if bound.LTE != nil {
			value := *bound.LTE
			bound.LTE = &value
		}
		result[key] = bound
	}
	return result
}

func compareSignalValues(expected map[string]SignalValueBounds, actual map[string]any, signalErrors map[string]string) (bool, []string) {
	failures := []string{}
	keys := make([]string, 0, len(expected))
	for key := range expected {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		bound := expected[key]
		for errorKey := range signalErrors {
			related := key == errorKey || strings.HasPrefix(key, errorKey+":")
			if strings.HasPrefix(key, "complexity:") {
				related = complexityValueErrorRelated(key, errorKey)
			}
			if related {
				failures = append(failures, fmt.Sprintf("signal value %s: signal error %s", key, errorKey))
			}
		}
		raw, exists := actual[key]
		if !exists {
			failures = append(failures, fmt.Sprintf("signal value %s: missing", key))
			continue
		}
		value, numeric := raw.(float64)
		if !numeric || math.IsNaN(value) || math.IsInf(value, 0) {
			failures = append(failures, fmt.Sprintf("signal value %s: expected finite number, got %v", key, raw))
			continue
		}
		if bound.GTE != nil && value < *bound.GTE {
			failures = append(failures, fmt.Sprintf("signal value %s: %g is below gte %g", key, value, *bound.GTE))
		}
		if bound.LTE != nil && value > *bound.LTE {
			failures = append(failures, fmt.Sprintf("signal value %s: %g exceeds lte %g", key, value, *bound.LTE))
		}
	}
	return len(failures) == 0, failures
}

// Complexity values end in a metric; errors end in a verdict. Only those
// suffixes are removed, so colons inside a rule name stay part of its identity.
func complexityValueErrorRelated(key, errorKey string) bool {
	base := key
	last := strings.LastIndex(key, ":")
	if strings.Count(key, ":") >= 2 {
		switch key[last+1:] {
		case "text_hard_score", "text_easy_score", "text_margin", "image_hard_score", "image_easy_score", "image_margin", "margin", "score":
			base = key[:last]
		}
	}
	if errorKey == key || errorKey == "complexity" {
		return true
	}
	last = strings.LastIndex(errorKey, ":")
	if last < 0 || errorKey[:last] != base {
		return false
	}
	switch errorKey[last+1:] {
	case "easy", "medium", "hard":
		return true
	default:
		return false
	}
}
