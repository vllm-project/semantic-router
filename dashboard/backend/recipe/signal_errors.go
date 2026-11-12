package recipe

import (
	"errors"
	"fmt"
	"maps"
	"strings"

	"gopkg.in/yaml.v3"
)

func validatedSignalErrors(node yaml.Node, label string, issues *[]string) map[string]string {
	result, err := parseSignalErrorExpectations(node)
	if err != nil {
		*issues = append(*issues, label+"."+err.Error())
	}
	return result
}

// Preserve omitted versus explicit {}, and reject YAML's scalar coercions.
func parseSignalErrorExpectations(node yaml.Node) (map[string]string, error) {
	if node.Kind == 0 {
		return nil, nil
	}
	if node.Kind != yaml.MappingNode {
		return nil, errors.New("expected_signal_errors must be a mapping")
	}
	result := make(map[string]string, len(node.Content)/2)
	for i := 0; i < len(node.Content); i += 2 {
		key, code := node.Content[i], node.Content[i+1]
		for _, value := range []*yaml.Node{key, code} {
			if value.Tag != "!!str" || value.Value == "" || value.Value != strings.TrimSpace(value.Value) {
				return nil, errors.New("expected_signal_errors requires non-empty string keys and error codes without surrounding whitespace")
			}
		}
		if _, duplicate := result[key.Value]; duplicate {
			return nil, fmt.Errorf("expected_signal_errors duplicate key %q", key.Value)
		}
		result[key.Value] = code.Value
	}
	return result, nil
}

func cloneSignalErrors(source map[string]string) map[string]string {
	return maps.Clone(source)
}
