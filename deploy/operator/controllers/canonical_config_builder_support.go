package controllers

import (
	"gopkg.in/yaml.v3"
)

func convertToTypedConfig[T any](r *SemanticRouterReconciler, value interface{}) (T, error) {
	var result T

	normalized := r.convertToConfigMap(value)
	data, err := yaml.Marshal(normalized)
	if err != nil {
		return result, err
	}
	if err := yaml.Unmarshal(data, &result); err != nil {
		return result, err
	}

	return result, nil
}
