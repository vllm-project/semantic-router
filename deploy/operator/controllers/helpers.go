/*
Copyright 2026 vLLM Semantic Router Contributors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package controllers

import (
	"encoding/json"
	"strconv"

	"k8s.io/apimachinery/pkg/api/resource"
)

// getInt32OrDefault returns the value if not nil, otherwise returns the default
func (r *SemanticRouterReconciler) getInt32OrDefault(val *int32, def int32) int32 {
	if val != nil {
		return *val
	}
	return def
}

// parseQuantity parses a quantity string (e.g., "1Gi", "500Mi")
func (r *SemanticRouterReconciler) parseQuantity(s string) (resource.Quantity, error) {
	return resource.ParseQuantity(s)
}

// convertToConfigMap converts structs to maps, converting string threshold values to floats
func (r *SemanticRouterReconciler) convertToConfigMap(v interface{}) interface{} {
	// Marshal to JSON first (which respects json struct tags), then unmarshal to map[string]interface{}
	// This preserves the JSON tag field names (snake_case) and structure
	data, err := json.Marshal(v)
	if err != nil {
		return v
	}

	var result map[string]interface{}
	if err := json.Unmarshal(data, &result); err != nil {
		return v
	}

	// Recursively convert string values that look like floats to actual floats
	r.convertStringsToFloats(result)
	return result
}

// convertStringsToFloats recursively converts string values to appropriate types
// This handles the mismatch between Kubernetes CRD string fields (to avoid float precision issues)
// and the semantic router app's expectation of actual numeric types in YAML
func (r *SemanticRouterReconciler) convertStringsToFloats(m map[string]interface{}) {
	for k, v := range m {
		switch val := v.(type) {
		case string:
			// Try to convert string to appropriate type
			m[k] = r.convertStringValue(val)
		case map[string]interface{}:
			// Recursively process nested maps
			r.convertStringsToFloats(val)
		case []interface{}:
			// Process arrays
			for i, item := range val {
				if nestedMap, ok := item.(map[string]interface{}); ok {
					r.convertStringsToFloats(nestedMap)
				} else if str, ok := item.(string); ok {
					val[i] = r.convertStringValue(str)
				}
			}
		}
	}
}

// convertStringValue attempts to convert a string to the most appropriate type
func (r *SemanticRouterReconciler) convertStringValue(s string) interface{} {
	// Empty string stays as empty string
	if s == "" {
		return s
	}

	// Try to parse as integer first (for values like "100", "5", etc.)
	if intVal, err := strconv.ParseInt(s, 10, 64); err == nil {
		// Check if the string representation matches (no decimal point)
		if strconv.FormatInt(intVal, 10) == s {
			return intVal
		}
	}

	// Try to parse as float (for values like "0.6", "1.0", "0.8", etc.)
	if floatVal, err := strconv.ParseFloat(s, 64); err == nil {
		return floatVal
	}

	// Try to parse as bool (for values like "true", "false")
	if boolVal, err := strconv.ParseBool(s); err == nil {
		return boolVal
	}

	// If none of the above, keep as string
	return s
}
