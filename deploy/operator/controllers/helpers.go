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
	"context"
	"encoding/json"
	"fmt"
	"strconv"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
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

// getSecretValue retrieves a value from a Kubernetes Secret
func (r *SemanticRouterReconciler) getSecretValue(
	ctx context.Context,
	namespace string,
	selector *corev1.SecretKeySelector,
) (string, error) {
	if selector == nil {
		return "", fmt.Errorf("secret selector is nil")
	}

	secret := &corev1.Secret{}
	err := r.Get(ctx, types.NamespacedName{
		Name:      selector.Name,
		Namespace: namespace,
	}, secret)
	if err != nil {
		return "", fmt.Errorf("failed to get secret %s: %w", selector.Name, err)
	}

	value, ok := secret.Data[selector.Key]
	if !ok {
		return "", fmt.Errorf("key %s not found in secret %s", selector.Key, selector.Name)
	}

	return string(value), nil
}

// resolveSemanticCacheSecrets resolves all SecretKeySelector references in cache config
func (r *SemanticRouterReconciler) resolveSemanticCacheSecrets(
	ctx context.Context,
	sr *vllmv1alpha1.SemanticRouter,
) error {
	if sr.Spec.Config.SemanticCache == nil {
		return nil
	}

	cache := sr.Spec.Config.SemanticCache

	// Resolve Redis password from Secret
	if cache.Redis != nil && cache.Redis.Connection.PasswordSecretRef != nil {
		password, err := r.getSecretValue(
			ctx,
			sr.Namespace,
			cache.Redis.Connection.PasswordSecretRef,
		)
		if err != nil {
			return fmt.Errorf("failed to resolve Redis password: %w", err)
		}
		cache.Redis.Connection.Password = password
		// Clear the SecretRef after resolution so it doesn't appear in the ConfigMap
		cache.Redis.Connection.PasswordSecretRef = nil
	}

	// Resolve Milvus password from Secret
	if cache.Milvus != nil &&
		cache.Milvus.Connection.Auth.PasswordSecretRef != nil {
		password, err := r.getSecretValue(
			ctx,
			sr.Namespace,
			cache.Milvus.Connection.Auth.PasswordSecretRef,
		)
		if err != nil {
			return fmt.Errorf("failed to resolve Milvus password: %w", err)
		}
		cache.Milvus.Connection.Auth.Password = password
		// Clear the SecretRef after resolution so it doesn't appear in the ConfigMap
		cache.Milvus.Connection.Auth.PasswordSecretRef = nil
	}

	return nil
}

// validateSemanticCacheConfig validates cache configuration consistency
func validateSemanticCacheConfig(cache *vllmv1alpha1.SemanticCacheConfig) error {
	if cache == nil || !cache.Enabled {
		return nil
	}

	switch cache.BackendType {
	case "redis":
		if cache.Redis == nil {
			return fmt.Errorf("redis configuration required when backend_type is 'redis'")
		}
		if cache.Redis.Connection.Host == "" {
			return fmt.Errorf("redis.connection.host is required")
		}
	case "milvus":
		if cache.Milvus == nil {
			return fmt.Errorf("milvus configuration required when backend_type is 'milvus'")
		}
		if cache.Milvus.Connection.Host == "" {
			return fmt.Errorf("milvus.connection.host is required")
		}
	case "hybrid":
		if cache.Milvus == nil {
			return fmt.Errorf("milvus configuration required for hybrid backend")
		}
		if cache.Milvus.Connection.Host == "" {
			return fmt.Errorf("milvus.connection.host is required for hybrid backend")
		}
		// HNSW config is optional but recommended
	case "memory", "":
		// No additional validation needed for memory backend
	default:
		return fmt.Errorf("unsupported backend_type: %s (must be one of: memory, redis, milvus, hybrid)", cache.BackendType)
	}

	return nil
}
