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
	"fmt"

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
	cache, err := operatorResponseCacheConfig(sr.Spec.Config)
	if err != nil {
		return err
	}
	if cache == nil {
		return nil
	}

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

	// Resolve Valkey password from Secret
	if cache.Valkey != nil && cache.Valkey.Connection.PasswordSecretRef != nil {
		password, err := r.getSecretValue(
			ctx,
			sr.Namespace,
			cache.Valkey.Connection.PasswordSecretRef,
		)
		if err != nil {
			return fmt.Errorf("failed to resolve Valkey password: %w", err)
		}
		cache.Valkey.Connection.Password = password
		// Clear the SecretRef after resolution so it doesn't appear in the ConfigMap
		cache.Valkey.Connection.PasswordSecretRef = nil
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
		return validateRedisCacheBackend(cache.Redis)
	case "valkey":
		return validateValkeyCacheBackend(cache.Valkey)
	case "milvus":
		return validateMilvusCacheBackend(cache.Milvus, "milvus")
	case "qdrant":
		return validateQdrantCacheBackend(cache.Qdrant)
	case "hybrid":
		return validateMilvusCacheBackend(cache.Milvus, "hybrid")
	case "memory", "":
		return nil
	default:
		return fmt.Errorf("unsupported backend_type: %s (must be one of: memory, redis, valkey, milvus, qdrant, hybrid)", cache.BackendType)
	}
}

func validateRedisCacheBackend(cfg *vllmv1alpha1.RedisCacheConfig) error {
	if cfg == nil {
		return fmt.Errorf("redis configuration required when backend_type is 'redis'")
	}
	if cfg.Connection.Host == "" {
		return fmt.Errorf("redis.connection.host is required")
	}
	return nil
}

func validateValkeyCacheBackend(cfg *vllmv1alpha1.ValkeyCacheConfig) error {
	if cfg == nil {
		return fmt.Errorf("valkey configuration required when backend_type is 'valkey'")
	}
	if cfg.Connection.Host == "" {
		return fmt.Errorf("valkey.connection.host is required")
	}
	return nil
}

func validateMilvusCacheBackend(cfg *vllmv1alpha1.MilvusCacheConfig, backend string) error {
	isHybrid := backend == "hybrid"
	if cfg == nil {
		if isHybrid {
			return fmt.Errorf("milvus configuration required for hybrid backend")
		}
		return fmt.Errorf("milvus configuration required when backend_type is 'milvus'")
	}
	if cfg.Connection.Host == "" {
		if isHybrid {
			return fmt.Errorf("milvus.connection.host is required for hybrid backend")
		}
		return fmt.Errorf("milvus.connection.host is required")
	}
	return nil
}

func validateQdrantCacheBackend(cfg *vllmv1alpha1.QdrantCacheConfig) error {
	if cfg == nil {
		return fmt.Errorf("qdrant configuration required when backend_type is 'qdrant'")
	}
	if cfg.Host == "" {
		return fmt.Errorf("qdrant.host is required")
	}
	return nil
}
