package controllers

import (
	"testing"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

func TestValidateSemanticCacheConfigAcceptsValkeyWithRedisConfig(t *testing.T) {
	cache := &vllmv1alpha1.SemanticCacheConfig{
		Enabled:     true,
		BackendType: "valkey",
		Redis: &vllmv1alpha1.RedisCacheConfig{
			Connection: vllmv1alpha1.RedisCacheConnection{
				Host: "valkey.default.svc",
				Port: 6379,
			},
		},
	}

	if err := validateSemanticCacheConfig(cache); err != nil {
		t.Fatalf("expected valkey cache config to validate, got error: %v", err)
	}
}
