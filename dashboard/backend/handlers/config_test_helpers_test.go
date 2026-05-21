package handlers

import (
	"os"
	"path/filepath"
	"testing"
)

// createValidTestConfig creates a minimal canonical v0.3 config file for testing.
func createValidTestConfig(t *testing.T, dir string) string {
	configPath := filepath.Join(dir, "config.yaml")
	validConfig := `
version: v0.3
listeners:
  - name: public
    address: 0.0.0.0
    port: 8801
providers:
  defaults:
    default_model: test-model
    reasoning_families:
      qwen3:
        type: reasoning_effort
        parameter: reasoning_effort
  models:
    - name: test-model
      reasoning_family: qwen3
      provider_model_id: test-model
      backend_refs:
        - name: endpoint1
          endpoint: 127.0.0.1:8000
          protocol: http
          weight: 1
routing:
  modelCards:
    - name: test-model
  signals:
    domains:
      - name: business
        description: Business and management related queries
  decisions:
    - name: default-business
      description: Route business requests to the default model
      priority: 1
      rules:
        operator: OR
        conditions:
          - type: domain
            name: business
      modelRefs:
        - model: test-model
          use_reasoning: false
`
	if err := os.WriteFile(configPath, []byte(validConfig), 0o644); err != nil {
		t.Fatalf("Failed to create test config file: %v", err)
	}
	return configPath
}

func createLegacyTestConfig(t *testing.T, dir string) string {
	configPath := filepath.Join(dir, "config.yaml")
	legacyConfig := `
categories:
  - name: business
    description: Business and management related queries

vllm_endpoints:
  - name: endpoint1
    address: 127.0.0.1
    port: 8000
    weight: 1

default_model: test-model

model_config:
  test-model:
    reasoning_family: qwen3
`
	if err := os.WriteFile(configPath, []byte(legacyConfig), 0o644); err != nil {
		t.Fatalf("Failed to create legacy test config file: %v", err)
	}
	return configPath
}

func canonicalConfigBody(endpoint string) map[string]interface{} {
	return map[string]interface{}{
		"version": "v0.3",
		"listeners": []map[string]interface{}{
			{
				"name":    "public",
				"address": "0.0.0.0",
				"port":    8801,
			},
		},
		"providers": map[string]interface{}{
			"defaults": map[string]interface{}{
				"default_model": "test-model",
				"reasoning_families": map[string]interface{}{
					"qwen3": map[string]interface{}{
						"type":      "reasoning_effort",
						"parameter": "reasoning_effort",
					},
				},
			},
			"models": []map[string]interface{}{
				{
					"name":              "test-model",
					"reasoning_family":  "qwen3",
					"provider_model_id": "test-model",
					"backend_refs": []map[string]interface{}{
						{
							"name":     "endpoint1",
							"endpoint": endpoint,
							"protocol": "http",
							"weight":   1,
						},
					},
				},
			},
		},
		"routing": map[string]interface{}{
			"modelCards": []map[string]interface{}{
				{
					"name": "test-model",
				},
			},
			"signals": map[string]interface{}{
				"domains": []map[string]interface{}{
					{
						"name":        "business",
						"description": "Business and management related queries",
					},
				},
			},
			"decisions": []map[string]interface{}{
				{
					"name":        "default-business",
					"description": "Route business requests to the default model",
					"priority":    1,
					"rules": map[string]interface{}{
						"operator": "OR",
						"conditions": []map[string]interface{}{
							{
								"type": "domain",
								"name": "business",
							},
						},
					},
					"modelRefs": []map[string]interface{}{
						{
							"model":         "test-model",
							"use_reasoning": false,
						},
					},
				},
			},
		},
	}
}

func contains(s, substr string) bool {
	if len(substr) == 0 {
		return true
	}
	if len(s) < len(substr) {
		return false
	}
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
