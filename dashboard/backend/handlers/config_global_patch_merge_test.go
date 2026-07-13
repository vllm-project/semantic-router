package handlers

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestMergeGlobalOverridePatchUsesNullOnlyForEmbeddingEndpointDeletes(t *testing.T) {
	existing := []byte(`version: v0.3
global:
  model_catalog:
    embeddings:
      semantic:
        endpoint:
          api_key_env: VLLM_SR_EMBEDDING_API_KEY
          timeout_seconds: 30
          future_transport_option: preserved
          future_nullable: configured
  integrations:
    future_service:
      nullable_option: configured
`)
	patch := []byte(`model_catalog:
  embeddings:
    semantic:
      endpoint:
        api_key_env: null
        timeout_seconds: null
        future_nullable: null
integrations:
  future_service:
    nullable_option: null
`)

	merged, err := mergeGlobalOverridePatchYAML(existing, patch)
	if err != nil {
		t.Fatalf("mergeGlobalOverridePatchYAML: %v", err)
	}
	doc, err := parseYAMLDocument(merged)
	if err != nil {
		t.Fatalf("parse merged YAML: %v", err)
	}
	root, err := documentMappingNode(doc)
	if err != nil {
		t.Fatalf("documentMappingNode: %v", err)
	}
	global := requiredMappingPath(t, root, "global")
	endpoint := requiredMappingPath(
		t,
		global,
		"model_catalog",
		"embeddings",
		"semantic",
		"endpoint",
	)
	for _, key := range []string{"api_key_env", "timeout_seconds"} {
		if mappingValueNode(endpoint, key) != nil {
			t.Fatalf("endpoint tombstone %q was persisted:\n%s", key, merged)
		}
	}
	if got := mappingValueNode(endpoint, "future_transport_option"); got == nil || got.Value != "preserved" {
		t.Fatalf("unknown endpoint field was not preserved: %#v\n%s", got, merged)
	}
	if got := mappingValueNode(endpoint, "future_nullable"); got == nil || got.Tag != "!!null" {
		t.Fatalf("unknown endpoint null was treated as a tombstone: %#v\n%s", got, merged)
	}
	nullable := mappingValueNode(
		requiredMappingPath(t, global, "integrations", "future_service"),
		"nullable_option",
	)
	if nullable == nil || nullable.Tag != "!!null" {
		t.Fatalf("unrelated legal null was treated as a tombstone: %#v\n%s", nullable, merged)
	}
}

func TestMergeGlobalOverridePatchDeletesOnlyTheEmbeddingEndpointMapping(t *testing.T) {
	existing := []byte(`version: v0.3
global:
  model_catalog:
    embeddings:
      semantic:
        future_semantic_option: preserved
        endpoint:
          base_url: https://embeddings.example/v1
          model: embedding-model
          api_key_env: VLLM_SR_EMBEDDING_API_KEY
`)
	patch := []byte(`model_catalog:
  embeddings:
    semantic:
      endpoint: null
`)

	merged, err := mergeGlobalOverridePatchYAML(existing, patch)
	if err != nil {
		t.Fatalf("mergeGlobalOverridePatchYAML: %v", err)
	}
	doc, err := parseYAMLDocument(merged)
	if err != nil {
		t.Fatalf("parse merged YAML: %v", err)
	}
	root, err := documentMappingNode(doc)
	if err != nil {
		t.Fatalf("documentMappingNode: %v", err)
	}
	semantic := requiredMappingPath(
		t,
		requiredMappingPath(t, root, "global"),
		"model_catalog",
		"embeddings",
		"semantic",
	)
	if mappingValueNode(semantic, "endpoint") != nil {
		t.Fatalf("endpoint mapping tombstone was persisted:\n%s", merged)
	}
	if got := mappingValueNode(semantic, "future_semantic_option"); got == nil || got.Value != "preserved" {
		t.Fatalf("semantic sibling was not preserved: %#v\n%s", got, merged)
	}
}

func TestUpdateRouterDefaultsHandlerDeletesClearedEmbeddingEndpointNumbers(t *testing.T) {
	useMissingManagedDockerCLI(t)
	for _, name := range []string{
		"VLLM_SR_RUNTIME_CONFIG_PATH",
		"VLLM_SR_ALGORITHM_OVERRIDE",
		"VLLM_SR_PLATFORM",
		"DASHBOARD_PLATFORM",
	} {
		t.Setenv(name, "")
	}

	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)
	base, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read base config: %v", err)
	}
	base = append(base, []byte(`
global:
  model_catalog:
    embeddings:
      semantic:
        embedding_config:
          backend: openai_compatible
          model_type: remote
          target_dimension: 1536
        endpoint:
          base_url: https://embeddings.example/v1
          model: embedding-model
          api_key_env: VLLM_SR_EMBEDDING_API_KEY
          timeout_seconds: 30
          max_retries: 2
          dimensions: 1536
          future_transport_option: preserved
`)...)
	if writeErr := os.WriteFile(configPath, base, 0o644); writeErr != nil {
		t.Fatalf("write embedding config: %v", writeErr)
	}

	requestBody := []byte(`{
  "model_catalog": {
    "embeddings": {
      "semantic": {
        "endpoint": {
          "base_url": "https://embeddings.example/v1",
          "model": "embedding-model",
          "api_key_env": "VLLM_SR_EMBEDDING_API_KEY",
          "timeout_seconds": null,
          "max_retries": null,
          "dimensions": null,
          "future_transport_option": "preserved"
        }
      }
    }
  }
}`)
	req := httptest.NewRequest(
		http.MethodPost,
		"/api/router/config/global/update",
		bytes.NewReader(requestBody),
	)
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	UpdateRouterDefaultsHandler(tempDir, false)(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", w.Code, w.Body.String())
	}
	updated, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read updated config: %v", err)
	}
	doc, err := parseYAMLDocument(updated)
	if err != nil {
		t.Fatalf("parse updated config: %v", err)
	}
	root, err := documentMappingNode(doc)
	if err != nil {
		t.Fatalf("documentMappingNode: %v", err)
	}
	endpoint := requiredMappingPath(
		t,
		requiredMappingPath(t, root, "global"),
		"model_catalog",
		"embeddings",
		"semantic",
		"endpoint",
	)
	for _, key := range []string{"timeout_seconds", "max_retries", "dimensions"} {
		if mappingValueNode(endpoint, key) != nil {
			t.Fatalf("cleared endpoint field %q was persisted:\n%s", key, updated)
		}
	}
	if got := mappingValueNode(endpoint, "future_transport_option"); got == nil || got.Value != "preserved" {
		t.Fatalf("unknown endpoint field was not preserved: %#v\n%s", got, updated)
	}
}

func requiredMappingPath(t *testing.T, root *yaml.Node, path ...string) *yaml.Node {
	t.Helper()
	current := root
	for _, key := range path {
		current = mappingValueNode(current, key)
		if current == nil || current.Kind != yaml.MappingNode {
			t.Fatalf("missing YAML mapping path %v at %q", path, key)
		}
	}
	return current
}
