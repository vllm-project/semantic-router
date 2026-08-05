package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestRouterDefaultsHandlerUsesExactConfiguredPath(t *testing.T) {
	tempDir := t.TempDir()
	_ = createValidTestConfig(t, tempDir)

	customPath := filepath.Join(tempDir, "remote-embedding-smoke.yaml")
	customConfig := `
version: v0.3
global:
  model_catalog:
    embeddings:
      semantic:
        embedding_config:
          backend: openai_compatible
          model_type: remote
          target_dimension: 1536
        endpoint:
          base_url: https://api.openai.com/v1
          model: text-embedding-3-small
          dimensions: 1536
`
	if err := os.WriteFile(customPath, []byte(customConfig), 0o644); err != nil {
		t.Fatalf("write custom config: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/api/router/config/global", nil)
	recorder := httptest.NewRecorder()
	RouterDefaultsHandler(customPath)(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", recorder.Code, recorder.Body.String())
	}

	var response map[string]any
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	modelCatalog := requireMapValue(t, response, "model_catalog")
	embeddings := requireMapValue(t, modelCatalog, "embeddings")
	semantic := requireMapValue(t, embeddings, "semantic")
	embeddingConfig := requireMapValue(t, semantic, "embedding_config")
	if got := embeddingConfig["backend"]; got != "openai_compatible" {
		t.Fatalf("embedding backend = %v, want openai_compatible", got)
	}
	if got := embeddingConfig["model_type"]; got != "remote" {
		t.Fatalf("embedding model_type = %v, want remote", got)
	}
}

func TestUpdateRouterDefaultsHandlerUsesExactConfiguredPath(t *testing.T) {
	tempDir := t.TempDir()
	decoyPath := createValidTestConfig(t, tempDir)
	decoyBefore, err := os.ReadFile(decoyPath)
	if err != nil {
		t.Fatalf("read decoy config: %v", err)
	}

	customPath := filepath.Join(tempDir, "remote-embedding-smoke.yaml")
	err = os.WriteFile(customPath, decoyBefore, 0o644)
	if err != nil {
		t.Fatalf("write custom config: %v", err)
	}

	patch := []byte(`{
  "model_catalog": {
    "embeddings": {
      "semantic": {
        "embedding_config": {
          "backend": "openai_compatible",
          "model_type": "remote",
          "target_dimension": 1536
        },
        "endpoint": {
          "base_url": "https://api.openai.com/v1",
          "model": "text-embedding-3-small",
          "dimensions": 1536
        }
      }
    }
  }
}`)
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/global/update", bytes.NewReader(patch))
	recorder := httptest.NewRecorder()
	UpdateRouterDefaultsHandler(customPath, false, tempDir)(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", recorder.Code, recorder.Body.String())
	}
	decoyAfter, err := os.ReadFile(decoyPath)
	if err != nil {
		t.Fatalf("read decoy config after update: %v", err)
	}
	if !bytes.Equal(decoyAfter, decoyBefore) {
		t.Fatal("default config.yaml changed instead of the configured custom file")
	}

	updatedGlobal, err := currentGlobalDefaults(customPath)
	if err != nil {
		t.Fatalf("read updated custom config: %v", err)
	}
	if got := updatedGlobal.ModelCatalog.Embeddings.Semantic.EmbeddingConfig.Backend; got != "openai_compatible" {
		t.Fatalf("updated embedding backend = %q, want openai_compatible", got)
	}
}

func requireMapValue(t *testing.T, value map[string]any, key string) map[string]any {
	t.Helper()
	nested, ok := value[key].(map[string]any)
	if !ok {
		t.Fatalf("%s = %#v, want object", key, value[key])
	}
	return nested
}
