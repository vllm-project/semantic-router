package config

import (
	"strings"
	"testing"
)

func embeddingBackendConfigYAML(backend string) []byte {
	return []byte(`
version: v0.3
providers:
  defaults:
    model: private-model
routing:
  modelCards:
    - name: private-model
      description: Metadata-only routing model
global:
  model_catalog:
    embeddings:
      semantic:
        embedding_config:
          backend: ` + backend + `
`)
}

func TestParseYAMLBytesRejectsRetiredEmbeddingBackends(t *testing.T) {
	for _, backend := range []string{"candle", "openvino", "OpenVINO"} {
		_, err := ParseYAMLBytes(embeddingBackendConfigYAML(backend))
		if err == nil {
			t.Fatalf("embedding_config.backend %q must be rejected", backend)
		}
		for _, fragment := range []string{
			"global.model_catalog.embeddings.semantic.embedding_config.backend: " + backend,
			"vllm-sr config migrate --config old-config.yaml",
		} {
			if !strings.Contains(err.Error(), fragment) {
				t.Fatalf("backend %q: expected error to mention %q, got: %v", backend, fragment, err)
			}
		}
	}
}

func TestParseYAMLBytesLocalEmbeddingsAreServedByTheModelRuntime(t *testing.T) {
	cfg, err := ParseYAMLBytes(embeddingBackendConfigYAML(EmbeddingBackendModelRuntime))
	if err != nil {
		t.Fatalf("explicit model_runtime embedding backend: %v", err)
	}
	if got := cfg.EmbeddingModels.EmbeddingBackend(); got != EmbeddingBackendModelRuntime {
		t.Fatalf("explicit backend = %q, want %q", got, EmbeddingBackendModelRuntime)
	}
	if cfg.EmbeddingModels.UsesRemoteEmbeddingBackend() {
		t.Fatal("model_runtime embeddings must not use the remote backend")
	}

	if got := (EmbeddingModels{}).EmbeddingBackend(); got != EmbeddingBackendModelRuntime {
		t.Fatalf("default backend = %q, want %q", got, EmbeddingBackendModelRuntime)
	}
	if got := (HNSWConfig{}).WithDefaults().Backend; got != EmbeddingBackendModelRuntime {
		t.Fatalf("defaulted backend = %q, want %q", got, EmbeddingBackendModelRuntime)
	}
}
