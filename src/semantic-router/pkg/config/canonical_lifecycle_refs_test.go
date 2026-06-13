package config

import (
	"strings"
	"testing"
)

func TestParseYAMLBytesResolvesCanonicalLifecycleModelRefs(t *testing.T) {
	canonicalYAML := []byte(`
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8888
providers:
  defaults:
    default_model: qwen3
  models:
    - name: qwen3
      provider_model_id: qwen3
      backend_refs:
        - endpoint: 127.0.0.1:8000
routing:
  modelCards:
    - name: qwen3
  decisions:
    - name: default_route
      priority: 1
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: qwen3
global:
  model_catalog:
    embeddings:
      semantic:
        model_refs:
          mmbert: mmbert_embedding
        use_cpu: true
    system:
      mmbert_embedding: models/custom-mmbert
      modality_classifier: models/custom-modality
    modules:
      modality_detector:
        enabled: true
        method: classifier
        classifier:
          model_ref: modality_classifier
          use_cpu: true
        confidence_threshold: 0.7
`)

	cfg, err := ParseYAMLBytes(canonicalYAML)
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}

	if cfg.MmBertModelPath != "models/custom-mmbert" {
		t.Fatalf("expected mmbert model_ref to resolve through system, got %q", cfg.MmBertModelPath)
	}
	if cfg.ModalityDetector.Classifier == nil {
		t.Fatal("expected modality classifier config")
	}
	if cfg.ModalityDetector.Classifier.ModelPath != "models/custom-modality" {
		t.Fatalf("expected modality classifier model_ref to resolve through system, got %q", cfg.ModalityDetector.Classifier.ModelPath)
	}
}

func TestParseYAMLBytesRejectsUnknownLifecycleModelRef(t *testing.T) {
	canonicalYAML := []byte(`
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8888
providers:
  defaults:
    default_model: qwen3
  models:
    - name: qwen3
      backend_refs:
        - endpoint: 127.0.0.1:8000
routing:
  modelCards:
    - name: qwen3
  decisions:
    - name: default_route
      priority: 1
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: qwen3
global:
  model_catalog:
    embeddings:
      semantic:
        model_refs:
          mmbert: not_a_lifecycle_role
        use_cpu: true
`)

	_, err := ParseYAMLBytes(canonicalYAML)
	if err == nil {
		t.Fatal("expected unknown model_ref to fail")
	}
	if !strings.Contains(err.Error(), "global.model_catalog.embeddings.semantic") ||
		!strings.Contains(err.Error(), "unknown model_ref") {
		t.Fatalf("expected lifecycle model_ref error, got %v", err)
	}
}
