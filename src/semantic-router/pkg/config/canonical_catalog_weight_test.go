package config

import (
	"strings"
	"testing"
)

func TestCanonicalCatalogMaterializesBackendRefWeights(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`
version: v0.3
providers:
  models:
    - name: weighted-model
      api_format: openai
      backend_refs:
        - name: twenty
          endpoint: 127.0.0.1:8000
          provider: vllm
          weight: 20
        - name: eighty
          endpoint: 127.0.0.1:8001
          provider: vllm
          weight: 80
routing: {}
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}

	weights := make(map[string]int, len(cfg.VLLMEndpoints))
	for _, endpoint := range cfg.VLLMEndpoints {
		weights[endpoint.Name] = endpoint.Weight
	}
	if got := weights["weighted-model_twenty"]; got != 20 {
		t.Fatalf("weighted-model_twenty weight = %d, want 20", got)
	}
	if got := weights["weighted-model_eighty"]; got != 80 {
		t.Fatalf("weighted-model_eighty weight = %d, want 80", got)
	}
}

func TestCanonicalCatalogRejectsNegativeBackendRefWeight(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(`
version: v0.3
providers:
  models:
    - name: weighted-model
      api_format: openai
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
          weight: -1
routing: {}
`))
	if err == nil || !strings.Contains(err.Error(), "weight cannot be negative") {
		t.Fatalf("ParseYAMLBytes() error = %v, want negative weight rejection", err)
	}
}
