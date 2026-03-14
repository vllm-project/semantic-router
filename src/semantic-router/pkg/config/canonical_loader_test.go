package config

import (
	"strings"
	"testing"
)

func TestParseYAMLBytesRejectsLegacyUserConfigLayout(t *testing.T) {
	legacyYAML := []byte(`
version: v0.3
signals:
  keywords:
    - name: urgent_keywords
      operator: OR
      keywords: ["urgent"]
decisions:
  - name: urgent_route
    rules:
      operator: AND
      conditions:
        - type: keyword
          name: urgent_keywords
    modelRefs:
      - model: qwen2.5:3b
        use_reasoning: false
providers:
  default_model: qwen2.5:3b
  models:
    - name: qwen2.5:3b
      endpoints:
        - endpoint: 127.0.0.1:11434
`)

	_, err := ParseYAMLBytes(legacyYAML)
	if err == nil {
		t.Fatal("expected legacy user config layout to be rejected")
	}

	message := err.Error()
	for _, fragment := range []string{
		"deprecated config fields are no longer supported",
		"providers.models[0].endpoints",
		"vllm-sr config migrate --config old-config.yaml",
	} {
		if !strings.Contains(message, fragment) {
			t.Fatalf("expected error to mention %q, got: %s", fragment, message)
		}
	}
}
