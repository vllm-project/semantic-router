package config

import (
	"strings"
	"testing"
)

func TestReadOnlyValidationRejectsAbsoluteKnowledgeBasePath(t *testing.T) {
	configYAML := `
version: v0.3
providers:
  defaults:
    default_model: model-a
  models:
    - name: model-a
      backend_refs:
        - endpoint: 127.0.0.1:8000
routing:
  modelCards:
    - name: model-a
  signals:
    kb:
      - name: private-signal
        kb: private
        target:
          kind: label
          value: private
global:
  model_catalog:
    kbs:
      - name: private
        source:
          path: /dev
          manifest: zero
        threshold: 0.5
`

	_, err := ParseYAMLBytesWithoutEnvExpansion([]byte(configYAML))
	if err == nil ||
		!strings.Contains(err.Error(), "absolute source.path is not allowed") {
		t.Fatalf("expected absolute KB path rejection, got %v", err)
	}
}
