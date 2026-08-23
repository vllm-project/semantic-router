package config

import (
	"strings"
	"testing"
)

func TestReadOnlyValidationRejectsAbsoluteKnowledgeBasePath(t *testing.T) {
	configYAML := canonicalRecipeFixture(`
signals:
  kb:
    - name: private-signal
      kb: private
      target:
        kind: label
        value: private
decisions:
  - name: route
    rules: {}
`, `
model_catalog:
  kbs:
    - name: private
      source:
        path: /dev
        manifest: zero
      threshold: 0.5
`)

	_, err := testAuthoringParser(t).ParseYAMLBytesWithoutEnvExpansion(configYAML)
	if err == nil ||
		!strings.Contains(err.Error(), "absolute source.path is not allowed") {
		t.Fatalf("expected absolute KB path rejection, got %v", err)
	}
}
