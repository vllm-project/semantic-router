package config

import (
	"strings"
	"testing"
)

func TestCanonicalExportPreservesCanonicalBackendRefName(t *testing.T) {
	const input = `
version: v0.3
routing:
  modelCards:
    - name: test-model
providers:
  defaults:
    default_model: test-model
  models:
    - name: test-model
      backend_refs:
        - name: test-model_primary
          endpoint: 127.0.0.1:8000
          protocol: http
`

	cfg, err := ParseYAMLBytes([]byte(input))
	if err != nil {
		t.Fatalf("parse canonical config: %v", err)
	}

	exported := CanonicalConfigFromRouterConfig(cfg)
	if len(exported.Providers.Models) != 1 || len(exported.Providers.Models[0].BackendRefs) != 1 {
		t.Fatalf("expected one provider backend ref, got %+v", exported.Providers.Models)
	}
	if got, want := exported.Providers.Models[0].BackendRefs[0].Name, "test-model_primary"; got != want {
		t.Fatalf("canonical backend ref name changed: got %q, want %q", got, want)
	}
}

func TestCanonicalConfigRejectsBackendRefNameCollisionAfterNormalization(t *testing.T) {
	const input = `
version: v0.3
routing:
  modelCards:
    - name: foo
providers:
  defaults:
    default_model: foo
  models:
    - name: foo
      backend_refs:
        - name: primary
          endpoint: 127.0.0.1:8000
        - name: foo_primary
          endpoint: 127.0.0.1:8001
`

	_, err := ParseYAMLBytes([]byte(input))
	if err == nil {
		t.Fatal("expected normalized backend ref name collision to be rejected")
	}
	for _, want := range []string{"providers.models[foo].backend_refs[1]", `normalized endpoint name "foo_primary"`, "backend_refs[0]"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("expected error to contain %q, got: %v", want, err)
		}
	}
}
