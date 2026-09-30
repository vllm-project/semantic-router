package modeldownload

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestDefaultPIIDownloadUsesPublishedVelaMapping(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(`
version: v0.3
providers:
  defaults:
    model: demo
  models:
    - name: demo
      backend_refs:
        - name: primary
          endpoint: localhost:8000
          protocol: http
          weight: 1
routing:
  signals:
    pii:
      - name: sensitive-contact
        threshold: 0.9
        pii_types_allowed: []
  decisions:
    - name: protect-contact
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: pii
            name: sensitive-contact
      modelRefs:
        - model: demo
`))
	if err != nil {
		t.Fatal(err)
	}
	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatal(err)
	}
	// The published Vela token-classification package contains pii_mapping.json,
	// not the legacy MOM pii_type_mapping.json. Model-directory presence alone
	// cannot establish that the download requirements are satisfiable.
	packageDir := t.TempDir()
	for _, name := range []string{"config.json", "pii_mapping.json"} {
		if err := os.WriteFile(filepath.Join(packageDir, name), []byte("{}"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	for _, spec := range specs {
		if spec.RepoID != "llm-semantic-router/Vela-1.0-Encoder-307M-PII" {
			continue
		}
		for _, name := range spec.RequiredFiles {
			if _, err := os.Stat(filepath.Join(packageDir, name)); err != nil {
				t.Fatalf("default PII provisioning requests absent published artifact %q: %v", name, err)
			}
		}
		if cfg.PIIModel.PIIMappingPath != filepath.Join(spec.LocalPath, "pii_mapping.json") {
			t.Fatalf("PII runtime mapping differs from the published package: %q", cfg.PIIModel.PIIMappingPath)
		}
		return
	}
	t.Fatal("PII signal did not provision the default Vela token classifier")
}
