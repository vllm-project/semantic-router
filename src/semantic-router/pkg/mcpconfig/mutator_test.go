package mcpconfig

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestMutatorGetDocument(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	writeTestConfig(t, configPath, map[string]any{"version": "v0.3"})

	mutator := NewMutator(configPath)
	doc, err := mutator.GetDocument()
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if doc["version"] != "v0.3" {
		t.Fatalf("version = %#v", doc["version"])
	}
}

func TestMutatorValidateRejectsInvalidDocument(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	writeTestConfig(t, configPath, map[string]any{"version": "v0.3"})

	mutator := NewMutator(configPath)
	_, err := mutator.ValidateDocument(map[string]any{"routing": "invalid"}, MutationMerge)
	if err == nil {
		t.Fatal("expected validation error")
	}
}

func TestMutatorApplyPatchCreatesBackup(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	initial := strings.TrimSpace(`
version: v0.3
listeners: []
providers:
  defaults: {}
routing:
  signals: {}
global:
  router:
    config_source: file
  services: {}
  stores:
    semantic_cache:
      enabled: false
  integrations:
    tools:
      enabled: false
  model_catalog:
    embeddings:
      semantic:
        use_cpu: true
`)
	if err := os.WriteFile(configPath, []byte(initial), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	mutator := NewMutator(configPath)
	patch := map[string]any{
		"global": map[string]any{
			"router": map[string]any{
				"auto_model_name": "router-mcp-test",
			},
		},
	}

	result, err := mutator.ApplyPatch(patch, MutationMerge, "")
	if err != nil {
		t.Fatalf("apply: %v", err)
	}
	if result.Version == "" {
		t.Fatal("expected version from backup")
	}

	backupPath := filepath.Join(dir, ".vllm-sr", "config-backups", "config."+result.Version+".yaml")
	if _, statErr := os.Stat(backupPath); statErr != nil {
		t.Fatalf("backup missing: %v", statErr)
	}

	updated, err := mutator.GetDocument()
	if err != nil {
		t.Fatalf("get updated: %v", err)
	}
	router, ok := updated["global"].(map[string]any)["router"].(map[string]any)
	if !ok || router["auto_model_name"] != "router-mcp-test" {
		t.Fatalf("expected patched auto_model_name, got %#v", updated)
	}
}

func writeTestConfig(t *testing.T, path string, doc map[string]any) {
	t.Helper()
	data, err := yaml.Marshal(doc)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
}
