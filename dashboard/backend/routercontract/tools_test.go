package routercontract

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReadToolSelectionReturnsToolsDBPath(t *testing.T) {
	configPath := writeRouterConfig(t, `version: v0.3
providers:
  defaults: {model: local}
  models:
    - name: local
      backend_refs: [{name: local, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
routing:
  modelCards:
    - {name: local, description: Local model}
  decisions:
    - name: agent
      priority: 100
      rules: {operator: AND, conditions: []}
      modelRefs: [{model: local}]
      plugins:
        - type: tools
          configuration:
            enabled: true
            select: {mode: add, tools_db_path: "/var/lib/vsr/tools.json"}
`)

	selection, err := ReadToolSelection(configPath)
	if err != nil {
		t.Fatalf("ReadToolSelection() error = %v", err)
	}
	if selection.ToolsDBPath != "/var/lib/vsr/tools.json" {
		t.Fatalf("ToolsDBPath = %q, want %q", selection.ToolsDBPath, "/var/lib/vsr/tools.json")
	}
}

func TestReadToolSelectionWithoutCatalogIsEmpty(t *testing.T) {
	selection, err := ReadToolSelection(writeRouterConfig(t, "version: v0.3\nglobal: {}\n"))
	if err != nil {
		t.Fatalf("ReadToolSelection() error = %v", err)
	}
	if selection.ToolsDBPath != "" {
		t.Fatalf("ToolsDBPath = %q, want none", selection.ToolsDBPath)
	}
}

func TestReadToolSelectionReturnsParseError(t *testing.T) {
	configPath := writeRouterConfig(t, "routing: [")

	_, err := ReadToolSelection(configPath)
	if err == nil {
		t.Fatal("ReadToolSelection() error = nil, want parse error")
	}
}

func writeRouterConfig(t *testing.T, content string) string {
	t.Helper()

	configPath := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(configPath, []byte(content), 0o644); err != nil {
		t.Fatalf("WriteFile(config): %v", err)
	}
	return configPath
}
