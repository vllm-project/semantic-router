package routercontract

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReadToolSelectionReturnsToolsDBPath(t *testing.T) {
	configPath := writeRouterConfig(t, `
version: "0.3"
global:
  integrations:
    tools:
      tools_db_path: "/var/lib/vsr/tools.json"
`)

	selection, err := ReadToolSelection(configPath)
	if err != nil {
		t.Fatalf("ReadToolSelection() error = %v", err)
	}
	if selection.ToolsDBPath != "/var/lib/vsr/tools.json" {
		t.Fatalf("ToolsDBPath = %q, want %q", selection.ToolsDBPath, "/var/lib/vsr/tools.json")
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
