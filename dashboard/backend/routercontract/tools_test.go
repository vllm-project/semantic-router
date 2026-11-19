package routercontract

import (
	"os"
	"path/filepath"
	"testing"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestToolSelectionFromConfigReturnsToolsDBPath(t *testing.T) {
	cfg := parseRouterConfig(t, `version: v0.5
providers:
  defaults: {model: local}
  models:
    - name: local
      backend_refs: [{name: local, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
      description: Local model
recipes:
  - name: default
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

	selection := ToolSelectionFromConfig(cfg)
	if selection.ToolsDBPath != "/var/lib/vsr/tools.json" {
		t.Fatalf("ToolsDBPath = %q, want %q", selection.ToolsDBPath, "/var/lib/vsr/tools.json")
	}
}

func TestToolSelectionFromConfigWithoutCatalogIsEmpty(t *testing.T) {
	selection := ToolSelectionFromConfig(parseRouterConfig(t, "version: v0.5\nglobal: {}\n"))
	if selection.ToolsDBPath != "" {
		t.Fatalf("ToolsDBPath = %q, want none", selection.ToolsDBPath)
	}
}

func parseRouterConfig(t *testing.T, content string) *routerconfig.RouterConfig {
	t.Helper()

	cfg, err := routerconfig.ParseYAMLBytes([]byte(content))
	if err != nil {
		t.Fatalf("ParseYAMLBytes(config): %v", err)
	}
	return cfg
}

func writeRouterConfig(t *testing.T, content string) string {
	t.Helper()

	configPath := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(configPath, []byte(content), 0o644); err != nil {
		t.Fatalf("WriteFile(config): %v", err)
	}
	return configPath
}
