package mcpconfig

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/mark3labs/mcp-go/mcp"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestServerRequiresEnabledConfig(t *testing.T) {
	_, err := NewServer(filepath.Join(t.TempDir(), "config.yaml"), config.MCPConfigServerConfig{})
	if err == nil {
		t.Fatal("expected disabled config to fail server creation")
	}
}

func TestServerGetConfigTool(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	writeTestConfig(t, configPath, map[string]any{
		"version":   "v0.3",
		"routing":   map[string]any{"signals": map[string]any{}, "decisions": []any{}},
		"providers": map[string]any{},
		"global":    map[string]any{},
	})

	server, err := NewServer(configPath, config.MCPConfigServerConfig{Enabled: true})
	if err != nil {
		t.Fatalf("new server: %v", err)
	}

	result, err := server.getConfigTool(context.Background(), mcp.CallToolRequest{})
	if err != nil {
		t.Fatalf("tool call: %v", err)
	}
	if result.IsError {
		t.Fatalf("unexpected tool error: %#v", result)
	}
}

func TestServerAuditLogWritten(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	auditPath := filepath.Join(dir, ".vllm-sr", "mcp-config-audit.jsonl")
	writeTestConfig(t, configPath, map[string]any{
		"version":   "v0.3",
		"routing":   map[string]any{"signals": map[string]any{}, "decisions": []any{}},
		"providers": map[string]any{},
		"global":    map[string]any{},
	})

	server, err := NewServer(configPath, config.MCPConfigServerConfig{Enabled: true})
	if err != nil {
		t.Fatalf("new server: %v", err)
	}

	_, err = server.getConfigTool(context.Background(), mcp.CallToolRequest{})
	if err != nil {
		t.Fatalf("tool call: %v", err)
	}

	data, err := os.ReadFile(auditPath)
	if err != nil {
		t.Fatalf("read audit log: %v", err)
	}
	if len(data) == 0 {
		t.Fatal("expected audit log entry")
	}
}
