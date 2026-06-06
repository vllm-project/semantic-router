package mcp

import (
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

func TestManagerPersistsServerConfigs(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "wf.sqlite")

	store1, err := workflowstore.Open(path, workflowstore.Options{})
	if err != nil {
		t.Fatal(err)
	}

	m1, err := NewManager(store1)
	if err != nil {
		t.Fatal(err)
	}

	config := &ServerConfig{
		ID:        "user-server-1",
		Name:      "Filesystem MCP",
		Transport: TransportStdio,
		Connection: ConnectionConfig{
			Command: "npx",
			Args:    []string{"-y", "@modelcontextprotocol/server-filesystem", "/tmp"},
		},
		Enabled: true,
	}
	if addErr := m1.AddServer(config); addErr != nil {
		t.Fatal(addErr)
	}

	m2, err := NewManager(store1)
	if err != nil {
		t.Fatal(err)
	}

	got, ok := m2.GetServer(config.ID)
	if !ok {
		t.Fatal("expected persisted server config after reload")
	}
	if got.Name != config.Name || got.Connection.Command != config.Connection.Command || !got.Enabled {
		t.Fatalf("unexpected config: %+v", got)
	}

	if delErr := m2.DeleteServer(config.ID); delErr != nil {
		t.Fatal(delErr)
	}

	m3, err := NewManager(store1)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := m3.GetServer(config.ID); ok {
		t.Fatal("expected deleted server config to be gone after reload")
	}
}

func TestManagerUpsertBuiltinServerRefreshesURL(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "wf.sqlite")

	store, err := workflowstore.Open(path, workflowstore.Options{})
	if err != nil {
		t.Fatal(err)
	}

	m, err := NewManager(store)
	if err != nil {
		t.Fatal(err)
	}

	first := &ServerConfig{
		ID:        BuiltinOpenClawServerID,
		Name:      BuiltinOpenClawServerName,
		Transport: TransportStreamableHTTP,
		Connection: ConnectionConfig{
			URL: "http://127.0.0.1:8700/_internal/openclaw/mcp",
		},
		Enabled: false,
	}
	if upsertErr := m.UpsertServer(first); upsertErr != nil {
		t.Fatal(upsertErr)
	}

	second := *first
	second.Connection.URL = "http://127.0.0.1:9001/_internal/openclaw/mcp"
	if upsertErr := m.UpsertServer(&second); upsertErr != nil {
		t.Fatal(upsertErr)
	}

	reloaded, err := NewManager(store)
	if err != nil {
		t.Fatal(err)
	}
	got, ok := reloaded.GetServer(BuiltinOpenClawServerID)
	if !ok {
		t.Fatal("expected builtin server after reload")
	}
	if got.Connection.URL != second.Connection.URL {
		t.Fatalf("url = %q, want %q", got.Connection.URL, second.Connection.URL)
	}
}
