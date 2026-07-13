package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

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

	m1, err := NewManagerWithOptions(store1, ManagerOptions{AllowStdio: true})
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

func TestManagerBlocksStdioByDefaultAcrossEntryPoints(t *testing.T) {
	t.Parallel()

	manager, err := NewManager(nil)
	if err != nil {
		t.Fatal(err)
	}
	config := &ServerConfig{
		ID:         "blocked-stdio",
		Name:       "Blocked stdio",
		Transport:  TransportStdio,
		Connection: ConnectionConfig{Command: "definitely-not-executed"},
		Enabled:    true,
	}
	if err := manager.AddServer(config); !errors.Is(err, ErrStdioTransportBlocked) {
		t.Fatalf("AddServer error = %v, want ErrStdioTransportBlocked", err)
	}
	if err := manager.UpsertServer(config); !errors.Is(err, ErrStdioTransportBlocked) {
		t.Fatalf("UpsertServer error = %v, want ErrStdioTransportBlocked", err)
	}
	if err := manager.UpsertEphemeralServer(config); !errors.Is(err, ErrStdioTransportBlocked) {
		t.Fatalf("UpsertEphemeralServer error = %v, want ErrStdioTransportBlocked", err)
	}
	if err := manager.TestConnection(context.Background(), config); !errors.Is(err, ErrStdioTransportBlocked) {
		t.Fatalf("TestConnection error = %v, want ErrStdioTransportBlocked", err)
	}
}

func TestManagerConfigSnapshotsCannotMutateStoredCredentials(t *testing.T) {
	t.Parallel()

	manager, err := NewManager(nil)
	if err != nil {
		t.Fatal(err)
	}
	config := &ServerConfig{
		ID:        "snapshot-server",
		Name:      "Snapshot",
		Transport: TransportStreamableHTTP,
		Connection: ConnectionConfig{
			URL:     "https://example.test/mcp",
			Headers: map[string]string{"Authorization": "stored-secret"},
		},
		Security: &SecurityConfig{OAuth: &OAuthConfig{ClientSecret: "oauth-secret"}},
	}
	if err := manager.AddServer(config); err != nil {
		t.Fatal(err)
	}
	first, ok := manager.GetServer(config.ID)
	if !ok {
		t.Fatal("server missing")
	}
	first.Connection.Headers["Authorization"] = "mutated"
	first.Security.OAuth.ClientSecret = "mutated"
	second, ok := manager.GetServer(config.ID)
	if !ok {
		t.Fatal("server missing on second read")
	}
	if second.Connection.Headers["Authorization"] != "stored-secret" ||
		second.Security.OAuth.ClientSecret != "oauth-secret" {
		t.Fatalf("stored config was mutated through a snapshot: %+v", second)
	}
}

func TestPersistedStdioCannotAutoConnectUnderSafePolicy(t *testing.T) {
	t.Parallel()

	store, err := workflowstore.Open(filepath.Join(t.TempDir(), "wf.sqlite"), workflowstore.Options{})
	if err != nil {
		t.Fatal(err)
	}
	config := &ServerConfig{
		ID:         "legacy-stdio",
		Name:       "Legacy stdio",
		Transport:  TransportStdio,
		Connection: ConnectionConfig{Command: "definitely-not-executed"},
		Enabled:    true,
	}
	payload, err := json.Marshal(config)
	if err != nil {
		t.Fatal(err)
	}
	if persistErr := store.PutMCPServerJSON(config.ID, string(payload)); persistErr != nil {
		t.Fatal(persistErr)
	}

	manager, err := NewManager(store)
	if err != nil {
		t.Fatal(err)
	}
	if connectErr := manager.Connect(context.Background(), config.ID); !errors.Is(connectErr, ErrStdioTransportBlocked) {
		t.Fatalf("Connect error = %v, want ErrStdioTransportBlocked", connectErr)
	}
	manager.ConnectEnabled(context.Background())
	state, err := manager.GetServerStatus(config.ID)
	if err != nil {
		t.Fatal(err)
	}
	if state.Status != StatusDisconnected {
		t.Fatalf("status = %q, want disconnected", state.Status)
	}
}

func TestEphemeralServerCredentialsAreNeverPersisted(t *testing.T) {
	t.Parallel()

	store, err := workflowstore.Open(filepath.Join(t.TempDir(), "wf.sqlite"), workflowstore.Options{})
	if err != nil {
		t.Fatal(err)
	}
	const capability = "process-only-capability"
	config := &ServerConfig{
		ID:        BuiltinOpenClawServerID,
		Name:      BuiltinOpenClawServerName,
		Transport: TransportStreamableHTTP,
		Connection: ConnectionConfig{
			URL:     "http://127.0.0.1:8700/_internal/openclaw/mcp",
			Headers: map[string]string{"X-Internal": capability},
		},
	}
	stale := *config
	stale.Connection.Headers = map[string]string{"X-Internal": "stale-persisted-value"}
	stalePayload, err := json.Marshal(&stale)
	if err != nil {
		t.Fatal(err)
	}
	if persistErr := store.PutMCPServerJSON(stale.ID, string(stalePayload)); persistErr != nil {
		t.Fatal(persistErr)
	}
	manager, err := NewManager(store)
	if err != nil {
		t.Fatal(err)
	}
	if upsertErr := manager.UpsertEphemeralServer(config); upsertErr != nil {
		t.Fatal(upsertErr)
	}
	rows, err := store.ListMCPServerJSON()
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 0 {
		t.Fatalf("ephemeral config persisted: %v", rows)
	}
	got, ok := manager.GetServer(config.ID)
	if !ok || got.Connection.Headers["X-Internal"] != capability {
		t.Fatal("expected process-local config to remain available in memory")
	}
}

func TestStdioBaseEnvironmentDoesNotInheritDashboardSecrets(t *testing.T) {
	t.Setenv("DASHBOARD_JWT_SECRET", "must-not-leak")
	t.Setenv("OPENCLAW_GATEWAY_TOKEN", "must-not-leak-either")
	t.Setenv("PATH", "/safe/test/path")

	environment := stdioBaseEnvironment()
	joined := strings.Join(environment, "\n")
	if strings.Contains(joined, "must-not-leak") || strings.Contains(joined, "DASHBOARD_JWT_SECRET=") ||
		strings.Contains(joined, "OPENCLAW_GATEWAY_TOKEN=") {
		t.Fatalf("ambient secret leaked into stdio environment: %q", joined)
	}
	if !containsEnvironmentEntry(environment, "PATH=/safe/test/path") {
		t.Fatalf("expected allowlisted PATH, got %v", environment)
	}
}

func TestManagerCloseCancelsInFlightConnectAndRejectsNewConnections(t *testing.T) {
	t.Parallel()

	requestStarted := make(chan struct{}, 1)
	releaseUpstream := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case requestStarted <- struct{}{}:
		default:
		}
		select {
		case <-r.Context().Done():
		case <-releaseUpstream:
		}
	}))
	t.Cleanup(upstream.Close)
	t.Cleanup(func() { close(releaseUpstream) })

	manager, err := NewManager(nil)
	if err != nil {
		t.Fatal(err)
	}
	config := &ServerConfig{
		ID:         "blocking-http",
		Name:       "Blocking HTTP",
		Transport:  TransportStreamableHTTP,
		Connection: ConnectionConfig{URL: upstream.URL},
	}
	if err := manager.AddServer(config); err != nil {
		t.Fatal(err)
	}
	connectDone := make(chan error, 1)
	go func() {
		connectDone <- manager.Connect(context.Background(), config.ID)
	}()
	select {
	case <-requestStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("MCP connect did not reach upstream")
	}

	closeDone := make(chan struct{})
	go func() {
		manager.Close()
		close(closeDone)
	}()
	select {
	case err := <-connectDone:
		if err == nil {
			t.Fatal("in-flight Connect unexpectedly succeeded after manager close")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("manager close did not cancel in-flight Connect")
	}
	select {
	case <-closeDone:
	case <-time.After(2 * time.Second):
		t.Fatal("manager Close did not finish")
	}
	if err := manager.Connect(context.Background(), config.ID); !errors.Is(err, ErrManagerClosed) {
		t.Fatalf("Connect after Close error = %v, want ErrManagerClosed", err)
	}
}

func containsEnvironmentEntry(environment []string, expected string) bool {
	for _, entry := range environment {
		if entry == expected {
			return true
		}
	}
	return false
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
