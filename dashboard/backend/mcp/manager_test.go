package mcp

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

func TestManagerRejectsConnectionsAfterClose(t *testing.T) {
	t.Parallel()

	manager, err := NewManager(nil)
	if err != nil {
		t.Fatal(err)
	}
	config := &ServerConfig{
		ID:        "delayed-server",
		Name:      "Delayed server",
		Transport: TransportStreamableHTTP,
		Connection: ConnectionConfig{
			URL: "https://mcp.example.test",
		},
		Enabled: true,
	}
	if err := manager.AddServer(config); err != nil {
		t.Fatal(err)
	}

	manager.Close()
	manager.Close()
	manager.ConnectEnabled(context.Background())

	if err := manager.Connect(context.Background(), config.ID); !errors.Is(err, errManagerClosed) {
		t.Fatalf("Connect() error = %v, want %v", err, errManagerClosed)
	}
}

func TestManagerCloseWaitsForInFlightConnection(t *testing.T) {
	t.Parallel()

	manager, err := NewManager(nil)
	if err != nil {
		t.Fatal(err)
	}
	config := &ServerConfig{
		ID:        "in-flight-server",
		Name:      "In-flight server",
		Transport: TransportStreamableHTTP,
		Connection: ConnectionConfig{
			URL: "https://mcp.example.test",
		},
	}
	if err := manager.AddServer(config); err != nil {
		t.Fatal(err)
	}

	connectStarted := make(chan struct{})
	cancelObserved := make(chan struct{})
	releaseConnect := make(chan struct{})
	manager.connectClientFn = func(ctx context.Context, _ *Client) error {
		close(connectStarted)
		<-ctx.Done()
		close(cancelObserved)
		<-releaseConnect
		return nil
	}

	connectDone := make(chan error, 1)
	go func() {
		connectDone <- manager.Connect(context.Background(), config.ID)
	}()
	<-connectStarted

	closeDone := make(chan struct{})
	go func() {
		manager.Close()
		close(closeDone)
	}()

	deadline := time.NewTimer(time.Second)
	defer deadline.Stop()
	for {
		manager.mu.RLock()
		closed := manager.closed
		manager.mu.RUnlock()
		if closed {
			break
		}
		select {
		case <-deadline.C:
			t.Fatal("manager did not enter closed state")
		default:
			time.Sleep(time.Millisecond)
		}
	}

	select {
	case <-cancelObserved:
	case <-deadline.C:
		t.Fatal("in-flight connection was not canceled")
	}
	select {
	case <-closeDone:
		t.Fatal("Close returned before in-flight connection finished")
	default:
	}

	close(releaseConnect)
	if err := <-connectDone; !errors.Is(err, errConnectionSuperseded) {
		t.Fatalf("Connect() error = %v, want %v", err, errConnectionSuperseded)
	}
	select {
	case <-closeDone:
	case <-time.After(time.Second):
		t.Fatal("Close did not return after in-flight connection finished")
	}

	manager.mu.RLock()
	_, clientExists := manager.clients[config.ID]
	_, attemptExists := manager.connectAttempts[config.ID]
	manager.mu.RUnlock()
	if clientExists || attemptExists {
		t.Fatal("closed manager retained an in-flight connection")
	}
}

func TestManagerRetiresConnectionsSupersededDuringConnect(t *testing.T) {
	t.Parallel()

	mutations := map[string]func(*testing.T, *Manager, *ServerConfig){
		"disconnect": func(t *testing.T, manager *Manager, config *ServerConfig) {
			t.Helper()
			if err := manager.Disconnect(config.ID); err != nil {
				t.Fatal(err)
			}
		},
		"update": func(t *testing.T, manager *Manager, config *ServerConfig) {
			t.Helper()
			updated := *config
			updated.Name = "Updated"
			if err := manager.UpdateServer(&updated); err != nil {
				t.Fatal(err)
			}
		},
		"upsert": func(t *testing.T, manager *Manager, config *ServerConfig) {
			t.Helper()
			updated := *config
			updated.Name = "Upserted"
			if err := manager.UpsertServer(&updated); err != nil {
				t.Fatal(err)
			}
		},
		"delete": func(t *testing.T, manager *Manager, config *ServerConfig) {
			t.Helper()
			if err := manager.DeleteServer(config.ID); err != nil {
				t.Fatal(err)
			}
		},
	}

	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			manager, err := NewManager(nil)
			if err != nil {
				t.Fatal(err)
			}
			config := &ServerConfig{
				ID:        "concurrent-server",
				Name:      "Original",
				Transport: TransportStreamableHTTP,
				Connection: ConnectionConfig{
					URL: "https://mcp.example.test",
				},
			}
			if err := manager.AddServer(config); err != nil {
				t.Fatal(err)
			}

			connectStarted := make(chan struct{})
			cancelObserved := make(chan struct{})
			releaseConnect := make(chan struct{})
			disconnected := make(chan *Client, 2)
			manager.connectClientFn = func(ctx context.Context, _ *Client) error {
				close(connectStarted)
				<-ctx.Done()
				close(cancelObserved)
				<-releaseConnect
				return nil
			}
			manager.disconnectClientFn = func(client *Client) error {
				disconnected <- client
				return nil
			}

			connectDone := make(chan error, 1)
			go func() {
				connectDone <- manager.Connect(context.Background(), config.ID)
			}()
			<-connectStarted

			mutate(t, manager, config)
			<-cancelObserved
			close(releaseConnect)
			if err := <-connectDone; !errors.Is(err, errConnectionSuperseded) {
				t.Fatalf("Connect() error = %v, want %v", err, errConnectionSuperseded)
			}

			if got := len(disconnected); got != 2 {
				t.Fatalf("stale client disconnect count = %d, want 2", got)
			}
			manager.mu.RLock()
			_, clientExists := manager.clients[config.ID]
			_, attemptExists := manager.connectAttempts[config.ID]
			manager.mu.RUnlock()
			if clientExists || attemptExists {
				t.Fatal("superseded connection remained published")
			}
		})
	}
}

func TestManagerPersistsServerConfigs(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "wf.sqlite")

	store1, err := workflowstore.Open(path)
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

func TestManagerPersistenceFailureDoesNotPublishMemoryChanges(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "wf.sqlite")

	store, err := workflowstore.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	manager, err := NewManager(store)
	if err != nil {
		t.Fatal(err)
	}
	original := &ServerConfig{
		ID:        "transactional-server",
		Name:      "Original",
		Transport: TransportStdio,
		Connection: ConnectionConfig{
			Command: "original-command",
		},
		Enabled: true,
	}
	if addErr := manager.AddServer(original); addErr != nil {
		t.Fatal(addErr)
	}
	if closeErr := store.Close(); closeErr != nil {
		t.Fatal(closeErr)
	}

	updated := *original
	updated.Name = "Updated"
	if updateErr := manager.UpdateServer(&updated); updateErr == nil {
		t.Fatal("update unexpectedly succeeded after persistence was closed")
	}
	assertStoredServerName(t, manager, original.ID, original.Name)

	if upsertErr := manager.UpsertServer(&updated); upsertErr == nil {
		t.Fatal("upsert unexpectedly succeeded after persistence was closed")
	}
	assertStoredServerName(t, manager, original.ID, original.Name)

	if deleteErr := manager.DeleteServer(original.ID); deleteErr == nil {
		t.Fatal("delete unexpectedly succeeded after persistence was closed")
	}
	assertStoredServerName(t, manager, original.ID, original.Name)

	newServer := *original
	newServer.ID = "new-server"
	if addErr := manager.AddServer(&newServer); addErr == nil {
		t.Fatal("add unexpectedly succeeded after persistence was closed")
	}
	if _, ok := manager.GetServer(newServer.ID); ok {
		t.Fatal("failed add was published to memory")
	}
}

func assertStoredServerName(t *testing.T, manager *Manager, id, want string) {
	t.Helper()
	stored, ok := manager.GetServer(id)
	if !ok {
		t.Fatalf("stored server %q missing after persistence failure", id)
	}
	if stored.Name != want {
		t.Fatalf("stored server name = %q, want %q", stored.Name, want)
	}
}
