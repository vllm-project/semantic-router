package mcp

import (
	"context"
	"errors"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

func TestManagerReportsMissingServerWithErrServerNotFound(t *testing.T) {
	t.Parallel()
	manager, err := NewManager(nil)
	if err != nil {
		t.Fatal(err)
	}
	const id = "missing-server"

	calls := map[string]func() error{
		"update": func() error { return manager.UpdateServer(&ServerConfig{ID: id}) },
		"delete": func() error { return manager.DeleteServer(id) },
		"status": func() error {
			_, statusErr := manager.GetServerStatus(id)
			return statusErr
		},
		"connect":    func() error { return manager.Connect(context.Background(), id) },
		"disconnect": func() error { return manager.Disconnect(id) },
	}
	for name, call := range calls {
		t.Run(name, func(t *testing.T) {
			got := call()
			if !errors.Is(got, ErrServerNotFound) {
				t.Fatalf("error = %v, want ErrServerNotFound", got)
			}
			if want := "server with ID missing-server not found"; got.Error() != want {
				t.Fatalf("error text = %q, want %q", got.Error(), want)
			}
		})
	}
}

func TestManagerStorageFailureIsNotReportedAsMissingServer(t *testing.T) {
	t.Parallel()
	store, err := workflowstore.Open(filepath.Join(t.TempDir(), "wf.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	manager, err := NewManager(store)
	if err != nil {
		t.Fatal(err)
	}
	config := &ServerConfig{
		ID:         "stored-server",
		Name:       "Stored server",
		Transport:  TransportStdio,
		Connection: ConnectionConfig{Command: "echo"},
	}
	if addErr := manager.AddServer(config); addErr != nil {
		t.Fatal(addErr)
	}
	if closeErr := store.Close(); closeErr != nil {
		t.Fatal(closeErr)
	}

	if updateErr := manager.UpdateServer(config); updateErr == nil || errors.Is(updateErr, ErrServerNotFound) {
		t.Fatalf("UpdateServer() error = %v, want a storage failure", updateErr)
	}
	if deleteErr := manager.DeleteServer(config.ID); deleteErr == nil || errors.Is(deleteErr, ErrServerNotFound) {
		t.Fatalf("DeleteServer() error = %v, want a storage failure", deleteErr)
	}
}

func TestManagerDisconnectSucceedsForServerWithoutClient(t *testing.T) {
	t.Parallel()
	manager, err := NewManager(nil)
	if err != nil {
		t.Fatal(err)
	}
	config := &ServerConfig{
		ID:         "idle-server",
		Name:       "Idle server",
		Transport:  TransportStdio,
		Connection: ConnectionConfig{Command: "echo"},
	}
	if addErr := manager.AddServer(config); addErr != nil {
		t.Fatal(addErr)
	}

	if disconnectErr := manager.Disconnect(config.ID); disconnectErr != nil {
		t.Fatalf("Disconnect() error = %v, want nil", disconnectErr)
	}
}
