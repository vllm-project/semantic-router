package mcp

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

func TestManagerRejectsConnectionTestsAfterClose(t *testing.T) {
	manager, err := NewManager(nil)
	if err != nil {
		t.Fatal(err)
	}
	manager.Close()
	if testErr := manager.TestConnection(context.Background(), &ServerConfig{ID: "closed"}); !errors.Is(testErr, errManagerClosed) {
		t.Fatalf("TestConnection() error = %v, want %v", testErr, errManagerClosed)
	}
}

func TestManagerCloseWaitsForConnectionTests(t *testing.T) {
	manager, err := NewManager(nil)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	started := make(chan struct{}, 2)
	canceled := make(chan struct{}, 2)
	releaseCleanup := make(chan struct{})
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(releaseCleanup) }) }
	t.Cleanup(func() {
		cancel()
		release()
		manager.Close()
	})
	manager.connectClientFn = func(connectCtx context.Context, _ *Client) error {
		started <- struct{}{}
		<-connectCtx.Done()
		canceled <- struct{}{}
		return connectCtx.Err()
	}
	manager.disconnectClientFn = func(_ *Client) error {
		<-releaseCleanup
		return nil
	}
	done := make(chan error, 2)
	for range 2 {
		go func() { done <- manager.TestConnection(ctx, &ServerConfig{ID: "same-server"}) }()
	}
	for range 2 {
		select {
		case <-started:
		case <-time.After(3 * time.Second):
			t.Fatal("connection test did not start")
		}
	}
	manager.mu.RLock()
	published, temporary := len(manager.clients), len(manager.testConnections)
	manager.mu.RUnlock()
	if published != 0 || temporary != 2 {
		t.Fatalf("connection tests changed published clients or collided: published=%d, temporary=%d", published, temporary)
	}
	closed := make(chan struct{})
	go func() { manager.Close(); close(closed) }()
	for range 2 {
		select {
		case <-canceled:
		case <-time.After(3 * time.Second):
			t.Fatal("Close did not cancel both connection tests")
		}
	}
	select {
	case <-closed:
		t.Fatal("Close returned before temporary client cleanup")
	default:
	}
	release()
	for range 2 {
		if testErr := <-done; !errors.Is(testErr, context.Canceled) {
			t.Errorf("TestConnection() error = %v, want cancellation", testErr)
		}
	}
	select {
	case <-closed:
	case <-time.After(3 * time.Second):
		t.Fatal("Close did not join connection test cleanup")
	}
	if len(manager.testConnections) != 0 {
		t.Fatal("closed manager retained temporary clients")
	}
}

func TestManagerConnectionTestPreservesPublishedClient(t *testing.T) {
	manager, err := NewManager(nil)
	if err != nil {
		t.Fatal(err)
	}
	config := &ServerConfig{ID: "same-server"}
	published, err := NewClient(config)
	if err != nil {
		t.Fatal(err)
	}
	manager.clients[config.ID] = published
	manager.connectClientFn = func(_ context.Context, client *Client) error {
		if client == published {
			t.Fatal("connection test reused the published client")
		}
		return nil
	}
	manager.disconnectClientFn = func(client *Client) error {
		if client == published {
			t.Error("connection test disconnected the published client")
		}
		return nil
	}
	if testErr := manager.TestConnection(context.Background(), config); testErr != nil {
		t.Fatal(testErr)
	}
	if manager.clients[config.ID] != published || len(manager.testConnections) != 0 {
		t.Fatal("connection test changed published clients or retained a temporary client")
	}
}
