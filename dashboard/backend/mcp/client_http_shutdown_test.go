package mcp

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	sdkclient "github.com/mark3labs/mcp-go/client"
)

// closeProbeClient makes a concurrent Close observable while delegating the
// actual cleanup to the SDK Streamable HTTP client.
type closeProbeClient struct {
	sdkclient.MCPClient

	started chan struct{}
	release chan struct{}
	count   chan struct{}

	once   sync.Once
	mu     sync.Mutex
	active bool
}

func (c *closeProbeClient) Close() error {
	c.mu.Lock()
	if c.active {
		c.mu.Unlock()
		return errors.New("concurrent transport cleanup")
	}
	c.active = true
	c.mu.Unlock()

	c.once.Do(func() { close(c.started) })
	c.count <- struct{}{}
	<-c.release

	c.mu.Lock()
	c.active = false
	c.mu.Unlock()
	return c.MCPClient.Close()
}

func TestClientSerializesConcurrentTransportCleanup(t *testing.T) {
	transportClient, err := sdkclient.NewStreamableHttpClient("http://mcp.example.test")
	if err != nil {
		t.Fatal(err)
	}
	probe := &closeProbeClient{
		MCPClient: transportClient,
		started:   make(chan struct{}),
		release:   make(chan struct{}),
		count:     make(chan struct{}, 2),
	}
	instance := &Client{}

	firstDone := make(chan error, 1)
	go func() { firstDone <- instance.closeMCPClient(probe) }()
	select {
	case <-probe.started:
	case <-time.After(time.Second):
		t.Fatal("first cleanup did not start")
	}
	<-probe.count

	secondDone := make(chan error, 1)
	go func() { secondDone <- instance.closeMCPClient(probe) }()
	select {
	case <-probe.count:
		t.Fatal("second cleanup entered the transport before the first finished")
	case <-time.After(20 * time.Millisecond):
	}

	close(probe.release)
	if err := <-firstDone; err != nil {
		t.Fatalf("first cleanup: %v", err)
	}
	if err := <-secondDone; err != nil {
		t.Fatalf("second cleanup: %v", err)
	}
}

func TestManagerHTTPInitializationShutdownDoesNotDoubleClose(t *testing.T) {
	initializing := make(chan struct{})
	releaseHandler := make(chan struct{})
	var initializingOnce sync.Once
	httpServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		initializingOnce.Do(func() { close(initializing) })
		select {
		case <-r.Context().Done():
		case <-releaseHandler:
		}
	}))
	t.Cleanup(httpServer.Close)
	t.Cleanup(func() { close(releaseHandler) })

	manager, err := NewManager(nil)
	if err != nil {
		t.Fatal(err)
	}
	config := &ServerConfig{
		ID:        "http-initialization-shutdown",
		Transport: TransportStreamableHTTP,
		Connection: ConnectionConfig{
			URL: httpServer.URL,
		},
	}
	if err := manager.AddServer(config); err != nil {
		t.Fatal(err)
	}

	connectDone := make(chan error, 1)
	go func() { connectDone <- manager.Connect(context.Background(), config.ID) }()
	select {
	case <-initializing:
	case <-time.After(3 * time.Second):
		t.Fatal("HTTP initialization did not start")
	}

	closeDone := make(chan struct{})
	go func() {
		manager.Close()
		close(closeDone)
	}()

	select {
	case err := <-connectDone:
		if err == nil {
			t.Fatal("Connect() unexpectedly succeeded after shutdown")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("Connect() did not finish after shutdown")
	}
	select {
	case <-closeDone:
	case <-time.After(3 * time.Second):
		t.Fatal("Manager.Close() did not finish after initialization cancellation")
	}
}
