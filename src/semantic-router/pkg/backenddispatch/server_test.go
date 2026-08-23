package backenddispatch

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestServerOwnsPrivateListenerLifecycle(t *testing.T) {
	server, testServerOwnsPrivateListenerLifecycleErr := NewServer(ServerOptions{
		BindAddress: "127.0.0.1",
		Handler: http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
			writer.WriteHeader(http.StatusNoContent)
		}),
	})
	if testServerOwnsPrivateListenerLifecycleErr != nil {
		t.Fatal(testServerOwnsPrivateListenerLifecycleErr)
	}
	if err := server.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := server.Ready(); err != nil {
		t.Fatal(err)
	}
	response, testServerOwnsPrivateListenerLifecycleErr := http.Get("http://" + server.Address() + "/v1/chat/completions")
	if testServerOwnsPrivateListenerLifecycleErr != nil {
		t.Fatal(testServerOwnsPrivateListenerLifecycleErr)
	}
	_, _ = io.Copy(io.Discard, response.Body)
	_ = response.Body.Close()
	if response.StatusCode != http.StatusNoContent {
		t.Fatalf("status = %d", response.StatusCode)
	}
	if err := server.Close(); err != nil {
		t.Fatal(err)
	}
	if err := server.Ready(); err == nil || !strings.Contains(err.Error(), "closed") {
		t.Fatalf("Ready() after Close = %v", err)
	}
}

func TestServerRejectsInvalidCompositionAndDoubleStart(t *testing.T) {
	if _, err := NewServer(ServerOptions{BindAddress: "localhost", Handler: http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})}); err == nil {
		t.Fatal("hostname bind unexpectedly accepted")
	}
	server, err := NewServer(ServerOptions{
		BindAddress: "127.0.0.1", ShutdownTimeout: time.Second,
		Handler: http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}),
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := server.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := server.Start(context.Background()); err == nil {
		t.Fatal("second Start unexpectedly succeeded")
	}
	if err := server.Close(); err != nil {
		t.Fatal(err)
	}
	if err := server.Close(); err != nil {
		t.Fatal(err)
	}
}
