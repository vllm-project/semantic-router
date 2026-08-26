package backenddispatch

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestServerOwnsPrivateListenerLifecycle(t *testing.T) {
	server, testServerOwnsPrivateListenerLifecycleErr := NewServer(ServerOptions{
		BindAddress: "127.0.0.1",
		Readiness:   func(context.Context) error { return nil },
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

func TestServerReadinessFailsClosedWithoutInvokingDispatch(t *testing.T) {
	readyErr := errors.New("routing publication is unavailable")
	dispatchCalls := 0
	server, err := NewServer(ServerOptions{
		BindAddress: "127.0.0.1",
		Readiness:   func(context.Context) error { return readyErr },
		Handler: http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
			dispatchCalls++
			writer.WriteHeader(http.StatusNoContent)
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := server.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer server.Close()

	response, err := http.Get("http://" + server.Address() + "/ready")
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(response.Body)
	_ = response.Body.Close()
	if response.StatusCode != http.StatusServiceUnavailable || dispatchCalls != 0 {
		t.Fatalf("unready response = %d %q, dispatch calls = %d", response.StatusCode, body, dispatchCalls)
	}
	if strings.Contains(string(body), readyErr.Error()) {
		t.Fatalf("readiness response exposed internal error: %s", body)
	}

	readyErr = nil
	response, err = http.Get("http://" + server.Address() + "/ready")
	if err != nil {
		t.Fatal(err)
	}
	_ = response.Body.Close()
	if response.StatusCode != http.StatusOK || dispatchCalls != 0 {
		t.Fatalf("ready response = %d, dispatch calls = %d", response.StatusCode, dispatchCalls)
	}
}

func TestServerRejectsInvalidCompositionAndDoubleStart(t *testing.T) {
	ready := func(context.Context) error { return nil }
	if _, err := NewServer(ServerOptions{BindAddress: "localhost", Readiness: ready, Handler: http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})}); err == nil {
		t.Fatal("hostname bind unexpectedly accepted")
	}
	if _, err := NewServer(ServerOptions{BindAddress: "127.0.0.1", Handler: http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})}); err == nil {
		t.Fatal("missing readiness unexpectedly accepted")
	}
	server, err := NewServer(ServerOptions{
		BindAddress: "127.0.0.1", ShutdownTimeout: time.Second,
		Readiness: ready,
		Handler:   http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}),
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
