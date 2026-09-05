//go:build !windows && cgo

package apiserver

import (
	"context"
	"errors"
	"net"
	"net/http"
	"testing"
	"time"
)

func TestServerShutdownDrainsAcceptedRequest(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	httpServer := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		close(started)
		<-release
		w.WriteHeader(http.StatusNoContent)
	})}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	server := &Server{
		httpServer: httpServer,
		done:       make(chan struct{}),
	}
	go func() {
		server.serveErr = httpServer.Serve(listener)
		close(server.done)
	}()

	requestDone := make(chan error, 1)
	go func() {
		response, err := http.Get("http://" + listener.Addr().String())
		if response != nil {
			_ = response.Body.Close()
		}
		requestDone <- err
	}()
	<-started

	shutdownDone := make(chan error, 1)
	shutdownCtx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	go func() { shutdownDone <- server.Shutdown(shutdownCtx) }()
	select {
	case err := <-shutdownDone:
		t.Fatalf("Shutdown() returned before the accepted request drained: %v", err)
	case <-time.After(25 * time.Millisecond):
	}

	close(release)
	if err := <-requestDone; err != nil {
		t.Fatalf("accepted request failed: %v", err)
	}
	if err := <-shutdownDone; err != nil {
		t.Fatalf("Shutdown() error = %v", err)
	}
}

func TestServerShutdownForcesBoundedRequestCancellation(t *testing.T) {
	started := make(chan struct{})
	cancelled := make(chan struct{})
	httpServer := &http.Server{Handler: http.HandlerFunc(func(_ http.ResponseWriter, request *http.Request) {
		close(started)
		<-request.Context().Done()
		close(cancelled)
	})}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	server := &Server{
		httpServer: httpServer,
		done:       make(chan struct{}),
	}
	go func() {
		server.serveErr = httpServer.Serve(listener)
		close(server.done)
	}()
	go func() {
		response, _ := http.Get("http://" + listener.Addr().String())
		if response != nil {
			_ = response.Body.Close()
		}
	}()
	<-started

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	defer cancel()
	err = server.Shutdown(shutdownCtx)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Shutdown() error = %v, want deadline exceeded", err)
	}
	select {
	case <-cancelled:
	case <-time.After(time.Second):
		t.Fatal("active request was not cancelled after the shutdown deadline")
	}
}
