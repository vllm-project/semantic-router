package proxy

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestWebSocketProxyClosesBothConnectionsOnRequestCancellation(t *testing.T) {
	backend, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen for backend: %v", err)
	}
	t.Cleanup(func() { _ = backend.Close() })

	type observedRequest struct {
		path          string
		authorization string
		err           error
	}
	requestSeen := make(chan observedRequest, 1)
	backendClosed := make(chan error, 1)
	go func() {
		conn, acceptErr := backend.Accept()
		if acceptErr != nil {
			requestSeen <- observedRequest{err: acceptErr}
			return
		}
		defer conn.Close()
		reader := bufio.NewReader(conn)
		request, readErr := http.ReadRequest(reader)
		if readErr != nil {
			requestSeen <- observedRequest{err: readErr}
			return
		}
		_ = request.Body.Close()
		if _, writeErr := io.WriteString(conn, "HTTP/1.1 101 Switching Protocols\r\nConnection: Upgrade\r\nUpgrade: websocket\r\n\r\n"); writeErr != nil {
			requestSeen <- observedRequest{err: writeErr}
			return
		}
		requestSeen <- observedRequest{path: request.URL.Path, authorization: request.Header.Get("Authorization")}
		_, readErr = reader.ReadByte()
		backendClosed <- readErr
	}()

	proxyHandler, err := NewWebSocketAwareHandlerWithHeaders(
		"http://"+backend.Addr().String(),
		"/embedded/openclaw/worker",
		map[string]string{"Authorization": "Bearer server-token"},
	)
	if err != nil {
		t.Fatalf("create proxy handler: %v", err)
	}
	cancelRequest := make(chan context.CancelFunc, 1)
	handlerDone := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer close(handlerDone)
		ctx, cancel := context.WithCancel(r.Context())
		defer cancel()
		cancelRequest <- cancel
		proxyHandler.ServeHTTP(w, r.WithContext(ctx))
	}))
	t.Cleanup(server.Close)

	client, err := net.DialTimeout("tcp", strings.TrimPrefix(server.URL, "http://"), 2*time.Second)
	if err != nil {
		t.Fatalf("connect to proxy: %v", err)
	}
	defer client.Close()
	_ = client.SetDeadline(time.Now().Add(2 * time.Second))
	if _, err := fmt.Fprintf(client,
		"GET /embedded/openclaw/worker/ws HTTP/1.1\r\nHost: %s\r\nConnection: Upgrade\r\nUpgrade: websocket\r\n\r\n",
		strings.TrimPrefix(server.URL, "http://"),
	); err != nil {
		t.Fatalf("write upgrade request: %v", err)
	}
	clientReader := bufio.NewReader(client)
	response, err := http.ReadResponse(clientReader, &http.Request{Method: http.MethodGet})
	if err != nil {
		t.Fatalf("read upgrade response: %v", err)
	}
	if response.StatusCode != http.StatusSwitchingProtocols {
		t.Fatalf("upgrade status = %d", response.StatusCode)
	}

	select {
	case observed := <-requestSeen:
		if observed.err != nil {
			t.Fatalf("backend handshake: %v", observed.err)
		}
		if observed.path != "/ws" || observed.authorization != "Bearer server-token" {
			t.Fatalf("backend received path %q and authorization %q", observed.path, observed.authorization)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("backend did not receive upgrade request")
	}

	select {
	case cancel := <-cancelRequest:
		cancel()
	case <-time.After(2 * time.Second):
		t.Fatal("proxy request cancel function was not captured")
	}
	select {
	case <-handlerDone:
	case <-time.After(2 * time.Second):
		t.Fatal("proxy handler remained blocked after cancellation")
	}
	select {
	case closeErr := <-backendClosed:
		if !errors.Is(closeErr, io.EOF) {
			t.Fatalf("backend connection did not close cleanly: %v", closeErr)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("backend connection remained open after cancellation")
	}
	_ = client.SetReadDeadline(time.Now().Add(2 * time.Second))
	if _, readErr := clientReader.ReadByte(); !errors.Is(readErr, io.EOF) {
		t.Fatalf("client connection did not close cleanly: %v", readErr)
	}
}
