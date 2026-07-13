package proxy

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

func TestWebSocketProxyClosesBothConnectionsWhenRequestContextIsCanceled(t *testing.T) {
	upstreamReady := make(chan error, 1)
	upstreamClosed := make(chan error, 1)
	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			upstreamReady <- err
			return
		}
		defer conn.Close()
		upstreamReady <- nil
		_, _, err = conn.ReadMessage()
		upstreamClosed <- err
	}))
	defer upstream.Close()

	proxyHandler, err := NewWebSocketAwareHandler(upstream.URL, "/embedded")
	if err != nil {
		t.Fatalf("NewWebSocketAwareHandler: %v", err)
	}
	cancelReady := make(chan context.CancelFunc, 1)
	proxyReturned := make(chan struct{}, 1)
	proxyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithCancel(r.Context())
		cancelReady <- cancel
		proxyHandler.ServeHTTP(w, r.WithContext(ctx))
		proxyReturned <- struct{}{}
	}))
	defer proxyServer.Close()

	conn, response, err := websocket.DefaultDialer.Dial(
		"ws"+strings.TrimPrefix(proxyServer.URL, "http")+"/embedded/socket",
		nil,
	)
	if response != nil && response.Body != nil {
		defer response.Body.Close()
	}
	if err != nil {
		t.Fatalf("WebSocket dial: %v", err)
	}
	defer conn.Close()
	if err := <-upstreamReady; err != nil {
		t.Fatalf("upstream upgrade: %v", err)
	}
	cancel := <-cancelReady
	defer cancel()

	cancel()
	_ = conn.SetReadDeadline(time.Now().Add(time.Second))
	_, _, clientErr := conn.ReadMessage()
	if clientErr == nil {
		t.Fatal("client WebSocket remained open after request context cancellation")
	}
	var timeoutErr net.Error
	if errors.As(clientErr, &timeoutErr) && timeoutErr.Timeout() {
		t.Fatalf("client WebSocket timed out instead of closing: %v", clientErr)
	}

	select {
	case upstreamErr := <-upstreamClosed:
		if upstreamErr == nil {
			t.Fatal("upstream WebSocket read completed without connection closure")
		}
	case <-time.After(time.Second):
		t.Fatal("upstream WebSocket remained open after request context cancellation")
	}
	select {
	case <-proxyReturned:
	case <-time.After(time.Second):
		t.Fatal("proxy handler did not return after request context cancellation")
	}
}
