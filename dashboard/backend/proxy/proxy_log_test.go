package proxy

import (
	"bytes"
	"errors"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

type proxyRoundTripperFunc func(*http.Request) (*http.Response, error)

func (fn proxyRoundTripperFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return fn(request)
}

func TestReverseProxyLogsExcludeQueriesHeadersAndErrors(t *testing.T) {
	handler, err := NewReverseProxy(
		"http://upstream.example/base?targetToken=target-query-secret",
		"/embedded/test",
		false,
	)
	if err != nil {
		t.Fatalf("NewReverseProxy: %v", err)
	}
	handler.Transport = proxyRoundTripperFunc(func(*http.Request) (*http.Response, error) {
		return nil, errors.New("transport-error-secret")
	})
	request := httptest.NewRequest(
		http.MethodGet,
		"http://dashboard.example/embedded/test/resource?safe=safe-query-value&authToken=dashboard-query-secret",
		nil,
	)
	request.Header.Set("Authorization", "Bearer dashboard-header-secret")
	request.Header.Set("Referer", "https://dashboard.example/?authToken=referer-secret")
	request.Header.Set("Cookie", "vsr_session=cookie-secret")
	request.Header.Set("Access-Control-Request-Private-Network", "pna-header-secret")
	recorder := httptest.NewRecorder()

	logs := captureProxyLogs(func() {
		handler.ServeHTTP(recorder, request)
	})
	if recorder.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusBadGateway)
	}
	assertProxyLogsExcludeCredentials(t, logs, []string{
		"target-query-secret",
		"safe-query-value",
		"dashboard-query-secret",
		"dashboard-header-secret",
		"referer-secret",
		"cookie-secret",
		"pna-header-secret",
		"transport-error-secret",
		"authToken",
	})
	for _, safeField := range []string{
		`method="GET"`,
		`target="http://upstream.example"`,
		`path="/base/resource"`,
	} {
		if !strings.Contains(logs, safeField) {
			t.Fatalf("proxy logs = %q, want safe field %q", logs, safeField)
		}
	}
}

func TestWebSocketProxyLogsExcludeQueriesAndCredentials(t *testing.T) {
	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		messageType, message, err := conn.ReadMessage()
		if err == nil {
			_ = conn.WriteMessage(messageType, message)
		}
	}))
	defer upstream.Close()

	proxyHandler, err := NewWebSocketAwareHandlerWithHeaders(
		upstream.URL,
		"/embedded",
		map[string]string{
			"Authorization":    "Bearer upstream-header-secret",
			"X-OpenClaw-Token": "upstream-token-secret",
		},
	)
	if err != nil {
		t.Fatalf("NewWebSocketAwareHandlerWithHeaders: %v", err)
	}
	proxyReturned := make(chan struct{}, 1)
	proxyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		proxyHandler.ServeHTTP(w, r)
		proxyReturned <- struct{}{}
	}))
	defer proxyServer.Close()

	logs := captureProxyLogs(func() {
		headers := http.Header{}
		headers.Set("Authorization", "Bearer dashboard-header-secret")
		headers.Set("Referer", "https://dashboard.example/?authToken=referer-secret")
		conn, response, err := websocket.DefaultDialer.Dial(
			"ws"+strings.TrimPrefix(proxyServer.URL, "http")+
				"/embedded/socket?roomId=room-query-secret&authToken=dashboard-query-secret",
			headers,
		)
		if response != nil && response.Body != nil {
			defer response.Body.Close()
		}
		if err != nil {
			t.Fatalf("WebSocket dial: %v", err)
		}
		if err := conn.WriteMessage(websocket.TextMessage, []byte("ping")); err != nil {
			_ = conn.Close()
			t.Fatalf("WebSocket write: %v", err)
		}
		if _, _, err := conn.ReadMessage(); err != nil {
			_ = conn.Close()
			t.Fatalf("WebSocket read: %v", err)
		}
		_ = conn.Close()
		select {
		case <-proxyReturned:
		case <-time.After(time.Second):
			t.Fatal("proxy handler did not return after WebSocket close")
		}
	})
	assertProxyLogsExcludeCredentials(t, logs, []string{
		"upstream-header-secret",
		"upstream-token-secret",
		"dashboard-header-secret",
		"referer-secret",
		"room-query-secret",
		"dashboard-query-secret",
		"authToken",
		"roomId=",
	})
	if !strings.Contains(logs, `method="GET"`) ||
		!strings.Contains(logs, `path="/socket"`) {
		t.Fatalf("WebSocket proxy logs lack safe request metadata: %q", logs)
	}
}

func captureProxyLogs(run func()) string {
	var output bytes.Buffer
	originalWriter := log.Writer()
	originalFlags := log.Flags()
	originalPrefix := log.Prefix()
	log.SetOutput(&output)
	log.SetFlags(0)
	log.SetPrefix("")
	defer func() {
		log.SetOutput(originalWriter)
		log.SetFlags(originalFlags)
		log.SetPrefix(originalPrefix)
	}()

	run()
	return output.String()
}

func assertProxyLogsExcludeCredentials(t *testing.T, logs string, forbidden []string) {
	t.Helper()
	for _, value := range forbidden {
		if strings.Contains(logs, value) {
			t.Fatalf("proxy logs leaked %q: %s", value, logs)
		}
	}
}
