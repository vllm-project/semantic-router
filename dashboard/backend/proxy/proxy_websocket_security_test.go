package proxy

import (
	"bufio"
	"bytes"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

func TestWebSocketProxyReplacesDashboardCredentialsWithStaticUpstreamAuth(t *testing.T) {
	testWebSocketCredentialIsolation(t, map[string]string{
		"Authorization":    "Bearer upstream-token",
		"X-OpenClaw-Token": "upstream-token",
	}, "Bearer upstream-token")
}

func TestWebSocketProxyDropsDashboardAuthorizationWithoutStaticAuth(t *testing.T) {
	testWebSocketCredentialIsolation(t, nil, "")
}

func TestWebSocketProxyRejectsInvalidStaticHeaders(t *testing.T) {
	for _, headers := range []map[string]string{
		{"Bad Header": "value"},
		{"Authorization": "Bearer safe\r\nX-Injected: yes"},
		{"Authorization": "Bearer safe\x7f"},
		{"Host": "attacker.example"},
		{"Connection": "keep-alive"},
		{"Proxy-Authenticate": "Basic realm=upstream"},
		{"Proxy-Authorization": "Basic cHJveHk6c2VjcmV0"},
		{"Transfer-Encoding": "chunked"},
		{"Sec-WebSocket-Key": "attacker-key"},
	} {
		if _, err := NewWebSocketAwareHandlerWithHeaders("http://127.0.0.1:1", "/embedded", headers); err == nil {
			t.Fatalf("expected invalid static headers %#v to fail", headers)
		}
	}
}

func TestWebSocketUpgradeRequestRegeneratesHopByHopHandshakeHeaders(t *testing.T) {
	r := httptest.NewRequest(http.MethodGet, "http://dashboard.example/embedded/socket", nil)
	r.Header.Set("Connection", "keep-alive, X-Remove, Upgrade")
	r.Header.Set("Upgrade", "websocket")
	r.Header.Set("X-Remove", "attacker-controlled")
	r.Header.Set("X-Preserve", "safe")
	r.Header.Set("Proxy-Authenticate", "Basic realm=dashboard")
	r.Header.Set("Proxy-Authorization", "Basic ZGFzaGJvYXJkOnNlY3JldA==")
	target, err := url.Parse("http://upstream.example")
	if err != nil {
		t.Fatalf("url.Parse: %v", err)
	}

	request, err := buildWebSocketUpgradeRequest(r, target, "/socket", nil)
	if err != nil {
		t.Fatalf("buildWebSocketUpgradeRequest: %v", err)
	}
	serialized := string(request)
	if strings.Count(serialized, "\r\nConnection: Upgrade\r\n") != 1 {
		t.Fatalf("serialized handshake must contain one canonical Connection header: %q", serialized)
	}
	if strings.Count(serialized, "\r\nUpgrade: websocket\r\n") != 1 {
		t.Fatalf("serialized handshake must contain one canonical Upgrade header: %q", serialized)
	}
	if strings.Contains(serialized, "X-Remove:") || strings.Contains(serialized, "keep-alive") {
		t.Fatalf("serialized handshake retained connection-nominated headers: %q", serialized)
	}
	if strings.Contains(serialized, "Proxy-Authenticate:") ||
		strings.Contains(serialized, "Proxy-Authorization:") {
		t.Fatalf("serialized handshake retained proxy credentials: %q", serialized)
	}
	if !strings.Contains(serialized, "\r\nX-Preserve: safe\r\n") {
		t.Fatalf("serialized handshake dropped end-to-end header: %q", serialized)
	}
}

func TestWebSocketUpgradeResponseRegeneratesHopByHopHandshakeHeaders(t *testing.T) {
	r := httptest.NewRequest(http.MethodGet, "http://dashboard.example/embedded/socket", nil)
	rawResponse := "HTTP/1.1 101 Switching Protocols\r\n" +
		"Connection: keep-alive, X-Remove, Upgrade\r\n" +
		"Upgrade: websocket\r\n" +
		"X-Remove: attacker-controlled\r\n" +
		"Proxy-Authenticate: Basic realm=upstream\r\n" +
		"Proxy-Authorization: Basic dXBzdHJlYW06c2VjcmV0\r\n" +
		"Sec-WebSocket-Accept: accepted\r\n" +
		"X-Preserve: safe\r\n\r\n"
	response, err := http.ReadResponse(bufio.NewReader(strings.NewReader(rawResponse)), r)
	if err != nil {
		t.Fatalf("http.ReadResponse: %v", err)
	}
	defer response.Body.Close()
	if !validWebSocketUpgradeResponse(response) {
		t.Fatal("malicious response fixture must remain a syntactically valid upgrade")
	}

	canonicalizeWebSocketUpgradeResponseHeaders(response.Header)
	var forwarded bytes.Buffer
	if err := response.Write(&forwarded); err != nil {
		t.Fatalf("response.Write: %v", err)
	}
	serialized := forwarded.String()
	if strings.Count(serialized, "\r\nConnection: Upgrade\r\n") != 1 ||
		strings.Count(serialized, "\r\nUpgrade: websocket\r\n") != 1 {
		t.Fatalf("forwarded handshake lacks canonical upgrade headers: %q", serialized)
	}
	for _, forbidden := range []string{
		"\r\nX-Remove:",
		"\r\nProxy-Authenticate:",
		"\r\nProxy-Authorization:",
		"keep-alive",
	} {
		if strings.Contains(serialized, forbidden) {
			t.Fatalf("forwarded handshake retained %q: %q", forbidden, serialized)
		}
	}
	if !strings.Contains(serialized, "\r\nSec-Websocket-Accept: accepted\r\n") ||
		!strings.Contains(serialized, "\r\nX-Preserve: safe\r\n") {
		t.Fatalf("forwarded handshake dropped end-to-end WebSocket headers: %q", serialized)
	}
}

func TestWebSocketProxyRejectsEncodedRequestLineControls(t *testing.T) {
	var upstreamRequests atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		upstreamRequests.Add(1)
	}))
	defer upstream.Close()

	handler, err := NewWebSocketAwareHandlerWithHeaders(upstream.URL, "/embedded", nil)
	if err != nil {
		t.Fatalf("NewWebSocketAwareHandlerWithHeaders: %v", err)
	}
	proxyServer := httptest.NewServer(handler)
	defer proxyServer.Close()

	wsURL := "ws" + strings.TrimPrefix(proxyServer.URL, "http") +
		"/embedded/socket%0d%0aX-Injected:%20yes"
	_, response, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if response != nil && response.Body != nil {
		defer response.Body.Close()
	}
	if err == nil {
		t.Fatal("expected encoded request-line controls to be rejected")
	}
	if response == nil || response.StatusCode != http.StatusBadRequest {
		t.Fatalf("response = %#v, want HTTP %d", response, http.StatusBadRequest)
	}
	if got := upstreamRequests.Load(); got != 0 {
		t.Fatalf("upstream requests = %d, want 0", got)
	}
}

func TestWebSocketProxyRejectsQueryDecodedDynamicHeaderControls(t *testing.T) {
	var upstreamRequests atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		upstreamRequests.Add(1)
	}))
	defer upstream.Close()

	proxyHandler, err := NewWebSocketAwareHandlerWithHeaders(upstream.URL, "/embedded", nil)
	if err != nil {
		t.Fatalf("NewWebSocketAwareHandlerWithHeaders: %v", err)
	}
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r.Header.Set("X-ClawOS-Room-Id", r.URL.Query().Get("roomId"))
		proxyHandler.ServeHTTP(w, r)
	})
	proxyServer := httptest.NewServer(handler)
	defer proxyServer.Close()

	wsURL := "ws" + strings.TrimPrefix(proxyServer.URL, "http") +
		"/embedded/socket?roomId=room%0d%0aX-Injected:%20yes"
	_, response, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if response != nil && response.Body != nil {
		defer response.Body.Close()
	}
	if err == nil {
		t.Fatal("expected query-decoded dynamic header controls to be rejected")
	}
	if response == nil || response.StatusCode != http.StatusBadRequest {
		t.Fatalf("response = %#v, want HTTP %d", response, http.StatusBadRequest)
	}
	if got := upstreamRequests.Load(); got != 0 {
		t.Fatalf("upstream requests = %d, want 0", got)
	}
}

func TestWebSocketUpgradeRequestEscapesPathAndStripsDashboardAuthQuery(t *testing.T) {
	r := httptest.NewRequest(
		http.MethodGet,
		"http://dashboard.example/embedded/socket%20room?roomId=r&auth%54oken=secret",
		nil,
	)
	target, err := url.Parse("http://upstream.example")
	if err != nil {
		t.Fatalf("url.Parse: %v", err)
	}

	request, err := buildWebSocketUpgradeRequest(
		r,
		target,
		stripProxyPath(r.URL.Path, "/embedded"),
		nil,
	)
	if err != nil {
		t.Fatalf("buildWebSocketUpgradeRequest: %v", err)
	}
	if got, want := strings.SplitN(string(request), "\r\n", 2)[0],
		"GET /socket%20room?roomId=r HTTP/1.1"; got != want {
		t.Fatalf("request line = %q, want %q", got, want)
	}
}

func TestWebSocketUpgradeRequestRejectsDecodedDynamicHeaderControls(t *testing.T) {
	r := httptest.NewRequest(http.MethodGet, "http://dashboard.example/embedded/socket", nil)
	r.Header.Set("Connection", "Upgrade")
	r.Header.Set("Upgrade", "websocket")
	r.Header["X-ClawOS-Room-Id"] = []string{"room\r\nX-Injected: yes"}
	target, err := url.Parse("http://upstream.example")
	if err != nil {
		t.Fatalf("url.Parse: %v", err)
	}

	if _, err := buildWebSocketUpgradeRequest(r, target, "/socket", nil); err == nil {
		t.Fatal("expected decoded dynamic header controls to be rejected")
	}
}

func TestWebSocketProxyBoundsUpstreamHandshake(t *testing.T) {
	release := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		<-release
	}))
	defer func() {
		close(release)
		upstream.Close()
	}()

	handler, err := newWebSocketAwareHandlerWithHeaders(
		upstream.URL,
		"/embedded",
		nil,
		50*time.Millisecond,
	)
	if err != nil {
		t.Fatalf("newWebSocketAwareHandlerWithHeaders: %v", err)
	}
	proxyServer := httptest.NewServer(handler)
	defer proxyServer.Close()

	started := time.Now()
	_, response, err := websocket.DefaultDialer.Dial(
		"ws"+strings.TrimPrefix(proxyServer.URL, "http")+"/embedded/socket",
		nil,
	)
	if response != nil && response.Body != nil {
		_ = response.Body.Close()
	}
	if err == nil {
		t.Fatal("expected stalled upstream handshake to fail")
	}
	if elapsed := time.Since(started); elapsed > time.Second {
		t.Fatalf("stalled handshake returned after %s, want at most 1s", elapsed)
	}
}

func testWebSocketCredentialIsolation(t *testing.T, staticHeaders map[string]string, wantAuthorization string) {
	t.Helper()
	captured := make(chan capturedProxyRequest, 1)
	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured <- captureProxyRequest(r)
		responseHeaders := http.Header{}
		responseHeaders.Add("Set-Cookie", "vsr_session=attacker; Path=/; HttpOnly")
		responseHeaders.Add("Set-Cookie", "openclaw_session=o; Path=/; HttpOnly")
		conn, err := upgrader.Upgrade(w, r, responseHeaders)
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

	handler, err := NewWebSocketAwareHandlerWithHeaders(
		upstream.URL,
		"/embedded/openclaw/demo",
		staticHeaders,
	)
	if err != nil {
		t.Fatalf("NewWebSocketAwareHandlerWithHeaders: %v", err)
	}
	proxyServer := httptest.NewServer(handler)
	defer proxyServer.Close()

	headers := http.Header{}
	headers.Set("Authorization", "Bearer dashboard-jwt")
	headers.Add("Cookie", "vsr_session=first; openclaw_session=o")
	headers.Add("Cookie", "theme=d; vsr_session=second")
	wsURL := "ws" + strings.TrimPrefix(proxyServer.URL, "http") +
		"/embedded/openclaw/demo/socket?roomId=r&auth%54oken=JWT"
	conn, response, err := websocket.DefaultDialer.Dial(wsURL, headers)
	if response != nil && response.Body != nil {
		defer response.Body.Close()
	}
	if err != nil {
		t.Fatalf("WebSocket dial: %v", err)
	}
	defer conn.Close()

	got := <-captured
	assertSanitizedProxyRequest(t, got, "/socket", "roomId=r", "openclaw_session=o", "theme=d")
	if got.authorization != wantAuthorization {
		t.Fatalf("upstream Authorization = %q, want %q", got.authorization, wantAuthorization)
	}
	wantOpenClawToken := ""
	if staticHeaders != nil {
		wantOpenClawToken = "upstream-token"
	}
	if got.openClawToken != wantOpenClawToken {
		t.Fatalf("upstream X-OpenClaw-Token = %q, want %q", got.openClawToken, wantOpenClawToken)
	}
	setCookies := response.Header.Values("Set-Cookie")
	if len(setCookies) != 1 || !strings.HasPrefix(setCookies[0], "openclaw_session=o;") {
		t.Fatalf("WebSocket Set-Cookie headers = %#v", setCookies)
	}

	deadline := time.Now().Add(2 * time.Second)
	_ = conn.SetReadDeadline(deadline)
	_ = conn.SetWriteDeadline(deadline)
	if writeErr := conn.WriteMessage(websocket.TextMessage, []byte("ping")); writeErr != nil {
		t.Fatalf("WebSocket write: %v", writeErr)
	}
	_, message, err := conn.ReadMessage()
	if err != nil {
		t.Fatalf("WebSocket read: %v", err)
	}
	if string(message) != "ping" {
		t.Fatalf("WebSocket echo = %q, want ping", message)
	}
}
