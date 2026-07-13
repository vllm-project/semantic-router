package proxy

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

const (
	webSocketHandshakeTimeout   = 10 * time.Second
	webSocketHandshakeByteLimit = 64 * 1024
)

// NewWebSocketAwareHandler returns an http.Handler that proxies both regular HTTP
// and WebSocket upgrade requests to the target. This is required for services
// like OpenClaw whose control UI uses WebSocket for real-time communication.
func NewWebSocketAwareHandler(targetBase, stripPrefix string) (http.Handler, error) {
	return NewWebSocketAwareHandlerWithHeaders(targetBase, stripPrefix, nil)
}

// NewWebSocketAwareHandlerWithHeaders behaves like NewWebSocketAwareHandler but
// also injects static headers into all proxied HTTP and WebSocket upgrade requests.
func NewWebSocketAwareHandlerWithHeaders(targetBase, stripPrefix string, staticHeaders map[string]string) (http.Handler, error) {
	return newWebSocketAwareHandlerWithHeaders(
		targetBase,
		stripPrefix,
		staticHeaders,
		webSocketHandshakeTimeout,
	)
}

func newWebSocketAwareHandlerWithHeaders(
	targetBase string,
	stripPrefix string,
	staticHeaders map[string]string,
	handshakeTimeout time.Duration,
) (http.Handler, error) {
	targetURL, err := url.Parse(targetBase)
	if err != nil {
		return nil, err
	}

	effectiveStaticHeaders, err := normalizeStaticHeaders(staticHeaders)
	if err != nil {
		return nil, err
	}

	httpProxy, err := NewReverseProxy(targetBase, stripPrefix, false)
	if err != nil {
		return nil, err
	}
	if len(effectiveStaticHeaders) > 0 {
		origDirector := httpProxy.Director
		httpProxy.Director = func(r *http.Request) {
			origDirector(r)
			for key, value := range effectiveStaticHeaders {
				// Static headers are authoritative in embedded mode to avoid stale
				// client-side gateway tokens causing auth mismatch loops.
				r.Header.Set(key, value)
			}
		}
	}
	origModify := httpProxy.ModifyResponse
	httpProxy.ModifyResponse = func(resp *http.Response) error {
		if origModify != nil {
			if err := origModify(resp); err != nil {
				return err
			}
		}

		// Rewrite control-ui-config.json to set basePath for embedded mode.
		if strings.HasSuffix(resp.Request.URL.Path, "/__openclaw/control-ui-config.json") {
			body, err := io.ReadAll(resp.Body)
			if err != nil {
				return err
			}
			_ = resp.Body.Close()

			var cfg map[string]interface{}
			if err := json.Unmarshal(body, &cfg); err == nil {
				embeddedBasePath := strings.TrimRight(strings.TrimSpace(stripPrefix), "/")
				if embeddedBasePath == "" {
					embeddedBasePath = "/"
				}

				// Always force embedded basePath. OpenClaw may default basePath to "/",
				// which breaks WebSocket routing when loaded behind /embedded/openclaw/{name}/.
				cfg["basePath"] = embeddedBasePath

				// Provide a relative gateway URL so Control UI resolves WebSocket requests
				// through the embedded proxy path (no host/port exposure in client config).
				cfg["gatewayUrl"] = embeddedBasePath
				updated, err := json.Marshal(cfg)
				if err == nil {
					resp.Body = io.NopCloser(bytes.NewReader(updated))
					resp.ContentLength = int64(len(updated))
					resp.Header.Set("Content-Length", strconv.Itoa(len(updated)))
					resp.Header.Set("Content-Type", "application/json; charset=utf-8")
				} else {
					resp.Body = io.NopCloser(bytes.NewReader(body))
				}
			} else {
				resp.Body = io.NopCloser(bytes.NewReader(body))
			}
			return nil
		}
		return nil
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if isWebSocketUpgrade(r) {
			proxyWebSocket(w, r, targetURL, stripPrefix, effectiveStaticHeaders, handshakeTimeout)
			return
		}
		httpProxy.ServeHTTP(w, r)
	}), nil
}

func normalizeStaticHeaders(staticHeaders map[string]string) (map[string]string, error) {
	normalized := make(map[string]string, len(staticHeaders))
	for key, value := range staticHeaders {
		key = strings.TrimSpace(key)
		value = strings.TrimSpace(value)
		if key == "" || value == "" {
			continue
		}
		if !validHTTPHeaderName(key) || !validHTTPHeaderValue(value) {
			return nil, fmt.Errorf("invalid static proxy header")
		}
		normalized[http.CanonicalHeaderKey(key)] = value
	}
	return normalized, nil
}

func validHTTPHeaderName(name string) bool {
	for i := 0; i < len(name); i++ {
		c := name[i]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') {
			continue
		}
		switch c {
		case '!', '#', '$', '%', '&', '\'', '*', '+', '-', '.', '^', '_', '`', '|', '~':
			continue
		default:
			return false
		}
	}
	return name != ""
}

func validHTTPHeaderValue(value string) bool {
	for i := 0; i < len(value); i++ {
		if value[i] == '\r' || value[i] == '\n' || value[i] == 0 || (value[i] < 0x20 && value[i] != '\t') {
			return false
		}
	}
	return true
}

func isWebSocketUpgrade(r *http.Request) bool {
	for _, v := range r.Header.Values("Connection") {
		for _, token := range strings.Split(v, ",") {
			if strings.EqualFold(strings.TrimSpace(token), "upgrade") {
				if strings.EqualFold(r.Header.Get("Upgrade"), "websocket") {
					return true
				}
			}
		}
	}
	return false
}

func proxyWebSocket(
	w http.ResponseWriter,
	r *http.Request,
	target *url.URL,
	stripPrefix string,
	staticHeaders map[string]string,
	handshakeTimeout time.Duration,
) {
	targetHost := webSocketTargetHost(target)
	path := stripProxyPath(r.URL.Path, stripPrefix)
	targetConn, err := net.DialTimeout("tcp", targetHost, 10*time.Second)
	if err != nil {
		log.Printf("WebSocket proxy: failed to connect to %s: %v", targetHost, err)
		http.Error(w, "Bad Gateway", http.StatusBadGateway)
		return
	}
	defer targetConn.Close()
	_ = targetConn.SetDeadline(time.Now().Add(handshakeTimeout))

	clientConn, clientBuf, ok := hijackWebSocketClient(w)
	if !ok {
		return
	}
	defer clientConn.Close()

	upgradeRequest := buildWebSocketUpgradeRequest(r, target, path, staticHeaders)
	if _, writeErr := targetConn.Write(upgradeRequest); writeErr != nil {
		log.Printf("WebSocket proxy: failed to write upgrade request: %v", writeErr)
		return
	}

	log.Printf("WebSocket proxy: %s %s -> %s%s", r.Method, r.URL.Path, target.Host, path)
	targetReader, ok := forwardWebSocketHandshake(targetConn, clientConn, r, handshakeTimeout)
	if !ok {
		return
	}
	relayWebSocketConnections(clientConn, clientBuf, targetConn, targetReader)
}

func webSocketTargetHost(target *url.URL) string {
	targetHost := target.Host
	if strings.Contains(targetHost, ":") {
		return targetHost
	}
	if target.Scheme == "https" || target.Scheme == "wss" {
		return targetHost + ":443"
	}
	return targetHost + ":80"
}

func hijackWebSocketClient(w http.ResponseWriter) (net.Conn, *bufio.ReadWriter, bool) {
	hijacker, ok := w.(http.Hijacker)
	if !ok {
		log.Printf("WebSocket proxy: hijacking not supported")
		http.Error(w, "WebSocket proxy error", http.StatusInternalServerError)
		return nil, nil, false
	}
	clientConn, clientBuf, err := hijacker.Hijack()
	if err != nil {
		log.Printf("WebSocket proxy: hijack failed: %v", err)
		http.Error(w, "WebSocket proxy error", http.StatusInternalServerError)
		return nil, nil, false
	}
	return clientConn, clientBuf, true
}

func buildWebSocketUpgradeRequest(
	r *http.Request,
	target *url.URL,
	path string,
	staticHeaders map[string]string,
) []byte {
	requestURL := path
	if rawQuery := stripDashboardAuthQuery(r.URL.RawQuery); rawQuery != "" {
		requestURL += "?" + rawQuery
	}

	var request strings.Builder
	request.WriteString(r.Method + " " + requestURL + " HTTP/1.1\r\n")
	request.WriteString("Host: " + target.Host + "\r\n")
	appendWebSocketRequestHeaders(&request, r.Header, staticHeaders)
	for key, value := range staticHeaders {
		request.WriteString(key + ": " + value + "\r\n")
	}
	request.WriteString("\r\n")
	return []byte(request.String())
}

func appendWebSocketRequestHeaders(
	request *strings.Builder,
	header http.Header,
	staticHeaders map[string]string,
) {
	overriddenHeaders := normalizedHeaderKeySet(staticHeaders)
	for key, values := range header {
		if shouldDropWebSocketHeader(key, overriddenHeaders) {
			continue
		}
		if strings.EqualFold(key, "Cookie") {
			cookieHeader := http.Header{"Cookie": append([]string(nil), values...)}
			stripDashboardSessionCookies(cookieHeader)
			values = cookieHeader.Values("Cookie")
		}
		for _, value := range values {
			request.WriteString(key + ": " + value + "\r\n")
		}
	}
}

func normalizedHeaderKeySet(headers map[string]string) map[string]struct{} {
	keys := make(map[string]struct{}, len(headers))
	for key := range headers {
		key = strings.ToLower(strings.TrimSpace(key))
		if key != "" {
			keys[key] = struct{}{}
		}
	}
	return keys
}

func shouldDropWebSocketHeader(key string, overriddenHeaders map[string]struct{}) bool {
	if strings.EqualFold(key, "Host") || strings.EqualFold(key, "Authorization") || strings.EqualFold(key, "Referer") {
		return true
	}
	_, overridden := overriddenHeaders[strings.ToLower(strings.TrimSpace(key))]
	return overridden
}

func forwardWebSocketHandshake(
	targetConn net.Conn,
	clientConn net.Conn,
	r *http.Request,
	handshakeTimeout time.Duration,
) (*bufio.Reader, bool) {
	handshakeDeadline := time.Now().Add(handshakeTimeout)
	_ = targetConn.SetDeadline(handshakeDeadline)
	_ = clientConn.SetDeadline(handshakeDeadline)
	limitedTarget := &io.LimitedReader{R: targetConn, N: webSocketHandshakeByteLimit}
	targetReader := bufio.NewReader(limitedTarget)
	upgradeResponse, err := http.ReadResponse(targetReader, r)
	if err != nil {
		log.Printf("WebSocket proxy: failed to read upgrade response: %v", err)
		return nil, false
	}
	if limitedTarget.N == 0 {
		log.Printf("WebSocket proxy: upgrade response exceeded header limit")
		return nil, false
	}
	stripDashboardSessionSetCookies(upgradeResponse.Header)
	if !validWebSocketUpgradeResponse(upgradeResponse) {
		_ = upgradeResponse.Body.Close()
		_, _ = io.WriteString(clientConn, "HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
		return nil, false
	}
	if err := upgradeResponse.Write(clientConn); err != nil {
		log.Printf("WebSocket proxy: failed to write upgrade response: %v", err)
		return nil, false
	}
	_ = targetConn.SetDeadline(time.Time{})
	_ = clientConn.SetDeadline(time.Time{})
	return targetReader, true
}

func validWebSocketUpgradeResponse(response *http.Response) bool {
	return response.StatusCode == http.StatusSwitchingProtocols &&
		headerHasToken(response.Header, "Connection", "upgrade") &&
		headerHasToken(response.Header, "Upgrade", "websocket")
}

func relayWebSocketConnections(
	clientConn net.Conn,
	clientBuf *bufio.ReadWriter,
	targetConn net.Conn,
	targetReader *bufio.Reader,
) {
	done := make(chan struct{}, 2)
	go func() {
		_, _ = io.Copy(targetConn, clientBuf)
		done <- struct{}{}
	}()
	go func() {
		_, _ = io.Copy(clientConn, io.MultiReader(targetReader, targetConn))
		done <- struct{}{}
	}()
	<-done
}

func headerHasToken(header http.Header, name string, want string) bool {
	for _, value := range header.Values(name) {
		for _, token := range strings.Split(value, ",") {
			if strings.EqualFold(strings.TrimSpace(token), want) {
				return true
			}
		}
	}
	return false
}
