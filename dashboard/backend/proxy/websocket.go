package proxy

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
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
	targetURL, err := url.Parse(targetBase)
	if err != nil {
		return nil, err
	}

	httpProxy, err := NewReverseProxy(targetBase, stripPrefix, false)
	if err != nil {
		return nil, err
	}
	if len(staticHeaders) > 0 {
		origDirector := httpProxy.Director
		httpProxy.Director = func(r *http.Request) {
			origDirector(r)
			for key, value := range staticHeaders {
				normalizedKey := strings.TrimSpace(key)
				normalizedValue := strings.TrimSpace(value)
				if normalizedKey == "" || normalizedValue == "" {
					continue
				}
				// Static headers are authoritative in embedded mode to avoid stale
				// client-side gateway tokens causing auth mismatch loops.
				r.Header.Set(normalizedKey, normalizedValue)
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
			proxyWebSocket(w, r, targetURL, stripPrefix, staticHeaders)
			return
		}
		httpProxy.ServeHTTP(w, r)
	}), nil
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

func proxyWebSocket(w http.ResponseWriter, r *http.Request, target *url.URL, stripPrefix string, staticHeaders map[string]string) {
	// Build the target address
	targetHost := target.Host
	if !strings.Contains(targetHost, ":") {
		if target.Scheme == "https" || target.Scheme == "wss" {
			targetHost += ":443"
		} else {
			targetHost += ":80"
		}
	}

	// Strip prefix from path
	path := r.URL.Path
	path = strings.TrimPrefix(path, stripPrefix)
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}

	// Connect to target
	targetConn, err := net.DialTimeout("tcp", targetHost, 10*time.Second)
	if err != nil {
		log.Printf("WebSocket proxy: failed to connect to %s: %v", targetHost, err)
		http.Error(w, "Bad Gateway", http.StatusBadGateway)
		return
	}
	defer targetConn.Close()

	// Hijack the client connection
	hijacker, ok := w.(http.Hijacker)
	if !ok {
		log.Printf("WebSocket proxy: hijacking not supported")
		http.Error(w, "WebSocket proxy error", http.StatusInternalServerError)
		return
	}
	clientConn, clientBuf, err := hijacker.Hijack()
	if err != nil {
		log.Printf("WebSocket proxy: hijack failed: %v", err)
		http.Error(w, "WebSocket proxy error", http.StatusInternalServerError)
		return
	}
	defer clientConn.Close()

	// Rebuild the original HTTP request and forward to target
	reqURL := path
	if r.URL.RawQuery != "" {
		reqURL += "?" + r.URL.RawQuery
	}

	var reqBuf strings.Builder
	reqBuf.WriteString(r.Method + " " + reqURL + " HTTP/1.1\r\n")
	reqBuf.WriteString("Host: " + target.Host + "\r\n")

	effectiveStaticHeaders := map[string]string{}
	for key, value := range staticHeaders {
		normalizedKey := strings.TrimSpace(key)
		normalizedValue := strings.TrimSpace(value)
		if normalizedKey == "" || normalizedValue == "" {
			continue
		}
		effectiveStaticHeaders[normalizedKey] = normalizedValue
	}

	normalizedStaticHeaderKeys := map[string]struct{}{}
	for key := range effectiveStaticHeaders {
		trimmed := strings.ToLower(strings.TrimSpace(key))
		if trimmed != "" {
			normalizedStaticHeaderKeys[trimmed] = struct{}{}
		}
	}

	for key, vals := range r.Header {
		if strings.EqualFold(key, "Host") {
			continue
		}
		if _, overridden := normalizedStaticHeaderKeys[strings.ToLower(strings.TrimSpace(key))]; overridden {
			continue
		}
		for _, val := range vals {
			reqBuf.WriteString(key + ": " + val + "\r\n")
		}
	}
	for key, value := range effectiveStaticHeaders {
		normalizedKey := strings.TrimSpace(key)
		normalizedValue := strings.TrimSpace(value)
		if normalizedKey == "" || normalizedValue == "" {
			continue
		}
		reqBuf.WriteString(normalizedKey + ": " + normalizedValue + "\r\n")
	}
	reqBuf.WriteString("\r\n")

	if _, err := targetConn.Write([]byte(reqBuf.String())); err != nil {
		log.Printf("WebSocket proxy: failed to write upgrade request: %v", err)
		return
	}

	log.Printf("WebSocket proxy: %s %s -> %s%s", r.Method, r.URL.Path, target.Host, path)

	// Bidirectional copy
	done := make(chan struct{}, 2)

	go func() {
		_, _ = io.Copy(targetConn, clientBuf)
		done <- struct{}{}
	}()
	go func() {
		_, _ = io.Copy(clientConn, targetConn)
		done <- struct{}{}
	}()

	<-done
}
