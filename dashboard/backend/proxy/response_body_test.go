package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
)

func TestJaegerProxyInjectsThemeWithinResponseLimit(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = io.WriteString(w, "<!doctype html><html><head><title>Jaeger</title></head><body></body></html>")
	}))
	defer upstream.Close()

	handler, err := NewJaegerProxy(upstream.URL, "/embedded/jaeger")
	if err != nil {
		t.Fatalf("NewJaegerProxy: %v", err)
	}
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(
		recorder,
		httptest.NewRequest(http.MethodGet, "http://dashboard.test/embedded/jaeger/", nil),
	)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
	}
	body := recorder.Body.String()
	if !strings.Contains(body, "localStorage.setItem('jaeger-ui-theme', 'light')") ||
		!strings.Contains(body, "</script></head>") {
		t.Fatalf("theme script was not injected before </head>: %q", body)
	}
	if got, want := recorder.Header().Get("Content-Length"), strconv.Itoa(len(recorder.Body.Bytes())); got != want {
		t.Fatalf("Content-Length = %q, want %q", got, want)
	}
}

func TestJaegerProxyRejectsOversizedChunkedHTML(t *testing.T) {
	const canary = "oversized-jaeger-canary"
	upstream := newChunkedResponseServer(
		t,
		"text/html; charset=utf-8",
		jaegerHTMLResponseByteLimit+1,
		canary,
	)
	defer upstream.Close()

	handler, err := NewJaegerProxy(upstream.URL, "/embedded/jaeger")
	if err != nil {
		t.Fatalf("NewJaegerProxy: %v", err)
	}
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(
		recorder,
		httptest.NewRequest(http.MethodGet, "http://dashboard.test/embedded/jaeger/", nil),
	)

	assertGenericOversizedProxyFailure(t, recorder, canary)
}

func TestOpenClawControlConfigRewriteWithinResponseLimit(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"basePath":"/","gatewayUrl":"ws://internal.example"}`)
	}))
	defer upstream.Close()

	const stripPrefix = "/embedded/openclaw/demo"
	handler, err := NewWebSocketAwareHandler(upstream.URL, stripPrefix)
	if err != nil {
		t.Fatalf("NewWebSocketAwareHandler: %v", err)
	}
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(
		recorder,
		httptest.NewRequest(
			http.MethodGet,
			"http://dashboard.test"+stripPrefix+"/__openclaw/control-ui-config.json",
			nil,
		),
	)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
	}
	var config map[string]interface{}
	if err := json.Unmarshal(recorder.Body.Bytes(), &config); err != nil {
		t.Fatalf("decode rewritten config: %v", err)
	}
	if config["basePath"] != stripPrefix || config["gatewayUrl"] != stripPrefix {
		t.Fatalf("rewritten config = %#v", config)
	}
}

func TestOpenClawControlConfigRejectsOversizedChunkedResponse(t *testing.T) {
	const canary = "oversized-openclaw-canary"
	upstream := newChunkedResponseServer(
		t,
		"application/json",
		openClawControlConfigResponseByteLimit+1,
		canary,
	)
	defer upstream.Close()

	const stripPrefix = "/embedded/openclaw/demo"
	handler, err := NewWebSocketAwareHandler(upstream.URL, stripPrefix)
	if err != nil {
		t.Fatalf("NewWebSocketAwareHandler: %v", err)
	}
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(
		recorder,
		httptest.NewRequest(
			http.MethodGet,
			"http://dashboard.test"+stripPrefix+"/__openclaw/control-ui-config.json",
			nil,
		),
	)

	assertGenericOversizedProxyFailure(t, recorder, canary)
}

func newChunkedResponseServer(
	t *testing.T,
	contentType string,
	totalBytes int64,
	canary string,
) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", contentType)
		w.WriteHeader(http.StatusOK)
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
		reader := io.MultiReader(
			strings.NewReader(strings.Repeat("x", int(totalBytes)-len(canary))),
			strings.NewReader(canary),
		)
		if _, err := io.CopyN(w, reader, totalBytes); err != nil {
			// The bounded proxy is expected to close the response as soon as it has
			// observed limit+1 bytes, so a broken pipe is a successful test outcome.
			return
		}
	}))
}

func assertGenericOversizedProxyFailure(t *testing.T, recorder *httptest.ResponseRecorder, canary string) {
	t.Helper()
	if recorder.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want %d; body=%q", recorder.Code, http.StatusBadGateway, recorder.Body.String())
	}
	if got := recorder.Body.String(); got != "Bad Gateway\n" || strings.Contains(got, canary) {
		t.Fatalf("oversized upstream response was not replaced by a generic error: %q", got)
	}
}
