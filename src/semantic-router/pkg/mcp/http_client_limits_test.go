package mcp

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func newLimitedHTTPClient(server *httptest.Server, maxResponseBytes int64) *HTTPClient {
	return NewHTTPClient("test", ClientConfig{
		URL:              server.URL,
		MaxResponseBytes: maxResponseBytes,
	})
}

func TestHTTPClientDefaultsResponseLimit(t *testing.T) {
	client := NewHTTPClient("test", ClientConfig{})
	if client.maxResponseBytes != defaultMCPMaxResponseBytes {
		t.Fatalf("maxResponseBytes = %d, want %d", client.maxResponseBytes, defaultMCPMaxResponseBytes)
	}
}

func TestHTTPClientResponseOneByteOverLimitIsRejected(t *testing.T) {
	const limit = 1024
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(strings.Repeat("x", limit+1)))
	}))
	defer server.Close()

	client := newLimitedHTTPClient(server, limit)
	_, err := client.sendRequest(context.Background(), "tools/call", nil)
	if err == nil {
		t.Fatal("expected an error")
	}
	if !strings.Contains(err.Error(), "exceeds limit") {
		t.Fatalf("error = %v, want exceeded limit", err)
	}
}

func TestHTTPClientErrorBodyIsTruncated(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(strings.Repeat("e", int(maxMCPErrorBodyBytes)+1)))
	}))
	defer server.Close()

	client := newLimitedHTTPClient(server, 1)
	_, err := client.sendRequest(context.Background(), "tools/call", nil)
	if err == nil {
		t.Fatal("expected an error")
	}
	if !strings.Contains(err.Error(), "status 503") || !strings.Contains(err.Error(), "truncated=true") {
		t.Fatalf("error = %v, want status and truncation", err)
	}
	if len(err.Error()) > int(maxMCPErrorBodyBytes)+256 {
		t.Fatalf("error length = %d, want bounded error", len(err.Error()))
	}
}
