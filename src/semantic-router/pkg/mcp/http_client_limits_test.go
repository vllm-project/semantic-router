package mcp

import (
	"context"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"
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

func TestHTTPClientDefaultsListPageLimit(t *testing.T) {
	client := NewHTTPClient("test", ClientConfig{})
	if client.maxListPages != defaultMCPMaxListPages {
		t.Fatalf("maxListPages = %d, want %d", client.maxListPages, defaultMCPMaxListPages)
	}
}

func TestHTTPClientDefaultsListByteLimit(t *testing.T) {
	for _, tc := range []struct {
		name   string
		config ClientConfig
		want   int64
	}{
		{name: "unset", config: ClientConfig{}, want: defaultMCPMaxResponseBytes},
		{name: "follows response cap", config: ClientConfig{MaxResponseBytes: 64 << 20}, want: 64 << 20},
		{name: "explicit", config: ClientConfig{MaxResponseBytes: 64 << 20, MaxListBytes: 1 << 20}, want: 1 << 20},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := NewHTTPClient("test", tc.config).maxListBytes; got != tc.want {
				t.Fatalf("maxListBytes = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestHTTPClientStopsListingAtPageLimit(t *testing.T) {
	core, logs := observer.New(zapcore.WarnLevel)
	t.Cleanup(zap.ReplaceGlobals(zap.New(core)))

	server := newPagedListServer(t, 10)
	client := NewHTTPClient("paged", ClientConfig{URL: server.URL, MaxListPages: 2})
	if err := client.Connect(); err != nil {
		t.Fatalf("Connect() error = %v", err)
	}

	if got, want := toolNames(client.GetTools()), []string{"tool-1", "tool-2"}; !slices.Equal(got, want) {
		t.Errorf("loaded %v, want %v", got, want)
	}
	if got := len(server.cursorsSent("tools/list")); got != 2 {
		t.Errorf("tools/list requests = %d, want 2", got)
	}

	var warned []any
	for _, entry := range logs.FilterMessage("mcp_list_page_limit_reached").All() {
		warned = append(warned, entry.ContextMap()["method"])
	}
	if want := []any{"tools/list", "resources/list", "prompts/list"}; !slices.Equal(warned, want) {
		t.Errorf("page limit warnings for %v, want %v", warned, want)
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
