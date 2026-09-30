package connector

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

func TestDoRequestSendsOperationQueryWithBasePath(t *testing.T) {
	var seenPath, seenQuery string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seenPath = r.URL.Path
		seenQuery = r.URL.RawQuery
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer server.Close()

	client, err := New(server.URL+"/openai", nil, testOptions())
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	defer func() { _ = client.Close() }()

	operation := Operation{
		Name:   "chat",
		Method: http.MethodPost,
		Path:   "/deployments/gpt-4o/chat/completions",
		Query:  "api-version=2024-02-01",
	}
	if _, err := client.DoRequest(context.Background(), operation, Request{Body: []byte(`{}`)}); err != nil {
		t.Fatalf("DoRequest() error = %v", err)
	}
	if seenPath != "/openai/deployments/gpt-4o/chat/completions" {
		t.Fatalf("path = %q", seenPath)
	}
	if seenQuery != "api-version=2024-02-01" {
		t.Fatalf("query = %q", seenQuery)
	}
}

func TestDoRequestRejectsInvalidPathOrQueryBeforeSending(t *testing.T) {
	var hits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		hits.Add(1)
	}))
	defer server.Close()
	client := newTestClient(t, server, testOptions())

	tests := []struct {
		name  string
		path  string
		query string
	}{
		{name: "query inside path", path: "/chat/completions?api-version=2024-02-01"},
		{name: "fragment inside path", path: "/chat/completions#frag"},
		{name: "relative path", path: "chat/completions"},
		{name: "fragment inside query", path: "/chat/completions", query: "api-version=1#frag"},
		{name: "second query marker", path: "/chat/completions", query: "a=1?b=2"},
		{name: "malformed escape", path: "/chat/completions", query: "api-version=%zz"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			operation := Operation{Name: "chat", Method: http.MethodPost, Path: tt.path, Query: tt.query}
			_, err := client.DoRequest(context.Background(), operation, Request{Body: []byte(`{}`)})
			var connectorErr *Error
			if !errors.As(err, &connectorErr) || connectorErr.Kind != KindRequest {
				t.Fatalf("DoRequest() error = %v, want KindRequest", err)
			}
			if hits.Load() != 0 {
				t.Fatal("invalid operation reached the server")
			}
		})
	}
}
