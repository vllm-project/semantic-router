package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestSearxngProviderSearch(t *testing.T) {
	t.Parallel()
	var gotPath, gotQuery string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotQuery = r.URL.RawQuery
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"query":"vllm router","results":[
			{"title":"vLLM Semantic Router","url":"https://vllm-semantic-router.com/docs","content":"An Envoy ExtProc router.","engine":"duckduckgo"},
			{"title":"GitHub","url":"https://github.com/vllm-project/semantic-router","content":"Source code.","engine":"google"}
		]}`))
	}))
	defer srv.Close()

	p := newSearxngProvider(srv.URL, srv.Client())
	got, err := p.Search(context.Background(), "vllm router", 5)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if gotPath != "/search" {
		t.Fatalf("request path = %q, want /search", gotPath)
	}
	for _, want := range []string{"q=vllm+router", "format=json"} {
		if !strings.Contains(gotQuery, want) {
			t.Fatalf("request query %q lacks %q", gotQuery, want)
		}
	}
	want := []Result{
		{Title: "vLLM Semantic Router", URL: "https://vllm-semantic-router.com/docs", Snippet: "An Envoy ExtProc router.", Domain: "vllm-semantic-router.com"},
		{Title: "GitHub", URL: "https://github.com/vllm-project/semantic-router", Snippet: "Source code.", Domain: "github.com"},
	}
	if len(got) != len(want) {
		t.Fatalf("got %d results, want %d: %+v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("result %d = %+v, want %+v", i, got[i], want[i])
		}
	}
}

func TestSearxngProviderReportsUpstreamStatus(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "json format disabled", http.StatusForbidden)
	}))
	defer srv.Close()

	_, err := newSearxngProvider(srv.URL, srv.Client()).Search(context.Background(), "x", 5)
	if err == nil || !strings.Contains(err.Error(), "403") {
		t.Fatalf("Search error = %v, want one naming status 403", err)
	}
}

func TestSearxngProviderRejectsAReplyThatIsNotJSON(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("<html>not json</html>"))
	}))
	defer srv.Close()

	_, err := newSearxngProvider(srv.URL, srv.Client()).Search(context.Background(), "q", 1)
	if err == nil || !strings.Contains(err.Error(), "decode reply") {
		t.Fatalf("Search error = %v, want a decode error", err)
	}
}
