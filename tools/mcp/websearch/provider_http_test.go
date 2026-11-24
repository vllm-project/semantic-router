package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestHTTPProviderPostsQueryWithBearer(t *testing.T) {
	t.Parallel()
	var gotMethod, gotAuth, gotContentType string
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotAuth = r.Header.Get("Authorization")
		gotContentType = r.Header.Get("Content-Type")
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = w.Write([]byte(`{"query":"q","results":[{"title":"Doc","url":"https://docs.example.com/a","snippet":"About a."}]}`))
	}))
	defer srv.Close()

	got, err := newHTTPProvider(srv.URL+"/search", "secret", srv.Client()).Search(context.Background(), "vllm router", 3)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if gotMethod != http.MethodPost || gotContentType != "application/json" {
		t.Fatalf("request = %s %q, want POST application/json", gotMethod, gotContentType)
	}
	if gotAuth != "Bearer secret" {
		t.Fatalf("Authorization = %q, want Bearer secret", gotAuth)
	}
	if gotBody["query"] != "vllm router" || gotBody["num_results"] != float64(3) {
		t.Fatalf("body = %v, want query and num_results", gotBody)
	}
	want := Result{Title: "Doc", URL: "https://docs.example.com/a", Snippet: "About a.", Domain: "docs.example.com"}
	if len(got) != 1 || got[0] != want {
		t.Fatalf("results = %+v, want [%+v]", got, want)
	}
}

func TestHTTPProviderSendsNoBearerWithoutKey(t *testing.T) {
	t.Parallel()
	var sawHeader bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawHeader = len(r.Header["Authorization"]) > 0
		_, _ = w.Write([]byte(`{"results":[]}`))
	}))
	defer srv.Close()

	if _, err := newHTTPProvider(srv.URL, "", srv.Client()).Search(context.Background(), "q", 1); err != nil {
		t.Fatalf("Search: %v", err)
	}
	if sawHeader {
		t.Fatal("Authorization header sent although no key was configured")
	}
}

func TestHTTPProviderReportsUpstreamStatus(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "nope", http.StatusBadGateway)
	}))
	defer srv.Close()

	_, err := newHTTPProvider(srv.URL, "", srv.Client()).Search(context.Background(), "q", 1)
	if err == nil || !strings.Contains(err.Error(), "502") {
		t.Fatalf("Search error = %v, want one naming status 502", err)
	}
}

func TestHTTPProviderRejectsAReplyThatIsNotJSON(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("<html>not json</html>"))
	}))
	defer srv.Close()

	_, err := newHTTPProvider(srv.URL, "", srv.Client()).Search(context.Background(), "q", 1)
	if err == nil || !strings.Contains(err.Error(), "decode reply") {
		t.Fatalf("Search error = %v, want a decode error", err)
	}
}
