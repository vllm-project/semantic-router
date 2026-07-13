package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func TestTopologyQueryBudgetRejectsBeforeRouterCall(t *testing.T) {
	var hits atomic.Int32
	routerAPI := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		_ = json.NewEncoder(w).Encode(RouterEvalResponse{})
	}))
	defer routerAPI.Close()

	body, err := json.Marshal(TestQueryRequest{
		Query: strings.Repeat("q", topologyMaxQueryBytes+1),
		Mode:  TestQueryModeDryRun,
	})
	if err != nil {
		t.Fatal(err)
	}
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/api/topology/test-query", bytes.NewReader(body))
	TopologyTestQueryHandler("", routerAPI.URL)(recorder, request)

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", recorder.Code)
	}
	if hits.Load() != 0 {
		t.Fatalf("router hits = %d, want 0", hits.Load())
	}
}

func TestTopologyRouterClientDoesNotFollowRedirects(t *testing.T) {
	var redirectedHits atomic.Int32
	redirected := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		redirectedHits.Add(1)
		_ = json.NewEncoder(w).Encode(RouterEvalResponse{})
	}))
	defer redirected.Close()
	origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, redirected.URL, http.StatusTemporaryRedirect)
	}))
	defer origin.Close()

	result := callRouterAPI(
		context.Background(),
		newTopologyRouterHTTPClient(),
		TestQueryRequest{Query: "private prompt", Mode: TestQueryModeDryRun},
		origin.URL,
		"",
	)
	if result == nil || result.Warning == "" {
		t.Fatalf("redirect returned unexpected result: %#v", result)
	}
	if redirectedHits.Load() != 0 {
		t.Fatalf("redirect target received %d requests, want 0", redirectedHits.Load())
	}
}

func TestTopologyRouterResponseBudgetAndURLRedaction(t *testing.T) {
	routerAPI := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, strings.Repeat("x", int(topologyMaxResponseBodyBytes)+1))
	}))
	defer routerAPI.Close()

	result := callRouterAPI(
		context.Background(),
		newTopologyRouterHTTPClient(),
		TestQueryRequest{Query: "test", Mode: TestQueryModeDryRun},
		routerAPI.URL,
		"",
	)
	if result == nil || result.Warning != "Failed to parse Router API response" {
		t.Fatalf("oversized response result = %#v", result)
	}

	invalid := callRouterAPI(
		context.Background(),
		newTopologyRouterHTTPClient(),
		TestQueryRequest{Query: "test", Mode: TestQueryModeDryRun},
		"https://user:sentinel-secret@example.com",
		"",
	)
	if invalid == nil || invalid.Warning != "Router API unavailable" || strings.Contains(invalid.Warning, "sentinel-secret") {
		t.Fatalf("invalid URL was not sanitized: %#v", invalid)
	}
}

func TestTopologyRouterResponseAllowsForwardCompatibleFields(t *testing.T) {
	routerAPI := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"routing_decision":"fallback","future_field":{"enabled":true}}`)
	}))
	defer routerAPI.Close()

	result := callRouterAPI(
		context.Background(),
		newTopologyRouterHTTPClient(),
		TestQueryRequest{Query: "test", Mode: TestQueryModeDryRun},
		routerAPI.URL,
		"",
	)
	if result == nil || result.Warning != "" || result.MatchedDecision != "fallback" {
		t.Fatalf("forward-compatible response result = %#v", result)
	}
}

func TestTopologyRouterClientIgnoresAmbientProxy(t *testing.T) {
	var proxyHits atomic.Int32
	proxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		proxyHits.Add(1)
		http.Error(w, "proxy", http.StatusBadGateway)
	}))
	defer proxy.Close()
	t.Setenv("HTTP_PROXY", proxy.URL)
	t.Setenv("http_proxy", proxy.URL)

	routerAPI := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"routing_decision":"direct"}`)
	}))
	defer routerAPI.Close()

	result := callRouterAPI(
		context.Background(),
		newTopologyRouterHTTPClient(),
		TestQueryRequest{Query: "test", Mode: TestQueryModeDryRun},
		routerAPI.URL,
		"",
	)
	if result == nil || result.MatchedDecision != "direct" {
		t.Fatalf("direct router result = %#v", result)
	}
	if proxyHits.Load() != 0 {
		t.Fatalf("ambient proxy received %d requests, want 0", proxyHits.Load())
	}
}
