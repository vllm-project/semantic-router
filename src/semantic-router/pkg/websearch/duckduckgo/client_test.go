package duckduckgo

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/websearch"
)

func fixture(t *testing.T, name string) []byte {
	t.Helper()
	body, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("read fixture %s: %v", name, err)
	}
	return body
}

// newTestClient points a client at a fake DuckDuckGo endpoint with no
// inter-request delay and no retry backoff, so tests run in milliseconds.
func newTestClient(endpoint string, opts ...Option) *Client {
	base := []Option{
		WithEndpoint(endpoint),
		WithRequestDelay(0, 0),
		WithRetryBaseDelay(0),
	}
	return NewClient(append(base, opts...)...)
}

func TestSearchParsesResultsFromHTMLEndpoint(t *testing.T) {
	var gotQuery atomic.Value
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotQuery.Store(r.URL.Query().Get("q"))
		w.Header().Set("Content-Type", "text/html")
		_, _ = w.Write(fixture(t, "results.html"))
	}))
	defer server.Close()

	results, err := newTestClient(server.URL).Search(context.Background(), "vllm semantic router", 10)
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if gotQuery.Load() != "vllm semantic router" {
		t.Fatalf("query sent upstream = %q, want the caller's query", gotQuery.Load())
	}

	want := []Result{
		{Title: "vLLM Semantic Router", URL: "https://vllm-sr.ai/docs/", Snippet: "An Envoy ExtProc router for LLM inference.", Domain: "vllm-sr.ai"},
		{Title: "GitHub - vllm-project/semantic-router", URL: "https://github.com/vllm-project/semantic-router", Snippet: "Signals & decisions for MoM routing", Domain: "github.com"},
		{Title: "Direct link result", URL: "https://example.org/direct", Snippet: "", Domain: "example.org"},
	}
	if len(results) != len(want) {
		t.Fatalf("results = %+v, want %d entries", results, len(want))
	}
	for i := range want {
		if results[i] != want[i] {
			t.Fatalf("result %d = %+v, want %+v", i, results[i], want[i])
		}
	}
}

func TestSearchSkipsUnsafeAndInternalLinks(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(fixture(t, "unsafe.html"))
	}))
	defer server.Close()

	results, err := newTestClient(server.URL).Search(context.Background(), "q", 10)
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(results) != 1 || results[0].URL != "https://safe.example/page" {
		t.Fatalf("results = %+v, want only the https result", results)
	}
}

func TestSearchFallsBackWhenPrimaryMarkupDoesNotMatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(fixture(t, "fallback.html"))
	}))
	defer server.Close()

	results, err := newTestClient(server.URL).Search(context.Background(), "q", 10)
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	want := Result{Title: "Fallback title", URL: "https://fallback.example/a", Snippet: "Fallback snippet", Domain: "fallback.example"}
	if len(results) != 1 || results[0] != want {
		t.Fatalf("results = %+v, want [%+v]", results, want)
	}
}

func TestSearchClampsResultCount(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(fixture(t, "results.html"))
	}))
	defer server.Close()

	client := newTestClient(server.URL)
	results, err := client.Search(context.Background(), "q", 2)
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("len(results) = %d, want 2 (caller asked for 2)", len(results))
	}

	results, err = client.Search(context.Background(), "q", 0)
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(results) != 3 {
		t.Fatalf("len(results) = %d, want all 3 (default %d is larger than the fixture)", len(results), DefaultNumResults)
	}
}

func TestSearchReportsChallengePageAsStatusError(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusAccepted)
		_, _ = w.Write([]byte("<html><body>anomaly detected</body></html>"))
	}))
	defer server.Close()

	_, err := newTestClient(server.URL, WithMaxRetries(2)).Search(context.Background(), "q", 5)
	var statusErr *StatusError
	if !errors.As(err, &statusErr) || statusErr.Status != http.StatusAccepted {
		t.Fatalf("Search() error = %v, want *StatusError with status 202", err)
	}
	if calls.Load() != 2 {
		t.Fatalf("upstream calls = %d, want 2 (202 is retried like other transient failures)", calls.Load())
	}
}

func TestSearchDoesNotRetryClientErrors(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusForbidden)
	}))
	defer server.Close()

	_, err := newTestClient(server.URL, WithMaxRetries(3)).Search(context.Background(), "q", 5)
	var statusErr *StatusError
	if !errors.As(err, &statusErr) || statusErr.Status != http.StatusForbidden {
		t.Fatalf("Search() error = %v, want *StatusError with status 403", err)
	}
	if calls.Load() != 1 {
		t.Fatalf("upstream calls = %d, want 1 (4xx must not be retried)", calls.Load())
	}
}

func TestSearchRetriesServerErrorsThenSucceeds(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) < 3 {
			w.WriteHeader(http.StatusBadGateway)
			return
		}
		_, _ = w.Write(fixture(t, "results.html"))
	}))
	defer server.Close()

	results, err := newTestClient(server.URL, WithMaxRetries(3)).Search(context.Background(), "q", 5)
	if err != nil {
		t.Fatalf("Search() error = %v, want success on the third attempt", err)
	}
	if len(results) == 0 || calls.Load() != 3 {
		t.Fatalf("results = %d, calls = %d; want results after exactly 3 calls", len(results), calls.Load())
	}
}

func TestSearchReportsTimeout(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	defer server.Close()

	client := newTestClient(server.URL,
		WithHTTPClient(&http.Client{Timeout: 50 * time.Millisecond}),
		WithMaxRetries(1),
	)
	_, err := client.Search(context.Background(), "q", 5)
	if !errors.Is(err, ErrTimeout) {
		t.Fatalf("Search() error = %v, want ErrTimeout", err)
	}
}

func TestSearchLimitsConcurrentUpstreamRequests(t *testing.T) {
	arrived := make(chan struct{}, 1)
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case arrived <- struct{}{}:
		default:
		}
		<-release
		_, _ = w.Write(fixture(t, "results.html"))
	}))
	defer server.Close()

	client := newTestClient(server.URL, WithMaxConcurrent(1), WithMaxRetries(1))
	firstDone := make(chan error, 1)
	go func() {
		_, err := client.Search(context.Background(), "q", 5)
		firstDone <- err
	}()
	<-arrived // the first call has reached upstream, so it holds the only slot

	_, err := client.Search(context.Background(), "q", 5)
	if !errors.Is(err, ErrTooManyConcurrentRequests) {
		t.Fatalf("second Search() error = %v, want ErrTooManyConcurrentRequests", err)
	}

	close(release)
	if err := <-firstDone; err != nil {
		t.Fatalf("first Search() error = %v", err)
	}
}

func TestSearchSendsBrowserHeadersFromConfiguredAgents(t *testing.T) {
	var gotAgent, gotAccept atomic.Value
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAgent.Store(r.Header.Get("User-Agent"))
		gotAccept.Store(r.Header.Get("Accept"))
		_, _ = w.Write(fixture(t, "results.html"))
	}))
	defer server.Close()

	client := newTestClient(server.URL, WithUserAgents([]string{"TestAgent/1.0"}))
	if _, err := client.Search(context.Background(), "q", 5); err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if gotAgent.Load() != "TestAgent/1.0" {
		t.Fatalf("User-Agent = %q, want the configured agent", gotAgent.Load())
	}
	if gotAccept.Load() == "" {
		t.Fatal("Accept header was not sent")
	}
}

func TestSearchRotatesDefaultUserAgents(t *testing.T) {
	var mu sync.Mutex
	seen := map[string]bool{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		seen[r.Header.Get("User-Agent")] = true
		mu.Unlock()
		_, _ = w.Write(fixture(t, "results.html"))
	}))
	defer server.Close()

	known := map[string]bool{}
	for _, agent := range websearch.DefaultUserAgents() {
		known[agent] = true
	}
	client := newTestClient(server.URL)
	for i := 0; i < 50; i++ {
		if _, err := client.Search(context.Background(), "q", 5); err != nil {
			t.Fatalf("Search() error = %v", err)
		}
	}
	if len(seen) < 2 {
		t.Fatalf("50 searches used %d distinct user agents, want rotation", len(seen))
	}
	for agent := range seen {
		if !known[agent] {
			t.Fatalf("User-Agent %q is not one of the shared browser agents", agent)
		}
	}
}

func TestSearchChargesTheBudgetOncePerSearch(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) < 3 {
			w.WriteHeader(http.StatusBadGateway)
			return
		}
		_, _ = w.Write(fixture(t, "results.html"))
	}))
	defer server.Close()

	var charged atomic.Int32
	client := newTestClient(server.URL, WithMaxRetries(3), WithBudget(func() bool {
		charged.Add(1)
		return true
	}))
	if _, err := client.Search(context.Background(), "q", 5); err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if calls.Load() != 3 || charged.Load() != 1 {
		t.Fatalf("upstream calls = %d, budget charges = %d; want 3 calls charged once (retries share one reservation)",
			calls.Load(), charged.Load())
	}
}

func TestSearchRefusesWhenBudgetIsExhausted(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		_, _ = w.Write(fixture(t, "results.html"))
	}))
	defer server.Close()

	var charged atomic.Int32
	client := newTestClient(server.URL, WithMaxRetries(3), WithBudget(func() bool {
		charged.Add(1)
		return false
	}))
	_, err := client.Search(context.Background(), "q", 5)
	if !errors.Is(err, ErrBudgetExhausted) {
		t.Fatalf("Search() error = %v, want ErrBudgetExhausted", err)
	}
	if calls.Load() != 0 || charged.Load() != 1 {
		t.Fatalf("upstream calls = %d, budget charges = %d; want no upstream call and a single refused charge",
			calls.Load(), charged.Load())
	}
}

func TestSearchKeepsEndpointQueryParameters(t *testing.T) {
	var gotQuery, gotRegion atomic.Value
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotQuery.Store(r.URL.Query().Get("q"))
		gotRegion.Store(r.URL.Query().Get("kl"))
		_, _ = w.Write(fixture(t, "results.html"))
	}))
	defer server.Close()

	client := newTestClient(server.URL + "/html/?kl=us-en")
	if _, err := client.Search(context.Background(), "vllm router", 5); err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if gotQuery.Load() != "vllm router" || gotRegion.Load() != "us-en" {
		t.Fatalf("upstream received q=%q kl=%q, want the query and the endpoint's own parameter", gotQuery.Load(), gotRegion.Load())
	}
}

func TestSearchWrapsBodyReadFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "100")
		_, _ = w.Write([]byte("short"))
		w.(http.Flusher).Flush()
		conn, _, err := w.(http.Hijacker).Hijack()
		if err != nil {
			t.Errorf("hijack: %v", err)
			return
		}
		_ = conn.Close()
	}))
	defer server.Close()

	_, err := newTestClient(server.URL, WithMaxRetries(1)).Search(context.Background(), "q", 5)
	if !errors.Is(err, ErrRequestFailed) || !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("Search() error = %v, want ErrRequestFailed wrapping io.ErrUnexpectedEOF", err)
	}
}

func TestSearchLogsTransportFailureWithoutTheQuery(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	endpoint := server.URL
	server.Close() // every connection is now refused

	var lines []string
	client := newTestClient(endpoint, WithMaxRetries(1), WithLogf(func(format string, args ...any) {
		lines = append(lines, fmt.Sprintf(format, args...))
	}))
	_, err := client.Search(context.Background(), "secret query", 5)
	if !errors.Is(err, ErrRequestFailed) {
		t.Fatalf("Search() error = %v, want ErrRequestFailed", err)
	}
	joined := strings.Join(lines, "\n")
	if !strings.Contains(joined, "connection refused") {
		t.Fatalf("diagnostics %q do not name the transport failure", joined)
	}
	if strings.Contains(joined, "secret query") || strings.Contains(joined, "secret+query") {
		t.Fatalf("diagnostics %q leak the query", joined)
	}
}

func TestSearchLogsContentFreeDiagnostics(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(fixture(t, "results.html"))
	}))
	defer server.Close()

	var lines []string
	client := newTestClient(server.URL, WithLogf(func(format string, args ...any) {
		lines = append(lines, fmt.Sprintf(format, args...))
	}))
	if _, err := client.Search(context.Background(), "secret query", 5); err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(lines) == 0 {
		t.Fatal("no diagnostics were logged")
	}
	for _, line := range lines {
		if strings.Contains(line, "secret query") || strings.Contains(line, "vllm-sr.ai") {
			t.Fatalf("diagnostic %q leaks query or result content", line)
		}
	}
}
