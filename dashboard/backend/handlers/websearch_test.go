package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/websearch/duckduckgo"
)

const fakeResultsPage = `<html><body><div class="results">
<div class="result"><div class="result__body">
<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fvllm-sr.ai%2Fdocs%2F&amp;rut=1">vLLM Semantic Router</a>
<a class="result__snippet" href="#">Routes LLM requests</a>
</div></div>
</div></body></html>`

// useFakeSearch points the handler's shared client at a fake DuckDuckGo for
// the duration of the test, with no inter-request delay and a single attempt.
// The fake is wired to the upstream budget exactly like the production client.
func useFakeSearch(t *testing.T, upstream http.HandlerFunc, opts ...duckduckgo.Option) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(upstream)
	t.Cleanup(server.Close)

	original := searchClient
	base := []duckduckgo.Option{
		duckduckgo.WithEndpoint(server.URL),
		duckduckgo.WithRequestDelay(0, 0),
		duckduckgo.WithRetryBaseDelay(0),
		duckduckgo.WithMaxRetries(1),
		duckduckgo.WithBudget(reserveUpstreamBudget),
	}
	searchClient = duckduckgo.NewClient(append(base, opts...)...)
	t.Cleanup(func() { searchClient = original })
	return server
}

// authedContext carries the session the handler keys its per-client limit on.
func authedContext(userID string) context.Context {
	return auth.WithAuthContext(context.Background(), auth.AuthContext{UserID: userID})
}

func postSearch(t *testing.T, ctx context.Context, body string) (*httptest.ResponseRecorder, WebSearchResponse) {
	t.Helper()
	req := httptest.NewRequestWithContext(ctx, http.MethodPost, "/api/tools/web-search", strings.NewReader(body))
	req.RemoteAddr = "192.0.2.10:44321"

	recorder := httptest.NewRecorder()
	WebSearchHandler()(recorder, req)

	var payload WebSearchResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode response: %v; body=%s", err, recorder.Body.String())
	}
	return recorder, payload
}

func TestWebSearchHandlerReturnsResultsFromSharedClient(t *testing.T) {
	resetRateLimiter(t)
	useFakeSearch(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(fakeResultsPage))
	})

	recorder, payload := postSearch(t, authedContext("user-results"), `{"query":"vllm semantic router","num_results":3}`)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}
	if payload.Query != "vllm semantic router" || payload.Error != "" || payload.Code != "" {
		t.Fatalf("payload = %+v, want the query echoed and no error", payload)
	}
	want := SearchResult{Title: "vLLM Semantic Router", URL: "https://vllm-sr.ai/docs/", Snippet: "Routes LLM requests", Domain: "vllm-sr.ai"}
	if len(payload.Results) != 1 || payload.Results[0] != want {
		t.Fatalf("results = %+v, want [%+v]", payload.Results, want)
	}

	// The wire names are consumed by the frontend; keep them stable.
	var raw struct {
		Results []map[string]any `json:"results"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &raw); err != nil {
		t.Fatalf("decode raw response: %v", err)
	}
	for _, field := range []string{"title", "url", "snippet", "domain"} {
		if _, ok := raw.Results[0][field]; !ok {
			t.Fatalf("result is missing wire field %q: %v", field, raw.Results[0])
		}
	}
}

func TestWebSearchHandlerMapsChallengePageToUpstreamError(t *testing.T) {
	resetRateLimiter(t)
	useFakeSearch(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusAccepted)
		_, _ = w.Write([]byte("<html><body>anomaly</body></html>"))
	})

	recorder, payload := postSearch(t, authedContext("user-challenge"), `{"query":"vllm"}`)

	if recorder.Code != http.StatusBadGateway || payload.Code != ErrCodeUpstreamError {
		t.Fatalf("status = %d code = %q, want 502 %s", recorder.Code, payload.Code, ErrCodeUpstreamError)
	}
}

func TestWebSearchHandlerMapsTimeoutToGatewayTimeout(t *testing.T) {
	resetRateLimiter(t)
	useFakeSearch(t, func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}, duckduckgo.WithHTTPClient(&http.Client{Timeout: 50 * time.Millisecond}))

	recorder, payload := postSearch(t, authedContext("user-timeout"), `{"query":"vllm"}`)

	if recorder.Code != http.StatusGatewayTimeout || payload.Code != ErrCodeTimeout {
		t.Fatalf("status = %d code = %q, want 504 %s", recorder.Code, payload.Code, ErrCodeTimeout)
	}
}

func TestWebSearchHandlerMapsTransportFailureToSearchFailed(t *testing.T) {
	resetRateLimiter(t)
	server := useFakeSearch(t, func(w http.ResponseWriter, r *http.Request) {})
	server.Close()

	recorder, payload := postSearch(t, authedContext("user-down"), `{"query":"vllm"}`)

	if recorder.Code != http.StatusInternalServerError || payload.Code != ErrCodeSearchFailed {
		t.Fatalf("status = %d code = %q, want 500 %s", recorder.Code, payload.Code, ErrCodeSearchFailed)
	}
}

func TestWebSearchHandlerMapsDeadlineToGatewayTimeout(t *testing.T) {
	resetRateLimiter(t)
	useFakeSearch(t, func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	})
	ctx, cancel := context.WithTimeout(authedContext("user-deadline"), 50*time.Millisecond)
	defer cancel()

	recorder, payload := postSearch(t, ctx, `{"query":"vllm"}`)

	if recorder.Code != http.StatusGatewayTimeout || payload.Code != ErrCodeTimeout {
		t.Fatalf("status = %d code = %q, want 504 %s for a request deadline", recorder.Code, payload.Code, ErrCodeTimeout)
	}
}

func TestWebSearchHandlerSpendsOneUpstreamSlotPerSearch(t *testing.T) {
	resetRateLimiter(t)
	useFakeSearch(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(fakeResultsPage))
	})

	recorder, _ := postSearch(t, authedContext("user-budget"), `{"query":"vllm"}`)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
	}
	if _, spent := globalRateLimiter.getStats(); spent != 1 {
		t.Fatalf("upstream budget spent = %d, want exactly 1 for one search", spent)
	}
}

func TestWebSearchHandlerRejectsWhenUpstreamBudgetIsExhausted(t *testing.T) {
	resetRateLimiter(t)
	var calls atomic.Int32
	useFakeSearch(t, func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		_, _ = w.Write([]byte(fakeResultsPage))
	})
	for i := 0; i < globalRateLimit; i++ {
		if !globalRateLimiter.reserveUpstream() {
			t.Fatalf("upstream budget refused reservation %d of %d", i+1, globalRateLimit)
		}
	}

	recorder, payload := postSearch(t, authedContext("user-exhausted"), `{"query":"vllm"}`)

	if recorder.Code != http.StatusTooManyRequests || payload.Code != ErrCodeRateLimited || payload.Query != "vllm" {
		t.Fatalf("status = %d code = %q query = %q, want 429 %s with the query echoed",
			recorder.Code, payload.Code, payload.Query, ErrCodeRateLimited)
	}
	if calls.Load() != 0 {
		t.Fatalf("upstream calls = %d, want 0 when the budget is exhausted", calls.Load())
	}
}
