package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func newImmediateWebSearchOutbound(client outboundHTTPClient) *webSearchOutbound {
	search := newWebSearchOutbound(client)
	search.randomDelay = func(context.Context) error { return nil }
	return search
}

func TestGetClientIPIgnoresForwardedHeaders(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		remoteAddr string
		want       string
	}{
		{name: "ipv4", remoteAddr: "198.51.100.7:4321", want: "198.51.100.7"},
		{name: "ipv6", remoteAddr: "[2001:db8::7]:4321", want: "2001:db8::7"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			req := httptest.NewRequest(http.MethodPost, "/api/tools/web-search", nil)
			req.RemoteAddr = tt.remoteAddr
			req.Header.Set("X-Forwarded-For", "203.0.113.99")
			req.Header.Set("X-Real-IP", "203.0.113.98")
			if got := getClientIP(req); got != tt.want {
				t.Fatalf("getClientIP() = %q, want direct peer %q", got, tt.want)
			}
		})
	}
}

func TestWebSearchHandlerDoesNotSetWildcardCORS(t *testing.T) {
	t.Parallel()

	req := httptest.NewRequest(http.MethodOptions, "/api/tools/web-search", nil)
	w := httptest.NewRecorder()
	webSearchHandlerWithDependencies(nil, newRateLimiter())(w, req)

	if got := w.Header().Get("Access-Control-Allow-Origin"); got != "" {
		t.Fatalf("Access-Control-Allow-Origin = %q, want absent", got)
	}
}

func TestWebSearchRateLimitCannotBeBypassedWithForwardedHeaders(t *testing.T) {
	t.Parallel()

	client := &controlledTestOutboundClient{do: func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader("<html></html>")),
		}, nil
	}}
	handler := webSearchHandlerWithDependencies(
		newImmediateWebSearchOutbound(client),
		newRateLimiter(),
	)

	for attempt := 0; attempt <= rateLimitMaxReqs; attempt++ {
		body, err := json.Marshal(WebSearchRequest{Query: "public information"})
		if err != nil {
			t.Fatalf("marshal request: %v", err)
		}
		req := httptest.NewRequest(http.MethodPost, "/api/tools/web-search", bytes.NewReader(body))
		req.RemoteAddr = "198.51.100.7:4321"
		req.Header.Set("X-Forwarded-For", "203.0.113."+strconv.Itoa(attempt+1))
		w := httptest.NewRecorder()
		handler(w, req)

		wantStatus := http.StatusOK
		if attempt == rateLimitMaxReqs {
			wantStatus = http.StatusTooManyRequests
		}
		if w.Code != wantStatus {
			t.Fatalf("attempt %d status = %d, want %d: %s", attempt+1, w.Code, wantStatus, w.Body.String())
		}
	}
}

func TestWebSearchRejectsOversizedUpstreamBody(t *testing.T) {
	t.Parallel()

	var calls atomic.Int32
	client := &controlledTestOutboundClient{do: func(*http.Request) (*http.Response, error) {
		calls.Add(1)
		return &http.Response{
			StatusCode: http.StatusOK,
			Body: io.NopCloser(strings.NewReader(strings.Repeat(
				"x",
				webSearchMaxResponseSize+1,
			))),
		}, nil
	}}
	search := newImmediateWebSearchOutbound(client)

	_, err := search.searchDuckDuckGo(context.Background(), "query-secret", defaultNumResults)
	if got := webSearchErrorCode(err); got != ErrCodeUpstreamError {
		t.Fatalf("error code = %s, want %s", got, ErrCodeUpstreamError)
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("outbound calls = %d, want non-retryable oversize rejection", got)
	}
	if strings.Contains(err.Error(), "query-secret") || strings.Contains(err.Error(), strings.Repeat("x", 32)) {
		t.Fatalf("oversize error leaked query or body: %v", err)
	}
}

func TestWebSearchDelayAndRequestHonorCancellation(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	started := time.Now()
	if err := waitWebSearchDelay(ctx, time.Minute, time.Minute); !errors.Is(err, context.Canceled) {
		t.Fatalf("waitWebSearchDelay() error = %v, want canceled", err)
	}
	if elapsed := time.Since(started); elapsed > 100*time.Millisecond {
		t.Fatalf("canceled delay took %v", elapsed)
	}

	var calls atomic.Int32
	client := &controlledTestOutboundClient{do: func(*http.Request) (*http.Response, error) {
		calls.Add(1)
		return nil, errors.New("unexpected request")
	}}
	search := newWebSearchOutbound(client)
	_, err := search.searchDuckDuckGo(ctx, "query", defaultNumResults)
	if got := webSearchErrorCode(err); got != ErrCodeTimeout {
		t.Fatalf("canceled search code = %s, want timeout", got)
	}
	if got := calls.Load(); got != 0 {
		t.Fatalf("outbound calls after cancellation = %d, want zero", got)
	}
}

func TestWebSearchErrorResponseDoesNotEchoQueryOrUpstreamError(t *testing.T) {
	t.Parallel()

	client := &controlledTestOutboundClient{do: func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusBadRequest,
			Status:     "400 body-secret",
			Body:       io.NopCloser(strings.NewReader("body-secret")),
		}, nil
	}}
	handler := webSearchHandlerWithDependencies(
		newImmediateWebSearchOutbound(client),
		newRateLimiter(),
	)
	body, err := json.Marshal(WebSearchRequest{Query: "query-secret"})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/api/tools/web-search", bytes.NewReader(body))
	w := httptest.NewRecorder()

	handler(w, req)

	if w.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502: %s", w.Code, w.Body.String())
	}
	for _, secret := range []string{"query-secret", "body-secret"} {
		if strings.Contains(w.Body.String(), secret) {
			t.Fatalf("error response leaked %q: %s", secret, w.Body.String())
		}
	}
}

func TestWebSearchResultURLRejectsUserinfo(t *testing.T) {
	t.Parallel()

	if isValidURL("https://user:password@example.com/path") {
		t.Fatal("isValidURL accepted userinfo")
	}
}
