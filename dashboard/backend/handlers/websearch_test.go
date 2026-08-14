package handlers

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

// resetRateLimiter clears the shared limiter state so each test starts with a
// fresh budget. Tests here must not run in parallel: the limiter is a package
// global and the whole point is to observe its shared state.
func resetRateLimiter() {
	globalRateLimiter.mu.Lock()
	defer globalRateLimiter.mu.Unlock()
	globalRateLimiter.requests = make(map[string][]time.Time)
	globalRateLimiter.globalReqs = nil
}

// webSearchRequest issues a POST to the web search handler with the given
// identity and headers. An empty body is fine: the rate limit check runs before
// the body is parsed, and an empty query is rejected with 400 after it, so the
// test never touches DuckDuckGo.
func webSearchRequest(t *testing.T, userID, remoteAddr, xff string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/api/tools/web-search", strings.NewReader("{}"))
	if remoteAddr != "" {
		req.RemoteAddr = remoteAddr
	}
	if xff != "" {
		req.Header.Set("X-Forwarded-For", xff)
	}
	if userID != "" {
		req = req.WithContext(auth.WithAuthContext(req.Context(), auth.AuthContext{
			UserID: userID,
			Perms:  map[string]bool{auth.PermToolsUse: true},
		}))
	}

	rec := httptest.NewRecorder()
	WebSearchHandler().ServeHTTP(rec, req)
	return rec
}

// wantBudget checks that a request was let through (empty query -> 400) or was
// rate limited (429), depending on what the test expects.
func wantBudget(t *testing.T, rec *httptest.ResponseRecorder, want int, what string) {
	t.Helper()
	if rec.Code != want {
		t.Fatalf("%s: status = %d, want %d", what, rec.Code, want)
	}
}

func TestWebSearchPerClientLimitAppliesToAuthenticatedUser(t *testing.T) {
	resetRateLimiter()

	for i := 0; i < rateLimitMaxReqs; i++ {
		rec := webSearchRequest(t, "user-a", "10.0.0.1", "")
		wantBudget(t, rec, http.StatusBadRequest, fmt.Sprintf("request %d", i+1))
	}

	rec := webSearchRequest(t, "user-a", "10.0.0.1", "")
	wantBudget(t, rec, http.StatusTooManyRequests, "6th request from the same user")
}

func TestWebSearchPerClientLimitIgnoresSpoofedProxyHeaders(t *testing.T) {
	resetRateLimiter()

	// The per-client key must come from the authenticated session, not from a
	// caller-controlled header, so rotating X-Forwarded-For has no effect.
	for i := 0; i < rateLimitMaxReqs; i++ {
		xff := fmt.Sprintf("203.0.113.%d", i+1)
		rec := webSearchRequest(t, "user-a", "10.0.0.1", xff)
		wantBudget(t, rec, http.StatusBadRequest, fmt.Sprintf("request %d with spoofed XFF", i+1))
	}

	rec := webSearchRequest(t, "user-a", "10.0.0.1", "203.0.113.99")
	wantBudget(t, rec, http.StatusTooManyRequests, "6th request with yet another spoofed XFF")
}

func TestWebSearchPerClientLimitIsPerUser(t *testing.T) {
	resetRateLimiter()

	for i := 0; i < rateLimitMaxReqs; i++ {
		rec := webSearchRequest(t, "user-a", "10.0.0.1", "")
		wantBudget(t, rec, http.StatusBadRequest, fmt.Sprintf("user-a request %d", i+1))
	}

	// A second user sharing the same peer address starts with a fresh budget.
	for i := 0; i < rateLimitMaxReqs; i++ {
		rec := webSearchRequest(t, "user-b", "10.0.0.1", "")
		wantBudget(t, rec, http.StatusBadRequest, fmt.Sprintf("user-b request %d", i+1))
	}

	rec := webSearchRequest(t, "user-b", "10.0.0.1", "")
	wantBudget(t, rec, http.StatusTooManyRequests, "user-b 6th request")
}

func TestWebSearchAnonymousRequestsAreKeyedOnPeerAddress(t *testing.T) {
	resetRateLimiter()

	// Without a session the handler falls back to the real peer address, still
	// ignoring spoofed headers.
	for i := 0; i < rateLimitMaxReqs; i++ {
		xff := fmt.Sprintf("198.51.100.%d", i+1)
		rec := webSearchRequest(t, "", "192.0.2.10", xff)
		wantBudget(t, rec, http.StatusBadRequest, fmt.Sprintf("peer request %d with spoofed XFF", i+1))
	}

	rec := webSearchRequest(t, "", "192.0.2.10", "198.51.100.99")
	wantBudget(t, rec, http.StatusTooManyRequests, "same peer with spoofed XFF")

	rec = webSearchRequest(t, "", "192.0.2.11", "")
	wantBudget(t, rec, http.StatusBadRequest, "different peer")
}

func TestPeerIPStripsPortAndIgnoresHostnames(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{"192.0.2.1:54321", "192.0.2.1"},
		{"[2001:db8::1]:443", "2001:db8::1"},
		{"192.0.2.1", "192.0.2.1"},
		{"", "unknown"},
	}
	for _, tc := range cases {
		if got := peerIP(tc.in); got != tc.want {
			t.Errorf("peerIP(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}
