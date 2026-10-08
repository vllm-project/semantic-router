package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

func newTestRateLimiter() *rateLimiter {
	return &rateLimiter{
		requests:   make(map[string][]time.Time),
		globalReqs: make([]time.Time, 0),
	}
}

// resetRateLimiter swaps the package limiter for an empty one so handler tests
// do not inherit each other's windows.
func resetRateLimiter(t *testing.T) {
	t.Helper()

	original := globalRateLimiter
	globalRateLimiter = newTestRateLimiter()
	t.Cleanup(func() { globalRateLimiter = original })
}

// searchRequest builds a request with an empty query, which is rejected after
// the rate limit check and before any outbound call. 400 means the limiter let
// the request through, 429 means it refused it.
func searchRequest(remoteAddr, forwardedFor, userID string) *http.Request {
	req := httptest.NewRequest(http.MethodPost, "/api/tools/web-search", strings.NewReader("{}"))
	req.RemoteAddr = remoteAddr
	if forwardedFor != "" {
		req.Header.Set("X-Forwarded-For", forwardedFor)
		req.Header.Set("X-Real-IP", forwardedFor)
	}
	if userID != "" {
		req = req.WithContext(auth.WithAuthContext(req.Context(), auth.AuthContext{UserID: userID}))
	}
	return req
}

func statusesFor(t *testing.T, requests []*http.Request) []int {
	t.Helper()

	handler := WebSearchHandler()
	codes := make([]int, 0, len(requests))
	for _, req := range requests {
		recorder := httptest.NewRecorder()
		handler(recorder, req)
		codes = append(codes, recorder.Code)
	}
	return codes
}

func countStatus(codes []int, want int) int {
	total := 0
	for _, code := range codes {
		if code == want {
			total++
		}
	}
	return total
}

// A caller that varies X-Forwarded-For must not land in a fresh bucket each
// time. This is the bypass that let one client drain the shared budget.
func TestWebSearchRateLimitIgnoresForwardedHeaders(t *testing.T) {
	resetRateLimiter(t)

	requests := make([]*http.Request, 0, rateLimitMaxReqs+1)
	for i := 0; i <= rateLimitMaxReqs; i++ {
		requests = append(requests, searchRequest("192.0.2.10:44321", fmt.Sprintf("10.0.0.%d", i), ""))
	}

	codes := statusesFor(t, requests)

	if got := codes[len(codes)-1]; got != http.StatusTooManyRequests {
		t.Fatalf("request %d status = %d, want %d (spoofed header bypassed the limit)",
			len(codes), got, http.StatusTooManyRequests)
	}
	if got := countStatus(codes, http.StatusTooManyRequests); got != 1 {
		t.Fatalf("refused %d of %d requests, want exactly 1", got, len(codes))
	}
}

// The same session must share one bucket regardless of where it connects from,
// and two sessions must not share one just because they share an address.
func TestWebSearchRateLimitKeysOnSession(t *testing.T) {
	t.Run("one user across many addresses shares a bucket", func(t *testing.T) {
		resetRateLimiter(t)

		requests := make([]*http.Request, 0, rateLimitMaxReqs+1)
		for i := 0; i <= rateLimitMaxReqs; i++ {
			requests = append(requests, searchRequest(
				fmt.Sprintf("192.0.2.%d:44321", i), fmt.Sprintf("10.0.0.%d", i), "user-a"))
		}

		codes := statusesFor(t, requests)
		if got := codes[len(codes)-1]; got != http.StatusTooManyRequests {
			t.Fatalf("request %d status = %d, want %d", len(codes), got, http.StatusTooManyRequests)
		}
	})

	t.Run("a second user behind the same address is unaffected", func(t *testing.T) {
		resetRateLimiter(t)

		requests := make([]*http.Request, 0, rateLimitMaxReqs+1)
		for i := 0; i <= rateLimitMaxReqs; i++ {
			requests = append(requests, searchRequest("192.0.2.10:44321", "", "user-a"))
		}
		statusesFor(t, requests)

		codes := statusesFor(t, []*http.Request{searchRequest("192.0.2.10:44321", "", "user-b")})
		if codes[0] == http.StatusTooManyRequests {
			t.Fatal("a second user was refused because another user exhausted their own window")
		}
	})
}

// One client must not be able to spend the whole upstream budget, which is what
// locked every other user out.
func TestUpstreamBudgetSurvivesOneNoisyClient(t *testing.T) {
	limiter := newTestRateLimiter()

	spent := 0
	for i := 0; i < globalRateLimit*2; i++ {
		if !limiter.allowClient("user:attacker") {
			continue
		}
		if limiter.reserveUpstream() {
			spent++
		}
	}

	if spent > rateLimitMaxReqs {
		t.Fatalf("one client spent %d of %d upstream slots, want at most %d",
			spent, globalRateLimit, rateLimitMaxReqs)
	}
	if !limiter.reserveUpstream() {
		t.Fatal("upstream budget was exhausted by a single client")
	}
}

// A request rejected before the outbound call must not spend upstream quota.
func TestRejectedRequestsDoNotSpendUpstreamBudget(t *testing.T) {
	resetRateLimiter(t)

	requests := make([]*http.Request, 0, globalRateLimit)
	for i := 0; i < globalRateLimit; i++ {
		requests = append(requests, searchRequest("192.0.2.10:44321", "", fmt.Sprintf("user-%d", i)))
	}
	statusesFor(t, requests)

	tracked, upstreamSpent := globalRateLimiter.getStats()
	if upstreamSpent != 0 {
		t.Fatalf("upstream budget spent = %d, want 0 for requests that never reached the search",
			upstreamSpent)
	}
	if tracked != globalRateLimit {
		t.Fatalf("tracked clients = %d, want %d", tracked, globalRateLimit)
	}
}

// The tracked-client cap is a memory guard. It must evict, never refuse a
// client it has not seen before.
func TestClientCapEvictsInsteadOfRefusing(t *testing.T) {
	limiter := newTestRateLimiter()

	stale := time.Now().Add(-rateLimitWindow - time.Minute)
	for i := 0; i < maxTrackedClients; i++ {
		limiter.requests[fmt.Sprintf("user:filler-%d", i)] = []time.Time{stale}
	}

	if !limiter.allowClient("user:newcomer") {
		t.Fatal("a client the limiter had never seen was refused at capacity")
	}
	if len(limiter.requests) > maxTrackedClients {
		t.Fatalf("tracked clients = %d, want at most %d", len(limiter.requests), maxTrackedClients)
	}
}

func TestClientCapEvictsLeastRecentlyActiveWhenAllAreFresh(t *testing.T) {
	limiter := newTestRateLimiter()

	now := time.Now()
	for i := 0; i < maxTrackedClients; i++ {
		limiter.requests[fmt.Sprintf("user:filler-%d", i)] = []time.Time{now.Add(-time.Duration(i) * time.Millisecond)}
	}
	oldest := fmt.Sprintf("user:filler-%d", maxTrackedClients-1)

	if !limiter.allowClient("user:newcomer") {
		t.Fatal("a new client was refused while every tracked window was still active")
	}
	if _, exists := limiter.requests[oldest]; exists {
		t.Fatal("the least recently active client was not the one evicted")
	}
}

func TestRateLimitKeyPrefersSessionOverAddress(t *testing.T) {
	tests := []struct {
		name         string
		remoteAddr   string
		forwardedFor string
		userID       string
		want         string
	}{
		{name: "session wins", remoteAddr: "192.0.2.10:44321", forwardedFor: "10.0.0.1", userID: "abc", want: "user:abc"},
		{name: "forwarded header ignored", remoteAddr: "192.0.2.10:44321", forwardedFor: "10.0.0.1", want: "ip:192.0.2.10"},
		{name: "ipv6 peer keeps its address", remoteAddr: "[2001:db8::1]:44321", want: "ip:2001:db8::1"},
		{name: "address without a port", remoteAddr: "192.0.2.10", want: "ip:192.0.2.10"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := rateLimitKey(searchRequest(tt.remoteAddr, tt.forwardedFor, tt.userID)); got != tt.want {
				t.Fatalf("rateLimitKey() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestRateLimiterStatsExcludeExpiredEntries(t *testing.T) {
	limiter := newTestRateLimiter()
	now := time.Now()
	limiter.requests["user:fresh"] = []time.Time{now.Add(-time.Second)}
	limiter.requests["user:mixed"] = []time.Time{
		now.Add(-rateLimitWindow - time.Second),
		now.Add(-time.Second),
	}
	limiter.requests["user:stale"] = []time.Time{now.Add(-rateLimitWindow - time.Second)}
	limiter.globalReqs = []time.Time{
		now.Add(-rateLimitWindow - time.Second),
		now.Add(-time.Second),
	}

	trackedClients, globalRequests := limiter.getStats()
	if trackedClients != 2 {
		t.Fatalf("tracked clients = %d, want 2", trackedClients)
	}
	if globalRequests != 1 {
		t.Fatalf("global requests = %d, want 1", globalRequests)
	}
}

func TestWebSearchRateLimitStatsHandlerReturnsAggregateOnly(t *testing.T) {
	resetRateLimiter(t)
	now := time.Now()
	globalRateLimiter.requests["user:secret"] = []time.Time{now.Add(-time.Second)}
	globalRateLimiter.globalReqs = []time.Time{now.Add(-time.Second), now.Add(-2 * time.Second)}

	beforeClients, beforeGlobal := globalRateLimiter.getStats()
	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/api/tools/web-search/stats", nil)
	WebSearchRateLimitStatsHandler()(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body=%s", response.Code, http.StatusOK, response.Body.String())
	}
	if got := response.Header().Get("Content-Type"); got != "application/json" {
		t.Fatalf("Content-Type = %q, want application/json", got)
	}
	if got := response.Header().Get("Cache-Control"); got != "no-store" {
		t.Fatalf("Cache-Control = %q, want no-store", got)
	}

	body := response.Body.String()
	var payload WebSearchRateLimitStatsResponse
	if err := json.Unmarshal([]byte(body), &payload); err != nil {
		t.Fatalf("decode stats response: %v; body=%s", err, body)
	}
	if payload.TrackedClients != beforeClients || payload.GlobalRequests != beforeGlobal {
		t.Fatalf("stats = %+v, want tracked=%d global=%d", payload, beforeClients, beforeGlobal)
	}
	if payload.WindowSeconds != int(rateLimitWindow/time.Second) ||
		payload.PerClientLimit != rateLimitMaxReqs || payload.GlobalLimit != globalRateLimit {
		t.Fatalf("stats limits = %+v, want window=%d per-client=%d global=%d", payload,
			int(rateLimitWindow/time.Second), rateLimitMaxReqs, globalRateLimit)
	}

	var fields map[string]json.RawMessage
	if err := json.Unmarshal([]byte(body), &fields); err != nil {
		t.Fatalf("decode stats fields: %v", err)
	}
	if len(fields) != 5 {
		t.Fatalf("response fields = %d, want exactly 5: %v", len(fields), fields)
	}
	for _, field := range []string{"tracked_clients", "global_requests", "window_seconds", "per_client_limit", "global_limit"} {
		if _, ok := fields[field]; !ok {
			t.Fatalf("response is missing field %q: %v", field, fields)
		}
	}
	for _, forbidden := range []string{"user:secret", "query", "192.0.2."} {
		if strings.Contains(body, forbidden) {
			t.Fatalf("response contains forbidden value %q: %s", forbidden, body)
		}
	}

	afterClients, afterGlobal := globalRateLimiter.getStats()
	if afterClients != beforeClients || afterGlobal != beforeGlobal {
		t.Fatalf("stats read changed limiter: before=(%d,%d) after=(%d,%d)",
			beforeClients, beforeGlobal, afterClients, afterGlobal)
	}
}

func TestWebSearchRateLimitStatsHandlerRejectsUnsupportedMethods(t *testing.T) {
	resetRateLimiter(t)
	for _, method := range []string{http.MethodPost, http.MethodOptions} {
		t.Run(method, func(t *testing.T) {
			response := httptest.NewRecorder()
			request := httptest.NewRequest(method, "/api/tools/web-search/stats", nil)
			WebSearchRateLimitStatsHandler()(response, request)
			if response.Code != http.StatusMethodNotAllowed {
				t.Fatalf("status = %d, want %d", response.Code, http.StatusMethodNotAllowed)
			}
			if response.Header().Get("Allow") != http.MethodGet {
				t.Fatalf("Allow = %q, want GET", response.Header().Get("Allow"))
			}
		})
	}
}
