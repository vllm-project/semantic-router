package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"log"
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/websearch/duckduckgo"
)

// ========================
// Error codes for better user feedback
// ========================

type SearchErrorCode string

const (
	ErrCodeInvalidRequest SearchErrorCode = "INVALID_REQUEST"
	ErrCodeEmptyQuery     SearchErrorCode = "EMPTY_QUERY"
	ErrCodeQueryTooLong   SearchErrorCode = "QUERY_TOO_LONG"
	ErrCodeRateLimited    SearchErrorCode = "RATE_LIMITED"
	ErrCodeSearchFailed   SearchErrorCode = "SEARCH_FAILED"
	ErrCodeTimeout        SearchErrorCode = "TIMEOUT"
	ErrCodeUpstreamError  SearchErrorCode = "UPSTREAM_ERROR"
	ErrCodeInvalidURL     SearchErrorCode = "INVALID_URL"
)

// Error messages in both English and Chinese for better UX
var errorMessages = map[SearchErrorCode]string{
	ErrCodeInvalidRequest: "Invalid request body / 无效的请求体",
	ErrCodeEmptyQuery:     "Search query is required / 搜索查询不能为空",
	ErrCodeQueryTooLong:   "Query too long (max 500 chars) / 查询过长（最多500字符）",
	ErrCodeRateLimited:    "Search service is busy, please try again in a moment / 搜索服务繁忙，请稍后再试",
	ErrCodeSearchFailed:   "Search failed / 搜索失败",
	ErrCodeTimeout:        "Search timeout, please try again / 搜索超时，请重试",
	ErrCodeUpstreamError:  "Search service temporarily unavailable / 搜索服务暂时不可用",
	ErrCodeInvalidURL:     "Invalid URL detected / 检测到无效URL",
}

// ========================
// Configuration constants
// ========================

const (
	rateLimitWindow   = time.Minute // Rate limit window
	rateLimitMaxReqs  = 5           // Max requests per window per client (strict for public service)
	globalRateLimit   = 30          // Global max requests per window (0.5 req/sec to avoid ban)
	maxTrackedClients = 10000       // Max clients to track (memory protection)
)

// searchClient performs the upstream search. The DuckDuckGo client is shared
// with other repository tools and owns the browser headers, the pause between
// requests, the concurrency cap and the retry policy; this handler owns who
// may search and how often, and lends the client its process-wide budget.
// Tests swap it for one aimed at a fake endpoint.
var searchClient = duckduckgo.NewClient(
	duckduckgo.WithLogf(log.Printf),
	duckduckgo.WithBudget(reserveUpstreamBudget),
)

// reserveUpstreamBudget charges the shared budget that keeps DuckDuckGo from
// banning this process's egress address. It looks the limiter up on every
// call so tests that swap globalRateLimiter are charged on their own copy.
func reserveUpstreamBudget() bool {
	return globalRateLimiter.reserveUpstream()
}

// ========================
// Rate limiter with global limit
// ========================

type rateLimiter struct {
	mu         sync.RWMutex
	requests   map[string][]time.Time
	globalReqs []time.Time // Track all requests globally
}

var globalRateLimiter = &rateLimiter{
	requests:   make(map[string][]time.Time),
	globalReqs: make([]time.Time, 0),
}

// allowClient charges one request against the caller's own window. It is the
// abuse control, so it is checked before the request body is read: a malformed
// request still costs the sender a slot.
func (rl *rateLimiter) allowClient(key string) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	windowStart := now.Add(-rateLimitWindow)

	if _, tracked := rl.requests[key]; !tracked && len(rl.requests) >= maxTrackedClients {
		rl.evictLocked(windowStart)
	}

	recent := withinWindow(rl.requests[key], windowStart)
	if len(recent) >= rateLimitMaxReqs {
		rl.requests[key] = recent
		return false
	}

	recent = append(recent, now)
	rl.requests[key] = recent
	return true
}

// reserveUpstream charges one request against the shared budget that keeps
// DuckDuckGo from banning us. It is taken immediately before the outbound call
// so requests rejected earlier do not spend somebody else's quota.
func (rl *rateLimiter) reserveUpstream() bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	recent := withinWindow(rl.globalReqs, now.Add(-rateLimitWindow))
	if len(recent) >= globalRateLimit {
		rl.globalReqs = recent
		return false
	}

	recent = append(recent, now)
	rl.globalReqs = recent
	return true
}

// evictLocked frees a slot for an untracked client. Refusing unknown clients
// would turn a memory guard into a way to deny service, so drop expired windows
// first and the least recently active client only if that was not enough.
func (rl *rateLimiter) evictLocked(windowStart time.Time) {
	for key, times := range rl.requests {
		if len(times) == 0 || !times[len(times)-1].After(windowStart) {
			delete(rl.requests, key)
		}
	}
	if len(rl.requests) < maxTrackedClients {
		return
	}

	oldestKey := ""
	var oldestSeen time.Time
	for key, times := range rl.requests {
		lastSeen := times[len(times)-1]
		if oldestKey == "" || lastSeen.Before(oldestSeen) {
			oldestKey, oldestSeen = key, lastSeen
		}
	}
	delete(rl.requests, oldestKey)
}

// withinWindow returns the timestamps that are still inside the window.
func withinWindow(times []time.Time, windowStart time.Time) []time.Time {
	kept := times[:0:0]
	for _, t := range times {
		if t.After(windowStart) {
			kept = append(kept, t)
		}
	}
	return kept
}

// countWithinWindow counts timestamps that are still inside the rolling
// limiter window without allocating a filtered slice.
func countWithinWindow(times []time.Time, windowStart time.Time) int {
	count := 0
	for _, t := range times {
		if t.After(windowStart) {
			count++
		}
	}
	return count
}

// cleanup removes stale entries (call periodically)
func (rl *rateLimiter) cleanup() {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	windowStart := time.Now().Add(-rateLimitWindow)

	for key, times := range rl.requests {
		valid := withinWindow(times, windowStart)
		if len(valid) == 0 {
			delete(rl.requests, key)
		} else {
			rl.requests[key] = valid
		}
	}

	rl.globalReqs = withinWindow(rl.globalReqs, windowStart)
}

// getStats returns current process-local rate limiter statistics for the
// rolling window. Expired client keys remain eligible for periodic cleanup but
// are excluded from the snapshot.
func (rl *rateLimiter) getStats() (trackedClients int, globalReqCount int) {
	rl.mu.RLock()
	defer rl.mu.RUnlock()

	windowStart := time.Now().Add(-rateLimitWindow)
	for _, times := range rl.requests {
		if countWithinWindow(times, windowStart) > 0 {
			trackedClients++
		}
	}
	return trackedClients, countWithinWindow(rl.globalReqs, windowStart)
}

// Start cleanup goroutine
func init() {
	go func() {
		ticker := time.NewTicker(5 * time.Minute)
		for range ticker.C {
			globalRateLimiter.cleanup()
		}
	}()
}

// ========================
// Data structures
// ========================

// SearchResult represents a single search result. Its wire names are the
// shared client's, so the Dashboard response stays stable across providers.
type SearchResult = duckduckgo.Result

// WebSearchRequest represents the incoming search request
type WebSearchRequest struct {
	Query      string `json:"query"`
	NumResults int    `json:"num_results,omitempty"`
}

// WebSearchResponse represents the search response
type WebSearchResponse struct {
	Query   string          `json:"query"`
	Results []SearchResult  `json:"results"`
	Error   string          `json:"error,omitempty"`
	Code    SearchErrorCode `json:"code,omitempty"`
}

// WebSearchRateLimitStatsResponse is the safe aggregate response for the
// WebSearch rate limiter. Counts are process-local and cover the current
// rolling window only.
type WebSearchRateLimitStatsResponse struct {
	TrackedClients int `json:"tracked_clients"`
	GlobalRequests int `json:"global_requests"`
	WindowSeconds  int `json:"window_seconds"`
	PerClientLimit int `json:"per_client_limit"`
	GlobalLimit    int `json:"global_limit"`
}

// ========================
// HTTP Handler
// ========================

// rateLimitKey identifies who to charge for a request.
//
// Every /api/ route is authenticated, so the session user ID is available and,
// unlike a forwarded header, cannot be chosen by the caller. Forwarded headers
// are deliberately not consulted: a key the caller picks is no limit at all.
func rateLimitKey(r *http.Request) string {
	if ac, ok := auth.AuthFromContext(r); ok && ac.UserID != "" {
		return "user:" + ac.UserID
	}
	return "ip:" + remoteHost(r)
}

// remoteHost returns the peer address without its port.
func remoteHost(r *http.Request) string {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}

// sendErrorResponse sends a JSON error response with code
func sendErrorResponse(w http.ResponseWriter, query string, code SearchErrorCode, status int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(WebSearchResponse{
		Query: query,
		Error: errorMessages[code],
		Code:  code,
	})
}

// searchErrorStatus maps a shared-client failure onto the response contract:
// an exhausted upstream budget is a rate limit, a timeout (the client's own
// or the request's deadline) is a gateway timeout, a non-200 upstream answer
// (including the challenge page a restricted egress address receives) is an
// upstream error, and anything else is a generic search failure.
func searchErrorStatus(err error) (SearchErrorCode, int) {
	var statusErr *duckduckgo.StatusError
	switch {
	case errors.Is(err, duckduckgo.ErrBudgetExhausted):
		return ErrCodeRateLimited, http.StatusTooManyRequests
	case errors.Is(err, duckduckgo.ErrTimeout), errors.Is(err, context.DeadlineExceeded):
		return ErrCodeTimeout, http.StatusGatewayTimeout
	case errors.As(err, &statusErr):
		return ErrCodeUpstreamError, http.StatusBadGateway
	default:
		return ErrCodeSearchFailed, http.StatusInternalServerError
	}
}

// WebSearchRateLimitStatsHandler returns an authenticated, aggregate-only
// snapshot of the WebSearch rate limiter.
func WebSearchRateLimitStatsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// PolicyMux may pass OPTIONS through as a preflight candidate; stats must
		// remain GET-only so preflight cannot expose the aggregate snapshot.
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			w.Header().Set("Allow", "GET, HEAD")
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		trackedClients, globalRequests := globalRateLimiter.getStats()
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		if err := json.NewEncoder(w).Encode(WebSearchRateLimitStatsResponse{
			TrackedClients: trackedClients,
			GlobalRequests: globalRequests,
			WindowSeconds:  int(rateLimitWindow / time.Second),
			PerClientLimit: rateLimitMaxReqs,
			GlobalLimit:    globalRateLimit,
		}); err != nil {
			log.Printf("Error encoding web search rate-limit stats: %v", err)
		}
	}
}

// WebSearchHandler handles web search requests
func WebSearchHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Set CORS headers
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		// Handle preflight
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		// Only allow POST requests
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Per-client rate limiting
		clientKey := rateLimitKey(r)
		if !globalRateLimiter.allowClient(clientKey) {
			log.Printf("Rate limit exceeded for client: %s", clientKey)
			sendErrorResponse(w, "", ErrCodeRateLimited, http.StatusTooManyRequests)
			return
		}

		// Parse request body
		var req WebSearchRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			log.Printf("Error parsing search request: %v", err)
			sendErrorResponse(w, "", ErrCodeInvalidRequest, http.StatusBadRequest)
			return
		}

		// Validate query
		if req.Query == "" {
			sendErrorResponse(w, req.Query, ErrCodeEmptyQuery, http.StatusBadRequest)
			return
		}

		if auth.RejectRevokedMutation(w, r) {
			return
		}
		log.Printf("Web search request: query_bytes=%d, num_results=%d, client=%s", len(req.Query), req.NumResults, clientKey)

		// The client charges the shared upstream budget right before its first
		// attempt, so requests rejected above never spend somebody else's quota.
		results, err := searchClient.Search(r.Context(), req.Query, req.NumResults)
		if err != nil {
			if errors.Is(err, duckduckgo.ErrBudgetExhausted) {
				log.Printf("Upstream search budget exhausted, rejecting client: %s", clientKey)
			} else {
				log.Printf("Search failed: error_class=%T", err)
			}
			errCode, statusCode := searchErrorStatus(err)
			sendErrorResponse(w, req.Query, errCode, statusCode)
			return
		}

		log.Printf("Web search completed: query_bytes=%d, results=%d", len(req.Query), len(results))

		// Send response
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(WebSearchResponse{
			Query:   req.Query,
			Results: results,
		})
	}
}
