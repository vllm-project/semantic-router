package handlers

import (
	"context"
	"encoding/json"
	"log"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
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
	maxQueryLength    = 500         // Maximum query length
	maxResultsLimit   = 10          // Maximum results per request
	defaultNumResults = 5           // Default number of results
	rateLimitWindow   = time.Minute // Rate limit window
	rateLimitMaxReqs  = 5           // Max requests per window per IP (strict for public service)
	globalRateLimit   = 30          // Global max requests per window (0.5 req/sec to avoid ban)
	maxTrackedIPs     = 10000       // Max IPs to track (memory protection)
)

// ========================
// Rate limiter with global limit
// ========================

type rateLimiter struct {
	mu         sync.RWMutex
	requests   map[string][]time.Time
	globalReqs []time.Time // Track all requests globally
}

var globalRateLimiter = newRateLimiter()

func newRateLimiter() *rateLimiter {
	return &rateLimiter{
		requests:   make(map[string][]time.Time),
		globalReqs: make([]time.Time, 0),
	}
}

// isAllowed checks if a request from the given IP is allowed
func (rl *rateLimiter) isAllowed(ip string) (bool, SearchErrorCode) {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	windowStart := now.Add(-rateLimitWindow)

	// Check global rate limit first
	var recentGlobal []time.Time
	for _, t := range rl.globalReqs {
		if t.After(windowStart) {
			recentGlobal = append(recentGlobal, t)
		}
	}
	if len(recentGlobal) >= globalRateLimit {
		rl.globalReqs = recentGlobal
		return false, ErrCodeRateLimited
	}

	// Memory protection: limit tracked IPs
	if len(rl.requests) >= maxTrackedIPs {
		if _, exists := rl.requests[ip]; !exists {
			return false, ErrCodeRateLimited
		}
	}

	// Check per-IP rate limit
	var recentReqs []time.Time
	for _, t := range rl.requests[ip] {
		if t.After(windowStart) {
			recentReqs = append(recentReqs, t)
		}
	}

	if len(recentReqs) >= rateLimitMaxReqs {
		rl.requests[ip] = recentReqs
		return false, ErrCodeRateLimited
	}

	// Record the request
	rl.requests[ip] = append(recentReqs, now)
	recentGlobal = append(recentGlobal, now)
	rl.globalReqs = recentGlobal
	return true, ""
}

// cleanup removes stale entries (call periodically)
func (rl *rateLimiter) cleanup() {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	windowStart := time.Now().Add(-rateLimitWindow)

	// Clean per-IP entries
	for ip, times := range rl.requests {
		var valid []time.Time
		for _, t := range times {
			if t.After(windowStart) {
				valid = append(valid, t)
			}
		}
		if len(valid) == 0 {
			delete(rl.requests, ip)
		} else {
			rl.requests[ip] = valid
		}
	}

	// Clean global entries
	var validGlobal []time.Time
	for _, t := range rl.globalReqs {
		if t.After(windowStart) {
			validGlobal = append(validGlobal, t)
		}
	}
	rl.globalReqs = validGlobal
}

// getStats returns current rate limiter statistics (for monitoring/debugging)
// nolint:unused // Reserved for future monitoring endpoint
func (rl *rateLimiter) getStats() (trackedIPs int, globalReqCount int) {
	rl.mu.RLock()
	defer rl.mu.RUnlock()
	return len(rl.requests), len(rl.globalReqs)
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

// SearchResult represents a single search result
type SearchResult struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Snippet string `json:"snippet"`
	Domain  string `json:"domain"`
}

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

// ========================
// HTTP Handler
// ========================

// getClientIP extracts client IP from request
func getClientIP(r *http.Request) string {
	if r == nil {
		return "unknown"
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err == nil && host != "" {
		return host
	}
	if parsed := net.ParseIP(r.RemoteAddr); parsed != nil {
		return parsed.String()
	}
	return "unknown"
}

// sendErrorResponse sends a JSON error response with code
func sendErrorResponse(w http.ResponseWriter, code SearchErrorCode, status int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(WebSearchResponse{
		Error: errorMessages[code],
		Code:  code,
	})
}

// WebSearchHandler handles web search requests
func WebSearchHandler() http.HandlerFunc {
	search := newWebSearchOutbound(newPublicOutboundHTTPClient(webSearchHTTPTimeout))
	return webSearchHandlerWithDependencies(search, globalRateLimiter)
}

func webSearchHandlerWithDependencies(
	search *webSearchOutbound,
	limiter *rateLimiter,
) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
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

		// Rate limiting check (per-IP + global)
		clientIP := getClientIP(r)
		allowed, errCode := limiter.isAllowed(clientIP)
		if !allowed {
			log.Printf("[WebSearch] Rate limit exceeded: code=%s", errCode)
			sendErrorResponse(w, errCode, http.StatusTooManyRequests)
			return
		}

		// Parse request body
		var req WebSearchRequest
		if status, err := decodeBoundedJSON(w, r, outboundMaxRequestBodyBytes, &req); err != nil {
			log.Printf("[WebSearch] Invalid request body")
			sendErrorResponse(w, ErrCodeInvalidRequest, status)
			return
		}

		// Validate query
		req.Query = strings.TrimSpace(req.Query)
		if req.Query == "" {
			sendErrorResponse(w, ErrCodeEmptyQuery, http.StatusBadRequest)
			return
		}
		if utf8.RuneCountInString(req.Query) > maxQueryLength {
			sendErrorResponse(w, ErrCodeQueryTooLong, http.StatusBadRequest)
			return
		}

		log.Printf(
			"[WebSearch] Request: query_length=%d num_results=%d",
			utf8.RuneCountInString(req.Query),
			req.NumResults,
		)

		// Perform search with retry
		searchCtx, cancel := context.WithTimeout(r.Context(), webSearchHTTPTimeout)
		defer cancel()
		results, err := search.searchDuckDuckGo(searchCtx, req.Query, req.NumResults)
		if err != nil {
			errCode := webSearchErrorCode(err)
			log.Printf("[WebSearch] Search failed: code=%s", errCode)
			statusCode := http.StatusInternalServerError
			switch errCode {
			case ErrCodeRateLimited:
				statusCode = http.StatusTooManyRequests
			case ErrCodeTimeout:
				statusCode = http.StatusGatewayTimeout
			case ErrCodeUpstreamError:
				statusCode = http.StatusBadGateway
			}

			sendErrorResponse(w, errCode, statusCode)
			return
		}

		log.Printf("[WebSearch] Completed: results=%d", len(results))

		// Send response
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(WebSearchResponse{
			Query:   req.Query,
			Results: results,
		})
	}
}
