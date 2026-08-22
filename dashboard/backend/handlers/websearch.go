package handlers

import (
	"encoding/json"
	"errors"
	"fmt"
	"html"
	"io"
	"log"
	"math/rand"
	"net"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
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
	maxQueryLength     = 500 // Maximum query length
	maxResultsLimit    = 10  // Maximum results per request
	defaultNumResults  = 5   // Default number of results
	httpTimeout        = 15 * time.Second
	maxRetries         = 3 // Retry attempts for transient failures
	retryBaseDelay     = 1 * time.Second
	rateLimitWindow    = time.Minute // Rate limit window
	rateLimitMaxReqs   = 5           // Max requests per window per client (strict for public service)
	globalRateLimit    = 30          // Global max requests per window (0.5 req/sec to avoid ban)
	maxConcurrent      = 2           // Max concurrent outgoing requests (very conservative)
	maxTrackedClients  = 10000       // Max clients to track (memory protection)
	minRequestInterval = 1 * time.Second
	maxRequestInterval = 3 * time.Second
)

// ========================
// Pre-compiled regex patterns (performance optimization)
// ========================

var (
	// Pattern to match search result links
	linkPattern = regexp.MustCompile(`<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>([^<]*)</a>`)
	// Pattern to match snippets
	snippetPattern = regexp.MustCompile(`<a[^>]*class="result__snippet"[^>]*>([^<]*(?:<[^>]*>[^<]*</[^>]*>)*[^<]*)</a>`)
	// Alternative snippet pattern
	snippetAltPattern = regexp.MustCompile(`class="result__snippet"[^>]*>([^<]+)`)
	// Pattern for uddg URL extraction
	uddgPattern = regexp.MustCompile(`uddg=([^&"]+)`)
	// Pattern for title extraction in fallback
	titlePattern = regexp.MustCompile(`result__a[^>]*>([^<]+)`)
	// Pattern for snippet extraction in fallback
	snippetFallbackPattern = regexp.MustCompile(`result__snippet[^>]*>([^<]+)`)
	// Pattern for HTML tag removal
	htmlTagPattern = regexp.MustCompile(`<[^>]*>`)
	// Pattern for whitespace normalization
	whitespacePattern = regexp.MustCompile(`\s+`)
)

// User-Agent rotation to reduce detection risk
var userAgents = []string{
	"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
	"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
	"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
	"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
	"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
	"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
}

// getRandomUserAgent returns a random User-Agent string
func getRandomUserAgent() string {
	return userAgents[rand.Intn(len(userAgents))]
}

// randomDelay adds a random delay between min and max intervals to avoid detection
func randomDelay() {
	delay := minRequestInterval + time.Duration(rand.Int63n(int64(maxRequestInterval-minRequestInterval)))
	time.Sleep(delay)
}

// ========================
// HTTP Client pool (connection reuse)
// ========================

var (
	httpClient     *http.Client
	httpClientOnce sync.Once
)

// getHTTPClient returns a shared HTTP client with connection pooling
func getHTTPClient() *http.Client {
	httpClientOnce.Do(func() {
		transport := &http.Transport{
			MaxIdleConns:        100,
			MaxIdleConnsPerHost: 10,
			IdleConnTimeout:     90 * time.Second,
			DisableCompression:  false,
		}
		httpClient = &http.Client{
			Timeout:   httpTimeout,
			Transport: transport,
		}
	})
	return httpClient
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

// Semaphore for concurrent request limiting
var concurrentSem = make(chan struct{}, maxConcurrent)

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

// getStats returns current rate limiter statistics (for monitoring/debugging)
// nolint:unused // Reserved for future monitoring endpoint
func (rl *rateLimiter) getStats() (trackedClients int, globalReqCount int) {
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
// URL validation (XSS prevention)
// ========================

// isValidURL validates URL to prevent XSS attacks
// Only allows http and https schemes, rejects javascript:, data:, vbscript:, etc.
func isValidURL(urlStr string) bool {
	if urlStr == "" {
		return false
	}

	parsed, err := url.Parse(urlStr)
	if err != nil {
		return false
	}

	// Only allow safe schemes
	scheme := strings.ToLower(parsed.Scheme)
	if scheme != "http" && scheme != "https" {
		return false
	}

	// Must have a valid host
	if parsed.Host == "" {
		return false
	}

	return true
}

// extractDomain extracts the domain from a URL
func extractDomain(urlStr string) string {
	parsed, err := url.Parse(urlStr)
	if err != nil {
		return ""
	}
	return parsed.Host
}

// ========================
// Search with retry mechanism
// ========================

// searchDuckDuckGo performs a search using DuckDuckGo's HTML interface with retry
func searchDuckDuckGo(query string, numResults int) ([]SearchResult, error) {
	if numResults <= 0 {
		numResults = defaultNumResults
	}
	if numResults > maxResultsLimit {
		numResults = maxResultsLimit
	}

	var lastErr error
	for attempt := 0; attempt < maxRetries; attempt++ {
		results, err := doSearchRequest(query, numResults)
		if err == nil {
			return results, nil
		}

		lastErr = err

		// Don't retry on non-transient errors
		if strings.Contains(err.Error(), "status 4") {
			break
		}

		// Exponential backoff
		if attempt < maxRetries-1 {
			delay := retryBaseDelay * time.Duration(1<<attempt)
			time.Sleep(delay)
			log.Printf("Retrying search (attempt %d/%d, error_class=%T)", attempt+2, maxRetries, err)
		}
	}

	return nil, lastErr
}

// doSearchRequest performs a single search request with concurrency control
func doSearchRequest(query string, numResults int) ([]SearchResult, error) {
	// Acquire semaphore to limit concurrent outgoing requests
	select {
	case concurrentSem <- struct{}{}:
		defer func() { <-concurrentSem }()
	default:
		// If semaphore is full, return rate limit error immediately
		return nil, fmt.Errorf("%s: too many concurrent requests", ErrCodeRateLimited)
	}

	// Add random delay to avoid detection (1-3 seconds)
	randomDelay()

	// Use DuckDuckGo HTML interface
	searchURL := fmt.Sprintf("https://html.duckduckgo.com/html/?q=%s", url.QueryEscape(query))

	client := getHTTPClient()

	req, err := http.NewRequest("GET", searchURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers to mimic a browser request with rotating User-Agent
	// Note: Do NOT set Accept-Encoding manually, let Go's http.Transport handle compression
	// automatically. When Accept-Encoding is set manually, Go won't auto-decompress the response,
	// which causes garbled data if the server returns gzip/deflate compressed content.
	req.Header.Set("User-Agent", getRandomUserAgent())
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8")
	req.Header.Set("Accept-Language", "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7")
	req.Header.Set("DNT", "1")
	req.Header.Set("Connection", "keep-alive")
	req.Header.Set("Upgrade-Insecure-Requests", "1")

	resp, err := client.Do(req)
	if err != nil {
		var networkError net.Error
		if errors.As(err, &networkError) && networkError.Timeout() {
			return nil, fmt.Errorf("%s: upstream request timed out", ErrCodeTimeout)
		}
		return nil, fmt.Errorf("%s: upstream request failed", ErrCodeSearchFailed)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s: status %d", ErrCodeUpstreamError, resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	// Keep response diagnostics content-free. Search HTML may contain the full
	// user query, snippets, credentials, or tenant data.
	bodyStr := string(body)
	log.Printf("DuckDuckGo response: status=%d, length=%d", resp.StatusCode, len(body))
	// Check for common blocking indicators
	if strings.Contains(bodyStr, "captcha") || strings.Contains(bodyStr, "robot") || strings.Contains(bodyStr, "blocked") {
		log.Printf("WARNING: DuckDuckGo may be blocking requests (captcha/robot detected)")
	}

	return parseDuckDuckGoHTML(string(body), numResults)
}

// ========================
// HTML parsing (using pre-compiled regex)
// ========================

// parseDuckDuckGoHTML parses DuckDuckGo HTML search results
func parseDuckDuckGoHTML(htmlContent string, maxResults int) ([]SearchResult, error) {
	var results []SearchResult

	// Match result links - the href contains a redirect URL
	linkMatches := linkPattern.FindAllStringSubmatch(htmlContent, -1)

	// Match snippets
	snippetMatches := snippetPattern.FindAllStringSubmatch(htmlContent, -1)

	// Alternative pattern for snippets (sometimes in span)
	if len(snippetMatches) == 0 {
		snippetMatches = snippetAltPattern.FindAllStringSubmatch(htmlContent, -1)
	}

	for i := 0; i < len(linkMatches) && i < maxResults; i++ {
		if len(linkMatches[i]) < 3 {
			continue
		}

		rawURL := linkMatches[i][1]
		title := strings.TrimSpace(linkMatches[i][2])

		// Clean up title - decode HTML entities using standard library
		title = html.UnescapeString(title)
		title = cleanExtraWhitespace(title)

		// Extract actual URL from DuckDuckGo redirect
		actualURL := extractActualURL(rawURL)
		if actualURL == "" {
			continue
		}

		// Security: Validate URL to prevent XSS
		if !isValidURL(actualURL) {
			log.Printf("Skipping invalid search-result URL (url_bytes=%d)", len(actualURL))
			continue
		}

		// Get snippet if available
		snippet := ""
		if i < len(snippetMatches) && len(snippetMatches[i]) > 1 {
			snippet = cleanHTMLTags(snippetMatches[i][1])
			snippet = html.UnescapeString(snippet)
			snippet = cleanExtraWhitespace(snippet)
		}

		// Skip if URL is DuckDuckGo internal
		if strings.Contains(actualURL, "duckduckgo.com") {
			continue
		}

		results = append(results, SearchResult{
			Title:   title,
			URL:     actualURL,
			Snippet: snippet,
			Domain:  extractDomain(actualURL),
		})
	}

	// If regex parsing failed, try a simpler approach
	if len(results) == 0 {
		results = parseSimpleHTML(htmlContent, maxResults)
	}

	return results, nil
}

// parseSimpleHTML is a fallback parser for DuckDuckGo results
func parseSimpleHTML(htmlContent string, maxResults int) []SearchResult {
	var results []SearchResult

	// Split by result blocks
	blocks := strings.Split(htmlContent, "result__body")

	for i, block := range blocks {
		if i == 0 || i > maxResults {
			continue
		}

		// Extract URL using pre-compiled pattern
		urlMatch := uddgPattern.FindStringSubmatch(block)
		if len(urlMatch) < 2 {
			continue
		}
		actualURL, _ := url.QueryUnescape(urlMatch[1])
		if actualURL == "" || strings.Contains(actualURL, "duckduckgo") {
			continue
		}

		// Security: Validate URL
		if !isValidURL(actualURL) {
			continue
		}

		// Extract title using pre-compiled pattern
		titleMatch := titlePattern.FindStringSubmatch(block)
		title := ""
		if len(titleMatch) > 1 {
			title = html.UnescapeString(strings.TrimSpace(titleMatch[1]))
		}

		// Extract snippet using pre-compiled pattern
		snippetMatch := snippetFallbackPattern.FindStringSubmatch(block)
		snippet := ""
		if len(snippetMatch) > 1 {
			snippet = html.UnescapeString(strings.TrimSpace(snippetMatch[1]))
		}

		if title != "" {
			results = append(results, SearchResult{
				Title:   title,
				URL:     actualURL,
				Snippet: snippet,
				Domain:  extractDomain(actualURL),
			})
		}
	}

	return results
}

// extractActualURL extracts the actual URL from DuckDuckGo's redirect URL
func extractActualURL(ddgURL string) string {
	// DuckDuckGo uses a redirect format like: //duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com
	if strings.Contains(ddgURL, "uddg=") {
		parsed, err := url.Parse(ddgURL)
		if err == nil {
			uddg := parsed.Query().Get("uddg")
			if uddg != "" {
				return uddg
			}
		}
		// Try regex extraction as fallback using pre-compiled pattern
		matches := uddgPattern.FindStringSubmatch(ddgURL)
		if len(matches) > 1 {
			decoded, err := url.QueryUnescape(matches[1])
			if err == nil {
				return decoded
			}
			return matches[1]
		}
	}

	// If it's already a direct URL
	if strings.HasPrefix(ddgURL, "http://") || strings.HasPrefix(ddgURL, "https://") {
		return ddgURL
	}

	return ""
}

// ========================
// Utility functions (using pre-compiled regex)
// ========================

// cleanHTMLTags removes HTML tags from a string
func cleanHTMLTags(s string) string {
	return strings.TrimSpace(htmlTagPattern.ReplaceAllString(s, " "))
}

// cleanExtraWhitespace normalizes whitespace in a string
func cleanExtraWhitespace(s string) string {
	return strings.TrimSpace(whitespacePattern.ReplaceAllString(s, " "))
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

		// Shared upstream budget, charged only for requests that reach DuckDuckGo
		if !globalRateLimiter.reserveUpstream() {
			log.Printf("Upstream search budget exhausted, rejecting client: %s", clientKey)
			sendErrorResponse(w, req.Query, ErrCodeRateLimited, http.StatusTooManyRequests)
			return
		}

		log.Printf("Web search request: query_bytes=%d, num_results=%d, client=%s", len(req.Query), req.NumResults, clientKey)

		// Perform search with retry
		results, err := searchDuckDuckGo(req.Query, req.NumResults)
		if err != nil {
			log.Printf("Search failed: error_class=%T", err)

			// Determine appropriate error code
			errCode := ErrCodeSearchFailed
			statusCode := http.StatusInternalServerError

			errStr := err.Error()
			if strings.Contains(errStr, string(ErrCodeTimeout)) {
				errCode = ErrCodeTimeout
				statusCode = http.StatusGatewayTimeout
			} else if strings.Contains(errStr, string(ErrCodeUpstreamError)) {
				errCode = ErrCodeUpstreamError
				statusCode = http.StatusBadGateway
			}

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
