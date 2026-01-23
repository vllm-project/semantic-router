package handlers

import (
	"encoding/json"
	"fmt"
	"html"
	"io"
	"log"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"sync"
	"time"
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
	ErrCodeRateLimited:    "Rate limit exceeded, please try again later / 请求过于频繁，请稍后重试",
	ErrCodeSearchFailed:   "Search failed / 搜索失败",
	ErrCodeTimeout:        "Search timeout / 搜索超时",
	ErrCodeUpstreamError:  "Search service unavailable / 搜索服务不可用",
	ErrCodeInvalidURL:     "Invalid URL detected / 检测到无效URL",
}

// ========================
// Configuration constants
// ========================

const (
	maxQueryLength    = 500 // Maximum query length
	maxResultsLimit   = 10  // Maximum results per request
	defaultNumResults = 5   // Default number of results
	httpTimeout       = 10 * time.Second
	maxRetries        = 3 // Retry attempts for transient failures
	retryBaseDelay    = 500 * time.Millisecond
	rateLimitWindow   = time.Minute // Rate limit window
	rateLimitMaxReqs  = 30          // Max requests per window per IP
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
// Rate limiter
// ========================

type rateLimiter struct {
	mu       sync.RWMutex
	requests map[string][]time.Time
}

var globalRateLimiter = &rateLimiter{
	requests: make(map[string][]time.Time),
}

// isAllowed checks if a request from the given IP is allowed
func (rl *rateLimiter) isAllowed(ip string) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	windowStart := now.Add(-rateLimitWindow)

	// Clean old entries and count recent requests
	var recentReqs []time.Time
	for _, t := range rl.requests[ip] {
		if t.After(windowStart) {
			recentReqs = append(recentReqs, t)
		}
	}

	if len(recentReqs) >= rateLimitMaxReqs {
		rl.requests[ip] = recentReqs
		return false
	}

	rl.requests[ip] = append(recentReqs, now)
	return true
}

// cleanup removes stale entries (call periodically)
func (rl *rateLimiter) cleanup() {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	windowStart := time.Now().Add(-rateLimitWindow)
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
			log.Printf("Retrying search (attempt %d/%d) after error: %v", attempt+2, maxRetries, err)
		}
	}

	return nil, lastErr
}

// doSearchRequest performs a single search request
func doSearchRequest(query string, numResults int) ([]SearchResult, error) {
	// Use DuckDuckGo HTML interface
	searchURL := fmt.Sprintf("https://html.duckduckgo.com/html/?q=%s", url.QueryEscape(query))

	client := getHTTPClient()

	req, err := http.NewRequest("GET", searchURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers to mimic a browser request
	req.Header.Set("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
	req.Header.Set("Accept-Language", "en-US,en;q=0.5")

	resp, err := client.Do(req)
	if err != nil {
		if strings.Contains(err.Error(), "timeout") {
			return nil, fmt.Errorf("%s: %w", ErrCodeTimeout, err)
		}
		return nil, fmt.Errorf("%s: %w", ErrCodeSearchFailed, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s: status %d", ErrCodeUpstreamError, resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
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
			log.Printf("Skipping invalid URL: %s", actualURL)
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

// getClientIP extracts client IP from request
func getClientIP(r *http.Request) string {
	// Check X-Forwarded-For header (for proxied requests)
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		parts := strings.Split(xff, ",")
		return strings.TrimSpace(parts[0])
	}
	// Check X-Real-IP header
	if xri := r.Header.Get("X-Real-IP"); xri != "" {
		return xri
	}
	// Fall back to RemoteAddr
	ip := r.RemoteAddr
	if idx := strings.LastIndex(ip, ":"); idx != -1 {
		ip = ip[:idx]
	}
	return ip
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

		// Rate limiting check
		clientIP := getClientIP(r)
		if !globalRateLimiter.isAllowed(clientIP) {
			log.Printf("Rate limit exceeded for IP: %s", clientIP)
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

		log.Printf("Web search request: query=%q, num_results=%d, ip=%s", req.Query, req.NumResults, clientIP)

		// Perform search with retry
		results, err := searchDuckDuckGo(req.Query, req.NumResults)
		if err != nil {
			log.Printf("Search error: %v", err)

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

		log.Printf("Web search completed: query=%q, results=%d", req.Query, len(results))

		// Send response
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(WebSearchResponse{
			Query:   req.Query,
			Results: results,
		})
	}
}
