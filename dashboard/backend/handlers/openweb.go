package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"
)

// ========================
// Configuration constants for OpenWeb
// ========================

const (
	openWebMaxContentLength = 15000               // Maximum content length (characters)
	openWebDefaultTimeout   = 10 * time.Second    // Default timeout
	openWebMaxTimeout       = 30 * time.Second    // Maximum timeout
	jinaReaderBaseURL       = "https://r.jina.ai" // Jina Reader API
)

// ========================
// Pre-compiled regex patterns for HTML cleaning
// ========================

var (
	// Tags to be removed (Go regexp doesn't support backreferences, so process each tag separately)
	scriptPattern   = regexp.MustCompile(`(?is)<script[^>]*>.*?</script>`)
	stylePattern    = regexp.MustCompile(`(?is)<style[^>]*>.*?</style>`)
	navPattern      = regexp.MustCompile(`(?is)<nav[^>]*>.*?</nav>`)
	headerPattern   = regexp.MustCompile(`(?is)<header[^>]*>.*?</header>`)
	footerPattern   = regexp.MustCompile(`(?is)<footer[^>]*>.*?</footer>`)
	noscriptPattern = regexp.MustCompile(`(?is)<noscript[^>]*>.*?</noscript>`)
	iframePattern   = regexp.MustCompile(`(?is)<iframe[^>]*>.*?</iframe>`)
	svgPattern      = regexp.MustCompile(`(?is)<svg[^>]*>.*?</svg>`)
	canvasPattern   = regexp.MustCompile(`(?is)<canvas[^>]*>.*?</canvas>`)
	// HTML comments
	htmlCommentPattern = regexp.MustCompile(`<!--[\s\S]*?-->`)
	// HTML tags (for extracting plain text)
	htmlTagsPattern = regexp.MustCompile(`<[^>]*>`)
	// Multiple whitespace characters
	multiWhitespacePattern = regexp.MustCompile(`\s+`)
	// Multiple newlines
	multiNewlinePattern = regexp.MustCompile(`\n{3,}`)
	// Title extraction
	titleTagPattern = regexp.MustCompile(`(?is)<title[^>]*>([^<]*)</title>`)
	h1TagPattern    = regexp.MustCompile(`(?is)<h1[^>]*>([^<]*)</h1>`)
)

// ========================
// Data structures
// ========================

// OpenWebRequest represents a web page fetch request
type OpenWebRequest struct {
	URL       string `json:"url"`
	Timeout   int    `json:"timeout,omitempty"`    // Timeout (seconds)
	ForceJina bool   `json:"force_jina,omitempty"` // Force using Jina
}

// OpenWebResponse represents a web page fetch response
type OpenWebResponse struct {
	URL       string `json:"url"`
	Title     string `json:"title"`
	Content   string `json:"content"`
	Length    int    `json:"length"`
	Truncated bool   `json:"truncated"`
	Method    string `json:"method"` // "direct" or "jina"
	Error     string `json:"error,omitempty"`
}

// ========================
// HTML Cleaning Functions
// ========================

// cleanHTMLContent cleans HTML content and extracts plain text
func cleanHTMLContent(html string) (title string, content string) {
	// Extract title
	titleMatch := titleTagPattern.FindStringSubmatch(html)
	if len(titleMatch) > 1 {
		title = strings.TrimSpace(titleMatch[1])
	}
	if title == "" {
		h1Match := h1TagPattern.FindStringSubmatch(html)
		if len(h1Match) > 1 {
			title = strings.TrimSpace(h1Match[1])
		}
	}
	if title == "" {
		title = "Untitled"
	}

	// Remove script, style and other tags with content (Go regexp doesn't support backreferences, process separately)
	content = scriptPattern.ReplaceAllString(html, "")
	content = stylePattern.ReplaceAllString(content, "")
	content = navPattern.ReplaceAllString(content, "")
	content = headerPattern.ReplaceAllString(content, "")
	content = footerPattern.ReplaceAllString(content, "")
	content = noscriptPattern.ReplaceAllString(content, "")
	content = iframePattern.ReplaceAllString(content, "")
	content = svgPattern.ReplaceAllString(content, "")
	content = canvasPattern.ReplaceAllString(content, "")

	// Remove HTML comments
	content = htmlCommentPattern.ReplaceAllString(content, "")

	// Remove all HTML tags
	content = htmlTagsPattern.ReplaceAllString(content, " ")

	// Clean whitespace characters
	content = multiWhitespacePattern.ReplaceAllString(content, " ")
	content = strings.ReplaceAll(content, " \n", "\n")
	content = strings.ReplaceAll(content, "\n ", "\n")
	content = multiNewlinePattern.ReplaceAllString(content, "\n\n")
	content = strings.TrimSpace(content)

	return title, content
}

// ========================
// Fetch Functions
// ========================

// fetchDirect fetches web page directly
func fetchWebDirect(targetURL string, timeout time.Duration) (*OpenWebResponse, error) {
	log.Printf("[OpenWeb:Direct] Starting fetch: %s", targetURL)
	startTime := time.Now()

	client := &http.Client{
		Timeout: timeout,
		// Don't follow too many redirects
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 10 {
				return fmt.Errorf("too many redirects")
			}
			return nil
		},
	}

	req, err := http.NewRequest("GET", targetURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set request headers to mimic browser
	req.Header.Set("User-Agent", getRandomUserAgent())
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
	req.Header.Set("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8")
	req.Header.Set("DNT", "1")
	req.Header.Set("Connection", "keep-alive")

	resp, err := client.Do(req)
	if err != nil {
		if strings.Contains(err.Error(), "timeout") {
			return nil, fmt.Errorf("request timeout")
		}
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	log.Printf("[OpenWeb:Direct] Response status: %d, elapsed: %v", resp.StatusCode, time.Since(startTime))

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, resp.Status)
	}

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if len(body) == 0 {
		return nil, fmt.Errorf("response content is empty")
	}

	log.Printf("[OpenWeb:Direct] Original HTML length: %d characters", len(body))

	// Clean HTML
	title, content := cleanHTMLContent(string(body))

	log.Printf("[OpenWeb:Direct] Cleaned content length: %d characters", len(content))

	truncated := false
	if len(content) > openWebMaxContentLength {
		content = content[:openWebMaxContentLength] + "\n\n...[Content truncated]"
		truncated = true
		log.Printf("[OpenWeb:Direct] Content truncated to %d characters", openWebMaxContentLength)
	}

	log.Printf("[OpenWeb:Direct] ✅ Fetch succeeded, total elapsed: %v", time.Since(startTime))

	return &OpenWebResponse{
		URL:       targetURL,
		Title:     title,
		Content:   content,
		Length:    len(content),
		Truncated: truncated,
		Method:    "direct",
	}, nil
}

// fetchWithJina fetches web page using Jina Reader API
func fetchWebWithJina(targetURL string, timeout time.Duration) (*OpenWebResponse, error) {
	log.Printf("[OpenWeb:Jina] Starting fetch: %s", targetURL)
	startTime := time.Now()

	jinaURL := fmt.Sprintf("%s/%s", jinaReaderBaseURL, targetURL)
	log.Printf("[OpenWeb:Jina] Jina URL: %s", jinaURL)

	client := &http.Client{Timeout: timeout}

	req, err := http.NewRequest("GET", jinaURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Accept", "application/json")
	req.Header.Set("X-Timeout", fmt.Sprintf("%d", int(timeout.Seconds())))
	req.Header.Set("X-No-Cache", "true")

	resp, err := client.Do(req)
	if err != nil {
		if strings.Contains(err.Error(), "timeout") {
			return nil, fmt.Errorf("request timeout")
		}
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	log.Printf("[OpenWeb:Jina] Response status: %d, elapsed: %v", resp.StatusCode, time.Since(startTime))

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	// Parse JSON response
	var result struct {
		Data struct {
			URL     string `json:"url"`
			Title   string `json:"title"`
			Content string `json:"content"`
		} `json:"data"`
		URL     string `json:"url"`
		Title   string `json:"title"`
		Content string `json:"content"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Prefer data field
	content := result.Data.Content
	if content == "" {
		content = result.Content
	}
	title := result.Data.Title
	if title == "" {
		title = result.Title
	}
	if title == "" {
		title = "Untitled"
	}
	actualURL := result.Data.URL
	if actualURL == "" {
		actualURL = result.URL
	}
	if actualURL == "" {
		actualURL = targetURL
	}

	if content == "" {
		return nil, fmt.Errorf("response content is empty")
	}

	log.Printf("[OpenWeb:Jina] Got title: %s", title)
	log.Printf("[OpenWeb:Jina] Original content length: %d characters", len(content))

	truncated := false
	if len(content) > openWebMaxContentLength {
		content = content[:openWebMaxContentLength] + "\n\n...[Content truncated]"
		truncated = true
		log.Printf("[OpenWeb:Jina] Content truncated to %d characters", openWebMaxContentLength)
	}

	log.Printf("[OpenWeb:Jina] ✅ Fetch succeeded, total elapsed: %v", time.Since(startTime))

	return &OpenWebResponse{
		URL:       actualURL,
		Title:     title,
		Content:   content,
		Length:    len(content),
		Truncated: truncated,
		Method:    "jina",
	}, nil
}

// ========================
// HTTP Handler
// ========================

// OpenWebHandler handles web page fetch requests
func OpenWebHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Set CORS headers
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		// Handle preflight request
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		// Only allow POST requests
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Parse request
		var req OpenWebRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			log.Printf("[OpenWeb] Failed to parse request: %v", err)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(OpenWebResponse{
				Error: "Invalid request format",
			})
			return
		}

		// Validate URL
		if req.URL == "" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(OpenWebResponse{
				Error: "URL cannot be empty",
			})
			return
		}

		// Validate URL format
		parsedURL, err := url.Parse(req.URL)
		if err != nil || (parsedURL.Scheme != "http" && parsedURL.Scheme != "https") {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(OpenWebResponse{
				URL:   req.URL,
				Error: "Invalid URL format",
			})
			return
		}

		// Calculate timeout
		timeout := openWebDefaultTimeout
		if req.Timeout > 0 {
			timeout = time.Duration(req.Timeout) * time.Second
			if timeout > openWebMaxTimeout {
				timeout = openWebMaxTimeout
			}
		}

		log.Printf("[OpenWeb] Request: url=%s, timeout=%v, force_jina=%v", req.URL, timeout, req.ForceJina)

		var result *OpenWebResponse
		var fetchErr error

		// Strategy 1: If not forcing Jina, try direct fetch first
		if !req.ForceJina {
			log.Printf("[OpenWeb] Strategy 1: Trying direct fetch...")
			result, fetchErr = fetchWebDirect(req.URL, timeout)
			if fetchErr == nil {
				log.Printf("[OpenWeb] ✅ Direct fetch succeeded")
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(result)
				return
			}
			log.Printf("[OpenWeb] ⚠️ Direct fetch failed: %v", fetchErr)
			log.Printf("[OpenWeb] Strategy 2: Falling back to Jina Reader...")
		} else {
			log.Printf("[OpenWeb] Skipping direct fetch, using Jina Reader directly")
		}

		// Strategy 2: Use Jina Reader
		result, fetchErr = fetchWebWithJina(req.URL, timeout)
		if fetchErr == nil {
			log.Printf("[OpenWeb] ✅ Jina Reader fetch succeeded")
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(result)
			return
		}

		// Both methods failed
		log.Printf("[OpenWeb] ❌ All fetch methods failed: %v", fetchErr)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadGateway)
		_ = json.NewEncoder(w).Encode(OpenWebResponse{
			URL:   req.URL,
			Error: fmt.Sprintf("Unable to fetch web content: %v", fetchErr),
		})
	}
}
