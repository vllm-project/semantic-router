package handlers

import (
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
	URL        string `json:"url"`
	Timeout    int    `json:"timeout,omitempty"`     // Timeout (seconds)
	ForceJina  bool   `json:"force_jina,omitempty"`  // Force using Jina
	Format     string `json:"format,omitempty"`      // markdown (default) or json
	MaxLength  int    `json:"max_length,omitempty"`  // Maximum returned content length
	WithImages bool   `json:"with_images,omitempty"` // Enable Jina image-alt enrichment
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

func normalizeOpenWebFormat(format string) string {
	if strings.EqualFold(format, "json") {
		return "json"
	}
	return "markdown"
}

func normalizeOpenWebMaxLength(maxLength int) int {
	if maxLength <= 0 || maxLength > openWebMaxContentLength {
		return openWebMaxContentLength
	}
	return maxLength
}

func truncateOpenWebContent(content string, maxLength int) (string, bool) {
	if len(content) > maxLength {
		return content[:maxLength] + "\n\n...[Content truncated]", true
	}
	return content, false
}

func extractMarkdownTitle(content string) string {
	for _, line := range strings.Split(content, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "# ") {
			return strings.TrimSpace(strings.TrimPrefix(trimmed, "# "))
		}
	}
	return "Untitled"
}

func shouldPreferJinaFetch(targetURL string, req OpenWebRequest) bool {
	if req.ForceJina || req.WithImages {
		return true
	}

	parsedURL, err := url.Parse(targetURL)
	if err != nil {
		return false
	}

	return strings.HasSuffix(strings.ToLower(parsedURL.Path), ".pdf")
}

// ========================
// Fetch Functions
// ========================

// fetchDirect fetches web page directly
func fetchWebDirect(targetURL string, timeout time.Duration, maxLength int) (*OpenWebResponse, error) {
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
		return nil, wrapOpenWebRequestError(err)
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

	result := buildOpenWebResponse(
		openWebFetchedContent{
			url:     targetURL,
			title:   title,
			content: content,
		},
		maxLength,
		"direct",
	)
	if result.Truncated {
		log.Printf("[OpenWeb:Direct] Content truncated to %d characters", maxLength)
	}

	log.Printf("[OpenWeb:Direct] ✅ Fetch succeeded, total elapsed: %v", time.Since(startTime))

	return result, nil
}
