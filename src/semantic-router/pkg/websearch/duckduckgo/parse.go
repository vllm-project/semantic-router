package duckduckgo

import (
	"html"
	"net/url"
	"regexp"
	"strings"
)

// The HTML page has no stable API. These patterns match the result markup
// DuckDuckGo serves today; parseSimpleResults is the fallback when the
// primary patterns stop matching.
var (
	linkPattern            = regexp.MustCompile(`<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>([^<]*)</a>`)
	snippetPattern         = regexp.MustCompile(`<a[^>]*class="result__snippet"[^>]*>([^<]*(?:<[^>]*>[^<]*</[^>]*>)*[^<]*)</a>`)
	snippetAltPattern      = regexp.MustCompile(`class="result__snippet"[^>]*>([^<]+)`)
	uddgPattern            = regexp.MustCompile(`uddg=([^&"]+)`)
	titlePattern           = regexp.MustCompile(`result__a[^>]*>([^<]+)`)
	snippetFallbackPattern = regexp.MustCompile(`result__snippet[^>]*>([^<]+)`)
	htmlTagPattern         = regexp.MustCompile(`<[^>]*>`)
	whitespacePattern      = regexp.MustCompile(`\s+`)
)

// parseResults extracts up to maxResults hits from a DuckDuckGo HTML page.
// Hits whose URL is not plain http(s), or that point back at DuckDuckGo, are
// dropped. logf receives content-free diagnostics only.
func parseResults(page string, maxResults int, logf func(string, ...any)) []Result {
	var results []Result

	linkMatches := linkPattern.FindAllStringSubmatch(page, -1)
	snippetMatches := snippetPattern.FindAllStringSubmatch(page, -1)
	if len(snippetMatches) == 0 {
		snippetMatches = snippetAltPattern.FindAllStringSubmatch(page, -1)
	}

	for i := 0; i < len(linkMatches) && i < maxResults; i++ {
		if len(linkMatches[i]) < 3 {
			continue
		}
		title := cleanExtraWhitespace(html.UnescapeString(strings.TrimSpace(linkMatches[i][2])))

		actualURL := extractActualURL(linkMatches[i][1])
		if actualURL == "" {
			continue
		}
		if !isValidURL(actualURL) {
			logf("Skipping invalid search-result URL (url_bytes=%d)", len(actualURL))
			continue
		}
		if strings.Contains(actualURL, "duckduckgo.com") {
			continue
		}

		snippet := ""
		if i < len(snippetMatches) && len(snippetMatches[i]) > 1 {
			snippet = cleanExtraWhitespace(html.UnescapeString(cleanHTMLTags(snippetMatches[i][1])))
		}

		results = append(results, Result{
			Title:   title,
			URL:     actualURL,
			Snippet: snippet,
			Domain:  extractDomain(actualURL),
		})
	}

	if len(results) == 0 {
		results = parseSimpleResults(page, maxResults)
	}
	return results
}

// parseSimpleResults splits the page into result blocks and reads each with
// looser patterns. It is used only when the primary patterns found nothing.
func parseSimpleResults(page string, maxResults int) []Result {
	var results []Result

	blocks := strings.Split(page, "result__body")
	for i, block := range blocks {
		if i == 0 || i > maxResults {
			continue
		}

		urlMatch := uddgPattern.FindStringSubmatch(block)
		if len(urlMatch) < 2 {
			continue
		}
		actualURL, _ := url.QueryUnescape(urlMatch[1])
		if actualURL == "" || strings.Contains(actualURL, "duckduckgo") || !isValidURL(actualURL) {
			continue
		}

		title := ""
		if titleMatch := titlePattern.FindStringSubmatch(block); len(titleMatch) > 1 {
			title = html.UnescapeString(strings.TrimSpace(titleMatch[1]))
		}
		snippet := ""
		if snippetMatch := snippetFallbackPattern.FindStringSubmatch(block); len(snippetMatch) > 1 {
			snippet = html.UnescapeString(strings.TrimSpace(snippetMatch[1]))
		}

		if title != "" {
			results = append(results, Result{
				Title:   title,
				URL:     actualURL,
				Snippet: snippet,
				Domain:  extractDomain(actualURL),
			})
		}
	}
	return results
}

// extractActualURL unwraps DuckDuckGo's redirect form
// (//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com) or returns a direct
// http(s) link unchanged. Anything else yields "".
func extractActualURL(ddgURL string) string {
	if strings.Contains(ddgURL, "uddg=") {
		if parsed, err := url.Parse(ddgURL); err == nil {
			if uddg := parsed.Query().Get("uddg"); uddg != "" {
				return uddg
			}
		}
		if matches := uddgPattern.FindStringSubmatch(ddgURL); len(matches) > 1 {
			if decoded, err := url.QueryUnescape(matches[1]); err == nil {
				return decoded
			}
			return matches[1]
		}
	}
	if strings.HasPrefix(ddgURL, "http://") || strings.HasPrefix(ddgURL, "https://") {
		return ddgURL
	}
	return ""
}

// isValidURL accepts only http and https links with a host, which keeps
// javascript:, data: and similar schemes out of results shown to users.
func isValidURL(raw string) bool {
	if raw == "" {
		return false
	}
	parsed, err := url.Parse(raw)
	if err != nil {
		return false
	}
	scheme := strings.ToLower(parsed.Scheme)
	return (scheme == "http" || scheme == "https") && parsed.Host != ""
}

func extractDomain(raw string) string {
	parsed, err := url.Parse(raw)
	if err != nil {
		return ""
	}
	return parsed.Host
}

func cleanHTMLTags(s string) string {
	return strings.TrimSpace(htmlTagPattern.ReplaceAllString(s, " "))
}

func cleanExtraWhitespace(s string) string {
	return strings.TrimSpace(whitespacePattern.ReplaceAllString(s, " "))
}
