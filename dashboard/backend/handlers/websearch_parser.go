package handlers

import (
	"html"
	"log"
	"net/url"
	"regexp"
	"strings"
)

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

func isValidURL(urlStr string) bool {
	_, err := parseOutboundHTTPURL(urlStr)
	return err == nil
}

func extractDomain(urlStr string) string {
	parsed, err := url.Parse(urlStr)
	if err != nil {
		return ""
	}
	return parsed.Hostname()
}

func parseDuckDuckGoHTML(htmlContent string, maxResults int) ([]SearchResult, error) {
	var results []SearchResult
	linkMatches := linkPattern.FindAllStringSubmatch(htmlContent, -1)
	snippetMatches := snippetPattern.FindAllStringSubmatch(htmlContent, -1)
	if len(snippetMatches) == 0 {
		snippetMatches = snippetAltPattern.FindAllStringSubmatch(htmlContent, -1)
	}

	for i := 0; i < len(linkMatches) && i < maxResults; i++ {
		if len(linkMatches[i]) < 3 {
			continue
		}
		rawURL := linkMatches[i][1]
		title := cleanExtraWhitespace(html.UnescapeString(strings.TrimSpace(linkMatches[i][2])))
		actualURL := extractActualURL(rawURL)
		if actualURL == "" {
			continue
		}
		if !isValidURL(actualURL) {
			log.Printf("[WebSearch] Skipping invalid result URL")
			continue
		}

		snippet := ""
		if i < len(snippetMatches) && len(snippetMatches[i]) > 1 {
			snippet = cleanExtraWhitespace(html.UnescapeString(cleanHTMLTags(snippetMatches[i][1])))
		}
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
	if len(results) == 0 {
		results = parseSimpleHTML(htmlContent, maxResults)
	}
	return results, nil
}

func parseSimpleHTML(htmlContent string, maxResults int) []SearchResult {
	var results []SearchResult
	blocks := strings.Split(htmlContent, "result__body")
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

		titleMatch := titlePattern.FindStringSubmatch(block)
		title := ""
		if len(titleMatch) > 1 {
			title = html.UnescapeString(strings.TrimSpace(titleMatch[1]))
		}
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

func extractActualURL(ddgURL string) string {
	if strings.Contains(ddgURL, "uddg=") {
		parsed, err := url.Parse(ddgURL)
		if err == nil {
			if uddg := parsed.Query().Get("uddg"); uddg != "" {
				return uddg
			}
		}
		matches := uddgPattern.FindStringSubmatch(ddgURL)
		if len(matches) > 1 {
			decoded, decodeErr := url.QueryUnescape(matches[1])
			if decodeErr == nil {
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

func cleanHTMLTags(value string) string {
	return strings.TrimSpace(htmlTagPattern.ReplaceAllString(value, " "))
}

func cleanExtraWhitespace(value string) string {
	return strings.TrimSpace(whitespacePattern.ReplaceAllString(value, " "))
}
