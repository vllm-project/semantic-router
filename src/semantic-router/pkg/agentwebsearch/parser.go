package agentwebsearch

import (
	"fmt"
	"io"
	"net/url"
	"strings"
	"unicode/utf8"

	"golang.org/x/net/html"
)

const maximumSearchResponseBytes = 2 << 20

func parseResults(reader io.Reader, maximum int) ([]searchResult, error) {
	limited := &io.LimitedReader{R: reader, N: maximumSearchResponseBytes + 1}
	document, err := html.Parse(limited)
	if err != nil {
		return nil, err
	}
	if limited.N <= 0 {
		return nil, fmt.Errorf("search response exceeds %d bytes", maximumSearchResponseBytes)
	}
	results := make([]searchResult, 0, maximum)
	seen := make(map[string]struct{}, maximum)
	var visit func(*html.Node)
	visit = func(node *html.Node) {
		if len(results) >= maximum {
			return
		}
		if node.Type == html.ElementNode && hasClass(node, "result") {
			if result, ok := resultFromNode(node); ok {
				if _, duplicate := seen[result.URL]; !duplicate {
					seen[result.URL] = struct{}{}
					results = append(results, result)
				}
				return
			}
		}
		for child := node.FirstChild; child != nil && len(results) < maximum; child = child.NextSibling {
			visit(child)
		}
	}
	visit(document)
	return results, nil
}
func resultFromNode(node *html.Node) (searchResult, bool) {
	link := firstDescendant(node, func(candidate *html.Node) bool {
		return candidate.Type == html.ElementNode && candidate.Data == "a" && hasClass(candidate, "result__a")
	})
	if link == nil {
		return searchResult{}, false
	}
	resolvedURL := canonicalResultURL(attribute(link, "href"))
	if resolvedURL == "" {
		return searchResult{}, false
	}
	parsed, err := url.Parse(resolvedURL)
	if err != nil || parsed.Hostname() == "" {
		return searchResult{}, false
	}
	title := boundedText(textContent(link), 512)
	if title == "" {
		return searchResult{}, false
	}
	snippetNode := firstDescendant(node, func(candidate *html.Node) bool {
		return candidate.Type == html.ElementNode && hasClass(candidate, "result__snippet")
	})
	snippet := ""
	if snippetNode != nil {
		snippet = boundedText(textContent(snippetNode), 2_048)
	}
	return searchResult{
		Title: title, URL: resolvedURL, Snippet: snippet,
		Domain: strings.ToLower(parsed.Hostname()),
	}, true
}

func canonicalResultURL(raw string) string {
	raw = strings.TrimSpace(raw)
	if strings.HasPrefix(raw, "//") {
		raw = "https:" + raw
	}
	parsed, err := url.Parse(raw)
	if err != nil {
		return ""
	}
	if redirected := parsed.Query().Get("uddg"); redirected != "" {
		parsed, err = url.Parse(redirected)
		if err != nil {
			return ""
		}
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return ""
	}
	if parsed.Hostname() == "" || strings.EqualFold(parsed.Hostname(), "duckduckgo.com") ||
		strings.HasSuffix(strings.ToLower(parsed.Hostname()), ".duckduckgo.com") {
		return ""
	}
	parsed.Fragment = ""
	return parsed.String()
}

func firstDescendant(node *html.Node, matches func(*html.Node) bool) *html.Node {
	for child := node.FirstChild; child != nil; child = child.NextSibling {
		if matches(child) {
			return child
		}
		if nested := firstDescendant(child, matches); nested != nil {
			return nested
		}
	}
	return nil
}

func hasClass(node *html.Node, expected string) bool {
	for _, value := range strings.Fields(attribute(node, "class")) {
		if value == expected {
			return true
		}
	}
	return false
}

func attribute(node *html.Node, name string) string {
	for _, item := range node.Attr {
		if item.Key == name {
			return item.Val
		}
	}
	return ""
}

func textContent(node *html.Node) string {
	var builder strings.Builder
	var visit func(*html.Node)
	visit = func(current *html.Node) {
		if current.Type == html.TextNode {
			builder.WriteString(current.Data)
			builder.WriteByte(' ')
		}
		for child := current.FirstChild; child != nil; child = child.NextSibling {
			visit(child)
		}
	}
	visit(node)
	return strings.Join(strings.Fields(builder.String()), " ")
}

func boundedText(value string, maximumRunes int) string {
	value = strings.Join(strings.Fields(strings.ToValidUTF8(value, "")), " ")
	if utf8.RuneCountInString(value) <= maximumRunes {
		return value
	}
	runes := []rune(value)
	return strings.TrimSpace(string(runes[:maximumRunes-1])) + "…"
}
