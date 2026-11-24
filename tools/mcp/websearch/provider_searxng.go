package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
)

// searxngProvider queries a SearXNG instance's JSON search API.
// The instance must list json under search.formats, or it answers 403.
type searxngProvider struct {
	baseURL string
	client  *http.Client
}

func newSearxngProvider(baseURL string, client *http.Client) *searxngProvider {
	return &searxngProvider{baseURL: strings.TrimRight(baseURL, "/"), client: client}
}

func (p *searxngProvider) Search(ctx context.Context, query string, _ int) ([]Result, error) {
	params := url.Values{"q": {query}, "format": {"json"}}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.baseURL+"/search?"+params.Encode(), nil)
	if err != nil {
		return nil, fmt.Errorf("searxng: build request: %w", err)
	}
	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("searxng: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("searxng: upstream status %d", resp.StatusCode)
	}
	var body struct {
		Results []struct {
			Title   string `json:"title"`
			URL     string `json:"url"`
			Content string `json:"content"`
		} `json:"results"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, maxResponseBytes)).Decode(&body); err != nil {
		return nil, fmt.Errorf("searxng: decode reply: %w", err)
	}
	results := make([]Result, 0, len(body.Results))
	for _, hit := range body.Results {
		results = append(results, Result{Title: hit.Title, URL: hit.URL, Snippet: hit.Content, Domain: domainOf(hit.URL)})
	}
	return results, nil
}
