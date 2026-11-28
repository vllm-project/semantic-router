package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// httpProvider posts to a bring-your-own search endpoint:
// POST {"query","num_results"} -> {"results":[{"title","url","snippet"}]}.
type httpProvider struct {
	endpoint string
	apiKey   string
	client   *http.Client
}

func newHTTPProvider(endpoint, apiKey string, client *http.Client) *httpProvider {
	return &httpProvider{endpoint: endpoint, apiKey: apiKey, client: client}
}

func (p *httpProvider) Search(ctx context.Context, query string, numResults int) ([]Result, error) {
	payload, err := json.Marshal(map[string]any{"query": query, "num_results": numResults})
	if err != nil {
		return nil, fmt.Errorf("http provider: encode request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("http provider: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.apiKey)
	}
	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("http provider: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("http provider: upstream status %d", resp.StatusCode)
	}
	var body struct {
		Results []struct {
			Title   string `json:"title"`
			URL     string `json:"url"`
			Snippet string `json:"snippet"`
		} `json:"results"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, maxResponseBytes)).Decode(&body); err != nil {
		return nil, fmt.Errorf("http provider: decode reply: %w", err)
	}
	results := make([]Result, 0, len(body.Results))
	for _, hit := range body.Results {
		results = append(results, Result{Title: hit.Title, URL: hit.URL, Snippet: hit.Snippet, Domain: domainOf(hit.URL)})
	}
	return results, nil
}
