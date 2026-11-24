package main

import (
	"context"
	"net/url"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/websearch/duckduckgo"
)

// Result is one search hit in the shape every provider returns.
type Result = duckduckgo.Result

// Provider answers a query with at most numResults hits.
// *duckduckgo.Client satisfies it directly.
type Provider interface {
	Search(ctx context.Context, query string, numResults int) ([]Result, error)
}

// maxResponseBytes bounds what a provider reads from an upstream reply.
const maxResponseBytes = 1 << 20

func domainOf(raw string) string {
	parsed, err := url.Parse(raw)
	if err != nil {
		return ""
	}
	return parsed.Hostname()
}
