package main

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/websearch/duckduckgo"
)

func TestNewProviderMatchesConfig(t *testing.T) {
	t.Parallel()
	if _, ok := mustProvider(t, Config{Provider: "duckduckgo"}).(*duckduckgo.Client); !ok {
		t.Fatal("duckduckgo config did not build the shared DuckDuckGo client")
	}
	sx, ok := mustProvider(t, Config{Provider: "searxng", SearxngURL: "http://searx.local/"}).(*searxngProvider)
	if !ok || sx.baseURL != "http://searx.local" {
		t.Fatalf("searxng config built %#v, want a SearXNG provider on http://searx.local", sx)
	}
	hp, ok := mustProvider(t, Config{Provider: "http", HTTPURL: "http://byo.local/search", HTTPAPIKey: "k"}).(*httpProvider)
	if !ok || hp.endpoint != "http://byo.local/search" || hp.apiKey != "k" {
		t.Fatalf("http config built %#v, want an HTTP provider with endpoint and key", hp)
	}
	if _, err := newProvider(Config{Provider: "bing"}); err == nil || !strings.Contains(err.Error(), "bing") {
		t.Fatalf("unknown provider error = %v, want one naming bing", err)
	}
}

func mustProvider(t *testing.T, cfg Config) Provider {
	t.Helper()
	p, err := newProvider(cfg)
	if err != nil {
		t.Fatalf("newProvider(%+v): %v", cfg, err)
	}
	return p
}

func TestRunStopsWhenTheContextEnds(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		done <- run(ctx, Config{Listen: "127.0.0.1:0", Provider: "http", HTTPURL: "http://127.0.0.1:9/search"})
	}()
	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("run returned %v after the context ended, want nil", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("run did not stop within 5s of the context ending")
	}
}

func TestRunReportsWhatItCannotStart(t *testing.T) {
	t.Parallel()
	for _, cfg := range []Config{
		{Listen: "127.0.0.1:0", Provider: "bing"},
		{Listen: "not an address", Provider: "duckduckgo"},
	} {
		if err := run(context.Background(), cfg); err == nil {
			t.Fatalf("run(%+v) returned nil, want an error", cfg)
		}
	}
}
