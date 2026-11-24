// Command websearch-mcp serves one MCP tool, web_search, over streamable HTTP
// so coding agents keep web search when the Router selects a model whose
// provider has no search of its own.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/websearch/duckduckgo"
)

const (
	upstreamTimeout   = 20 * time.Second
	readHeaderTimeout = 10 * time.Second
	shutdownTimeout   = 5 * time.Second
)

func newProvider(cfg Config) (Provider, error) {
	client := &http.Client{Timeout: upstreamTimeout}
	switch cfg.Provider {
	case providerDuckDuckGo:
		return duckduckgo.NewClient(duckduckgo.WithLogf(log.Printf)), nil
	case providerSearxng:
		return newSearxngProvider(cfg.SearxngURL, client), nil
	case providerHTTP:
		return newHTTPProvider(cfg.HTTPURL, cfg.HTTPAPIKey, client), nil
	}
	return nil, fmt.Errorf("unknown provider %q", cfg.Provider)
}

// run serves until ctx ends; no write timeout, since MCP streams over long-lived responses.
func run(ctx context.Context, cfg Config) error {
	provider, err := newProvider(cfg)
	if err != nil {
		return err
	}
	listener, err := net.Listen("tcp", cfg.Listen)
	if err != nil {
		return err
	}
	srv := &http.Server{Handler: newHTTPHandler(provider), ReadHeaderTimeout: readHeaderTimeout}
	served := make(chan error, 1)
	go func() { served <- srv.Serve(listener) }()
	log.Printf("websearch-mcp: provider %s, listening on http://%s%s", cfg.Provider, listener.Addr(), endpointPath)
	select {
	case err := <-served:
		return err
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
		defer cancel()
		return srv.Shutdown(shutdownCtx)
	}
}

func main() {
	cfg, err := parseConfig(os.Args[1:], os.Getenv, os.Stderr)
	if errors.Is(err, flag.ErrHelp) {
		return
	}
	if err != nil {
		log.Fatalf("websearch-mcp: %v", err)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	err = run(ctx, cfg)
	stop()
	if err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatalf("websearch-mcp: %v", err)
	}
}
