package main

import (
	"flag"
	"fmt"
	"io"
	"net/url"
)

const (
	defaultListen = "127.0.0.1:8710"

	providerDuckDuckGo = "duckduckgo"
	providerSearxng    = "searxng"
	providerHTTP       = "http"
)

// Config is the server's settings, read from flags with environment fallbacks.
type Config struct {
	Listen     string
	Provider   string
	SearxngURL string
	HTTPURL    string
	HTTPAPIKey string
}

// parseConfig reads flags from args; a flag left out takes its VLLM_SR_* variable.
// Usage and flag errors are written to usage.
func parseConfig(args []string, getenv func(string) string, usage io.Writer) (Config, error) {
	withDefault := func(key, fallback string) string {
		if value := getenv(key); value != "" {
			return value
		}
		return fallback
	}
	var cfg Config
	fs := flag.NewFlagSet("websearch-mcp", flag.ContinueOnError)
	fs.SetOutput(usage)
	fs.StringVar(&cfg.Listen, "listen", withDefault("VLLM_SR_SEARCH_LISTEN", defaultListen), "address the MCP server listens on")
	fs.StringVar(&cfg.Provider, "provider", withDefault("VLLM_SR_SEARCH_PROVIDER", providerDuckDuckGo), "search provider: duckduckgo, searxng or http")
	fs.StringVar(&cfg.SearxngURL, "searxng-url", getenv("VLLM_SR_SEARXNG_URL"), "SearXNG instance base URL")
	fs.StringVar(&cfg.HTTPURL, "http-url", getenv("VLLM_SR_SEARCH_HTTP_URL"), "search endpoint of a bring-your-own HTTP provider")
	fs.StringVar(&cfg.HTTPAPIKey, "http-api-key", getenv("VLLM_SR_SEARCH_HTTP_API_KEY"), "bearer token sent to the HTTP provider")
	if err := fs.Parse(args); err != nil {
		return Config{}, err
	}
	return cfg, cfg.validate()
}

func (c Config) validate() error {
	switch c.Provider {
	case providerDuckDuckGo:
		return nil
	case providerSearxng:
		return requireWebURL(c.Provider, "--searxng-url or VLLM_SR_SEARXNG_URL", c.SearxngURL)
	case providerHTTP:
		return requireWebURL(c.Provider, "--http-url or VLLM_SR_SEARCH_HTTP_URL", c.HTTPURL)
	default:
		return fmt.Errorf("unknown provider %q: use duckduckgo, searxng or http", c.Provider)
	}
}

func requireWebURL(provider, setting, value string) error {
	if value == "" {
		return fmt.Errorf("provider %s needs %s", provider, setting)
	}
	parsed, err := url.Parse(value)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
		return fmt.Errorf("%s must be an absolute http(s) URL, got %q", setting, value)
	}
	return nil
}
