package main

import (
	"errors"
	"flag"
	"io"
	"strings"
	"testing"
)

func TestParseConfig(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name    string
		args    []string
		env     map[string]string
		want    Config
		wantErr string
	}{
		{
			name: "defaults to duckduckgo on loopback",
			want: Config{Listen: "127.0.0.1:8710", Provider: "duckduckgo"},
		},
		{
			name:    "searxng needs its url",
			args:    []string{"--provider", "searxng"},
			wantErr: "--searxng-url",
		},
		{
			name:    "http needs its url",
			args:    []string{"--provider", "http"},
			wantErr: "--http-url",
		},
		{
			name: "http provider with url and key",
			args: []string{"--provider", "http", "--http-url", "http://search.local/search", "--http-api-key", "k"},
			want: Config{Listen: "127.0.0.1:8710", Provider: "http", HTTPURL: "http://search.local/search", HTTPAPIKey: "k"},
		},
		{
			name: "environment fills flags left out",
			env: map[string]string{
				"VLLM_SR_SEARCH_PROVIDER": "searxng",
				"VLLM_SR_SEARXNG_URL":     "http://searx.local",
				"VLLM_SR_SEARCH_LISTEN":   "0.0.0.0:9000",
			},
			want: Config{Listen: "0.0.0.0:9000", Provider: "searxng", SearxngURL: "http://searx.local"},
		},
		{
			name: "flag wins over environment",
			args: []string{"--provider", "duckduckgo"},
			env:  map[string]string{"VLLM_SR_SEARCH_PROVIDER": "searxng"},
			want: Config{Listen: "127.0.0.1:8710", Provider: "duckduckgo"},
		},
		{
			name:    "searxng url must be absolute http(s)",
			args:    []string{"--provider", "searxng", "--searxng-url", "searx.local:8080"},
			wantErr: "absolute http(s) URL",
		},
		{
			name:    "http url must be absolute http(s)",
			args:    []string{"--provider", "http", "--http-url", "byo.local/search"},
			wantErr: "absolute http(s) URL",
		},
		{
			name:    "unknown provider is refused",
			args:    []string{"--provider", "bing"},
			wantErr: `provider "bing"`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			getenv := func(key string) string { return tc.env[key] }
			got, err := parseConfig(tc.args, getenv, io.Discard)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("parseConfig(%v) error = %v, want one containing %q", tc.args, err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseConfig(%v) error = %v", tc.args, err)
			}
			if got != tc.want {
				t.Fatalf("parseConfig(%v) = %+v, want %+v", tc.args, got, tc.want)
			}
		})
	}
}

func TestHelpPrintsTheFlags(t *testing.T) {
	t.Parallel()
	var usage strings.Builder
	_, err := parseConfig([]string{"--help"}, func(string) string { return "" }, &usage)
	if !errors.Is(err, flag.ErrHelp) {
		t.Fatalf("parseConfig(--help) error = %v, want flag.ErrHelp", err)
	}
	for _, flagName := range []string{"-listen", "-provider", "-searxng-url", "-http-url", "-http-api-key"} {
		if !strings.Contains(usage.String(), flagName) {
			t.Fatalf("usage lacks %s:\n%s", flagName, usage.String())
		}
	}
}
