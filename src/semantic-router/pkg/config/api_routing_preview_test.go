package config

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

func TestCanonicalRoutingPreviewBudget(t *testing.T) {
	for _, test := range []struct {
		name, fields string
		timeout      time.Duration
		concurrency  int
		invalid      bool
	}{
		{name: "defaults", timeout: 120 * time.Second, concurrency: 16},
		{name: "long request", fields: "request_timeout_seconds: 1800\n        max_concurrency: 8", timeout: 1800 * time.Second, concurrency: 8},
		{name: "maximum timeout", fields: "request_timeout_seconds: 3600", timeout: 3600 * time.Second, concurrency: 16},
		{name: "zero timeout", fields: "request_timeout_seconds: 0", invalid: true},
		{name: "excessive timeout", fields: "request_timeout_seconds: 3601", invalid: true},
		{name: "zero capacity", fields: "max_concurrency: 0", invalid: true},
		{name: "negative capacity", fields: "max_concurrency: -1", invalid: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			yaml := fmt.Sprintf("version: v0.3\nglobal:\n  services:\n    api:\n      routing_preview:\n        %s\n", test.fields)
			cfg, err := ParseYAMLBytes([]byte(yaml))
			if test.invalid {
				if err == nil || !strings.Contains(err.Error(), "routing_preview") {
					t.Fatalf("expected Preview config rejection, got %v", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got := cfg.API.RoutingPreview; got.RequestTimeout() != test.timeout || got.ConcurrencyLimit() != test.concurrency {
				t.Fatalf("resolved Preview = timeout %v, concurrency %d", got.RequestTimeout(), got.ConcurrencyLimit())
			}
		})
	}
}
