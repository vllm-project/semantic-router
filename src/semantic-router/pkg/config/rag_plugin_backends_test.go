package config

import (
	"math"
	"testing"
	"time"
)

func TestRAGTimeoutHonorsConfig(t *testing.T) {
	sixty := 60
	zero := 0
	huge := math.MaxInt
	cases := []struct {
		name    string
		seconds *int
		want    time.Duration
	}{
		{"unset_uses_default", nil, 30 * time.Second},
		{"configured_passes_through", &sixty, 60 * time.Second},
		{"zero_uses_default", &zero, 30 * time.Second},
		{"overflowing_clamps_to_max", &huge, time.Duration(MaxRAGTimeoutSeconds) * time.Second},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &ExternalAPIRAGConfig{TimeoutSeconds: tc.seconds}
			if got := cfg.GetTimeout(); got != tc.want {
				t.Fatalf("GetTimeout() = %v, want %v", got, tc.want)
			}
		})
	}
	if got := (&OpenAIRAGConfig{}).GetTimeout(); got != 60*time.Second {
		t.Fatalf("OpenAI default GetTimeout() = %v, want 60s", got)
	}
}
