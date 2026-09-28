package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestApplyRequestCacheControlsClampsAuthorizedFreshness(t *testing.T) {
	decision := &config.Decision{
		Name: "bounded-cache",
		Plugins: []config.DecisionPlugin{{
			Type: config.DecisionPluginResponseCache,
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"request_controls": map[string]interface{}{
					"enabled":         true,
					"allowed":         []string{"max-age", "ttl"},
					"max_ttl_seconds": 60,
				},
			}),
		}},
	}
	ctx := &RequestContext{
		Headers: map[string]string{
			defaultCacheControlHeader: "max-age=10, ttl=120, no-store",
		},
		VSRSelectedDecision: decision,
	}
	applyRequestCacheControls(ctx)
	if ctx.CacheMaxAgeSeconds == nil || *ctx.CacheMaxAgeSeconds != 10 {
		t.Fatalf("CacheMaxAgeSeconds = %v, want 10", ctx.CacheMaxAgeSeconds)
	}
	if ctx.CacheWriteTTLSeconds == nil || *ctx.CacheWriteTTLSeconds != 60 {
		t.Fatalf("CacheWriteTTLSeconds = %v, want clamped 60", ctx.CacheWriteTTLSeconds)
	}
	if ctx.CacheWriteBypass {
		t.Fatal("disallowed no-store directive was applied")
	}
}
