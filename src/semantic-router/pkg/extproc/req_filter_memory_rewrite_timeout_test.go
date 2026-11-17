package extproc

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestResolveMemoryRewriteTimeoutIsTheDeployments(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"rewriter": {
		Provider: config.OpenAIChatProvider, Endpoint: "http://127.0.0.1:8000/v1", Model: "rewriter", TimeoutMs: 800,
	}}
	cfg.Memory.QueryRewrite = &config.MemoryQueryRewriteConfig{Deployment: "rewriter"}

	resolved := resolveMemoryRewriteServiceConfig(cfg)
	if resolved == nil {
		t.Fatal("resolveMemoryRewriteServiceConfig() = nil")
	}
	if got := getTimeout(resolved); got != 800*time.Millisecond {
		t.Fatalf("getTimeout() = %v, want 800ms", got)
	}
}

func TestMemoryRewriteTimeoutDefaultsWhenUnset(t *testing.T) {
	if got := getTimeout(&memoryRewriteServiceConfig{}); got != 5*time.Second {
		t.Fatalf("getTimeout() = %v, want 5s", got)
	}
}
