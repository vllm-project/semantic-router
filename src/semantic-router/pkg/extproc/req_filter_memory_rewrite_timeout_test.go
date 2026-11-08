package extproc

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestResolveMemoryRewriteTimeoutKeepsFraction(t *testing.T) {
	cfg := &config.RouterConfig{
		ExternalModels: []config.ExternalModelConfig{{
			ModelRole:      config.ModelRoleMemoryRewrite,
			ModelName:      "rewriter",
			TimeoutSeconds: 0.8,
			ModelEndpoint:  config.ClassifierVLLMEndpoint{Address: "127.0.0.1", Port: 8000},
		}},
	}

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
