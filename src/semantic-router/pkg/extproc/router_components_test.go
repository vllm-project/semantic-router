package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestDetectSemanticCacheEmbeddingModelUsesLifecycleDefault(t *testing.T) {
	cfg := config.DefaultGlobalConfig()

	if got := detectSemanticCacheEmbeddingModel(&cfg); got != "mmbert" {
		t.Fatalf("detectSemanticCacheEmbeddingModel() = %q, want mmbert", got)
	}
}

func TestDetectSemanticCacheEmbeddingModelHonorsExplicitModel(t *testing.T) {
	cfg := config.DefaultGlobalConfig()
	cfg.SemanticCache.EmbeddingModel = "bert"

	if got := detectSemanticCacheEmbeddingModel(&cfg); got != "bert" {
		t.Fatalf("detectSemanticCacheEmbeddingModel() = %q, want bert", got)
	}
}
