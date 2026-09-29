//go:build !windows && cgo

package cache

import (
	"context"
	"os"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
)

// Select either prepared Nano or Mini with VELA_OMNI_ARTIFACT. This checks the
// real provider boundary for cache and memory without requiring a database.
func TestOmniStorageIntegrationUsesArtifactDimensionAndIdentity(t *testing.T) {
	artifact := os.Getenv("VELA_OMNI_ARTIFACT")
	if artifact == "" {
		if os.Getenv("REQUIRE_OMNI_TESTS") == "1" {
			t.Fatal("VELA_OMNI_ARTIFACT must select a prepared artifact")
		}
		t.Skip("set VELA_OMNI_ARTIFACT to select native storage integration")
	}
	provider, err := native.New(nil).Embedding(context.Background(), config.ResolvedModelBinding{
		Recipe: "storage-test", Name: "embedding",
		Binding:    config.ModelBinding{Deployment: "omni", Contract: "embedding.v1", Adapter: "vela_omni"},
		Deployment: config.ModelDeployment{Artifact: artifact, Provider: "ort", Device: "cpu", Precision: "native", Input: config.ModelInputBudget{Overflow: "reject"}},
	}, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if closeErr := provider.Close(); closeErr != nil {
			t.Error(closeErr)
		}
	})
	size := provider.Dimension()
	if size != 384 && size != 768 {
		t.Fatalf("unexpected published output %d", size)
	}
	cache := NewInMemoryCache(InMemoryCacheOptions{Enabled: true, EmbeddingModel: "multimodal", EmbeddingProvider: provider})
	t.Cleanup(func() { _ = cache.Close() })
	if validationErr := ValidateBackendEmbedding(context.Background(), cache); validationErr != nil {
		t.Fatal(validationErr)
	}
	cfg := namespaceFixture(RedisCacheType, 0)
	cfg.EmbeddingModel = "multimodal"
	cfg.EmbeddingProvider = provider
	bound, identity, err := PrepareEmbeddingNamespace(cfg, func(settings embedding.ConsumerSettings) (embedding.ContentIdentity, error) {
		return embedding.ResolveProviderIdentity(provider, settings)
	})
	if err != nil || identity == "" || bound.Redis.Index.VectorField.Dimension != size || cfg.Redis.Index.VectorField.Dimension != 0 {
		t.Fatalf("bound size/identity: %+v %s %v", bound.Redis, identity, err)
	}
	memoryConfig := memory.EmbeddingConfig{Model: memory.EmbeddingModelMulti, Provider: provider}
	vector, err := memory.GenerateEmbedding("A Chinese and English memory: 会议时间。", memoryConfig)
	if err != nil || len(vector) != size {
		t.Fatalf("memory vector width=%d want=%d err=%v", len(vector), size, err)
	}
	wrong := 384
	if size == 384 {
		wrong = 768
	}
	memoryConfig.Dimension = wrong
	if _, dimensionErr := memory.GenerateEmbedding("must reject unsupported output", memoryConfig); dimensionErr == nil {
		t.Fatal("unsupported dimension accepted")
	}
	if _, dimensionErr := resolveCacheDimension(wrong, provider); dimensionErr == nil {
		t.Fatal("cache accepted unsupported dimension")
	}
}
