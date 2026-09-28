package native

import (
	"context"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
)

func TestCandleStorageUsesLoadedMatryoshkaDimensions(t *testing.T) {
	ctx := context.Background()
	spec := config.ResolvedModelBinding{
		Recipe: "storage-dimensions", Name: "embedding",
		Binding:    config.ModelBinding{Deployment: "encoder", Contract: "embedding.v1", Adapter: "mmbert"},
		Deployment: config.ModelDeployment{Artifact: nativeHeadlessWidthFixture(t, 0, 512), Provider: "candle", Device: "cpu", Precision: "float32", Input: config.ModelInputBudget{MaxTokens: 16, Overflow: "reject"}},
	}
	provider, err := New(nil).Embedding(ctx, spec, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer provider.Close()
	if provider.Dimension() != 512 || !reflect.DeepEqual(provider.EmbeddingInfo().Dimensions, []int{64, 128, 256, 512}) {
		t.Fatalf("actual loaded dimensions not published: %+v", provider.EmbeddingInfo())
	}
	// Memory prepares the complete provider but retains its established 256-D view.
	cfg := memory.EmbeddingConfig{Model: memory.EmbeddingModelMMBERT, Provider: provider}
	dimension, err := memory.StorageDimension(0, cfg)
	if err != nil || dimension != 256 {
		t.Fatalf("memory default rejected: %d %v", dimension, err)
	}
	cfg.Dimension = dimension
	vector, err := memory.GenerateEmbeddingWithContext(ctx, "hello world", cfg)
	if err != nil || len(vector) != dimension {
		t.Fatalf("native storage view: %d %v", len(vector), err)
	}
	for _, invalid := range []int{63, 513, 768} {
		if _, dimensionErr := memory.StorageDimension(invalid, cfg); dimensionErr == nil {
			t.Fatalf("unsupported storage width %d accepted", invalid)
		}
	}
	// Cache and other fixed-width consumers use the same checked representation.
	if dimension, err = embedding.ResolveDimension(provider, 128); err != nil || dimension != 128 {
		t.Fatalf("advertised cache width rejected: %d %v", dimension, err)
	}
	info := provider.EmbeddingInfo()
	info.Dimensions[0] = 63
	if _, dimensionErr := embedding.ResolveDimension(provider, 63); dimensionErr == nil {
		t.Fatal("caller mutated prepared capabilities")
	}
}
