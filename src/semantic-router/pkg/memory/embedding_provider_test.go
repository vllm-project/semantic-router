package memory

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

type recordingMemoryProvider struct {
	options embedding.Options
	ctx     context.Context
	calls   int
}

func (p *recordingMemoryProvider) Embed(ctx context.Context, text string) ([]float32, error) {
	return p.EmbedWithOptions(ctx, text, embedding.Options{})
}

func (p *recordingMemoryProvider) EmbedWithOptions(ctx context.Context, _ string, options embedding.Options) ([]float32, error) {
	p.options, p.ctx = options, ctx
	p.calls++
	return []float32{3, 4}, nil
}

func (*recordingMemoryProvider) EmbedBatch(context.Context, []string) ([][]float32, error) {
	return nil, errors.New("unused")
}
func (*recordingMemoryProvider) Dimension() int  { return 2 }
func (*recordingMemoryProvider) Backend() string { return "test" }

func TestOwnedMemoryProviderPreservesModelOptionsAndRawVector(t *testing.T) {
	for _, test := range []struct {
		cfg  EmbeddingConfig
		want embedding.Options
	}{
		{EmbeddingConfig{Model: EmbeddingModelMMBERT}, embedding.Options{Dimension: 256}},
		{EmbeddingConfig{Model: EmbeddingModelMMBERT, Dimension: 128, Layer: 6}, embedding.Options{Dimension: 128, Layer: 6}},
		{EmbeddingConfig{Model: EmbeddingModelMulti}, embedding.Options{Dimension: 384}},
		{EmbeddingConfig{Model: EmbeddingModelBERT, Dimension: 10}, embedding.Options{}},
		{EmbeddingConfig{Model: EmbeddingModelQwen3, Dimension: 10}, embedding.Options{}},
		{EmbeddingConfig{Model: EmbeddingModelGemma, Dimension: 10}, embedding.Options{}},
	} {
		p := &recordingMemoryProvider{}
		cfg := test.cfg
		cfg.Provider = p
		ctx, cancel := context.WithCancel(context.Background())
		vector, err := GenerateEmbeddingWithContext(ctx, "text", cfg)
		if err != nil || !reflect.DeepEqual(vector, []float32{3, 4}) || p.options != test.want || p.ctx != ctx {
			t.Fatalf("%s contract changed: vector=%v options=%+v err=%v", cfg.Model, vector, p.options, err)
		}
		cancel()
		if _, err := GenerateEmbeddingWithContext(ctx, "text", cfg); !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled embedding returned %v", err)
		}
		if p.calls != 1 {
			t.Fatal("cancelled context reached provider")
		}
	}
}

func TestOwnedMemoryProviderDoesNotUseNativeGlobalFallback(t *testing.T) {
	t.Setenv(deterministicEmbeddingsEnv, "")
	if _, err := GenerateEmbedding("text", EmbeddingConfig{Model: EmbeddingModelBERT}); err == nil {
		t.Fatal("nil provider accepted")
	}
	p := &recordingMemoryProvider{}
	t.Setenv(deterministicEmbeddingsEnv, "true")
	vector, err := GenerateEmbedding("text", EmbeddingConfig{Model: EmbeddingModelBERT, Provider: p})
	if err != nil || !reflect.DeepEqual(vector, []float32{3, 4}) || p.calls != 1 {
		t.Fatal("test environment replaced explicitly prepared provider")
	}
}

type reducedWidthMemoryProvider struct {
	options embedding.Options
}

func (p *reducedWidthMemoryProvider) Embed(ctx context.Context, text string) ([]float32, error) {
	return p.EmbedWithOptions(ctx, text, embedding.Options{})
}

func (p *reducedWidthMemoryProvider) EmbedWithOptions(_ context.Context, _ string, options embedding.Options) ([]float32, error) {
	p.options = options
	dimension := options.Dimension
	if dimension <= 0 {
		dimension = 768
	}
	return make([]float32, dimension), nil
}

func (*reducedWidthMemoryProvider) EmbedBatch(context.Context, []string) ([][]float32, error) {
	return nil, errors.New("unused")
}
func (*reducedWidthMemoryProvider) Dimension() int  { return 768 }
func (*reducedWidthMemoryProvider) Backend() string { return "test" }

func (p *reducedWidthMemoryProvider) EmbeddingDimensionContract() (embedding.DimensionContract, error) {
	return embedding.DimensionContract{
		NativeDimension:     768,
		SupportedDimensions: []int{768, 512, 256, 128, 64},
	}, nil
}

func TestGenerateEmbeddingUsesResolvedReducedWidth(t *testing.T) {
	for _, model := range []EmbeddingModelType{EmbeddingModelQwen3, EmbeddingModelGemma} {
		t.Run(string(model), func(t *testing.T) {
			provider := &reducedWidthMemoryProvider{}
			cfg := EmbeddingConfig{
				Provider:  provider,
				Model:     model,
				Dimension: 256,
			}
			resolved, err := resolveMilvusStoreEmbeddingDimension(cfg, 768)
			if err != nil {
				t.Fatalf("resolveMilvusStoreEmbeddingDimension() error = %v", err)
			}
			cfg.Dimension = resolved
			vector, err := GenerateEmbedding("reduced width", cfg)
			if err != nil {
				t.Fatalf("GenerateEmbedding() error = %v", err)
			}
			if len(vector) != 256 || provider.options.Dimension != 256 {
				t.Fatalf("resolved width = %d, options=%+v; want 256", len(vector), provider.options)
			}
		})
	}
}
