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
		{EmbeddingConfig{Deployment: "vela-embedding", Dimension: 2}, embedding.Options{Dimension: 2}},
		{EmbeddingConfig{Deployment: "vela-embedding", Dimension: 2, Layer: 6}, embedding.Options{Dimension: 2, Layer: 6}},
		{EmbeddingConfig{Deployment: "omni"}, embedding.Options{}},
	} {
		p := &recordingMemoryProvider{}
		cfg := test.cfg
		cfg.Provider = p
		ctx, cancel := context.WithCancel(context.Background())
		vector, err := GenerateEmbeddingWithContext(ctx, "text", cfg)
		if err != nil || !reflect.DeepEqual(vector, []float32{3, 4}) || p.options != test.want || p.ctx != ctx {
			t.Fatalf("%s contract changed: vector=%v options=%+v err=%v", cfg.Deployment, vector, p.options, err)
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

// A provider that answers another width than the view the store reads is
// refused before the vector is stored.
func TestMemoryEmbeddingWidthIsTheStoresView(t *testing.T) {
	p := &recordingMemoryProvider{}
	if _, err := GenerateEmbedding("text", EmbeddingConfig{Deployment: "vela-embedding", Dimension: 256, Provider: p}); err == nil {
		t.Fatal("a 2-value vector was accepted for a 256-dimension store")
	}
}

func TestOwnedMemoryProviderDoesNotUseNativeGlobalFallback(t *testing.T) {
	t.Setenv(deterministicEmbeddingsEnv, "")
	if _, err := GenerateEmbedding("text", EmbeddingConfig{Deployment: "qwen3-embedding"}); err == nil {
		t.Fatal("nil provider accepted")
	}
	p := &recordingMemoryProvider{}
	t.Setenv(deterministicEmbeddingsEnv, "true")
	vector, err := GenerateEmbedding("text", EmbeddingConfig{Deployment: "qwen3-embedding", Provider: p})
	if err != nil || !reflect.DeepEqual(vector, []float32{3, 4}) || p.calls != 1 {
		t.Fatal("test environment replaced explicitly prepared provider")
	}
}

func TestOmniMemoryUsesPreparedDimensionWithoutSlicing(t *testing.T) {
	for _, size := range []int{384, 768} {
		calls := 0
		provider, _ := embedding.NewFuncProvider("test", size, func(context.Context, string) ([]float32, error) { calls++; return make([]float32, size), nil })
		cfg := EmbeddingConfig{Deployment: "omni", Provider: provider}
		vector, err := GenerateEmbedding("text", cfg)
		if err != nil || len(vector) != size {
			t.Fatalf("dimension=%d vector=%d err=%v", size, len(vector), err)
		}
		if got, err := StorageDimension(0, cfg); err != nil || got != size {
			t.Fatalf("storage=%d err=%v", got, err)
		}
		cfg.Dimension = 128
		if _, err := GenerateEmbedding("text", cfg); err == nil || calls != 1 {
			t.Fatal("unsupported reduced dimension reached inference")
		}
		if _, err := StorageDimension(128, cfg); err == nil {
			t.Fatal("unsupported storage width accepted")
		}
	}
}
