//go:build !windows && cgo

package memory

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

type milvusContractTestProvider struct{}

func (milvusContractTestProvider) Embed(context.Context, string) ([]float32, error) {
	return []float32{0}, nil
}

func (milvusContractTestProvider) EmbedBatch(context.Context, []string) ([][]float32, error) {
	return [][]float32{{0}}, nil
}

func (milvusContractTestProvider) Dimension() int  { return 768 }
func (milvusContractTestProvider) Backend() string { return "test" }

func (milvusContractTestProvider) EmbeddingDimensionContract() (embedding.DimensionContract, error) {
	return embedding.DimensionContract{
		NativeDimension:     768,
		SupportedDimensions: []int{768, 512, 256, 128, 64},
	}, nil
}

func TestMilvusStoreUsesPreparedEmbeddingContract(t *testing.T) {
	provider := embedding.WithOptions(milvusContractTestProvider{}, embedding.Options{})

	got, err := resolveMilvusStoreEmbeddingDimension(
		EmbeddingConfig{Provider: provider},
		384,
	)
	if err != nil {
		t.Fatalf("omitted dimension returned error: %v", err)
	}
	if got != 768 {
		t.Fatalf("omitted dimension = %d, want native dimension 768", got)
	}

	got, err = resolveMilvusStoreEmbeddingDimension(
		EmbeddingConfig{Provider: provider, Dimension: 256},
		384,
	)
	if err != nil || got != 256 {
		t.Fatalf("declared dimension = %d, err=%v; want 256", got, err)
	}
}

func TestMilvusStoreRejectsUndeclaredPreparedDimension(t *testing.T) {
	provider := embedding.WithOptions(milvusContractTestProvider{}, embedding.Options{})

	if _, err := resolveMilvusStoreEmbeddingDimension(
		EmbeddingConfig{Provider: provider, Dimension: 384},
		0,
	); err == nil {
		t.Fatal("undeclared dimension was accepted")
	}
}
