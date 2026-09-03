package candle_binding

import (
	"strings"
	"testing"
)

func TestResolveEmbeddingDimensionPreservesLegacyConfiguration(t *testing.T) {
	originalLoader := getEmbeddingDimensionContract
	t.Cleanup(func() {
		getEmbeddingDimensionContract = originalLoader
	})

	getEmbeddingDimensionContract = func(modelType string) (EmbeddingDimensionContract, error) {
		if modelType != "mmbert" {
			t.Fatalf("contract requested for %q, want mmbert", modelType)
		}
		return EmbeddingDimensionContract{
			Model:               "mmbert",
			NativeDimension:     768,
			SupportedDimensions: []int{768, 384},
		}, nil
	}

	tests := []struct {
		name      string
		requested int
		want      int
	}{
		{
			name:      "legacy configuration keeps explicit native dimension",
			requested: 768,
			want:      768,
		},
		{
			name:      "legacy configuration keeps explicit supported dimension",
			requested: 384,
			want:      384,
		},
		{
			name:      "new omitted dimension uses model native dimension",
			requested: 0,
			want:      768,
		},
		{
			name:      "negative omitted dimension uses model native dimension",
			requested: -1,
			want:      768,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := ResolveEmbeddingDimension(" MmBert ", test.requested)
			if err != nil {
				t.Fatalf("ResolveEmbeddingDimension() error = %v", err)
			}
			if got != test.want {
				t.Fatalf("ResolveEmbeddingDimension() = %d, want %d", got, test.want)
			}
		})
	}

	if _, err := ResolveEmbeddingDimension("mmbert", 512); err == nil || !strings.Contains(err.Error(), "does not support dimension 512") {
		t.Fatalf("unsupported dimension error = %v", err)
	}
}
