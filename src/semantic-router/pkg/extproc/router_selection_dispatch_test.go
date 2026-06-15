package extproc

import "testing"

// TestCandleEmbeddingSupportsBatched guards the selection-embedding dispatch:
// the candle batched FFI (GetEmbeddingBatched) is implemented only for qwen3,
// so every other model type — including the default "mmbert" — must be routed
// to the single-text FFI (GetEmbeddingWithModelType). Regressing this makes
// selection embeddings fail for the default config and collapses RouterDC /
// hybrid / session_aware base scores to a flat tie.
func TestCandleEmbeddingSupportsBatched(t *testing.T) {
	cases := []struct {
		modelType string
		batched   bool
	}{
		{"qwen3", true},
		{"mmbert", false},
		{"gemma", false},
		{"bert", false},
		{"modernbert", false},
		{"", false},
	}

	for _, tc := range cases {
		if got := candleEmbeddingSupportsBatched(tc.modelType); got != tc.batched {
			t.Errorf("candleEmbeddingSupportsBatched(%q) = %v, want %v", tc.modelType, got, tc.batched)
		}
	}
}
