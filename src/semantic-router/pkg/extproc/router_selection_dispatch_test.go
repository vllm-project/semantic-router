package extproc

import "testing"

// TestCandleEmbeddingSupportsBatched checks only qwen3 uses the batched FFI;
// other types (incl. default mmbert) must use the single-text FFI.
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
