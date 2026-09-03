package extproc

import "testing"

func TestOpenVINOEmbeddingDispatchNormalizesModelType(t *testing.T) {
	tests := []struct {
		name           string
		modelType      string
		usesModernBERT bool
	}{
		{name: "padded uppercase mmbert", modelType: " MMBERT ", usesModernBERT: true},
		{name: "mixed case modernbert", modelType: " MoDeRnBeRt ", usesModernBERT: true},
		{name: "bert keeps default path", modelType: " BERT ", usesModernBERT: false},
		{name: "unknown keeps default path", modelType: "custom", usesModernBERT: false},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := openvinoEmbeddingUsesModernBERT(test.modelType); got != test.usesModernBERT {
				t.Fatalf("openvinoEmbeddingUsesModernBERT(%q) = %t, want %t", test.modelType, got, test.usesModernBERT)
			}
		})
	}
}
