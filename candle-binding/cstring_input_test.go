//go:build !windows && cgo && (amd64 || arm64)
// +build !windows
// +build cgo
// +build amd64 arm64

package candle_binding

import (
	"errors"
	"testing"
)

func TestCandleRequestBoundariesRejectEmbeddedNULBeforeFFI(t *testing.T) {
	t.Parallel()

	invalid := "trusted-prefix\x00hidden-suffix"
	tests := []struct {
		name string
		call func() error
	}{
		{"tokenization", func() error { _, err := TokenizeText(invalid, 512); return err }},
		{"embedding", func() error { _, err := GetEmbedding(invalid, 512); return err }},
		{"smart embedding", func() error { _, err := GetEmbeddingSmart(invalid, 0, 0); return err }},
		{"batched embedding selector", func() error { _, err := GetEmbeddingBatched("valid", invalid, 0); return err }},
		{"multi-modal text", func() error { _, err := MultiModalEncodeText(invalid, 0); return err }},
		{"embedding metadata", func() error { _, err := GetEmbeddingWithMetadata(invalid, 0, 0, 0); return err }},
		{"2D matryoshka", func() error { _, err := GetEmbedding2DMatryoshka(invalid, "mmbert", 0, 0); return err }},
		{"embedding similarity second operand", func() error {
			_, err := CalculateEmbeddingSimilarity("valid", invalid, "auto", 0)
			return err
		}},
		{"batch similarity candidate", func() error {
			_, err := CalculateSimilarityBatch("valid", []string{"valid", invalid}, 1, "auto", 0)
			return err
		}},
		{"sequence classification", func() error { _, err := ClassifyText(invalid); return err }},
		{"PII classification", func() error { _, err := ClassifyMmBert32KPII(invalid); return err }},
		{"hallucination answer", func() error {
			_, err := DetectHallucinations("context", "question", invalid, 0)
			return err
		}},
		{"NLI hypothesis", func() error { _, err := ClassifyNLI("premise", invalid); return err }},
		{"ModernBERT classification", func() error { _, err := ClassifyModernBertText(invalid); return err }},
		{"ModernBERT PII config path", func() error {
			_, err := ClassifyModernBertPIITokens("valid", invalid)
			return err
		}},
		{"BERT label JSON", func() error { _, err := ClassifyBertPIITokens("valid", invalid); return err }},
		{"Candle BERT label JSON", func() error {
			_, err := ClassifyCandleBertTokensWithLabels("valid", invalid)
			return err
		}},
		{"LoRA batch item", func() error { _, err := ClassifyBatchWithLoRA([]string{"valid", invalid}); return err }},
		{"Qwen adapter name", func() error { _, err := ClassifyWithQwen3Adapter("valid", invalid); return err }},
		{"zero-shot category", func() error { _, err := ClassifyZeroShotQwen3("valid", []string{"safe", invalid}); return err }},
		{"guard mode", func() error { _, err := GetGuardRawOutput("valid", invalid); return err }},
		{"MLP JSON", func() error { _, err := MLPFromJSON(invalid); return err }},
		{"MLP JSON with device", func() error { _, err := MLPFromJSONWithDevice(invalid, MLPDeviceCPU); return err }},
		{"MLP JSON with device and dtype", func() error {
			_, err := MLPFromJSONWithDeviceAndDType(invalid, MLPDeviceCPU, MLPF32)
			return err
		}},
	}

	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if err := test.call(); !errors.Is(err, errEmbeddedNULByte) {
				t.Fatalf("expected embedded-NUL validation error, got %v", err)
			}
		})
	}
}

func TestCandleLegacyNoErrorAPIsRejectEmbeddedNUL(t *testing.T) {
	t.Parallel()

	if got := CalculateSimilarity("valid", "hidden\x00suffix", 512); got != -1.0 {
		t.Fatalf("CalculateSimilarity must fail closed with -1, got %f", got)
	}
	if got := FindMostSimilar("valid", []string{"hidden\x00suffix"}, 512); got.Index != -1 || got.Score != -1.0 {
		t.Fatalf("FindMostSimilar must fail closed with its legacy sentinel, got %+v", got)
	}
}

func TestValidateCStringInputsIdentifiesEmbeddedNUL(t *testing.T) {
	t.Parallel()

	err := validateCStringInputs(
		cStringInput{"first", "valid"},
		cStringInput{"second", "prefix\x00suffix"},
	)
	if !errors.Is(err, errEmbeddedNULByte) {
		t.Fatalf("expected embedded-NUL validation error, got %v", err)
	}
}
