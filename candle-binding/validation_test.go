package candle_binding

import (
	"strings"
	"testing"
)

// This test file carries no build constraint, so it runs under both the CGO and
// non-CGO builds. It pins the shared request validation contract from #2619:
// the same checks must apply regardless of whether the native backend is linked.

func TestValidateRequiredText(t *testing.T) {
	cases := []struct {
		name    string
		field   string
		value   string
		wantErr string // substring expected in the error, "" means no error
	}{
		{name: "valid", field: "text", value: "hello", wantErr: ""},
		{name: "empty", field: "text", value: "", wantErr: "text cannot be empty"},
		{name: "nul in middle", field: "text", value: "a\x00b", wantErr: "text cannot contain NUL bytes"},
		{name: "nul only", field: "url", value: "\x00", wantErr: "url cannot contain NUL bytes"},
		{name: "field name is used", field: "base64Str", value: "", wantErr: "base64Str cannot be empty"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateRequiredText(tc.field, tc.value)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("expected no error, got %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("expected error containing %q, got %q", tc.wantErr, err.Error())
			}
		})
	}
}

// TestValidateSemanticHelpers covers the shared validation helper functions directly.
func TestValidateSemanticHelpers(t *testing.T) {
	t.Run("validateTargetDim", func(t *testing.T) {
		if err := validateTargetDim(0); err != nil {
			t.Fatalf("expected 0 to be valid, got %v", err)
		}
		if err := validateTargetDim(128); err != nil {
			t.Fatalf("expected 128 to be valid, got %v", err)
		}
		if err := validateTargetDim(-1); err == nil || !strings.Contains(err.Error(), "targetDim cannot be negative") {
			t.Fatalf("expected negative error, got %v", err)
		}
	})

	t.Run("validateTopK", func(t *testing.T) {
		if err := validateTopK(0); err != nil {
			t.Fatalf("expected 0 to be valid, got %v", err)
		}
		if err := validateTopK(5); err != nil {
			t.Fatalf("expected 5 to be valid, got %v", err)
		}
		if err := validateTopK(-1); err == nil || !strings.Contains(err.Error(), "topK cannot be negative") {
			t.Fatalf("expected negative error, got %v", err)
		}
	})

	t.Run("validateCandidates", func(t *testing.T) {
		if err := validateCandidates([]string{"a", "b"}); err != nil {
			t.Fatalf("expected valid candidates, got %v", err)
		}
		if err := validateCandidates(nil); err == nil || !strings.Contains(err.Error(), "candidates array cannot be empty") {
			t.Fatalf("expected empty array error, got %v", err)
		}
		if err := validateCandidates([]string{}); err == nil || !strings.Contains(err.Error(), "candidates array cannot be empty") {
			t.Fatalf("expected empty array error, got %v", err)
		}
		if err := validateCandidates([]string{""}); err == nil || !strings.Contains(err.Error(), "cannot be empty") {
			t.Fatalf("expected empty candidate error, got %v", err)
		}
		if err := validateCandidates([]string{"valid", "bad\x00candidate"}); err == nil || !strings.Contains(err.Error(), "cannot contain NUL bytes") {
			t.Fatalf("expected NUL candidate error, got %v", err)
		}
	})

	t.Run("validateImageTensor", func(t *testing.T) {
		pixelData := make([]float32, 3*512*512)
		if err := validateImageTensor(pixelData, 512, 512, 0); err != nil {
			t.Fatalf("expected valid image tensor, got %v", err)
		}
		if err := validateImageTensor(nil, 512, 512, 0); err == nil || !strings.Contains(err.Error(), "pixelData cannot be empty") {
			t.Fatalf("expected empty pixelData error, got %v", err)
		}
		if err := validateImageTensor(pixelData, 0, 512, 0); err == nil || !strings.Contains(err.Error(), "height must be positive") {
			t.Fatalf("expected positive height error, got %v", err)
		}
		if err := validateImageTensor(pixelData, 512, -1, 0); err == nil || !strings.Contains(err.Error(), "width must be positive") {
			t.Fatalf("expected positive width error, got %v", err)
		}
		if err := validateImageTensor(make([]float32, 100), 512, 512, 0); err == nil || !strings.Contains(err.Error(), "pixelData length 100 != expected") {
			t.Fatalf("expected length mismatch error, got %v", err)
		}
		if err := validateImageTensor(pixelData, 512, 512, -5); err == nil || !strings.Contains(err.Error(), "targetDim cannot be negative") {
			t.Fatalf("expected targetDim negative error, got %v", err)
		}
	})

	t.Run("validateAudioTensor", func(t *testing.T) {
		melData := make([]float32, 80*100)
		if err := validateAudioTensor(melData, 80, 100, 0); err != nil {
			t.Fatalf("expected valid audio tensor, got %v", err)
		}
		if err := validateAudioTensor(nil, 80, 100, 0); err == nil || !strings.Contains(err.Error(), "melData cannot be empty") {
			t.Fatalf("expected empty melData error, got %v", err)
		}
		if err := validateAudioTensor(melData, 0, 100, 0); err == nil || !strings.Contains(err.Error(), "nMels must be positive") {
			t.Fatalf("expected positive nMels error, got %v", err)
		}
		if err := validateAudioTensor(melData, 80, -1, 0); err == nil || !strings.Contains(err.Error(), "timeFrames must be positive") {
			t.Fatalf("expected positive timeFrames error, got %v", err)
		}
		if err := validateAudioTensor(make([]float32, 50), 80, 100, 0); err == nil || !strings.Contains(err.Error(), "melData length 50 != expected") {
			t.Fatalf("expected length mismatch error, got %v", err)
		}
		if err := validateAudioTensor(melData, 80, 100, -10); err == nil || !strings.Contains(err.Error(), "targetDim cannot be negative") {
			t.Fatalf("expected targetDim negative error, got %v", err)
		}
	})

	t.Run("validateImageBytes", func(t *testing.T) {
		if err := validateImageBytes([]byte{1, 2, 3}, 0); err != nil {
			t.Fatalf("expected valid image bytes, got %v", err)
		}
		if err := validateImageBytes(nil, 0); err == nil || !strings.Contains(err.Error(), "imageBytes cannot be empty") {
			t.Fatalf("expected empty imageBytes error, got %v", err)
		}
		if err := validateImageBytes([]byte{1, 2, 3}, -1); err == nil || !strings.Contains(err.Error(), "targetDim cannot be negative") {
			t.Fatalf("expected targetDim negative error, got %v", err)
		}
	})

	t.Run("validateSimilarityBatch", func(t *testing.T) {
		if err := validateSimilarityBatch("q", []string{"a", "b"}, 1, "auto", 0); err != nil {
			t.Fatalf("expected valid batch similarity, got %v", err)
		}
		if err := validateSimilarityBatch("", []string{"a"}, 1, "auto", 0); err == nil || !strings.Contains(err.Error(), "query cannot be empty") {
			t.Fatalf("expected empty query error, got %v", err)
		}
		if err := validateSimilarityBatch("q\x00x", []string{"a"}, 1, "auto", 0); err == nil || !strings.Contains(err.Error(), "cannot contain NUL bytes") {
			t.Fatalf("expected NUL query error, got %v", err)
		}
		if err := validateSimilarityBatch("q", []string{"a"}, 1, "invalid_model", 0); err == nil || !strings.Contains(err.Error(), "invalid model type") {
			t.Fatalf("expected invalid model error, got %v", err)
		}
		if err := validateSimilarityBatch("q", nil, 1, "auto", 0); err == nil || !strings.Contains(err.Error(), "candidates array cannot be empty") {
			t.Fatalf("expected empty candidates error, got %v", err)
		}
		if err := validateSimilarityBatch("q", []string{"a"}, -1, "auto", 0); err == nil || !strings.Contains(err.Error(), "topK cannot be negative") {
			t.Fatalf("expected topK negative error, got %v", err)
		}
		if err := validateSimilarityBatch("q", []string{"a"}, 1, "auto", -1); err == nil || !strings.Contains(err.Error(), "targetDim cannot be negative") {
			t.Fatalf("expected targetDim negative error, got %v", err)
		}
	})

	t.Run("validateEmbeddingSimilarity", func(t *testing.T) {
		if err := validateEmbeddingSimilarity("t1", "t2", "qwen3", 0); err != nil {
			t.Fatalf("expected valid embedding similarity, got %v", err)
		}
		if err := validateEmbeddingSimilarity("", "t2", "qwen3", 0); err == nil || !strings.Contains(err.Error(), "text1 cannot be empty") {
			t.Fatalf("expected empty text1 error, got %v", err)
		}
		if err := validateEmbeddingSimilarity("t1", "", "qwen3", 0); err == nil || !strings.Contains(err.Error(), "text2 cannot be empty") {
			t.Fatalf("expected empty text2 error, got %v", err)
		}
		if err := validateEmbeddingSimilarity("t1", "t2", "unsupported", 0); err == nil || !strings.Contains(err.Error(), "invalid model type") {
			t.Fatalf("expected invalid model error, got %v", err)
		}
		if err := validateEmbeddingSimilarity("t1", "t2", "qwen3", -1); err == nil || !strings.Contains(err.Error(), "targetDim cannot be negative") {
			t.Fatalf("expected targetDim negative error, got %v", err)
		}
	})
}

// TestMultiModalValidationRunsInBothModes verifies the public multimodal APIs
// reject malformed input via the shared validator before any backend dispatch.
// Because validation precedes the native call, invalid input is rejected the
// same way whether the CGO backend or the fail-closed stub is compiled in, and
// the assertions need no linked backend.
func TestMultiModalValidationRunsInBothModes(t *testing.T) {
	t.Run("empty rejected", func(t *testing.T) {
		if _, err := MultiModalEncodeText("", 0); err == nil ||
			!strings.Contains(err.Error(), "text cannot be empty") {
			t.Fatalf("MultiModalEncodeText: want empty-text error, got %v", err)
		}
		if _, err := MultiModalEncodeImage(nil, 512, 512, 0); err == nil ||
			!strings.Contains(err.Error(), "pixelData cannot be empty") {
			t.Fatalf("MultiModalEncodeImage: want empty pixelData error, got %v", err)
		}
		if _, err := MultiModalEncodeAudio(nil, 80, 100, 0); err == nil ||
			!strings.Contains(err.Error(), "melData cannot be empty") {
			t.Fatalf("MultiModalEncodeAudio: want empty melData error, got %v", err)
		}
		if _, err := MultiModalEncodeImageFromBytes(nil, 0); err == nil ||
			!strings.Contains(err.Error(), "imageBytes cannot be empty") {
			t.Fatalf("MultiModalEncodeImageFromBytes: want empty error, got %v", err)
		}
		if _, err := MultiModalEncodeImageFromBase64("", 0); err == nil ||
			!strings.Contains(err.Error(), "base64Str cannot be empty") {
			t.Fatalf("MultiModalEncodeImageFromBase64: want empty error, got %v", err)
		}
		if _, err := MultiModalEncodeImageFromURL("", 0); err == nil ||
			!strings.Contains(err.Error(), "url cannot be empty") {
			t.Fatalf("MultiModalEncodeImageFromURL: want empty-url error, got %v", err)
		}
	})

	t.Run("NUL rejected", func(t *testing.T) {
		if _, err := MultiModalEncodeText("a\x00b", 0); err == nil ||
			!strings.Contains(err.Error(), "cannot contain NUL bytes") {
			t.Fatalf("MultiModalEncodeText: want NUL error, got %v", err)
		}
		if _, err := MultiModalEncodeImageFromURL("http://x\x00y", 0); err == nil ||
			!strings.Contains(err.Error(), "cannot contain NUL bytes") {
			t.Fatalf("MultiModalEncodeImageFromURL: want NUL error, got %v", err)
		}
		if _, err := MultiModalEncodeImageFromBase64("a\x00b", 0); err == nil ||
			!strings.Contains(err.Error(), "cannot contain NUL bytes") {
			t.Fatalf("MultiModalEncodeImageFromBase64: want NUL error, got %v", err)
		}
	})

	t.Run("negative targetDim rejected", func(t *testing.T) {
		if _, err := MultiModalEncodeText("valid text", -1); err == nil ||
			!strings.Contains(err.Error(), "targetDim cannot be negative") {
			t.Fatalf("MultiModalEncodeText: want targetDim error, got %v", err)
		}
		if _, err := MultiModalEncodeImage(make([]float32, 3*512*512), 512, 512, -1); err == nil ||
			!strings.Contains(err.Error(), "targetDim cannot be negative") {
			t.Fatalf("MultiModalEncodeImage: want targetDim error, got %v", err)
		}
		if _, err := MultiModalEncodeAudio(make([]float32, 80*100), 80, 100, -1); err == nil ||
			!strings.Contains(err.Error(), "targetDim cannot be negative") {
			t.Fatalf("MultiModalEncodeAudio: want targetDim error, got %v", err)
		}
		if _, err := MultiModalEncodeImageFromBytes([]byte{1, 2, 3}, -1); err == nil ||
			!strings.Contains(err.Error(), "targetDim cannot be negative") {
			t.Fatalf("MultiModalEncodeImageFromBytes: want targetDim error, got %v", err)
		}
		if _, err := MultiModalEncodeImageFromBase64("valid", -1); err == nil ||
			!strings.Contains(err.Error(), "targetDim cannot be negative") {
			t.Fatalf("MultiModalEncodeImageFromBase64: want targetDim error, got %v", err)
		}
		if _, err := MultiModalEncodeImageFromURL("http://example.com/img.png", -1); err == nil ||
			!strings.Contains(err.Error(), "targetDim cannot be negative") {
			t.Fatalf("MultiModalEncodeImageFromURL: want targetDim error, got %v", err)
		}
	})

	t.Run("shape mismatch rejected", func(t *testing.T) {
		if _, err := MultiModalEncodeImage(make([]float32, 10), 512, 512, 0); err == nil ||
			!strings.Contains(err.Error(), "pixelData length 10 != expected") {
			t.Fatalf("MultiModalEncodeImage: want shape mismatch error, got %v", err)
		}
		if _, err := MultiModalEncodeAudio(make([]float32, 10), 80, 100, 0); err == nil ||
			!strings.Contains(err.Error(), "melData length 10 != expected") {
			t.Fatalf("MultiModalEncodeAudio: want shape mismatch error, got %v", err)
		}
	})
}

// TestSharedEmbeddingValidationRunsInBothModes tests embedding and similarity validation.
func TestSharedEmbeddingValidationRunsInBothModes(t *testing.T) {
	t.Run("CalculateSimilarityBatch validation", func(t *testing.T) {
		if _, err := CalculateSimilarityBatch("", []string{"a"}, 1, "auto", 0); err == nil ||
			!strings.Contains(err.Error(), "query cannot be empty") {
			t.Fatalf("CalculateSimilarityBatch: want empty query error, got %v", err)
		}
		if _, err := CalculateSimilarityBatch("q", nil, 1, "auto", 0); err == nil ||
			!strings.Contains(err.Error(), "candidates array cannot be empty") {
			t.Fatalf("CalculateSimilarityBatch: want empty candidates error, got %v", err)
		}
		if _, err := CalculateSimilarityBatch("q", []string{"a"}, -1, "auto", 0); err == nil ||
			!strings.Contains(err.Error(), "topK cannot be negative") {
			t.Fatalf("CalculateSimilarityBatch: want negative topK error, got %v", err)
		}
		if _, err := CalculateSimilarityBatch("q", []string{"a"}, 1, "auto", -1); err == nil ||
			!strings.Contains(err.Error(), "targetDim cannot be negative") {
			t.Fatalf("CalculateSimilarityBatch: want negative targetDim error, got %v", err)
		}
		if _, err := CalculateSimilarityBatch("q", []string{"a"}, 1, "invalid", 0); err == nil ||
			!strings.Contains(err.Error(), "invalid model type") {
			t.Fatalf("CalculateSimilarityBatch: want invalid model error, got %v", err)
		}
	})

	t.Run("CalculateEmbeddingSimilarity validation", func(t *testing.T) {
		if _, err := CalculateEmbeddingSimilarity("", "b", "auto", 0); err == nil ||
			!strings.Contains(err.Error(), "text1 cannot be empty") {
			t.Fatalf("CalculateEmbeddingSimilarity: want empty text1 error, got %v", err)
		}
		if _, err := CalculateEmbeddingSimilarity("a", "", "auto", 0); err == nil ||
			!strings.Contains(err.Error(), "text2 cannot be empty") {
			t.Fatalf("CalculateEmbeddingSimilarity: want empty text2 error, got %v", err)
		}
		if _, err := CalculateEmbeddingSimilarity("a", "b", "auto", -1); err == nil ||
			!strings.Contains(err.Error(), "targetDim cannot be negative") {
			t.Fatalf("CalculateEmbeddingSimilarity: want negative targetDim error, got %v", err)
		}
	})

	t.Run("GetEmbedding APIs validation", func(t *testing.T) {
		if _, err := GetEmbeddingWithDim("", 0.5, 0.5, 0); err == nil ||
			!strings.Contains(err.Error(), "text cannot be empty") {
			t.Fatalf("GetEmbeddingWithDim: want empty text error, got %v", err)
		}
		if _, err := GetEmbeddingWithDim("valid", 0.5, 0.5, -1); err == nil ||
			!strings.Contains(err.Error(), "targetDim cannot be negative") {
			t.Fatalf("GetEmbeddingWithDim: want negative targetDim error, got %v", err)
		}
		if _, err := GetEmbeddingWithMetadata("", 0.5, 0.5, 0); err == nil ||
			!strings.Contains(err.Error(), "text cannot be empty") {
			t.Fatalf("GetEmbeddingWithMetadata: want empty text error, got %v", err)
		}
		if _, err := GetEmbeddingWithMetadata("valid", 0.5, 0.5, -1); err == nil ||
			!strings.Contains(err.Error(), "targetDim cannot be negative") {
			t.Fatalf("GetEmbeddingWithMetadata: want negative targetDim error, got %v", err)
		}
		if _, err := GetEmbeddingWithModelType("", "qwen3", 0); err == nil ||
			!strings.Contains(err.Error(), "text cannot be empty") {
			t.Fatalf("GetEmbeddingWithModelType: want empty text error, got %v", err)
		}
		if _, err := GetEmbeddingWithModelType("valid", "qwen3", -1); err == nil ||
			!strings.Contains(err.Error(), "targetDim cannot be negative") {
			t.Fatalf("GetEmbeddingWithModelType: want negative targetDim error, got %v", err)
		}
		if _, err := GetEmbedding2DMatryoshka("valid", "qwen3", -1, 0); err == nil ||
			!strings.Contains(err.Error(), "targetLayer cannot be negative") {
			t.Fatalf("GetEmbedding2DMatryoshka: want negative targetLayer error, got %v", err)
		}
		if _, err := GetEmbeddingBatched("", "qwen3", 0); err == nil ||
			!strings.Contains(err.Error(), "text cannot be empty") {
			t.Fatalf("GetEmbeddingBatched: want empty text error, got %v", err)
		}
		if _, err := GetEmbeddingBatched("valid", "qwen3", -1); err == nil ||
			!strings.Contains(err.Error(), "targetDim cannot be negative") {
			t.Fatalf("GetEmbeddingBatched: want negative targetDim error, got %v", err)
		}
	})
}
