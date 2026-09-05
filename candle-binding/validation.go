package candle_binding

// This file carries no build constraint on purpose. Request validation must run
// identically whether the native Candle backend is linked (CGO build) or the
// fail-closed stub is compiled in (non-CGO build). Both semantic-router.go and
// semantic-router_mock.go call these helpers before dispatching a request, so a
// malformed input is rejected the same way in either mode rather than only when
// the native backend happens to be present (issue #2619, issue #2675).

import (
	"fmt"
	"strings"
)

// validateRequiredText validates a required string argument to a public
// entry point. It rejects empty values and values containing a NUL byte.
// The field name is interpolated verbatim so the message matches historical checks.
func validateRequiredText(field, value string) error {
	if value == "" {
		return fmt.Errorf("%s cannot be empty", field)
	}
	if strings.IndexByte(value, 0) >= 0 {
		return fmt.Errorf("%s cannot contain NUL bytes", field)
	}
	return nil
}

// validateTargetDim validates target embedding dimension. Negative values are invalid;
// 0 represents auto/default dimension.
func validateTargetDim(targetDim int) error {
	if targetDim < 0 {
		return fmt.Errorf("targetDim cannot be negative, got %d", targetDim)
	}
	return nil
}

// validateTopK validates top-k parameter. Negative values are invalid;
// 0 represents return-all / default.
func validateTopK(topK int) error {
	if topK < 0 {
		return fmt.Errorf("topK cannot be negative, got %d", topK)
	}
	return nil
}

// validateCandidates validates a candidate string slice. It rejects empty slices
// and candidate strings containing empty/NUL content.
func validateCandidates(candidates []string) error {
	if len(candidates) == 0 {
		return fmt.Errorf("candidates array cannot be empty")
	}
	for i, c := range candidates {
		if c == "" {
			return fmt.Errorf("candidate at index %d cannot be empty", i)
		}
		if strings.IndexByte(c, 0) >= 0 {
			return fmt.Errorf("candidate at index %d cannot contain NUL bytes", i)
		}
	}
	return nil
}

// validateImageTensor validates image pixel data and tensor dimensions (C=3, H, W).
func validateImageTensor(pixelData []float32, height, width, targetDim int) error {
	if len(pixelData) == 0 {
		return fmt.Errorf("pixelData cannot be empty")
	}
	if height <= 0 {
		return fmt.Errorf("height must be positive, got %d", height)
	}
	if width <= 0 {
		return fmt.Errorf("width must be positive, got %d", width)
	}
	expected := 3 * height * width
	if len(pixelData) != expected {
		return fmt.Errorf("pixelData length %d != expected %d (3*%d*%d)", len(pixelData), expected, height, width)
	}
	if err := validateTargetDim(targetDim); err != nil {
		return err
	}
	return nil
}

// validateAudioTensor validates audio mel spectrogram data and tensor dimensions.
func validateAudioTensor(melData []float32, nMels, timeFrames, targetDim int) error {
	if len(melData) == 0 {
		return fmt.Errorf("melData cannot be empty")
	}
	if nMels <= 0 {
		return fmt.Errorf("nMels must be positive, got %d", nMels)
	}
	if timeFrames <= 0 {
		return fmt.Errorf("timeFrames must be positive, got %d", timeFrames)
	}
	expected := nMels * timeFrames
	if len(melData) != expected {
		return fmt.Errorf("melData length %d != expected %d (%d*%d)", len(melData), expected, nMels, timeFrames)
	}
	if err := validateTargetDim(targetDim); err != nil {
		return err
	}
	return nil
}

// validateImageBytes validates raw image bytes and target dimension.
func validateImageBytes(imageBytes []byte, targetDim int) error {
	if len(imageBytes) == 0 {
		return fmt.Errorf("imageBytes cannot be empty")
	}
	if err := validateTargetDim(targetDim); err != nil {
		return err
	}
	return nil
}

// validateSimilarityBatch validates all inputs to CalculateSimilarityBatch.
func validateSimilarityBatch(query string, candidates []string, topK int, modelType string, targetDim int) error {
	if err := validateRequiredText("query", query); err != nil {
		return err
	}
	if modelType != "auto" && modelType != "qwen3" && modelType != "gemma" {
		return fmt.Errorf("invalid model type: %s (must be 'auto', 'qwen3', or 'gemma')", modelType)
	}
	if err := validateCandidates(candidates); err != nil {
		return err
	}
	if err := validateTopK(topK); err != nil {
		return err
	}
	if err := validateTargetDim(targetDim); err != nil {
		return err
	}
	return nil
}

// validateEmbeddingSimilarity validates inputs to CalculateEmbeddingSimilarity.
func validateEmbeddingSimilarity(text1, text2, modelType string, targetDim int) error {
	if err := validateRequiredText("text1", text1); err != nil {
		return err
	}
	if err := validateRequiredText("text2", text2); err != nil {
		return err
	}
	if modelType != "auto" && modelType != "qwen3" && modelType != "gemma" {
		return fmt.Errorf("invalid model type: %s (must be 'auto', 'qwen3', or 'gemma')", modelType)
	}
	if err := validateTargetDim(targetDim); err != nil {
		return err
	}
	return nil
}
