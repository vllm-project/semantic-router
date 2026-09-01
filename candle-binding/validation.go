package candle_binding

// This file carries no build constraint on purpose. Request validation must run
// identically whether the native Candle backend is linked (CGO build) or the
// fail-closed stub is compiled in (non-CGO build). Both semantic-router.go and
// semantic-router_mock.go call these helpers before dispatching a request, so a
// malformed input is rejected the same way in either mode rather than only when
// the native backend happens to be present (issue #2619, issue #2675).
//
// Scope: this file defines shared validators for input presence, NUL-byte checks,
// target-dimension legal sets, top-k bounds, tensor-shape constraints, and candidate slices.
//
// The NUL check does tighten one previously accepted case: a data URI whose
// prefix contained a NUL, such as "data:image/png\x00;base64,...", used to
// decode successfully because everything before ";base64," is discarded. That
// input is now rejected. See validateRequiredText for why that is intended.

import (
	"fmt"
	"math"
	"strings"
)

// validateRequiredText validates a required string argument to a public
// entry point. It rejects empty values and values containing a NUL byte.
// The field name is interpolated verbatim so the message matches the
// historical inline checks, e.g. "text cannot be empty".
//
// The reason for rejecting NUL differs by argument, and only one of them is a
// cgo concern:
//
//   - text is passed to C.CString, where a NUL terminates the string early, so
//     the native backend would silently receive a truncated prompt.
//   - base64Str and url never reach cgo: they are consumed in Go by
//     base64.StdEncoding.DecodeString and http.NewRequest respectively. A NUL
//     there is not a truncation risk but a malformed-input signal — in a URL it
//     is a classic request-smuggling smell, and in a data URI it can hide bytes
//     ahead of the ";base64," separator that the parser then discards.
//
// Rejecting NUL uniformly is deliberate: a shared validator that applied
// different rules per argument would be harder to reason about than one that
// refuses a byte no legitimate caller sends. Callers that genuinely need to
// carry NUL in a payload should encode it rather than pass it raw.
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
	if targetDim > math.MaxInt32 {
		return fmt.Errorf("targetDim %d exceeds maximum 32-bit integer range (%d)", targetDim, math.MaxInt32)
	}
	return nil
}

// validateTargetLayer validates layer early-exit parameter. Negative values are invalid;
// 0 represents full model (no early exit).
func validateTargetLayer(targetLayer int) error {
	if targetLayer < 0 {
		return fmt.Errorf("targetLayer cannot be negative, got %d", targetLayer)
	}
	if targetLayer > math.MaxInt32 {
		return fmt.Errorf("targetLayer %d exceeds maximum 32-bit integer range (%d)", targetLayer, math.MaxInt32)
	}
	return nil
}

// validateTopK validates top-k parameter. Negative values are invalid;
// 0 represents return-all / default.
func validateTopK(topK int) error {
	if topK < 0 {
		return fmt.Errorf("topK cannot be negative, got %d", topK)
	}
	if topK > math.MaxInt32 {
		return fmt.Errorf("topK %d exceeds maximum 32-bit integer range (%d)", topK, math.MaxInt32)
	}
	return nil
}

// validateCandidates validates a candidate string slice. It rejects empty slices
// and candidate strings containing empty/NUL content.
func validateCandidates(candidates []string) error {
	if len(candidates) == 0 {
		return fmt.Errorf("candidates array cannot be empty")
	}
	if len(candidates) > math.MaxInt32 {
		return fmt.Errorf("candidates array length %d exceeds maximum 32-bit integer range (%d)", len(candidates), math.MaxInt32)
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
	if height > math.MaxInt32 {
		return fmt.Errorf("height %d exceeds maximum 32-bit integer range (%d)", height, math.MaxInt32)
	}
	if width <= 0 {
		return fmt.Errorf("width must be positive, got %d", width)
	}
	if width > math.MaxInt32 {
		return fmt.Errorf("width %d exceeds maximum 32-bit integer range (%d)", width, math.MaxInt32)
	}
	// Overflow-safe check for 3 * height * width:
	if height > (math.MaxInt/3)/width {
		return fmt.Errorf("image tensor dimensions %dx%d overflow integer range", height, width)
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
	if nMels > math.MaxInt32 {
		return fmt.Errorf("nMels %d exceeds maximum 32-bit integer range (%d)", nMels, math.MaxInt32)
	}
	if timeFrames <= 0 {
		return fmt.Errorf("timeFrames must be positive, got %d", timeFrames)
	}
	if timeFrames > math.MaxInt32 {
		return fmt.Errorf("timeFrames %d exceeds maximum 32-bit integer range (%d)", timeFrames, math.MaxInt32)
	}
	// Overflow-safe check for nMels * timeFrames:
	if timeFrames > math.MaxInt/nMels {
		return fmt.Errorf("audio tensor dimensions %dx%d overflow integer range", nMels, timeFrames)
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
