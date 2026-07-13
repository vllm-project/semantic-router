package onnx_binding

import (
	"fmt"
	"math"
	"strings"
)

const maxONNXCInt = int64(1<<31 - 1)

// SimilarityOptions is the shared pair/batch similarity contract used by the
// native backends. ONNX currently has one similarity model, mmBERT: "auto"
// deterministically resolves to mmBERT and TargetLayer controls its early exit.
// QualityPriority and LatencyPriority are validated for API parity, but they do
// not select or rank models in this single-model backend.
type SimilarityOptions struct {
	ModelType       string
	TargetLayer     int
	TargetDim       int
	QualityPriority float32
	LatencyPriority float32
}

// validateONNXSimilarityModelType preserves the public API's "auto" default
// while keeping model selection honest for the ONNX backend.
func validateONNXSimilarityModelType(modelType string) (string, error) {
	normalized := strings.ToLower(strings.TrimSpace(modelType))
	switch normalized {
	case "auto", "mmbert":
		return "mmbert", nil
	default:
		return "", fmt.Errorf("unsupported ONNX embedding model type %q", modelType)
	}
}

func normalizeONNXSimilarityOptions(options SimilarityOptions) (SimilarityOptions, error) {
	modelType, err := validateONNXSimilarityModelType(options.ModelType)
	if err != nil {
		return SimilarityOptions{}, err
	}
	options.ModelType = modelType

	if err := validateONNXEmbeddingControls(options.TargetLayer, options.TargetDim); err != nil {
		return SimilarityOptions{}, err
	}
	if options.TargetLayer != 0 && options.ModelType != "mmbert" {
		return SimilarityOptions{}, fmt.Errorf("target layer is only supported for ONNX model type mmbert")
	}
	if !validONNXSimilarityPriority(options.QualityPriority) ||
		!validONNXSimilarityPriority(options.LatencyPriority) {
		return SimilarityOptions{}, fmt.Errorf("similarity priorities must be finite values between 0 and 1")
	}

	return options, nil
}

func validateONNXNonNegativeCInt(name string, value int) error {
	if value < 0 {
		return fmt.Errorf("%s cannot be negative", name)
	}
	if int64(value) > maxONNXCInt {
		return fmt.Errorf("%s must fit a signed 32-bit C int", name)
	}
	return nil
}

func validateONNXEmbeddingControls(targetLayer, targetDim int) error {
	if err := validateONNXNonNegativeCInt("target layer", targetLayer); err != nil {
		return err
	}
	return validateONNXNonNegativeCInt("target dimension", targetDim)
}

func validateONNXEmbeddingPriorities(qualityPriority, latencyPriority float32) error {
	if !validONNXSimilarityPriority(qualityPriority) || !validONNXSimilarityPriority(latencyPriority) {
		return fmt.Errorf("embedding priorities must be finite values between 0 and 1")
	}
	return nil
}

func validONNXSimilarityPriority(priority float32) bool {
	value := float64(priority)
	return !math.IsNaN(value) && !math.IsInf(value, 0) && priority >= 0 && priority <= 1
}
