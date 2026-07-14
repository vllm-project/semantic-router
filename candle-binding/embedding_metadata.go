package candle_binding

import (
	"fmt"
	"math"
)

const maxCInt = int64(1<<31 - 1)

// SimilarityOptions is the shared pair/batch routing contract. TargetLayer is
// meaningful only for mmBERT; priorities are meaningful only for auto routing.
type SimilarityOptions struct {
	ModelType       string
	TargetLayer     int
	TargetDim       int
	QualityPriority float32
	LatencyPriority float32
}

func embeddingModelTypeName(modelType int) string {
	switch modelType {
	case 0:
		return "qwen3"
	case 1:
		return "gemma"
	case 2:
		return "mmbert"
	case 3:
		return "multimodal"
	default:
		return "unknown"
	}
}

func validateSimilarityModelType(modelType string) error {
	switch modelType {
	case "auto", "qwen3", "gemma", "mmbert":
		return nil
	default:
		return fmt.Errorf("invalid model type: %s (must be 'auto', 'qwen3', 'gemma', or 'mmbert')", modelType)
	}
}

func normalizeSimilarityOptions(options SimilarityOptions) (SimilarityOptions, error) {
	if err := validateSimilarityModelType(options.ModelType); err != nil {
		return SimilarityOptions{}, err
	}
	if err := validateSimilarityIntegerOptions(options); err != nil {
		return SimilarityOptions{}, err
	}
	if options.TargetLayer != 0 && options.ModelType != "mmbert" {
		return SimilarityOptions{}, fmt.Errorf("target layer is only supported for model type mmbert")
	}
	if !validSimilarityPriority(options.QualityPriority) ||
		!validSimilarityPriority(options.LatencyPriority) {
		return SimilarityOptions{}, fmt.Errorf("similarity priorities must be finite values between 0 and 1")
	}
	if options.ModelType == "auto" &&
		options.QualityPriority == 0 && options.LatencyPriority == 0 {
		options.QualityPriority = 0.5
		options.LatencyPriority = 0.5
	}
	return options, nil
}

func validateSimilarityIntegerOptions(options SimilarityOptions) error {
	if err := validateNonNegativeCInt("target layer", options.TargetLayer); err != nil {
		return err
	}
	return validateNonNegativeCInt("target dimension", options.TargetDim)
}

func validateNonNegativeCInt(name string, value int) error {
	if value < 0 {
		return fmt.Errorf("%s cannot be negative", name)
	}
	if int64(value) > maxCInt {
		return fmt.Errorf("%s must fit a signed 32-bit C int", name)
	}
	return nil
}

func validateEmbeddingControls(targetLayer, targetDim int) error {
	if err := validateNonNegativeCInt("target layer", targetLayer); err != nil {
		return err
	}
	return validateNonNegativeCInt("target dimension", targetDim)
}

func validateEmbeddingPriorities(qualityPriority, latencyPriority float32) error {
	if !validSimilarityPriority(qualityPriority) || !validSimilarityPriority(latencyPriority) {
		return fmt.Errorf("embedding priorities must be finite values between 0 and 1")
	}
	return nil
}

func validSimilarityPriority(priority float32) bool {
	value := float64(priority)
	return !math.IsNaN(value) && !math.IsInf(value, 0) && priority >= 0 && priority <= 1
}
