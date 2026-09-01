package modelruntime

import (
	"fmt"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type embeddingDimensionContract struct {
	Default   int
	Supported []int
}

var multimodalDimensionContract = func() embeddingDimensionContract {
	return embeddingDimensionContract{
		Default:   candle_binding.MultiModalGetEmbeddingDim(),
		Supported: candle_binding.MultiModalGetSupportedDimensions(),
	}
}

func initializeMultiModalEmbeddingModel(component string, useCPU bool, modelPath string, targetDimension int) bool {
	if modelPath == "" {
		return false
	}

	logging.ComponentEvent(component, "multimodal_embedding_init_started", map[string]interface{}{
		"model_ref": modelPath,
		"use_cpu":   useCPU,
	})
	if err := candle_binding.InitMultiModalEmbeddingModel(modelPath, useCPU); err != nil {
		logging.ComponentWarnEvent(component, "multimodal_embedding_init_failed", map[string]interface{}{
			"model_ref":               modelPath,
			"error":                   err.Error(),
			"multimodal_routes_ready": false,
		})
		return false
	}
	contract := multimodalDimensionContract()
	if err := validateConfiguredMultimodalDimension(targetDimension, contract); err != nil {
		logging.ComponentWarnEvent(component, "multimodal_embedding_dimension_invalid", map[string]interface{}{
			"model_ref": modelPath,
			"error":     err.Error(),
		})
		return false
	}
	logging.ComponentEvent(component, "multimodal_embedding_initialized", map[string]interface{}{
		"model_ref": modelPath,
	})
	return true
}

func validateConfiguredMultimodalDimension(dimension int, contract embeddingDimensionContract) error {
	if dimension == 0 {
		return nil
	}
	for _, supported := range contract.Supported {
		if dimension == supported {
			return nil
		}
	}
	return fmt.Errorf("configured target dimension %d is unsupported; model default is %d and supported dimensions are %v", dimension, contract.Default, contract.Supported)
}
