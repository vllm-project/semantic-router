package ttft

import (
	"testing"

	"github.com/vllm-project/semantic-router/semantic-router/pkg/config"
)

func TestComputeBaseTTFT(t *testing.T) {

	gpuConfig := config.GPUConfig{
		FLOPS: 1e12, // 1 TFLOP
		HBM:   1e11, // 100 GB/s
	}
	calculator := NewCalculator(gpuConfig)

	routerCfg := &config.RouterConfig{}
	// Mock config methods if needed, or set up fields so that
	// GetModelParamCount, GetModelBatchSize, GetModelContextSize return defaults

	ttft := calculator.ComputeBaseTTFT("test-model", routerCfg)
	if ttft <= 0 {
		t.Errorf("Expected TTFT > 0, got %f", ttft)
	}
}

func TestInitializeModelTTFT(t *testing.T) {
	gpuConfig := config.GPUConfig{
		FLOPS: 1e12,
		HBM:   1e11,
	}
	calculator := NewCalculator(gpuConfig)

	// Minimal mock config with two categories and models
	routerCfg := &config.RouterConfig{
		Categories: []config.Category{
			{
				ModelScores: []config.ModelScore{
					{Model: "model-a", Score: 0.9},
					{Model: "model-b", Score: 0.8},
				},
			},
		},
		DefaultModel: "model-default",
	}

	modelTTFT := calculator.InitializeModelTTFT(routerCfg)
	if len(modelTTFT) != 3 {
		t.Errorf("Expected 3 models in TTFT map, got %d", len(modelTTFT))
	}
	for model, ttft := range modelTTFT {
		if ttft <= 0 {
			t.Errorf("Model %s has non-positive TTFT: %f", model, ttft)
		}
	}
}
