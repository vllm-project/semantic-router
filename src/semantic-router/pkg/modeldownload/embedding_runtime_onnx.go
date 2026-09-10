//go:build onnx

package modeldownload

import (
	"fmt"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const compiledEmbeddingRuntime = "onnx"

var onnxMmBertPrimaryFiles = []string{
	"tokenizer.json",
	"onnx/layer-22/model.onnx",
	"onnx/layer-22/model.onnx.data",
}

var onnxMultimodalEmbeddingFiles = []string{
	"tokenizer.json",
	"onnx/text_encoder.onnx",
	"onnx/text_encoder.onnx.data",
	"onnx/image_encoder.onnx",
	"onnx/image_encoder.onnx.data",
	"onnx/audio_encoder.onnx",
}

func runtimeEmbeddingModelRequiredFiles(cfg *config.RouterConfig) map[string][]string {
	// Non-candle configuration selects a separate runtime path. Preserve its
	// existing full-snapshot completeness behavior.
	if cfg.EmbeddingModels.EmbeddingBackend() != config.EmbeddingBackendCandle {
		return candleEmbeddingModelRequiredFiles(cfg)
	}

	required := make(map[string][]string)
	required[cfg.MmBertModelPath] = append([]string(nil), onnxMmBertPrimaryFiles...)
	if targetLayer := cfg.EmbeddingModels.EmbeddingConfig.TargetLayer; targetLayer > 0 && targetLayer != 22 {
		for _, name := range []string{
			fmt.Sprintf("onnx/layer-%d/model.onnx", targetLayer),
			fmt.Sprintf("onnx/layer-%d/model.onnx.data", targetLayer),
		} {
			if !slices.Contains(required[cfg.MmBertModelPath], name) {
				required[cfg.MmBertModelPath] = append(required[cfg.MmBertModelPath], name)
			}
		}
	}
	required[cfg.MultiModalModelPath] = append([]string(nil), onnxMultimodalEmbeddingFiles...)
	return required
}

func runtimeEmbeddingModelExcludePatterns(_ *config.RouterConfig) map[string][]string {
	// ONNX Runtime consumes the exports that the Candle build deliberately
	// excludes. Keep the complete snapshot so optimized variants and optional
	// early-exit layers remain available at runtime.
	return nil
}
