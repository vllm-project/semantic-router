//go:build !onnx

package modeldownload

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

const compiledEmbeddingRuntime = "candle"

func runtimeEmbeddingModelRequiredFiles(cfg *config.RouterConfig) map[string][]string {
	return candleEmbeddingModelRequiredFiles(cfg)
}

func runtimeEmbeddingModelExcludePatterns(cfg *config.RouterConfig) map[string][]string {
	return candleEmbeddingModelExcludePatterns(cfg)
}
