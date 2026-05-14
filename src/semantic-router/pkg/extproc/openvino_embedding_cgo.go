//go:build openvino && !windows && cgo

package extproc

import (
	openvino_binding "github.com/vllm-project/semantic-router/openvino-binding"
)

func openvinoEmbeddingFunc(modelType string) func(string) ([]float32, error) {
	return func(text string) ([]float32, error) {
		switch modelType {
		case "mmbert", "modernbert":
			return openvino_binding.GetModernBertEmbedding(text, 32768)
		default:
			return openvino_binding.GetEmbedding(text, 32768)
		}
	}
}
