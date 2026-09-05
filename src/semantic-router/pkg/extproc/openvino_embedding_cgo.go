//go:build openvino && !windows && cgo

package extproc

import (
	openvino_binding "github.com/vllm-project/semantic-router/openvino-binding"
)

func openvinoEmbeddingFunc(modelType string) func(string) ([]float32, error) {
	usesModernBERT := openvinoEmbeddingUsesModernBERT(modelType)
	return func(text string) ([]float32, error) {
		if usesModernBERT {
			return openvino_binding.GetModernBertEmbedding(text, 32768)
		}
		return openvino_binding.GetEmbedding(text, 32768)
	}
}
