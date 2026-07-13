//go:build !windows && cgo && !onnx

package apiserver

func nativeEmbeddingBackendCapabilities() embeddingBackendCapabilities {
	return embeddingBackendCapabilities{
		name: "candle",
		models: map[string]embeddingModelCapability{
			"qwen3": {
				dimensions: []int{64, 128, 256, 512, 768, 1024},
			},
			"gemma": {
				dimensions: []int{128, 256, 512, 768},
			},
			"mmbert": {
				dimensions:          []int{64, 128, 256, 512, 768},
				supportsTargetLayer: true,
			},
		},
		// Auto can select Qwen3 or Gemma and can fall back to mmBERT. Only
		// their dimension intersection is safe before native model selection.
		autoModels:              []string{"qwen3", "gemma", "mmbert"},
		autoSupportsPriorities:  true,
		supportsMultimodalText:  true,
		supportsMultimodalImage: true,
		multimodalDimensions:    []int{64, 128, 256, 384},
	}
}
