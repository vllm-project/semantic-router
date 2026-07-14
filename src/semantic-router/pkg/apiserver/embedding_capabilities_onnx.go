//go:build !windows && cgo && onnx

package apiserver

func nativeEmbeddingBackendCapabilities() embeddingBackendCapabilities {
	return embeddingBackendCapabilities{
		name: "onnx",
		models: map[string]embeddingModelCapability{
			"mmbert": {
				dimensions:          []int{64, 128, 256, 512, 768},
				supportsTargetLayer: true,
			},
		},
		// ONNX auto deterministically resolves to mmBERT. Priorities cannot
		// influence a single-model backend, while target_layer remains valid.
		autoModels:              []string{"mmbert"},
		autoSupportsLayer:       true,
		supportsMultimodalText:  true,
		supportsMultimodalImage: true,
		multimodalDimensions:    []int{64, 128, 256, 384},
	}
}
