//go:build !windows && cgo && onnx

package apiserver

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

func TestONNXEmbeddingCapabilitiesExposeOnlyAutoAndMmBert(t *testing.T) {
	for _, model := range []string{"auto", "mmbert"} {
		if code, message, ok := validateSimilarityOptions(model, 768, 0, 0, 0, nil); !ok {
			t.Fatalf("model %q rejected: code=%q message=%q", model, code, message)
		}
	}
	for _, model := range []string{"qwen3", "gemma"} {
		code, message, ok := validateSimilarityOptions(model, 768, 0, 0, 0, nil)
		if ok || code != "INVALID_MODEL" || !strings.Contains(message, "onnx") {
			t.Fatalf("model %q validation = ok=%v code=%q message=%q", model, ok, code, message)
		}
	}
}

func TestONNXAutoRejectsIgnoredPrioritiesAndAcceptsMmBertLayer(t *testing.T) {
	if code, _, ok := validateSimilarityOptions("auto", 768, 0, 0.8, 0.2, nil); ok || code != "INVALID_PARAMETER" {
		t.Fatalf("ONNX auto priorities validation = ok=%v code=%q", ok, code)
	}
	if code, message, ok := validateSimilarityOptions("auto", 256, 16, 0, 0, []int{6, 11, 16, 22}); !ok {
		t.Fatalf("ONNX auto mmBERT layer rejected: code=%q message=%q", code, message)
	}
}

func TestONNXAutoEmbeddingHandlerAcceptsMmBertOutputContract(t *testing.T) {
	original := embeddingOutputForRequest
	embeddingOutputForRequest = func(req EmbeddingRequest, _ string) (*candle_binding.EmbeddingOutput, error) {
		if req.Model != "auto" || req.Dimension != defaultEmbeddingDimension {
			t.Fatalf("normalized request = %#v", req)
		}
		return &candle_binding.EmbeddingOutput{
			Embedding: make([]float32, req.Dimension),
			ModelType: "mmbert",
		}, nil
	}
	t.Cleanup(func() { embeddingOutputForRequest = original })

	server := &ClassificationAPIServer{embeddingAdmission: newEmbeddingProcessAdmission(1)}
	request := httptest.NewRequest(http.MethodPost, "/api/v1/embeddings", strings.NewReader(`{"texts":["hello"]}`))
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()

	server.handleEmbeddings(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	if !strings.Contains(response.Body.String(), `"model_used":"mmbert"`) ||
		!strings.Contains(response.Body.String(), `"dimension":768`) {
		t.Fatalf("response did not preserve ONNX mmBERT output contract: %s", response.Body.String())
	}
}

func TestONNXModelInfoUsesLoadedOnlyBackendCapabilityContract(t *testing.T) {
	capabilities := nativeEmbeddingBackendCapabilities()
	if capabilities.name != "onnx" {
		t.Fatalf("backend name = %q, want onnx", capabilities.name)
	}

	if _, ok := loadedEmbeddingModelInfo(candle_binding.ModelInfo{
		ModelName: "mmbert",
		IsLoaded:  false,
	}, capabilities); ok {
		t.Fatal("unloaded ONNX inventory placeholder was exposed as a model")
	}

	model, ok := loadedEmbeddingModelInfo(candle_binding.ModelInfo{
		ModelName:         "mmbert",
		IsLoaded:          true,
		MaxSequenceLength: 32768,
		DefaultDimension:  768,
		ModelPath:         "models/mmbert-embed-32k-2d-matryoshka",
	}, capabilities)
	if !ok {
		t.Fatal("loaded ONNX model was omitted")
	}
	if model.Metadata["backend"] != "onnx" ||
		model.Metadata["supported_dimensions"] != "64, 128, 256, 512, 768" ||
		model.Metadata["target_layer_supported"] != "true" {
		t.Fatalf("ONNX inventory metadata = %#v", model.Metadata)
	}
}
