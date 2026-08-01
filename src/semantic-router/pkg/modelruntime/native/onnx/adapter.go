package onnx

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
)

type Adapter struct{}

func NewAdapter() *Adapter {
	return &Adapter{}
}

func (a *Adapter) Name() native.Backend {
	return native.BackendONNX
}

func (a *Adapter) Capabilities() native.CapabilitySet {
	return native.CapabilitySet{
		Capabilities: []native.Capability{
			native.CapabilityEmbedding,
			native.CapabilitySequenceClassification,
		},
		SupportedFamilies: []native.Family{
			native.FamilyModernBERT,
			native.FamilyLlama,
			native.FamilyQwen,
			native.FamilyGemma,
		},
		SupportedArtifacts: []native.ArtifactFormat{
			native.ArtifactFormatONNX,
		},
		Features: map[string]bool{
			"matryoshka_2d": true,
		},
	}
}

func (a *Adapter) LoadModel(ctx context.Context, req native.LoadRequest) (native.ModelHandle, error) {
	if _, ok := req.Parameters["lora"]; ok {
		return nil, fmt.Errorf("ONNX adapter does not support LoRA")
	}

	switch req.Capability {
	case native.CapabilityEmbedding:
		return &onnxEmbeddingHandler{modelID: req.ModelID}, nil
	case native.CapabilitySequenceClassification:
		return &onnxClassificationHandler{modelID: req.ModelID}, nil
	default:
		return nil, fmt.Errorf("unsupported capability: %s", req.Capability)
	}
}

type onnxEmbeddingHandler struct {
	modelID string
}

func (h *onnxEmbeddingHandler) Info() native.ModelInfo {
	return native.ModelInfo{
		Backend:      native.BackendONNX,
		Capabilities: []native.Capability{native.CapabilityEmbedding},
		ModelID:      h.modelID,
		IsLoaded:     true,
	}
}

func (h *onnxEmbeddingHandler) Infer(ctx context.Context, req native.InferenceRequest) (native.InferenceResponse, error) {
	return native.EmbeddingResponse{}, nil
}

func (h *onnxEmbeddingHandler) Unload(ctx context.Context) error {
	return nil
}

type onnxClassificationHandler struct {
	modelID string
}

func (h *onnxClassificationHandler) Info() native.ModelInfo {
	return native.ModelInfo{
		Backend:      native.BackendONNX,
		Capabilities: []native.Capability{native.CapabilitySequenceClassification},
		ModelID:      h.modelID,
		IsLoaded:     true,
	}
}

func (h *onnxClassificationHandler) Infer(ctx context.Context, req native.InferenceRequest) (native.InferenceResponse, error) {
	return native.SequenceClassificationResponse{}, nil
}

func (h *onnxClassificationHandler) Unload(ctx context.Context) error {
	return nil
}

func init() {
	native.Registry.Register(NewAdapter())
}