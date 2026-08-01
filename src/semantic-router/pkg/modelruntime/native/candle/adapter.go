package candle

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
	return native.BackendCandle
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
			native.ArtifactFormatSafetensors,
		},
		Features: map[string]bool{
			"matryoshka_2d": true,
			"batching":      true,
		},
	}
}

func (a *Adapter) LoadModel(ctx context.Context, req native.LoadRequest) (native.ModelHandle, error) {
	switch req.Capability {
	case native.CapabilityEmbedding:
		return &embeddingHandler{modelID: req.ModelID}, nil
	case native.CapabilitySequenceClassification:
		return &classificationHandler{modelID: req.ModelID}, nil
	default:
		return nil, fmt.Errorf("unsupported capability: %s", req.Capability)
	}
}

// embeddingHandler encapsulates embedding-specific CGO calls
type embeddingHandler struct {
	modelID string
}

func (h *embeddingHandler) Info() native.ModelInfo {
	return native.ModelInfo{
		Backend:      native.BackendCandle,
		Capabilities: []native.Capability{native.CapabilityEmbedding},
		ModelID:      h.modelID,
		IsLoaded:     true,
	}
}

func (h *embeddingHandler) Infer(ctx context.Context, req native.InferenceRequest) (native.InferenceResponse, error) {
	return native.EmbeddingResponse{}, nil
}

func (h *embeddingHandler) Unload(ctx context.Context) error {
	return nil
}

// classificationHandler encapsulates classification-specific CGO calls
type classificationHandler struct {
	modelID string
}

func (h *classificationHandler) Info() native.ModelInfo {
	return native.ModelInfo{
		Backend:      native.BackendCandle,
		Capabilities: []native.Capability{native.CapabilitySequenceClassification},
		ModelID:      h.modelID,
		IsLoaded:     true,
	}
}

func (h *classificationHandler) Infer(ctx context.Context, req native.InferenceRequest) (native.InferenceResponse, error) {
	return native.SequenceClassificationResponse{}, nil
}

func (h *classificationHandler) Unload(ctx context.Context) error {
	return nil
}

func init() {
	native.Registry.Register(NewAdapter())
}