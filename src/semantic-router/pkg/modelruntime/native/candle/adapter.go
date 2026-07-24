package candle

import (
	"context"
	"errors"
	"fmt"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
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
			native.CapabilityTokenClassification,
			native.CapabilityMultimodalEmbedding,
			native.CapabilityModalityRouting,
			native.CapabilityReranking,
			native.CapabilityGuard,
			native.CapabilityMLPSelector,
		},
		SupportedFamilies: []native.Family{
			native.FamilyBERT,
			native.FamilyModernBERT,
			native.FamilymmBERT,
			native.FamilyQwen3,
			native.FamilyGemma,
			native.FamilyQwen3Guard,
		},
		SupportedArtifacts: []native.ArtifactFormat{
			native.ArtifactFullModel,
			native.ArtifactMergedLoRA,
		},
		Features: map[string]bool{
			"matryoshka_2d": false, // Handled differently in candle
			"batching":      true,
		},
	}
}

type candleHandle struct {
	id string
}

func (h *candleHandle) ID() string {
	return h.id
}

func (a *Adapter) LoadModel(ctx context.Context, req native.LoadRequest) (native.ModelHandle, error) {
	// For Phase 2, this adapter is intentionally unregistered and returns an explicit error
	// to prevent call sites from believing the capability is operational.
	return nil, errors.New("candle native adapter lifecycle is not yet wired (Phase 2)")
}

func (a *Adapter) UnloadModel(ctx context.Context, handle native.ModelHandle) error {
	return errors.New("candle native adapter lifecycle is not yet wired (Phase 2)")
}

func (a *Adapter) Inference(ctx context.Context, handle native.ModelHandle, req native.InferenceRequest) (native.InferenceResponse, error) {
	return nil, errors.New("candle native adapter inference is not yet wired (Phase 2)")
}

func (a *Adapter) Info() []native.ModelInfo {
	// In the future this maps candle_binding.GetEmbeddingModelsInfo() etc.
	// to the unified schema.
	info, err := candle_binding.GetEmbeddingModelsInfo()
	if err != nil {
		return nil
	}

	var results []native.ModelInfo
	for _, m := range info.Models {
		results = append(results, native.ModelInfo{
			Backend:           native.BackendCandle,
			Capability:        native.CapabilityEmbedding,
			ModelName:         m.ModelName,
			ModelPath:         m.ModelPath,
			IsLoaded:          m.IsLoaded,
			MaxSequenceLength: m.MaxSequenceLength,
			DefaultDimension:  m.DefaultDimension,
		})
	}
	return results
}
