package openvino

import (
	"context"
	"errors"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
)

type Adapter struct{}

func NewAdapter() *Adapter {
	return &Adapter{}
}

func (a *Adapter) Name() native.Backend {
	return native.BackendOpenVINO
}

func (a *Adapter) Capabilities() native.CapabilitySet {
	return native.CapabilitySet{
		Capabilities: []native.Capability{
			native.CapabilityEmbedding,
			native.CapabilitySequenceClassification,
			native.CapabilityTokenClassification,
		},
		SupportedFamilies: []native.Family{
			native.FamilyModernBERT,
		},
		SupportedArtifacts: []native.ArtifactFormat{
			native.ArtifactOpenVINOIR,
		},
		Features: map[string]bool{
			"matryoshka_2d": false,
			"batching":      true,
		},
	}
}

type openvinoHandle struct {
	id string
}

func (h *openvinoHandle) ID() string {
	return h.id
}

func (a *Adapter) LoadModel(ctx context.Context, req native.LoadRequest) (native.ModelHandle, error) {
	return nil, errors.New("openvino native adapter lifecycle is not yet wired (Phase 3)")
}

func (a *Adapter) UnloadModel(ctx context.Context, handle native.ModelHandle) error {
	return errors.New("openvino native adapter lifecycle is not yet wired (Phase 3)")
}

func (a *Adapter) Inference(ctx context.Context, handle native.ModelHandle, req native.InferenceRequest) (native.InferenceResponse, error) {
	return nil, errors.New("openvino native adapter inference is not yet wired (Phase 3)")
}

func (a *Adapter) Info() ([]native.ModelInfo, error) {
	return nil, nil
}

