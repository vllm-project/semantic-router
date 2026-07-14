package openvino

import (
	"context"

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
	// For Phase 3, this implements the contract but delegates down.
	return &openvinoHandle{id: req.ModelRef}, nil
}

func (a *Adapter) UnloadModel(ctx context.Context, handle native.ModelHandle) error {
	return nil
}

func (a *Adapter) Info() []native.ModelInfo {
	return nil
}

func init() {
	native.Registry.Register(NewAdapter())
}

