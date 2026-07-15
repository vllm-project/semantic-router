package onnx

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
	return native.BackendONNX
}

func (a *Adapter) Capabilities() native.CapabilitySet {
	return native.CapabilitySet{
		Capabilities: []native.Capability{
			native.CapabilityEmbedding,
			native.CapabilitySequenceClassification,
			native.CapabilityTokenClassification,
		},
		SupportedFamilies: []native.Family{
			native.FamilymmBERT,
		},
		SupportedArtifacts: []native.ArtifactFormat{
			native.ArtifactONNXIR,
			native.ArtifactQuantized,
		},
		Features: map[string]bool{
			"matryoshka_2d":      true,
			"batching":           true,
			"provider_selection": true, // CPU, CUDA, ROCm
		},
	}
}

type onnxHandle struct {
	id string
}

func (h *onnxHandle) ID() string {
	return h.id
}

func (a *Adapter) LoadModel(ctx context.Context, req native.LoadRequest) (native.ModelHandle, error) {
	return nil, errors.New("onnx native adapter lifecycle is not yet wired (Phase 3)")
}

func (a *Adapter) UnloadModel(ctx context.Context, handle native.ModelHandle) error {
	return errors.New("onnx native adapter lifecycle is not yet wired (Phase 3)")
}

func (a *Adapter) Inference(ctx context.Context, handle native.ModelHandle, req native.InferenceRequest) (native.InferenceResponse, error) {
	return nil, errors.New("onnx native adapter inference is not yet wired (Phase 3)")
}

func (a *Adapter) Info() []native.ModelInfo {
	return nil
}

