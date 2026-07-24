package native

import "context"

type CapabilitySet struct {
	Capabilities       []Capability
	SupportedFamilies  []Family
	SupportedArtifacts []ArtifactFormat
	Features           map[string]bool // e.g., "matryoshka_2d", "batching"
}

type BackendAdapter interface {
	Name() Backend
	Capabilities() CapabilitySet
	LoadModel(ctx context.Context, req LoadRequest) (ModelHandle, error)
}

type ModelInfo struct {
	Backend      Backend
	Capabilities []Capability
	ModelID      string
	IsLoaded     bool
}

type ModelHandle interface {
	Info() ModelInfo
	Infer(ctx context.Context, req InferenceRequest) (InferenceResponse, error)
	Unload(ctx context.Context) error
}
