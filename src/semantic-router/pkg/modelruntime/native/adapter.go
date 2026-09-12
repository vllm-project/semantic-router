package native

import "context"

// CapabilitySet defines the supported features of a specific backend.
type CapabilitySet struct {
	Capabilities       []Capability
	SupportedFamilies  []Family
	SupportedArtifacts []ArtifactFormat
	Features           map[string]bool // e.g., "matryoshka_2d", "batching", "provider_selection"
}

// ModelInfo describes a model currently loaded in a backend.
type ModelInfo struct {
	Backend             Backend
	Capabilities        []Capability
	Family              Family
	ModelName           string
	ModelPath           string
	IsLoaded            bool
	MaxSequenceLength   int
	DefaultDimension    int
	ArtifactFormat      ArtifactFormat
	Modality            Modality
	RequestedDimensions int
	RuntimeDimensions   int
	RequestedLayers     int
	RuntimeLayers       int
	Provider            string
	Device              string
	Version             string
	FeatureFlags        map[string]bool
	UnsupportedReasons  map[string]string
	RegistryMetadata    map[string]string
}

// BackendAdapter defines the neutral contract that all backends must implement.
type BackendAdapter interface {
	Name() Backend
	Capabilities() CapabilitySet
	LoadModel(ctx context.Context, req LoadRequest) (ModelHandle, error)
	UnloadModel(ctx context.Context, handle ModelHandle) error
	Inference(ctx context.Context, handle ModelHandle, req InferenceRequest) (InferenceResponse, error)
	Info() []ModelInfo
}

