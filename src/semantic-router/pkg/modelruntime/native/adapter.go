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

// BackendIdentifier provides discovery and capability metadata.
type BackendIdentifier interface {
	Name() Backend
	Capabilities() CapabilitySet
}

// BackendLifecycle manages model loading and unloading.
type BackendLifecycle interface {
	LoadModel(ctx context.Context, req LoadRequest) (ModelHandle, error)
	UnloadModel(ctx context.Context, handle ModelHandle) error
}

// BackendInference executes inference on a loaded model.
type BackendInference interface {
	Inference(ctx context.Context, handle ModelHandle, req InferenceRequest) (InferenceResponse, error)
}

// BackendAdapter defines the neutral contract that all backends must implement.
type BackendAdapter interface {
	BackendIdentifier
	BackendLifecycle
	BackendInference
}

// BackendDiscoverer is an optional interface for adapters that support model discovery.
type BackendDiscoverer interface {
	Info() ([]ModelInfo, error)
}
