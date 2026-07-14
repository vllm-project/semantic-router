package native

// LoadRequest encapsulates all configuration required to load a model.
type LoadRequest struct {
	ModelRef          string
	ResolvedPath      string
	Capability        Capability
	Family            Family
	ArtifactFormat    ArtifactFormat
	DevicePolicy      string
	ProviderPolicy    string
	BatchingHints     map[string]interface{}
	TargetDimensions  int
	TargetLayers      int
	LabelsMappings    map[string]string
}

// InferenceRequest encapsulates data sent for inference to any capability.
type InferenceRequest struct {
	Capability Capability
	Inputs     []string
	Parameters map[string]interface{}
}

// ModelHandle represents an opaque handle to a loaded model in a backend.
type ModelHandle interface {
	ID() string
}

