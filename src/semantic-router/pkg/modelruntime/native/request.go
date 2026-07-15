package native

// LoadRequest encapsulates all configuration required to load a model.
type LoadRequest struct {
	ModelRef         string
	ResolvedPath     string
	Capability       Capability
	Family           Family
	ArtifactFormat   ArtifactFormat
	DevicePolicy     string
	ProviderPolicy   string
	BatchingHints    map[string]interface{}
	TargetDimensions int
	TargetLayers     int
	LabelsMappings   map[string]string
}

// InferenceRequest encapsulates data sent for inference to any capability.
type InferenceRequest interface {
	Capability() Capability
}

// InferenceResponse encapsulates data returned from an inference operation.
type InferenceResponse interface {
	Capability() Capability
}

// EmbeddingRequest is the typed request for text embeddings.
type EmbeddingRequest struct {
	Inputs []string
}

func (r EmbeddingRequest) Capability() Capability { return CapabilityEmbedding }

// EmbeddingResponse is the typed response for text embeddings.
type EmbeddingResponse struct {
	Embeddings [][]float32
}

func (r EmbeddingResponse) Capability() Capability { return CapabilityEmbedding }

// SequenceClassificationRequest is the typed request for sequence classification.
type SequenceClassificationRequest struct {
	Inputs []string
}

func (r SequenceClassificationRequest) Capability() Capability { return CapabilitySequenceClassification }

// SequenceClassificationResponse is the typed response for sequence classification.
type SequenceClassificationResponse struct {
	Predictions [][]float32
}

func (r SequenceClassificationResponse) Capability() Capability { return CapabilitySequenceClassification }

// ModelHandle represents an opaque handle to a loaded model in a backend.
type ModelHandle interface {
	ID() string
}
