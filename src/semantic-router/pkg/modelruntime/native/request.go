package native

type LoadRequest struct {
	ModelID    string
	ModelPath  string
	Format     ArtifactFormat
	Family     Family
	Modality   Modality
	Capability Capability
	Parameters map[string]interface{}
}

type InferenceRequest interface {
	isInferenceRequest()
}

type InferenceResponse interface {
	isInferenceResponse()
}

type EmbeddingRequest struct {
	Inputs []string
}

func (EmbeddingRequest) isInferenceRequest() {}

type EmbeddingResponse struct {
	Embeddings [][]float32
}

func (EmbeddingResponse) isInferenceResponse() {}

type SequenceClassificationRequest struct {
	Inputs []string
}

func (SequenceClassificationRequest) isInferenceRequest() {}

type SequenceClassificationResponse struct {
	Scores [][]float32
	Labels [][]string
}

func (SequenceClassificationResponse) isInferenceResponse() {}
