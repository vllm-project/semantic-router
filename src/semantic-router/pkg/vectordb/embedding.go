package vectordb

type EmbeddingConfig struct {
	Provider *string // Can  be nil for local embeddings
	Model    string
	// Add credentials?
}

type EmbeddingService interface {
}
