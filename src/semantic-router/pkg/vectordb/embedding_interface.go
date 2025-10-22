package vectordb

type EmbeddingConfig struct {
	Endpoint *string
	// Add credentials?
}

type EmbeddingService interface {
	Embed(input string, model string) ([]float64, error) // TODO: Might need to adjust query params and output
}
