package vectordb

type VectorDbConfig struct {
	Type             VectorDbBackendType
	Endpoint         string
	Collection       string
	EmbeddingService EmbeddingService
}

type VectorDbBackend interface {
	Query(queryText string) ([]string, error) // TODO: Might need to adjust query params and output
}

type VectorDbBackendType string

const (
	ChromaVectorDbType VectorDbBackendType = "chroma"
	MilvusVectorDbType VectorDbBackendType = "milvus"
)
