package vectordb

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
)

func NewVectorDbBackend(config VectorDbConfig) (VectorDbBackend, error) {
	if err := ValidateVectorDbConfig(config); err != nil {
		return nil, fmt.Errorf("invalid vector DB config: %w", err)
	}
	switch config.Type {
	case ChromaVectorDbType:
		observability.Debugf("Creating chroma backend - Endpoint: %s, Collection: %s", config.Endpoint, config.Collection)
		options := ChromaVectorDbOptions{
			Endpoint:         config.Endpoint,
			Collection:       config.Collection,
			EmbeddingService: config.EmbeddingService,
			EmbeddingModel:   config.EmbeddingModel,
		}
		vectorDb, err := NewChromaVectorDb(options)
		if err != nil {
			observability.Errorf("Error instantiating Chroma DB: %w", err)
			return nil, err
		}
		return vectorDb, nil
	case MilvusVectorDbType:
		observability.Debugf("Creating milvus backend - Endpoint: %s, Collection: %s", config.Endpoint, config.Collection)
		options := MilvusVectorDbOptions{
			Endpoint:   config.Endpoint,
			Collection: config.Collection,
		}
		vectorDb, err := NewMilvusVectorDb(options)
		if err != nil {
			observability.Errorf("Error instantiating Chroma DB: %w", err)
			return nil, err
		}
		return vectorDb, nil
	default:
		return nil, fmt.Errorf("vector DB type %s is not implemented", config.Type)
	}
}

func ValidateVectorDbConfig(config VectorDbConfig) error {
	switch config.Type {
	// Add backend-specific validations once implemented
	case ChromaVectorDbType, MilvusVectorDbType:
		if config.Endpoint == "" {
			return fmt.Errorf("endpoint not specified")
		}
	default:
		return fmt.Errorf("vector DB type not specified")
	}
	return nil
}
