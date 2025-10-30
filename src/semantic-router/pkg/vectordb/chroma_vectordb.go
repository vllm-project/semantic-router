package vectordb

import (
	"context"

	chroma "github.com/amikos-tech/chroma-go/pkg/api/v2"
	chroma_embedding "github.com/amikos-tech/chroma-go/pkg/embeddings"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
)

type ChromaVectorDbOptions struct {
	Endpoint          string // Chroma server endpoint
	Tenant            string // Default tenant to use
	Database          string // Default database to use
	Collection        string // Collection
	EmbeddingEndpoint string // EmbeddingEndpoint
	EmbeddingModel    string // EmbeddingModel
}

type ChromaVectorDb struct {
	client           chroma.Client
	collection       string
	embeddingService *OpenAIEmbeddingService
	embeddingModel   string
}

func NewChromaVectorDb(options ChromaVectorDbOptions) (*ChromaVectorDb, error) {
	clientOptions := []chroma.ClientOption{
		chroma.WithBaseURL(options.Endpoint),
	}
	if options.Database != "" && options.Tenant != "" {
		clientOptions = append(clientOptions, chroma.WithDatabaseAndTenant(options.Database, options.Tenant))
	} else if options.Tenant != "" {
		clientOptions = append(clientOptions, chroma.WithTenant(options.Tenant))
	}
	c, err := chroma.NewHTTPClient(
		clientOptions...,
	)
	if err != nil {
		observability.Errorf("Error creating client: %s \n", err)
		return nil, err
	}
	v, err := c.GetVersion(context.Background())
	if err != nil {
		observability.Errorf("Error getting chroma client version: %s \n", err)
		return nil, err
	}
	observability.Debugf("Initialized Chroma client with API Version: %s \n", v)
	es := NewOpenAIEmbeddingService(NewOpenAIEmbeddingServiceOptions{Endpoint: "http://localhost:11434/v1/"})
	return &ChromaVectorDb{
		client:           c,
		collection:       options.Collection,
		embeddingService: es,
		embeddingModel:   options.EmbeddingModel,
	}, nil
}

func (c *ChromaVectorDb) Query(queryText string) ([]string, error) {
	// Use a dummy embedding function here. The Chroma client demands it even though we
	// would actually not use it. Instead, we fetch and pass embeddings ourselves.
	embeddingFunction := chroma_embedding.NewConsistentHashEmbeddingFunction()
	coll, err := c.client.GetCollection(context.Background(), c.collection, chroma.WithEmbeddingFunctionGet(embeddingFunction))
	if err != nil {
		observability.Errorf("Error getting collection: %s \n", err)
		return nil, err
	}
	embeddingResult, err := c.embeddingService.Embed(queryText, c.embeddingModel)
	if err != nil {
		observability.Errorf("Error getting embedding: %s \n", err)
		return nil, err
	}
	embeddings := chroma_embedding.NewEmbeddingFromFloat64(embeddingResult)
	qr, err := coll.Query(context.Background(), chroma.WithQueryEmbeddings(embeddings))
	if err != nil {
		observability.Errorf("Error querying collection: %s \n", err)
		return nil, err
	}
	results := make([]string, 0, len(qr.GetDocumentsGroups()[0]))
	for _, document := range qr.GetDocumentsGroups()[0] {
		results = append(results, document.ContentString())
	}
	return results, nil
}
