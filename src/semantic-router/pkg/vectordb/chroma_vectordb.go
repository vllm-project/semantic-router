package vectordb

import (
	"context"

	chroma "github.com/amikos-tech/chroma-go/pkg/api/v2"
	defaultef "github.com/amikos-tech/chroma-go/pkg/embeddings/default_ef"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
)

type ChromaVectorDbOptions struct {
	Endpoint   string // Chroma server endpoint
	Tenant     string // Default tenant to use
	Database   string // Default database to use
	Collection string // Collection
}

type ChromaVectorDb struct {
	client     chroma.Client
	collection string
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
	return &ChromaVectorDb{
		client:     c,
		collection: options.Collection,
	}, nil
}

func (c *ChromaVectorDb) Query(queryText string) ([]string, error) { // Assume single query text for now
	ef, closeef, err := defaultef.NewDefaultEmbeddingFunction()

	// make sure to call this to ensure proper resource release
	defer func() {
		err := closeef()
		if err != nil {
			observability.Warnf("Error closing default embedding function: %s \n", err)
		}
	}()
	if err != nil {
		observability.Errorf("Error creatings embedding function: %s \n", err)
		return nil, err
	}

	coll, err := c.client.GetCollection(context.Background(), c.collection, chroma.WithEmbeddingFunctionGet(ef))
	if err != nil {
		observability.Errorf("Error getting collection: %s \n", err)
		return nil, err
	}
	qr, err := coll.Query(context.Background(), chroma.WithQueryTexts(queryText))
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
