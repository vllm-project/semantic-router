package vectordb

import (
	"context"
	"fmt"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
)

// MilvusVectorDbOptions defines config parameters for Milvus connection
type MilvusVectorDbOptions struct {
	Endpoint   string // e.g. "127.0.0.1:19530"
	Collection string // e.g. "test_collection"
}

// MilvusVectorDb manages a Milvus client instance
type MilvusVectorDb struct {
	client     client.Client
	collection string
}

// NewMilvusVectorDb initializes a connection to Milvus
func NewMilvusVectorDb(options MilvusVectorDbOptions) (*MilvusVectorDb, error) {
	ctx := context.Background()

	cli, err := client.NewGrpcClient(ctx, options.Endpoint)
	if err != nil {
		observability.Errorf("Milvus connect error: %v", err)
		return nil, err
	}

	observability.Debugf("Connected to Milvus at %s", options.Endpoint)
	return &MilvusVectorDb{
		client:     cli,
		collection: options.Collection,
	}, nil
}

// Query is a stub for now — Phase 2 will implement it
func (m *MilvusVectorDb) Query(queryText string) ([]string, error) {
	fmt.Printf("[MilvusVectorDb] Query called with: %s\n", queryText)
	return []string{"milvus-stub"}, nil
}
