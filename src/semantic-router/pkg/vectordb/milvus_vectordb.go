package vectordb

import (
	"context"
	"fmt"
    "github.com/milvus-io/milvus-sdk-go/v2/entity"
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

// Query stub
func (m *MilvusVectorDb) Query(queryText string) ([]string, error) {
	fmt.Printf("[MilvusVectorDb] Query called with: %s\n", queryText)
	return []string{"milvus-stub"}, nil
}


func (m *MilvusVectorDb) CreateOrLoadCollection(ctx context.Context, collectionName string, dim int) error {
    has, err := m.client.HasCollection(ctx, collectionName)
    if err != nil {
        return err
    }

    if !has {
        // Define fields
        schema := &entity.Schema{
            CollectionName: collectionName,
            Description:    "Semantic Router embeddings",
            AutoID:         false,
            Fields: []*entity.Field{
                {
                    Name:       "id",
                    DataType:   entity.FieldTypeInt64,
                    PrimaryKey: true,
                    AutoID:     false,
                },
                {
                    Name:     "vector",
                    DataType: entity.FieldTypeFloatVector,
                    TypeParams: map[string]string{
                        "dim": fmt.Sprintf("%d", dim),
                    },
                },
                {
                    Name:     "content",
                    DataType: entity.FieldTypeVarChar,
                    TypeParams: map[string]string{
                        "max_length": "1024",
                    },
                },
            },
        }

        if err := m.client.CreateCollection(ctx, schema,1); err != nil {
            observability.Errorf("Error creating Milvus collection: %v", err)
            return err
        }

        observability.Infof("Created collection %s", collectionName)
    } else {
        observability.Infof("Collection %s already exists", collectionName)
    }
    return nil
}

func (m *MilvusVectorDb) ListCollections(ctx context.Context) ([]string, error) {
    cols, err := m.client.ListCollections(ctx)
    if err != nil {
        return nil, err
    }
    names := []string{}
    for _, col := range cols {
        names = append(names, col.Name)
    }
    return names, nil
}
