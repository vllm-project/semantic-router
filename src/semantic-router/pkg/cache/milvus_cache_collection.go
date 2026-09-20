package cache

import (
	"context"
	"fmt"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	milvuslifecycle "github.com/vllm-project/semantic-router/src/semantic-router/pkg/milvus"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (c *MilvusCache) initializeCollection() error {
	ctx := context.Background()

	// Verify collection existence
	hasCollection, err := c.client.HasCollection(ctx, c.collectionName)
	if err != nil {
		return fmt.Errorf("failed to check collection existence: %w", err)
	}

	// Handle development mode collection reset
	if c.config.Development.DropCollectionOnStartup && hasCollection {
		if err := c.client.DropCollection(ctx, c.collectionName); err != nil {
			logging.Debugf("MilvusCache: failed to drop collection: %v", err)
			return fmt.Errorf("failed to drop collection: %w", err)
		}
		logging.Debugf("MilvusCache: dropped existing collection '%s' for development", c.collectionName)
		logging.LogEvent("collection_dropped", map[string]interface{}{
			"backend":    "milvus",
			"collection": c.collectionName,
			"reason":     "development_mode",
		})
		hasCollection = false
	}

	expectedDimension, err := c.embeddingDimension()
	if err != nil {
		return fmt.Errorf("failed to resolve semantic cache embedding dimension: %w", err)
	}
	if c.exactOnly && hasCollection {
		expectedDimension, err = milvuslifecycle.VectorDimension(
			ctx,
			c.client,
			c.collectionName,
			c.vectorFieldName(),
		)
		if err != nil {
			return fmt.Errorf("failed to inspect exact-only collection dimension: %w", err)
		}
		c.effectiveDimension = expectedDimension
		c.config.Collection.VectorField.Dimension = expectedDimension
	}
	if expectedDimension <= 0 && !c.exactOnly {
		return fmt.Errorf("invalid semantic cache embedding dimension: %d", expectedDimension)
	}
	vectorFieldName := c.vectorFieldName()

	if err := milvuslifecycle.EnsureCollectionLoadedWithHooksRetry(
		ctx,
		c.client,
		c.collectionName,
		func(innerCtx context.Context) error {
			logging.Debugf("MilvusCache: collection '%s' does not exist. AutoCreateCollection=%v",
				c.collectionName, c.config.Development.AutoCreateCollection)
			if !c.config.Development.AutoCreateCollection {
				return fmt.Errorf("collection %s does not exist and auto-creation is disabled", c.collectionName)
			}
			if err := c.createCollection(innerCtx); err != nil {
				return err
			}
			logging.Debugf("MilvusCache: created new collection '%s' with dimension %d",
				c.collectionName, c.config.Collection.VectorField.Dimension)
			logging.LogEvent("collection_created", map[string]interface{}{
				"backend":    "milvus",
				"collection": c.collectionName,
				"dimension":  c.config.Collection.VectorField.Dimension,
			})
			return nil
		},
		func(innerCtx context.Context) error {
			if expectedDimension <= 0 || c.exactOnly {
				return nil
			}
			return milvuslifecycle.ValidateVectorDimension(
				innerCtx,
				c.client,
				c.collectionName,
				vectorFieldName,
				expectedDimension,
			)
		},
		milvuslifecycle.CollectionRetryOptions{},
	); err != nil {
		logging.Debugf("MilvusCache: failed to ensure/load collection: %v", err)
		return fmt.Errorf("failed to ensure/load collection: %w", err)
	}

	return nil
}

func (c *MilvusCache) vectorFieldName() string {
	if c != nil && c.config != nil && c.config.Collection.VectorField.Name != "" {
		return c.config.Collection.VectorField.Name
	}
	return "embedding"
}

func (c *MilvusCache) embeddingDimension() (int, error) {
	if c == nil {
		return 0, fmt.Errorf("milvus cache is nil")
	}
	if c.effectiveDimension > 0 {
		return c.effectiveDimension, nil
	}
	if c.config == nil {
		return 0, fmt.Errorf("milvus cache config is not initialized")
	}
	return resolveCacheBackendDimension(
		c.embeddingProvider,
		c.config.Collection.VectorField.Dimension,
		c.embeddingModel,
		c.exactOnly,
	)
}

// createCollection builds the Milvus collection with the appropriate schema
func (c *MilvusCache) createCollection(ctx context.Context) error {
	actualDimension, err := c.embeddingDimension()
	if err != nil {
		return fmt.Errorf("failed to resolve semantic cache embedding dimension: %w", err)
	}
	if actualDimension <= 0 && c.exactOnly {
		actualDimension = exactCacheSentinelDimension
		c.effectiveDimension = actualDimension
	}
	if actualDimension <= 0 {
		return fmt.Errorf("invalid semantic cache embedding dimension: %d", actualDimension)
	}
	c.config.Collection.VectorField.Dimension = actualDimension

	logging.Debugf("MilvusCache.createCollection: using embedding dimension: %d", actualDimension)

	// Define schema with auto-detected dimension
	schema := &entity.Schema{
		CollectionName: c.collectionName,
		Description:    c.config.Collection.Description,
		Fields: []*entity.Field{
			{
				Name:       "id",
				DataType:   entity.FieldTypeVarChar,
				PrimaryKey: true,
				TypeParams: map[string]string{"max_length": "64"},
			},
			{
				Name:       "request_id",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "64"},
			},
			{
				Name:       "model",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "256"},
			},
			{
				Name:       "query",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "65535"},
			},
			{
				Name:       "request_body",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "65535"},
			},
			{
				Name:       "response_body",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "65535"},
			},
			{
				Name:     c.vectorFieldName(),
				DataType: entity.FieldTypeFloatVector,
				TypeParams: map[string]string{
					"dim": fmt.Sprintf("%d", actualDimension),
				},
			},
			{
				Name:     "timestamp",
				DataType: entity.FieldTypeInt64,
			},
			{
				Name:     "ttl_seconds",
				DataType: entity.FieldTypeInt64,
			},
			{
				Name:     "expires_at",
				DataType: entity.FieldTypeInt64,
			},
		},
	}

	// Create collection at the configured consistency level (SDK default when unset)
	if createErr := c.client.CreateCollection(ctx, schema, 1, c.createCollectionOptions()...); createErr != nil {
		return createErr
	}

	// Create index with updated API
	index, err := entity.NewIndexHNSW(entity.MetricType(c.config.Collection.VectorField.MetricType), c.config.Collection.Index.Params.M, c.config.Collection.Index.Params.EfConstruction)
	if err != nil {
		return fmt.Errorf("failed to create HNSW index: %w", err)
	}
	if err := c.client.CreateIndex(ctx, c.collectionName, c.vectorFieldName(), index, false); err != nil {
		return err
	}

	return nil
}
