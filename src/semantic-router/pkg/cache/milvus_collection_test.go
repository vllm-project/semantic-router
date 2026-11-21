package cache

import (
	"context"
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type existingMilvusCacheCollection struct {
	client.Client
	dimension int
	loads     int
}

func (*existingMilvusCacheCollection) HasCollection(context.Context, string) (bool, error) {
	return true, nil
}

func (c *existingMilvusCacheCollection) DescribeCollection(context.Context, string) (*entity.Collection, error) {
	field := entity.NewField().WithName("cache_vector").WithDataType(entity.FieldTypeFloatVector).WithDim(int64(c.dimension))
	return &entity.Collection{Schema: entity.NewSchema().WithField(field)}, nil
}

func (c *existingMilvusCacheCollection) LoadCollection(context.Context, string, bool, ...client.LoadCollectionOption) error {
	c.loads++
	return nil
}

func TestMilvusCacheExistingCollectionDimension(t *testing.T) {
	for _, tc := range []struct {
		name     string
		stored   int
		mismatch bool
	}{
		{name: "matches", stored: 384},
		{name: "mismatch", stored: 768, mismatch: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			milvus := &existingMilvusCacheCollection{dimension: tc.stored}
			cfg := &config.MilvusConfig{}
			cfg.Collection.VectorField.Name = "cache_vector"
			cfg.Collection.VectorField.Dimension = 384
			cache := &MilvusCache{client: milvus, config: cfg, collectionName: "responses"}

			err := cache.initializeCollection()
			if tc.mismatch {
				assert.ErrorContains(t, err, `collection "responses" vector field "cache_vector" has dimension 768, expected 384`)
				assert.Zero(t, milvus.loads)
			} else {
				require.NoError(t, err)
				assert.Equal(t, 1, milvus.loads)
			}
		})
	}
}
