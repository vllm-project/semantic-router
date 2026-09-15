package extproc

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	milvuslifecycle "github.com/vllm-project/semantic-router/src/semantic-router/pkg/milvus"
)

type memoryStartupClient struct {
	client.Client
	dimension   int
	describeErr error
	loaded      bool
	closed      bool
}

func (c *memoryStartupClient) HasCollection(context.Context, string) (bool, error) {
	return true, nil
}

func (c *memoryStartupClient) DescribeCollection(context.Context, string) (*entity.Collection, error) {
	return &entity.Collection{Schema: &entity.Schema{Fields: []*entity.Field{{
		Name: "embedding", DataType: entity.FieldTypeFloatVector,
		TypeParams: map[string]string{"dim": strconv.Itoa(c.dimension)},
	}}}}, c.describeErr
}

func (c *memoryStartupClient) LoadCollection(context.Context, string, bool, ...client.LoadCollectionOption) error {
	c.loaded = true
	return nil
}

func (c *memoryStartupClient) Close() error {
	c.closed = true
	return nil
}

func TestMemoryRuntimeStartupDimensionPolicy(t *testing.T) {
	// Avoid native model loading while exercising the real collection validation
	// and NewMilvusStore error wrapping used by router initialization.
	t.Setenv("VLLM_SR_DETERMINISTIC_EMBEDDINGS", "1")
	for _, tc := range []struct {
		name            string
		storedDimension int
		connectErr      error
		describeErr     error
		disabled        bool
		wantFatal       bool
		wantStore       bool
	}{
		{name: "incompatible collection fails startup", storedDimension: 384, wantFatal: true},
		{name: "matching collection enables memory", storedDimension: 256, wantStore: true},
		{name: "connection failure remains fail open", connectErr: errors.New("connection refused")},
		{name: "schema lookup failure remains fail open", describeErr: errors.New("schema lookup failed")},
		{name: "error text alone is not incompatibility", connectErr: errors.New("vector dimension mismatch")},
		{name: "disabled memory skips initialization", disabled: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &config.RouterConfig{}
			cfg.Memory = memory.DefaultMemoryConfig()
			cfg.Memory.Enabled = !tc.disabled
			cfg.Memory.Milvus.Dimension = 256
			components := &routerComponents{cfg: cfg, resources: newResourceScope()}
			rolledBack := false
			components.resources.add(func() error { rolledBack = true; return nil })
			t.Cleanup(func() { require.NoError(t, components.resources.close()) })
			milvusClient := &memoryStartupClient{dimension: tc.storedDimension, describeErr: tc.describeErr}
			factoryCalled := false
			err := components.buildMemoryRuntime(func(cfg *config.RouterConfig) (memory.Store, error) {
				factoryCalled = true
				if tc.connectErr != nil {
					return nil, fmt.Errorf("failed to create Milvus client: %w", tc.connectErr)
				}
				store, err := memory.NewMilvusStore(memory.MilvusStoreOptions{
					Client: milvusClient, CollectionName: "existing_memory", Config: cfg.Memory, Enabled: true,
					EmbeddingConfig: &memory.EmbeddingConfig{Model: memory.EmbeddingModelMMBERT, Dimension: 256},
				})
				if err != nil {
					_ = milvusClient.Close()
					return nil, fmt.Errorf("failed to create memory store: %w", err)
				}
				return store, nil
			})
			require.Equal(t, !tc.disabled, factoryCalled)
			require.Equal(t, tc.wantFatal, rolledBack)
			require.Equal(t, tc.wantStore, milvusClient.loaded)
			if tc.wantFatal {
				var mismatch *milvuslifecycle.VectorDimensionMismatchError
				require.ErrorAs(t, err, &mismatch)
				require.Equal(t, "existing_memory", mismatch.CollectionName)
				require.Equal(t, 384, mismatch.StoredDimension)
				require.Equal(t, 256, mismatch.ExpectedDimension)
				require.True(t, milvusClient.closed)
			} else {
				require.NoError(t, err)
			}
			if tc.wantStore {
				require.NotNil(t, components.memoryStore)
				require.NotNil(t, components.memoryExtractor)
				require.NoError(t, components.resources.close())
				require.True(t, rolledBack)
				// MilvusStore.Close leaves a potentially shared client to its owner.
				require.False(t, milvusClient.closed)
				require.NoError(t, milvusClient.Close())
			} else {
				require.Nil(t, components.memoryStore)
				require.Nil(t, components.memoryExtractor)
			}
		})
	}
}
