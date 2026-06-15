package memory

import (
	"context"
	"fmt"
	"strings"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	milvuslifecycle "github.com/vllm-project/semantic-router/src/semantic-router/pkg/milvus"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (m *MilvusStore) ensureCollection(ctx context.Context) error {
	logging.Infof("MilvusStore: ensuring collection '%s' (dimension=%d)", m.collectionName, m.config.Milvus.Dimension)

	err := milvuslifecycle.EnsureCollectionLoadedWithRetry(ctx, m.client, m.collectionName, func(innerCtx context.Context) error {
		return m.createMemoryCollection(innerCtx)
	}, milvuslifecycle.CollectionRetryOptions{})
	if err != nil {
		return err
	}

	logging.Infof("MilvusStore: collection '%s' ensured and loaded successfully", m.collectionName)
	return nil
}

func (m *MilvusStore) createMemoryCollection(ctx context.Context) error {
	schema := memoryCollectionSchema(m.collectionName, m.config.Milvus.Dimension)

	numPartitions := int64(16)
	if m.config.Milvus.NumPartitions > 0 {
		numPartitions = int64(m.config.Milvus.NumPartitions)
	}
	logging.Infof("MilvusStore: creating collection with %d partitions (partition key: user_id)", numPartitions)

	if err := m.client.CreateCollection(ctx, schema, 1, client.WithPartitionNum(numPartitions)); err != nil {
		return err
	}

	index, err := entity.NewIndexHNSW(entity.COSINE, 16, 256)
	if err != nil {
		return fmt.Errorf("failed to create HNSW index: %w", err)
	}
	if err := m.client.CreateIndex(ctx, m.collectionName, "embedding", index, false); err != nil {
		return fmt.Errorf("failed to create index: %w", err)
	}
	return nil
}

func buildTypeFilter(types []MemoryType) string {
	if len(types) == 0 {
		return ""
	}
	parts := make([]string, len(types))
	for i, t := range types {
		parts[i] = fmt.Sprintf("memory_type == %q", string(t))
	}
	return "(" + strings.Join(parts, " || ") + ")"
}
