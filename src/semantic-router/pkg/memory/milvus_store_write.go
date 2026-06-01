package memory

import (
	"context"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (m *MilvusStore) Store(ctx context.Context, memory *Memory) error {
	startTime := time.Now()
	backend := "milvus"
	operation := "store"
	status := "success"

	defer func() {
		duration := time.Since(startTime).Seconds()
		RecordMemoryStoreOperation(backend, operation, status, duration)
	}()

	if !m.enabled {
		status = "error"
		return fmt.Errorf("milvus store is not enabled")
	}

	if err := prepareMemoryForStore(memory); err != nil {
		status = "error"
		return err
	}

	logging.Debugf("MilvusStore.Store: id=%s, user=%s, type=%s, content_len=%d",
		memory.ID, memory.UserID, memory.Type, len(memory.Content))

	embedding, err := memoryEmbedding(memory, m.embeddingConfig)
	if err != nil {
		status = "error"
		return err
	}

	metadataJSON, err := memoryMetadataJSON(memory)
	if err != nil {
		status = "error"
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	cols := newMemoryRowColumns(memory, embedding, string(metadataJSON))
	err = m.retryWithBackoff(ctx, func() error {
		_, insertErr := m.client.Insert(ctx, m.collectionName, "", cols.insertArgs()...)
		return insertErr
	})
	if err != nil {
		status = "error"
		return fmt.Errorf("milvus insert failed: %w", err)
	}

	logging.Debugf("MilvusStore.Store: successfully stored memory id=%s", memory.ID)
	return nil
}

func (m *MilvusStore) upsert(ctx context.Context, memory *Memory) error {
	metadataJSON, err := memoryMetadataJSON(memory)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	if len(memory.Embedding) == 0 {
		return fmt.Errorf("embedding is required for upsert")
	}

	cols := newMemoryRowColumns(memory, memory.Embedding, string(metadataJSON))
	err = m.retryWithBackoff(ctx, func() error {
		_, upsertErr := m.client.Upsert(ctx, m.collectionName, "", cols.insertArgs()...)
		return upsertErr
	})
	if err != nil {
		return fmt.Errorf("milvus upsert failed: %w", err)
	}

	logging.Debugf("MilvusStore.upsert: successfully upserted memory id=%s", memory.ID)
	return nil
}

func memoryEmbedding(memory *Memory, cfg EmbeddingConfig) ([]float32, error) {
	if len(memory.Embedding) > 0 {
		return memory.Embedding, nil
	}
	embedding, err := GenerateEmbedding(memory.Content, cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}
	return embedding, nil
}
