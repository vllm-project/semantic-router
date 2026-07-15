package memory

import (
	"context"
	"fmt"
	"sort"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (m *MilvusStore) Get(ctx context.Context, id string) (*Memory, error) {
	if !m.enabled {
		return nil, fmt.Errorf("milvus store is not enabled")
	}

	if id == "" {
		return nil, fmt.Errorf("memory ID is required")
	}

	logging.Debugf("MilvusStore.Get: retrieving memory id=%s", id)

	// Query by ID (includes embedding so the caller can Upsert without re-generating it)
	filterExpr := fmt.Sprintf("id == \"%s\"", id)
	outputFields := []string{"id", "content", "user_id", "memory_type", "metadata", "created_at", "updated_at", "embedding"}

	var queryResult []entity.Column
	err := m.retryWithBackoff(ctx, func() error {
		var retryErr error
		queryResult, retryErr = m.client.Query(
			ctx,
			m.collectionName,
			[]string{}, // All partitions
			filterExpr,
			outputFields,
		)
		return retryErr
	})
	if err != nil {
		return nil, fmt.Errorf("milvus query failed: %w", err)
	}

	if len(queryResult) == 0 {
		return nil, fmt.Errorf("memory not found: %s", id)
	}

	memory, err := memoryFromQueryColumns(queryResult)
	if err != nil {
		return nil, fmt.Errorf("memory not found: %s", id)
	}

	logging.Debugf("MilvusStore.Get: found memory id=%s, user_id=%s", memory.ID, memory.UserID)
	return memory, nil
}

func (m *MilvusStore) List(ctx context.Context, opts ListOptions) (*ListResult, error) {
	if !m.enabled {
		return nil, fmt.Errorf("milvus store is not enabled")
	}

	if opts.UserID == "" {
		return nil, fmt.Errorf("user ID is required for listing memories")
	}

	logging.Debugf("MilvusStore.List: user_id=%s, types=%v, limit=%d",
		opts.UserID, opts.Types, opts.Limit)

	// Build filter expression
	filterExpr := milvusUserScopeFilter(opts.UserID)

	if tf := buildTypeFilter(opts.Types); tf != "" {
		filterExpr = fmt.Sprintf("%s && %s", filterExpr, tf)
	}

	outputFields := []string{"id", "content", "user_id", "memory_type", "metadata", "created_at", "updated_at"}

	// Query all matching records to get total count and apply pagination
	var queryResult []entity.Column
	err := m.retryWithBackoff(ctx, func() error {
		var retryErr error
		queryResult, retryErr = m.client.Query(
			ctx,
			m.collectionName,
			[]string{}, // All partitions
			filterExpr,
			outputFields,
		)
		return retryErr
	})
	if err != nil {
		return nil, fmt.Errorf("milvus query failed: %w", err)
	}

	// Parse results into Memory objects (no project_id filtering — field is not populated)
	memories := m.parseListResults(queryResult, "")

	// Sort by created_at descending for deterministic results.
	// Milvus Query does not support server-side ORDER BY, so sorting is done client-side.
	sort.Slice(memories, func(i, j int) bool {
		return memories[i].CreatedAt.After(memories[j].CreatedAt)
	})

	total := len(memories)

	// Apply limit
	limit := opts.Limit
	if limit <= 0 {
		limit = 20
	}
	if limit > 100 {
		limit = 100
	}

	if limit < len(memories) {
		memories = memories[:limit]
	}

	logging.Debugf("MilvusStore.List: found %d total, returning %d (limit=%d)",
		total, len(memories), limit)

	return &ListResult{
		Memories: memories,
		Total:    total,
		Limit:    limit,
	}, nil
}

func (m *MilvusStore) parseListResults(queryResult []entity.Column, projectIDFilter string) []*Memory {
	if len(queryResult) == 0 {
		return []*Memory{}
	}

	rowCount, cols := indexListResultColumns(queryResult)
	memories := make([]*Memory, 0, rowCount)
	for i := 0; i < rowCount; i++ {
		mem := memoryFromListRow(cols, i)
		if mem.ID == "" {
			continue
		}
		if projectIDFilter != "" && mem.ProjectID != projectIDFilter {
			continue
		}
		memories = append(memories, mem)
	}
	return memories
}
