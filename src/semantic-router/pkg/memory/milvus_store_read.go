package memory

import (
	"context"
	"fmt"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (m *MilvusStore) Get(ctx context.Context, id string) (*Memory, error) {
	release, gateErr := m.life.begin(m.enabled)
	if gateErr != nil {
		return nil, fmt.Errorf("milvus: %w", gateErr)
	}
	defer release()

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
	release, gateErr := m.life.begin(m.enabled)
	if gateErr != nil {
		return nil, fmt.Errorf("milvus: %w", gateErr)
	}
	defer release()

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

	// Milvus Query in this SDK has no ORDER BY. The filtered set is sorted here
	// so offset pages share the same order as the other stores. Total is the
	// size of this read, not a snapshot token.
	limit, offset, err := normalizeListWindow(opts)
	if err != nil {
		return nil, err
	}
	sortMemoriesForList(memories)
	page := pageMemories(memories, offset, limit)

	logging.Debugf("MilvusStore.List: found %d total, returning %d (limit=%d offset=%d)",
		len(memories), len(page), limit, offset)

	return &ListResult{
		Memories: page,
		Total:    len(memories),
		Limit:    limit,
		Offset:   offset,
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
