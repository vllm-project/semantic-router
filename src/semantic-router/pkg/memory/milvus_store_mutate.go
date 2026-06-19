package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (m *MilvusStore) Update(ctx context.Context, id string, memory *Memory) error {
	startTime := time.Now()
	backend := "milvus"
	operation := "update"
	status := "success"

	// Defer metrics recording
	defer func() {
		duration := time.Since(startTime).Seconds()
		RecordMemoryStoreOperation(backend, operation, status, duration)
	}()

	if !m.enabled {
		status = "error"
		return fmt.Errorf("milvus store is not enabled")
	}

	if id == "" {
		status = "error"
		return fmt.Errorf("memory ID is required")
	}

	logging.Debugf("MilvusStore.Update: upserting memory id=%s", id)

	memory.ID = id
	memory.UpdatedAt = time.Now()

	// If CreatedAt or Embedding are missing, fetch from the existing row so we don't lose data
	if memory.CreatedAt.IsZero() || len(memory.Embedding) == 0 {
		existing, err := m.Get(ctx, id)
		if err != nil {
			status = "error"
			return fmt.Errorf("memory not found: %s", id)
		}
		if memory.CreatedAt.IsZero() {
			memory.CreatedAt = existing.CreatedAt
		}
		if len(memory.Embedding) == 0 {
			memory.Embedding = existing.Embedding
		}
	}

	err := m.upsert(ctx, memory)
	if err != nil {
		status = "error"
		return err
	}
	return nil
}

func (m *MilvusStore) recordRetrievalBatch(ids []string) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	for _, id := range ids {
		if err := m.recordRetrieval(ctx, id); err != nil {
			logging.Warnf("MilvusStore.recordRetrievalBatch: id=%s: %v", id, err)
		}
	}
}

func (m *MilvusStore) recordRetrieval(ctx context.Context, id string) error {
	existing, err := m.Get(ctx, id)
	if err != nil {
		return err
	}
	existing.AccessCount++
	existing.LastAccessed = time.Now()
	existing.UpdatedAt = existing.LastAccessed
	return m.Update(ctx, id, existing)
}

func (m *MilvusStore) Forget(ctx context.Context, id string) error {
	startTime := time.Now()
	backend := "milvus"
	operation := "forget"
	status := "success"

	// Defer metrics recording
	defer func() {
		duration := time.Since(startTime).Seconds()
		RecordMemoryStoreOperation(backend, operation, status, duration)
	}()

	if !m.enabled {
		status = "error"
		return fmt.Errorf("milvus store is not enabled")
	}

	if id == "" {
		status = "error"
		return fmt.Errorf("memory ID is required")
	}

	logging.Debugf("MilvusStore.Forget: deleting memory id=%s", id)

	// Build delete expression
	// NOTE: IDs are system-generated UUIDs, so injection risk is minimal.
	// For production with user-controlled IDs, consider escaping quotes or using parameterized queries.
	deleteExpr := fmt.Sprintf("id == \"%s\"", id)

	err := m.retryWithBackoff(ctx, func() error {
		return m.client.Delete(
			ctx,
			m.collectionName,
			"", // Default partition
			deleteExpr,
		)
	})
	if err != nil {
		status = "error"
		return fmt.Errorf("milvus delete failed: %w", err)
	}

	logging.Debugf("MilvusStore.Forget: successfully deleted memory id=%s", id)
	return nil
}

func (m *MilvusStore) ForgetByScope(ctx context.Context, scope MemoryScope) error {
	startTime := time.Now()
	backend := "milvus"
	operation := "forget_by_scope"
	status := "success"

	// Defer metrics recording
	defer func() {
		duration := time.Since(startTime).Seconds()
		RecordMemoryStoreOperation(backend, operation, status, duration)
	}()

	if !m.enabled {
		status = "error"
		return fmt.Errorf("milvus store is not enabled")
	}

	if scope.UserID == "" {
		status = "error"
		return fmt.Errorf("user ID is required for scope deletion")
	}

	logging.Debugf("MilvusStore.ForgetByScope: deleting memories for user_id=%s, project_id=%s, types=%v",
		scope.UserID, scope.ProjectID, scope.Types)

	// Build filter expression
	filterExpr := milvusUserScopeFilter(scope.UserID)

	// Add project filter if specified
	if scope.ProjectID != "" {
		// Note: project_id is in metadata JSON, so we need to query first then delete by ID
		// For simplicity, we'll query matching IDs first, then delete them
		err := m.forgetByScopeWithQuery(ctx, scope)
		if err != nil {
			status = "error"
		}
		return err
	}

	// Add type filter if specified
	if tf := buildTypeFilter(scope.Types); tf != "" {
		filterExpr = fmt.Sprintf("%s && %s", filterExpr, tf)
	}

	logging.Debugf("MilvusStore.ForgetByScope: delete expression: %s", filterExpr)

	err := m.retryWithBackoff(ctx, func() error {
		return m.client.Delete(
			ctx,
			m.collectionName,
			"", // Default partition
			filterExpr,
		)
	})
	if err != nil {
		status = "error"
		return fmt.Errorf("milvus delete by scope failed: %w", err)
	}

	logging.Debugf("MilvusStore.ForgetByScope: successfully deleted memories for user_id=%s", scope.UserID)
	return nil
}

func (m *MilvusStore) forgetByScopeWithQuery(ctx context.Context, scope MemoryScope) error {
	// Query all memories for the user
	filterExpr := milvusUserScopeFilter(scope.UserID)

	// Add type filter if specified
	if tf := buildTypeFilter(scope.Types); tf != "" {
		filterExpr = fmt.Sprintf("%s && %s", filterExpr, tf)
	}

	outputFields := []string{"id", "metadata"}

	var queryResult []entity.Column
	err := m.retryWithBackoff(ctx, func() error {
		var retryErr error
		queryResult, retryErr = m.client.Query(
			ctx,
			m.collectionName,
			[]string{},
			filterExpr,
			outputFields,
		)
		return retryErr
	})
	if err != nil {
		return fmt.Errorf("milvus query failed: %w", err)
	}

	// Find ID and metadata columns
	idCol, metadataCol := indexForgetScopeQueryColumns(queryResult)
	if idCol == nil {
		logging.Debugf("MilvusStore.ForgetByScope: no IDs found")
		return nil
	}

	idsToDelete := collectForgetScopeIDs(scope, idCol, metadataCol)

	// Delete each matching memory
	// NOTE: Deletes one-by-one for simplicity. For production at scale,
	// consider batch deletion using "id in [...]" expression for efficiency.
	for _, memID := range idsToDelete {
		if err := m.Forget(ctx, memID); err != nil {
			logging.Warnf("MilvusStore.ForgetByScope: failed to delete memory id=%s: %v", memID, err)
		}
	}

	logging.Debugf("MilvusStore.ForgetByScope: deleted %d memories", len(idsToDelete))
	return nil
}

func indexForgetScopeQueryColumns(queryResult []entity.Column) (*entity.ColumnVarChar, *entity.ColumnVarChar) {
	var idCol, metadataCol *entity.ColumnVarChar
	for _, col := range queryResult {
		switch col.Name() {
		case "id":
			if c, ok := col.(*entity.ColumnVarChar); ok {
				idCol = c
			}
		case "metadata":
			if c, ok := col.(*entity.ColumnVarChar); ok {
				metadataCol = c
			}
		}
	}
	return idCol, metadataCol
}

func collectForgetScopeIDs(scope MemoryScope, idCol, metadataCol *entity.ColumnVarChar) []string {
	idsToDelete := make([]string, 0, idCol.Len())
	for i := 0; i < idCol.Len(); i++ {
		memID, _ := idCol.ValueByIdx(i)
		if shouldForgetScopeMemory(scope, i, metadataCol) {
			idsToDelete = append(idsToDelete, memID)
		}
	}
	return idsToDelete
}

func shouldForgetScopeMemory(scope MemoryScope, idx int, metadataCol *entity.ColumnVarChar) bool {
	if scope.ProjectID == "" {
		return true
	}
	if metadataCol == nil || metadataCol.Len() <= idx {
		return false
	}
	metadataStr, _ := metadataCol.ValueByIdx(idx)
	var metadata map[string]interface{}
	if err := json.Unmarshal([]byte(metadataStr), &metadata); err != nil {
		return false
	}
	projectID, ok := metadata["project_id"].(string)
	return ok && projectID == scope.ProjectID
}
