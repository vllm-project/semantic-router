package memory

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

func memoryMetadataJSON(memory *Memory) ([]byte, error) {
	metadata := map[string]interface{}{
		"user_id":       memory.UserID,
		"project_id":    memory.ProjectID,
		"source":        memory.Source,
		"importance":    memory.Importance,
		"access_count":  memory.AccessCount,
		"last_accessed": memory.LastAccessed.Unix(),
	}
	return json.Marshal(metadata)
}

func normalizedMemoryScopeFields(memory *Memory) (projectID, source string) {
	projectID = memory.ProjectID
	if projectID == "" {
		projectID = "default"
	}
	source = memory.Source
	if source == "" {
		source = "extraction"
	}
	return projectID, source
}

type memoryRowColumns struct {
	id          entity.Column
	content     entity.Column
	userID      entity.Column
	projectID   entity.Column
	memoryType  entity.Column
	source      entity.Column
	metadata    entity.Column
	embedding   entity.Column
	createdAt   entity.Column
	updatedAt   entity.Column
	accessCount entity.Column
	importance  entity.Column
}

func newMemoryRowColumns(memory *Memory, embedding []float32, metadataJSON string) memoryRowColumns {
	projectID, source := normalizedMemoryScopeFields(memory)
	return memoryRowColumns{
		id:          entity.NewColumnVarChar("id", []string{memory.ID}),
		content:     entity.NewColumnVarChar("content", []string{memory.Content}),
		userID:      entity.NewColumnVarChar("user_id", []string{memory.UserID}),
		projectID:   entity.NewColumnVarChar("project_id", []string{projectID}),
		memoryType:  entity.NewColumnVarChar("memory_type", []string{string(memory.Type)}),
		source:      entity.NewColumnVarChar("source", []string{source}),
		metadata:    entity.NewColumnVarChar("metadata", []string{metadataJSON}),
		embedding:   entity.NewColumnFloatVector("embedding", len(embedding), [][]float32{embedding}),
		createdAt:   entity.NewColumnInt64("created_at", []int64{memory.CreatedAt.Unix()}),
		updatedAt:   entity.NewColumnInt64("updated_at", []int64{memory.UpdatedAt.Unix()}),
		accessCount: entity.NewColumnInt64("access_count", []int64{int64(memory.AccessCount)}),
		importance:  entity.NewColumnFloat("importance", []float32{float32(memory.Importance)}),
	}
}

func (cols memoryRowColumns) insertArgs() []entity.Column {
	return []entity.Column{
		cols.id, cols.content, cols.userID, cols.projectID, cols.memoryType,
		cols.source, cols.metadata, cols.embedding, cols.createdAt, cols.updatedAt,
		cols.accessCount, cols.importance,
	}
}

func prepareMemoryForStore(memory *Memory) error {
	if memory.ID == "" {
		return fmt.Errorf("memory ID is required")
	}
	if memory.Content == "" {
		return fmt.Errorf("memory content is required")
	}
	if memory.UserID == "" {
		return fmt.Errorf("user ID is required")
	}
	now := time.Now()
	if memory.CreatedAt.IsZero() {
		memory.CreatedAt = now
	}
	memory.UpdatedAt = now
	if memory.LastAccessed.IsZero() {
		memory.LastAccessed = now
	}
	return nil
}
