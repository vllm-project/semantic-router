package memory

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

func memoryFromQueryColumns(queryResult []entity.Column) (*Memory, error) {
	if !queryResultHasData(queryResult) {
		return nil, fmt.Errorf("memory not found")
	}

	memory := &Memory{}
	for _, col := range queryResult {
		applyQueryColumnToMemory(memory, col)
	}
	if memory.ID == "" {
		return nil, fmt.Errorf("memory not found")
	}
	return memory, nil
}

func queryResultHasData(queryResult []entity.Column) bool {
	for _, col := range queryResult {
		if col.Len() > 0 {
			return true
		}
	}
	return false
}

func applyQueryColumnToMemory(memory *Memory, col entity.Column) {
	switch col.Name() {
	case "id":
		assignVarCharAtIndex(col, 0, func(v string) { memory.ID = v })
	case "content":
		assignVarCharAtIndex(col, 0, func(v string) { memory.Content = v })
	case "user_id":
		assignVarCharAtIndex(col, 0, func(v string) { memory.UserID = v })
	case "memory_type":
		assignVarCharAtIndex(col, 0, func(v string) { memory.Type = MemoryType(v) })
	case "metadata":
		populateMemoryFromMetadataJSON(memory, col)
	case "created_at":
		assignInt64AtIndex(col, 0, func(v int64) { memory.CreatedAt = time.Unix(v, 0) })
	case "updated_at":
		assignInt64AtIndex(col, 0, func(v int64) { memory.UpdatedAt = time.Unix(v, 0) })
	case "embedding":
		assignEmbeddingAtIndex(col, 0, func(v []float32) { memory.Embedding = v })
	}
}

func assignVarCharAtIndex(col entity.Column, idx int, assign func(string)) {
	c, ok := col.(*entity.ColumnVarChar)
	if !ok || c.Len() <= idx {
		return
	}
	val, _ := c.ValueByIdx(idx)
	assign(val)
}

func assignInt64AtIndex(col entity.Column, idx int, assign func(int64)) {
	c, ok := col.(*entity.ColumnInt64)
	if !ok || c.Len() <= idx {
		return
	}
	val, _ := c.ValueByIdx(idx)
	assign(val)
}

func assignEmbeddingAtIndex(col entity.Column, idx int, assign func([]float32)) {
	c, ok := col.(*entity.ColumnFloatVector)
	if !ok || c.Len() <= idx {
		return
	}
	assign(c.Data()[idx])
}

func populateMemoryFromMetadataJSON(memory *Memory, col entity.Column) {
	c, ok := col.(*entity.ColumnVarChar)
	if !ok || c.Len() == 0 {
		return
	}
	val, _ := c.ValueByIdx(0)
	populateMemoryFromMetadataString(memory, val)
}

func populateMemoryFromMetadataString(memory *Memory, raw string) {
	if raw == "" {
		return
	}
	var metadata map[string]interface{}
	if err := json.Unmarshal([]byte(raw), &metadata); err != nil {
		return
	}
	applyMemoryMetadataMap(memory, metadata)
}

func applyMemoryMetadataMap(memory *Memory, metadata map[string]interface{}) {
	if projectID, ok := metadata["project_id"].(string); ok {
		memory.ProjectID = projectID
	}
	if source, ok := metadata["source"].(string); ok {
		memory.Source = source
	}
	if importance, ok := metadata["importance"].(float64); ok {
		memory.Importance = float32(importance)
	}
	if accessCount, ok := metadata["access_count"].(float64); ok {
		memory.AccessCount = int(accessCount)
	}
	if lastAccessed, ok := metadata["last_accessed"].(float64); ok {
		memory.LastAccessed = time.Unix(int64(lastAccessed), 0)
	}
}

type listResultColumns struct {
	id        *entity.ColumnVarChar
	content   *entity.ColumnVarChar
	userID    *entity.ColumnVarChar
	typeCol   *entity.ColumnVarChar
	metadata  *entity.ColumnVarChar
	createdAt *entity.ColumnInt64
	updatedAt *entity.ColumnInt64
}

func indexListResultColumns(queryResult []entity.Column) (rowCount int, cols listResultColumns) {
	for _, col := range queryResult {
		if col.Len() > rowCount {
			rowCount = col.Len()
		}
		switch col.Name() {
		case "id":
			cols.id, _ = col.(*entity.ColumnVarChar)
		case "content":
			cols.content, _ = col.(*entity.ColumnVarChar)
		case "user_id":
			cols.userID, _ = col.(*entity.ColumnVarChar)
		case "memory_type":
			cols.typeCol, _ = col.(*entity.ColumnVarChar)
		case "metadata":
			cols.metadata, _ = col.(*entity.ColumnVarChar)
		case "created_at":
			cols.createdAt, _ = col.(*entity.ColumnInt64)
		case "updated_at":
			cols.updatedAt, _ = col.(*entity.ColumnInt64)
		}
	}
	return rowCount, cols
}

func memoryFromListRow(cols listResultColumns, row int) *Memory {
	mem := &Memory{}
	assignVarCharColumnAt(cols.id, row, func(v string) { mem.ID = v })
	assignVarCharColumnAt(cols.content, row, func(v string) { mem.Content = v })
	assignVarCharColumnAt(cols.userID, row, func(v string) { mem.UserID = v })
	assignVarCharColumnAt(cols.typeCol, row, func(v string) { mem.Type = MemoryType(v) })
	assignListRowMetadata(cols.metadata, row, mem)
	assignInt64ColumnAt(cols.createdAt, row, func(v int64) { mem.CreatedAt = time.Unix(v, 0) })
	assignInt64ColumnAt(cols.updatedAt, row, func(v int64) { mem.UpdatedAt = time.Unix(v, 0) })
	return mem
}

func assignVarCharColumnAt(col *entity.ColumnVarChar, row int, assign func(string)) {
	if col == nil || col.Len() <= row {
		return
	}
	val, _ := col.ValueByIdx(row)
	assign(val)
}

func assignInt64ColumnAt(col *entity.ColumnInt64, row int, assign func(int64)) {
	if col == nil || col.Len() <= row {
		return
	}
	val, _ := col.ValueByIdx(row)
	assign(val)
}

func assignListRowMetadata(metadataCol *entity.ColumnVarChar, row int, mem *Memory) {
	if metadataCol == nil || metadataCol.Len() <= row {
		return
	}
	val, _ := metadataCol.ValueByIdx(row)
	populateMemoryFromMetadataString(mem, val)
}
