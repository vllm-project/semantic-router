//go:build !windows && cgo

package apiserver

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
)

// MemoryResponse wraps a single memory for API responses.
type MemoryResponse struct {
	ID          string            `json:"id"`
	Type        memory.MemoryType `json:"type"`
	Content     string            `json:"content"`
	UserID      string            `json:"user_id"`
	Source      string            `json:"source,omitempty"`
	Importance  float32           `json:"importance"`
	AccessCount int               `json:"access_count"`
	CreatedAt   string            `json:"created_at"`
	UpdatedAt   string            `json:"updated_at,omitempty"`
}

// MemoryListResponse wraps a list of memories with total count.
type MemoryListResponse struct {
	Memories []MemoryResponse `json:"memories"`
	Total    int              `json:"total"`
	Limit    int              `json:"limit"`
}

// MemoryDeleteResponse represents the response from a delete operation.
type MemoryDeleteResponse struct {
	Success bool   `json:"success"`
	Message string `json:"message"`
}

func memoryToResponse(mem *memory.Memory) MemoryResponse {
	resp := MemoryResponse{
		ID:          mem.ID,
		Type:        mem.Type,
		Content:     mem.Content,
		UserID:      mem.UserID,
		Source:      mem.Source,
		Importance:  mem.Importance,
		AccessCount: mem.AccessCount,
		CreatedAt:   mem.CreatedAt.UTC().Format(time.RFC3339),
	}
	if !mem.UpdatedAt.IsZero() {
		resp.UpdatedAt = mem.UpdatedAt.UTC().Format(time.RFC3339)
	}
	return resp
}
