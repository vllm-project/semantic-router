/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package apiserver

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
)

// =============================================================================
// Mock Memory Store
// =============================================================================

// mockMemoryStore implements memory.Store for testing API handlers
// without requiring BERT models or Milvus connections.
type mockMemoryStore struct {
	mu       sync.RWMutex
	memories map[string]*memory.Memory
}

func newMockMemoryStore() *mockMemoryStore {
	return &mockMemoryStore{
		memories: make(map[string]*memory.Memory),
	}
}

// addMemory is a test helper that directly inserts a memory (no embedding needed)
func (m *mockMemoryStore) addMemory(mem *memory.Memory) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.memories[mem.ID] = mem
}

func (m *mockMemoryStore) Store(_ context.Context, mem *memory.Memory) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, exists := m.memories[mem.ID]; exists {
		return fmt.Errorf("memory with ID %s already exists", mem.ID)
	}
	if mem.CreatedAt.IsZero() {
		mem.CreatedAt = time.Now()
	}
	m.memories[mem.ID] = mem
	return nil
}

func (m *mockMemoryStore) Retrieve(_ context.Context, _ memory.RetrieveOptions) ([]*memory.RetrieveResult, error) {
	return nil, nil
}

func (m *mockMemoryStore) Get(_ context.Context, id string) (*memory.Memory, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	mem, exists := m.memories[id]
	if !exists {
		return nil, fmt.Errorf("memory not found: %s", id)
	}
	return mem, nil
}

func (m *mockMemoryStore) Update(_ context.Context, id string, updated *memory.Memory) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	existing, exists := m.memories[id]
	if !exists {
		return fmt.Errorf("memory not found: %s", id)
	}
	existing.Content = updated.Content
	existing.Type = updated.Type
	existing.UpdatedAt = time.Now()
	if updated.ProjectID != "" {
		existing.ProjectID = updated.ProjectID
	}
	if updated.Source != "" {
		existing.Source = updated.Source
	}
	return nil
}

func (m *mockMemoryStore) List(_ context.Context, opts memory.ListOptions) (*memory.ListResult, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if opts.UserID == "" {
		return nil, fmt.Errorf("user ID is required")
	}

	var matching []*memory.Memory
	for _, mem := range m.memories {
		if mem.UserID != opts.UserID {
			continue
		}
		if len(opts.Types) > 0 {
			found := false
			for _, t := range opts.Types {
				if mem.Type == t {
					found = true
					break
				}
			}
			if !found {
				continue
			}
		}
		matching = append(matching, mem)
	}

	sort.Slice(matching, func(i, j int) bool {
		return matching[i].CreatedAt.After(matching[j].CreatedAt)
	})

	total := len(matching)
	limit := opts.Limit
	if limit <= 0 {
		limit = 20
	}
	if limit > 100 {
		limit = 100
	}
	if limit < len(matching) {
		matching = matching[:limit]
	}
	return &memory.ListResult{Memories: matching, Total: total, Limit: limit}, nil
}

func (m *mockMemoryStore) Forget(_ context.Context, id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, exists := m.memories[id]; !exists {
		return fmt.Errorf("memory not found: %s", id)
	}
	delete(m.memories, id)
	return nil
}

func (m *mockMemoryStore) ForgetByScope(_ context.Context, scope memory.MemoryScope) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if scope.UserID == "" {
		return fmt.Errorf("user ID is required")
	}
	var toDelete []string
	for id, mem := range m.memories {
		if mem.UserID != scope.UserID {
			continue
		}
		if scope.ProjectID != "" && mem.ProjectID != scope.ProjectID {
			continue
		}
		if len(scope.Types) > 0 {
			found := false
			for _, t := range scope.Types {
				if mem.Type == t {
					found = true
					break
				}
			}
			if !found {
				continue
			}
		}
		toDelete = append(toDelete, id)
	}
	for _, id := range toDelete {
		delete(m.memories, id)
	}
	return nil
}

func (m *mockMemoryStore) IsEnabled() bool                         { return true }
func (m *mockMemoryStore) CheckConnection(_ context.Context) error { return nil }
func (m *mockMemoryStore) Close() error                            { return nil }

// =============================================================================
// Test Helpers
// =============================================================================

// newTestServer creates a ClassificationAPIServer with a pre-populated mock store
func newTestServer() (*ClassificationAPIServer, *mockMemoryStore) {
	store := newMockMemoryStore()
	server := &ClassificationAPIServer{
		memoryStore: store,
	}
	return server, store
}

// seedTestMemories populates the mock store with test data
func seedTestMemories(store *mockMemoryStore) {
	now := time.Now()
	store.addMemory(&memory.Memory{
		ID:         "mem-1",
		Type:       memory.MemoryTypeSemantic,
		Content:    "User's budget for Hawaii is $10,000",
		UserID:     "user-alice",
		ProjectID:  "proj-travel",
		Source:     "conversation",
		CreatedAt:  now.Add(-3 * time.Hour),
		Importance: 0.8,
	})
	store.addMemory(&memory.Memory{
		ID:         "mem-2",
		Type:       memory.MemoryTypeProcedural,
		Content:    "To deploy: run npm build then docker push",
		UserID:     "user-alice",
		ProjectID:  "proj-devops",
		Source:     "conversation",
		CreatedAt:  now.Add(-2 * time.Hour),
		Importance: 0.6,
	})
	store.addMemory(&memory.Memory{
		ID:         "mem-3",
		Type:       memory.MemoryTypeEpisodic,
		Content:    "On Jan 5 2026 user discussed Hawaii trip options",
		UserID:     "user-alice",
		CreatedAt:  now.Add(-1 * time.Hour),
		Importance: 0.5,
	})
	store.addMemory(&memory.Memory{
		ID:         "mem-4",
		Type:       memory.MemoryTypeSemantic,
		Content:    "User prefers window seats",
		UserID:     "user-bob",
		CreatedAt:  now,
		Importance: 0.7,
	})
}

// parseErrorResponse extracts the error code from a standard error response body
func parseErrorResponse(t *testing.T, body []byte) string {
	t.Helper()
	var resp map[string]interface{}
	if err := json.Unmarshal(body, &resp); err != nil {
		t.Fatalf("Failed to parse error response: %v", err)
	}
	errObj, ok := resp["error"].(map[string]interface{})
	if !ok {
		t.Fatalf("Response missing 'error' object: %s", string(body))
	}
	code, _ := errObj["code"].(string)
	return code
}
