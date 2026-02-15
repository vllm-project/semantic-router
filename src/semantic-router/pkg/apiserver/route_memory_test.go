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
	"net/http"
	"net/http/httptest"
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

// =============================================================================
// GET /v1/memory (List)
// =============================================================================

func TestHandleListMemories_Success(t *testing.T) {
	server, store := newTestServer()
	seedTestMemories(store)

	req := httptest.NewRequest(http.MethodGet, "/v1/memory?user_id=user-alice", nil)
	w := httptest.NewRecorder()

	server.handleListMemories(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp MemoryListResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	if resp.Total != 3 {
		t.Errorf("Expected 3 total memories for user-alice, got %d", resp.Total)
	}
	if len(resp.Memories) != 3 {
		t.Errorf("Expected 3 memories returned, got %d", len(resp.Memories))
	}

	// All memories should belong to user-alice
	for _, mem := range resp.Memories {
		if mem.UserID != "user-alice" {
			t.Errorf("Expected user_id=user-alice, got %s", mem.UserID)
		}
	}
}

func TestHandleListMemories_FilterByType(t *testing.T) {
	server, store := newTestServer()
	seedTestMemories(store)

	req := httptest.NewRequest(http.MethodGet, "/v1/memory?user_id=user-alice&type=semantic", nil)
	w := httptest.NewRecorder()

	server.handleListMemories(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp MemoryListResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	if resp.Total != 1 {
		t.Errorf("Expected 1 semantic memory, got %d", resp.Total)
	}
	if len(resp.Memories) > 0 && resp.Memories[0].Type != memory.MemoryTypeSemantic {
		t.Errorf("Expected type=semantic, got %s", resp.Memories[0].Type)
	}
}

func TestHandleListMemories_FilterByMultipleTypes(t *testing.T) {
	server, store := newTestServer()
	seedTestMemories(store)

	req := httptest.NewRequest(http.MethodGet, "/v1/memory?user_id=user-alice&type=semantic,episodic", nil)
	w := httptest.NewRecorder()

	server.handleListMemories(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp MemoryListResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	if resp.Total != 2 {
		t.Errorf("Expected 2 memories (semantic + episodic), got %d", resp.Total)
	}
}

func TestHandleListMemories_Limit(t *testing.T) {
	server, store := newTestServer()
	seedTestMemories(store)

	req := httptest.NewRequest(http.MethodGet, "/v1/memory?user_id=user-alice&limit=2", nil)
	w := httptest.NewRecorder()

	server.handleListMemories(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp MemoryListResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	if resp.Total != 3 {
		t.Errorf("Expected total=3, got %d", resp.Total)
	}
	if len(resp.Memories) != 2 {
		t.Errorf("Expected 2 memories (limit=2), got %d", len(resp.Memories))
	}
	if resp.Limit != 2 {
		t.Errorf("Expected limit=2, got %d", resp.Limit)
	}
}

func TestHandleListMemories_EmptyResult(t *testing.T) {
	server, store := newTestServer()
	seedTestMemories(store)

	req := httptest.NewRequest(http.MethodGet, "/v1/memory?user_id=user-nobody", nil)
	w := httptest.NewRecorder()

	server.handleListMemories(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp MemoryListResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	if resp.Total != 0 {
		t.Errorf("Expected total=0, got %d", resp.Total)
	}
	if len(resp.Memories) != 0 {
		t.Errorf("Expected 0 memories, got %d", len(resp.Memories))
	}
}

func TestHandleListMemories_MissingUserID(t *testing.T) {
	server, _ := newTestServer()

	req := httptest.NewRequest(http.MethodGet, "/v1/memory", nil)
	w := httptest.NewRecorder()

	server.handleListMemories(w, req)

	if w.Code != http.StatusUnauthorized {
		t.Errorf("Expected 401, got %d", w.Code)
	}

	code := parseErrorResponse(t, w.Body.Bytes())
	if code != "MISSING_USER_ID" {
		t.Errorf("Expected error code MISSING_USER_ID, got %s", code)
	}
}

func TestHandleListMemories_UserIsolation(t *testing.T) {
	server, store := newTestServer()
	seedTestMemories(store)

	// user-bob should only see their own memories
	req := httptest.NewRequest(http.MethodGet, "/v1/memory?user_id=user-bob", nil)
	w := httptest.NewRecorder()

	server.handleListMemories(w, req)

	var resp MemoryListResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	if resp.Total != 1 {
		t.Errorf("Expected 1 memory for user-bob, got %d", resp.Total)
	}
	for _, mem := range resp.Memories {
		if mem.UserID != "user-bob" {
			t.Errorf("user-bob should not see memory from %s", mem.UserID)
		}
	}
}

func TestHandleListMemories_AuthHeaderPriority(t *testing.T) {
	server, store := newTestServer()
	seedTestMemories(store)

	// Query param says user-bob, but auth header says user-alice.
	// Auth header takes priority — should return user-alice's memories.
	req := httptest.NewRequest(http.MethodGet, "/v1/memory?user_id=user-bob", nil)
	req.Header.Set("x-authz-user-id", "user-alice")
	w := httptest.NewRecorder()

	server.handleListMemories(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp MemoryListResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	// user-alice has 3 memories, user-bob has 1
	if resp.Total != 3 {
		t.Errorf("Expected 3 memories for user-alice (from auth header), got %d", resp.Total)
	}
	for _, mem := range resp.Memories {
		if mem.UserID != "user-alice" {
			t.Errorf("Expected all memories to belong to user-alice, got %s", mem.UserID)
		}
	}
}

func TestHandleListMemories_AuthHeaderOnly(t *testing.T) {
	server, store := newTestServer()
	seedTestMemories(store)

	// No query param, only auth header — should work
	req := httptest.NewRequest(http.MethodGet, "/v1/memory", nil)
	req.Header.Set("x-authz-user-id", "user-alice")
	w := httptest.NewRecorder()

	server.handleListMemories(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp MemoryListResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	if resp.Total != 3 {
		t.Errorf("Expected 3 memories for user-alice, got %d", resp.Total)
	}
}

// =============================================================================
// GET /v1/memory/{id} (Get)
// =============================================================================

func TestHandleGetMemory_Success(t *testing.T) {
	server, store := newTestServer()
	seedTestMemories(store)

	// Use the mux to set up path value
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/memory/{id}", server.handleGetMemory)

	req := httptest.NewRequest(http.MethodGet, "/v1/memory/mem-1?user_id=user-alice", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp MemoryResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	if resp.ID != "mem-1" {
		t.Errorf("Expected id=mem-1, got %s", resp.ID)
	}
	if resp.Content != "User's budget for Hawaii is $10,000" {
		t.Errorf("Unexpected content: %s", resp.Content)
	}
	if resp.UserID != "user-alice" {
		t.Errorf("Expected user_id=user-alice, got %s", resp.UserID)
	}
	if resp.Type != memory.MemoryTypeSemantic {
		t.Errorf("Expected type=semantic, got %s", resp.Type)
	}
}

func TestHandleGetMemory_NotFound(t *testing.T) {
	server, store := newTestServer()
	seedTestMemories(store)

	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/memory/{id}", server.handleGetMemory)

	req := httptest.NewRequest(http.MethodGet, "/v1/memory/nonexistent?user_id=user-alice", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("Expected 404, got %d: %s", w.Code, w.Body.String())
	}

	code := parseErrorResponse(t, w.Body.Bytes())
	if code != "NOT_FOUND" {
		t.Errorf("Expected error code NOT_FOUND, got %s", code)
	}
}

func TestHandleGetMemory_WrongUser(t *testing.T) {
	server, store := newTestServer()
	seedTestMemories(store)

	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/memory/{id}", server.handleGetMemory)

	// mem-1 belongs to user-alice, but user-bob tries to access it
	req := httptest.NewRequest(http.MethodGet, "/v1/memory/mem-1?user_id=user-bob", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("Expected 404 for wrong user, got %d: %s", w.Code, w.Body.String())
	}
}

func TestHandleGetMemory_MissingUserID(t *testing.T) {
	server, store := newTestServer()
	seedTestMemories(store)

	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/memory/{id}", server.handleGetMemory)

	req := httptest.NewRequest(http.MethodGet, "/v1/memory/mem-1", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusUnauthorized {
		t.Errorf("Expected 401, got %d", w.Code)
	}
}

// =============================================================================
// DELETE /v1/memory/{id} (Delete single)
// =============================================================================

func TestHandleDeleteMemory_Success(t *testing.T) {
	server, store := newTestServer()
	seedTestMemories(store)

	mux := http.NewServeMux()
	mux.HandleFunc("DELETE /v1/memory/{id}", server.handleDeleteMemory)

	req := httptest.NewRequest(http.MethodDelete, "/v1/memory/mem-1?user_id=user-alice", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp MemoryDeleteResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	if !resp.Success {
		t.Errorf("Expected success=true")
	}

	// Verify memory is actually deleted
	_, err := store.Get(context.Background(), "mem-1")
	if err == nil {
		t.Errorf("Memory should have been deleted")
	}
}

func TestHandleDeleteMemory_NotFound(t *testing.T) {
	server, store := newTestServer()
	seedTestMemories(store)

	mux := http.NewServeMux()
	mux.HandleFunc("DELETE /v1/memory/{id}", server.handleDeleteMemory)

	req := httptest.NewRequest(http.MethodDelete, "/v1/memory/nonexistent?user_id=user-alice", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("Expected 404, got %d", w.Code)
	}
}

func TestHandleDeleteMemory_WrongUser(t *testing.T) {
	server, store := newTestServer()
	seedTestMemories(store)

	mux := http.NewServeMux()
	mux.HandleFunc("DELETE /v1/memory/{id}", server.handleDeleteMemory)

	// mem-1 belongs to user-alice, but user-bob tries to delete it
	req := httptest.NewRequest(http.MethodDelete, "/v1/memory/mem-1?user_id=user-bob", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("Expected 404 for wrong user, got %d", w.Code)
	}

	// Verify memory is NOT deleted
	mem, err := store.Get(context.Background(), "mem-1")
	if err != nil {
		t.Errorf("Memory should still exist: %v", err)
	}
	if mem.UserID != "user-alice" {
		t.Errorf("Memory owner should be user-alice")
	}
}

// =============================================================================
// DELETE /v1/memory (Delete by scope)
// =============================================================================

func TestHandleDeleteMemoriesByScope_AllForUser(t *testing.T) {
	server, store := newTestServer()
	seedTestMemories(store)

	req := httptest.NewRequest(http.MethodDelete, "/v1/memory?user_id=user-alice", nil)
	w := httptest.NewRecorder()

	server.handleDeleteMemoriesByScope(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected 200, got %d: %s", w.Code, w.Body.String())
	}

	// Verify all of user-alice's memories are deleted
	result, err := store.List(context.Background(), memory.ListOptions{UserID: "user-alice"})
	if err != nil {
		t.Fatalf("List failed: %v", err)
	}
	if result.Total != 0 {
		t.Errorf("Expected 0 memories for user-alice after delete, got %d", result.Total)
	}

	// Verify user-bob's memories are untouched
	result, err = store.List(context.Background(), memory.ListOptions{UserID: "user-bob"})
	if err != nil {
		t.Fatalf("List failed: %v", err)
	}
	if result.Total != 1 {
		t.Errorf("Expected 1 memory for user-bob (untouched), got %d", result.Total)
	}
}

func TestHandleDeleteMemoriesByScope_ByType(t *testing.T) {
	server, store := newTestServer()
	seedTestMemories(store)

	req := httptest.NewRequest(http.MethodDelete, "/v1/memory?user_id=user-alice&type=semantic", nil)
	w := httptest.NewRecorder()

	server.handleDeleteMemoriesByScope(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected 200, got %d: %s", w.Code, w.Body.String())
	}

	// Only semantic memories should be deleted
	result, err := store.List(context.Background(), memory.ListOptions{UserID: "user-alice"})
	if err != nil {
		t.Fatalf("List failed: %v", err)
	}
	if result.Total != 2 {
		t.Errorf("Expected 2 remaining memories (procedural + episodic), got %d", result.Total)
	}
	for _, mem := range result.Memories {
		if mem.Type == memory.MemoryTypeSemantic {
			t.Errorf("Semantic memory should have been deleted: %s", mem.ID)
		}
	}
}

func TestHandleDeleteMemoriesByScope_MissingUserID(t *testing.T) {
	server, _ := newTestServer()

	req := httptest.NewRequest(http.MethodDelete, "/v1/memory", nil)
	w := httptest.NewRecorder()

	server.handleDeleteMemoriesByScope(w, req)

	if w.Code != http.StatusUnauthorized {
		t.Errorf("Expected 401, got %d", w.Code)
	}
}

// =============================================================================
// Type Validation
// =============================================================================

func TestHandleListMemories_InvalidType(t *testing.T) {
	server, store := newTestServer()
	seedTestMemories(store)

	req := httptest.NewRequest(http.MethodGet, "/v1/memory?user_id=user-alice&type=invalid_type", nil)
	w := httptest.NewRecorder()

	server.handleListMemories(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected 400 for invalid type, got %d: %s", w.Code, w.Body.String())
	}

	code := parseErrorResponse(t, w.Body.Bytes())
	if code != "INVALID_TYPE" {
		t.Errorf("Expected error code INVALID_TYPE, got %s", code)
	}
}

func TestHandleListMemories_InvalidTypeInMultiple(t *testing.T) {
	server, store := newTestServer()
	seedTestMemories(store)

	req := httptest.NewRequest(http.MethodGet, "/v1/memory?user_id=user-alice&type=semantic,bogus", nil)
	w := httptest.NewRecorder()

	server.handleListMemories(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected 400 for invalid type in list, got %d: %s", w.Code, w.Body.String())
	}

	code := parseErrorResponse(t, w.Body.Bytes())
	if code != "INVALID_TYPE" {
		t.Errorf("Expected error code INVALID_TYPE, got %s", code)
	}
}

func TestHandleDeleteMemoriesByScope_InvalidType(t *testing.T) {
	server, store := newTestServer()
	seedTestMemories(store)

	req := httptest.NewRequest(http.MethodDelete, "/v1/memory?user_id=user-alice&type=fake", nil)
	w := httptest.NewRecorder()

	server.handleDeleteMemoriesByScope(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected 400 for invalid type in delete scope, got %d: %s", w.Code, w.Body.String())
	}

	code := parseErrorResponse(t, w.Body.Bytes())
	if code != "INVALID_TYPE" {
		t.Errorf("Expected error code INVALID_TYPE, got %s", code)
	}

	// Verify no memories were deleted
	result, _ := store.List(context.Background(), memory.ListOptions{UserID: "user-alice"})
	if result.Total != 3 {
		t.Errorf("Expected 3 memories still present, got %d", result.Total)
	}
}

// =============================================================================
// Input Sanitization (injection prevention)
// =============================================================================

func TestHandleListMemories_UserIDInjectionAttempt(t *testing.T) {
	server, store := newTestServer()
	seedTestMemories(store)

	// Attempt Milvus expression injection via x-authz-user-id header.
	// Using the header avoids URL-encoding issues in test setup while testing
	// the same extractUserID → safeIDPattern validation path.
	injectionPayloads := []string{
		`alice" || user_id != "`,        // Classic expression injection
		`alice" && memory_type == "`,    // Field manipulation
		`alice\"; drop collection; --`,  // Escape attempt
		`user" || 1==1 || user_id == "`, // Boolean injection
	}

	for _, payload := range injectionPayloads {
		req := httptest.NewRequest(http.MethodGet, "/v1/memory", nil)
		req.Header.Set("x-authz-user-id", payload)
		w := httptest.NewRecorder()

		server.handleListMemories(w, req)

		if w.Code != http.StatusBadRequest {
			t.Errorf("Payload %q: expected 400, got %d: %s", payload, w.Code, w.Body.String())
		}

		code := parseErrorResponse(t, w.Body.Bytes())
		if code != "INVALID_USER_ID" {
			t.Errorf("Payload %q: expected INVALID_USER_ID, got %s", payload, code)
		}
	}
}

func TestHandleListMemories_ValidUserIDFormats(t *testing.T) {
	server, store := newTestServer()
	_ = store

	// These are legitimate user_id formats that should be accepted
	validIDs := []string{
		"user-alice",
		"user_alice",
		"user.alice@example.com",
		"alice:default",
		"org/user-123",
		"550e8400-e29b-41d4-a716-446655440000", // UUID
	}

	for _, userID := range validIDs {
		req := httptest.NewRequest(http.MethodGet, "/v1/memory?user_id="+userID, nil)
		w := httptest.NewRecorder()

		server.handleListMemories(w, req)

		// Should not be rejected as invalid — 200 OK with empty results
		if w.Code != http.StatusOK {
			t.Errorf("user_id %q: expected 200, got %d: %s", userID, w.Code, w.Body.String())
		}
	}
}

func TestHandleGetMemory_MemoryIDInjectionAttempt(t *testing.T) {
	server, store := newTestServer()
	seedTestMemories(store)

	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/memory/{id}", server.handleGetMemory)

	// Test various injection-style memory IDs.
	// Characters like " break httptest.NewRequest URL parsing,
	// so we test with characters that are URL-safe but rejected by safeIDPattern.
	injectionIDs := []string{
		"mem-1%20||%20id!=", // URL-encoded spaces and operators (decoded by Go: "mem-1 || id!=")
		"mem-1;drop",        // Semicolon (not in allowlist)
		"mem-1&&true",       // Ampersand pair (not in allowlist)
	}

	for _, id := range injectionIDs {
		req := httptest.NewRequest(http.MethodGet, "/v1/memory/"+id+"?user_id=user-alice", nil)
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, req)

		if w.Code != http.StatusBadRequest {
			t.Errorf("ID %q: expected 400, got %d: %s", id, w.Code, w.Body.String())
		}

		code := parseErrorResponse(t, w.Body.Bytes())
		if code != "INVALID_ID" {
			t.Errorf("ID %q: expected INVALID_ID, got %s", id, code)
		}
	}
}

func TestHandleDeleteMemoriesByScope_UserIDInjection(t *testing.T) {
	server, store := newTestServer()
	seedTestMemories(store)

	// Injection attempt on the bulk delete endpoint (highest damage potential).
	// Uses header to deliver the payload, same validation path as query param.
	req := httptest.NewRequest(http.MethodDelete, "/v1/memory", nil)
	req.Header.Set("x-authz-user-id", `alice" || user_id != "`)
	w := httptest.NewRecorder()

	server.handleDeleteMemoriesByScope(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected 400, got %d: %s", w.Code, w.Body.String())
	}

	// Verify no memories were deleted
	result, _ := store.List(context.Background(), memory.ListOptions{UserID: "user-alice"})
	if result.Total != 3 {
		t.Errorf("Expected 3 memories still present after injection attempt, got %d", result.Total)
	}
	result, _ = store.List(context.Background(), memory.ListOptions{UserID: "user-bob"})
	if result.Total != 1 {
		t.Errorf("Expected 1 memory for user-bob still present, got %d", result.Total)
	}
}

// =============================================================================
// Memory Store Not Available (503)
// =============================================================================

func TestHandleMemory_StoreNotAvailable(t *testing.T) {
	server := &ClassificationAPIServer{
		memoryStore: nil, // No store configured
	}

	tests := []struct {
		name    string
		method  string
		path    string
		handler func(http.ResponseWriter, *http.Request)
	}{
		{"List", http.MethodGet, "/v1/memory?user_id=test", server.handleListMemories},
		{"DeleteByScope", http.MethodDelete, "/v1/memory?user_id=test", server.handleDeleteMemoriesByScope},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(tc.method, tc.path, nil)
			w := httptest.NewRecorder()
			tc.handler(w, req)

			if w.Code != http.StatusServiceUnavailable {
				t.Errorf("Expected 503, got %d", w.Code)
			}

			code := parseErrorResponse(t, w.Body.Bytes())
			if code != "MEMORY_NOT_AVAILABLE" {
				t.Errorf("Expected error code MEMORY_NOT_AVAILABLE, got %s", code)
			}
		})
	}

	// Test path-based handlers via mux
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/memory/{id}", server.handleGetMemory)
	mux.HandleFunc("DELETE /v1/memory/{id}", server.handleDeleteMemory)

	pathTests := []struct {
		name   string
		method string
		path   string
	}{
		{"Get", http.MethodGet, "/v1/memory/mem-1?user_id=test"},
		{"Delete", http.MethodDelete, "/v1/memory/mem-1?user_id=test"},
	}

	for _, tc := range pathTests {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(tc.method, tc.path, nil)
			w := httptest.NewRecorder()
			mux.ServeHTTP(w, req)

			if w.Code != http.StatusServiceUnavailable {
				t.Errorf("Expected 503, got %d: %s", w.Code, w.Body.String())
			}
		})
	}
}

// =============================================================================
// Integration-style: Full CRD (Create-Read-Delete) lifecycle
// =============================================================================

func TestMemoryAPI_CRDLifecycle(t *testing.T) {
	server, store := newTestServer()

	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/memory/{id}", server.handleGetMemory)
	mux.HandleFunc("GET /v1/memory", server.handleListMemories)
	mux.HandleFunc("DELETE /v1/memory/{id}", server.handleDeleteMemory)
	mux.HandleFunc("DELETE /v1/memory", server.handleDeleteMemoriesByScope)

	// Step 1: Seed a memory directly (Store is not exposed via API yet)
	store.addMemory(&memory.Memory{
		ID:        "lifecycle-1",
		Type:      memory.MemoryTypeSemantic,
		Content:   "Original content",
		UserID:    "user-test",
		CreatedAt: time.Now(),
	})

	// Step 2: List - should see 1 memory
	req := httptest.NewRequest(http.MethodGet, "/v1/memory?user_id=user-test", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	var listResp MemoryListResponse
	if err := json.Unmarshal(w.Body.Bytes(), &listResp); err != nil {
		t.Fatalf("Step 2: Failed to unmarshal list response: %v", err)
	}
	if listResp.Total != 1 {
		t.Fatalf("Step 2: Expected 1 memory, got %d", listResp.Total)
	}

	// Step 3: Get by ID
	req = httptest.NewRequest(http.MethodGet, "/v1/memory/lifecycle-1?user_id=user-test", nil)
	w = httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Step 3: Expected 200, got %d", w.Code)
	}

	var getResp MemoryResponse
	if err := json.Unmarshal(w.Body.Bytes(), &getResp); err != nil {
		t.Fatalf("Step 3: Failed to unmarshal get response: %v", err)
	}
	if getResp.Content != "Original content" {
		t.Fatalf("Step 3: Unexpected content: %s", getResp.Content)
	}

	// Step 4: Delete
	req = httptest.NewRequest(http.MethodDelete, "/v1/memory/lifecycle-1?user_id=user-test", nil)
	w = httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Step 4: Expected 200, got %d", w.Code)
	}

	// Step 5: Verify deleted
	req = httptest.NewRequest(http.MethodGet, "/v1/memory/lifecycle-1?user_id=user-test", nil)
	w = httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Fatalf("Step 5: Expected 404 after delete, got %d", w.Code)
	}

	// Step 6: List should be empty
	req = httptest.NewRequest(http.MethodGet, "/v1/memory?user_id=user-test", nil)
	w = httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if err := json.Unmarshal(w.Body.Bytes(), &listResp); err != nil {
		t.Fatalf("Step 6: Failed to unmarshal list response: %v", err)
	}
	if listResp.Total != 0 {
		t.Fatalf("Step 6: Expected 0 memories after delete, got %d", listResp.Total)
	}
}
