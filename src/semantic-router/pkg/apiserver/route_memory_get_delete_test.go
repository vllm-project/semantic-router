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
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
)

func TestHandleGetMemory_Success(t *testing.T) {
	server, store := newTestServer()
	seedTestMemories(store)

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

	req := httptest.NewRequest(http.MethodDelete, "/v1/memory/mem-1?user_id=user-bob", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("Expected 404 for wrong user, got %d", w.Code)
	}

	mem, err := store.Get(context.Background(), "mem-1")
	if err != nil {
		t.Errorf("Memory should still exist: %v", err)
	}
	if mem.UserID != "user-alice" {
		t.Errorf("Memory owner should be user-alice")
	}
}

func TestHandleDeleteMemoriesByScope_AllForUser(t *testing.T) {
	server, store := newTestServer()
	seedTestMemories(store)

	req := httptest.NewRequest(http.MethodDelete, "/v1/memory?user_id=user-alice", nil)
	w := httptest.NewRecorder()

	server.handleDeleteMemoriesByScope(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected 200, got %d: %s", w.Code, w.Body.String())
	}

	result, err := store.List(context.Background(), memory.ListOptions{UserID: "user-alice"})
	if err != nil {
		t.Fatalf("List failed: %v", err)
	}
	if result.Total != 0 {
		t.Errorf("Expected 0 memories for user-alice after delete, got %d", result.Total)
	}

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
