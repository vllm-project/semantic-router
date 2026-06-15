/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package apiserver

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
)

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

func TestHandleListMemories_LimitValidation(t *testing.T) {
	server, store := newTestServer()
	seedTestMemories(store)

	for _, limit := range []string{"abc", "0", "-1"} {
		req := httptest.NewRequest(http.MethodGet, "/v1/memory?user_id=user-alice&limit="+limit, nil)
		w := httptest.NewRecorder()

		server.handleListMemories(w, req)

		if w.Code != http.StatusBadRequest {
			t.Fatalf("limit %q: expected 400, got %d: %s", limit, w.Code, w.Body.String())
		}
		if code := parseErrorResponse(t, w.Body.Bytes()); code != "INVALID_LIMIT" {
			t.Fatalf("limit %q: expected INVALID_LIMIT, got %s", limit, code)
		}
	}
}

func TestHandleListMemories_LimitCapsAtMax(t *testing.T) {
	server, store := newTestServer()
	seedTestMemories(store)

	req := httptest.NewRequest(http.MethodGet, "/v1/memory?user_id=user-alice&limit=1000", nil)
	w := httptest.NewRecorder()

	server.handleListMemories(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp MemoryListResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}
	if resp.Limit != maxMemoryListLimit {
		t.Fatalf("expected capped limit %d, got %d", maxMemoryListLimit, resp.Limit)
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
