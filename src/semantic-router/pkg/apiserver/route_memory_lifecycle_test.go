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
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
)

func TestHandleMemory_StoreNotAvailable(t *testing.T) {
	server := &ClassificationAPIServer{memoryStore: nil}

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

func TestMemoryAPI_CRDLifecycle(t *testing.T) {
	server, store := newTestServer()
	mux := newMemoryTestMux(server)

	store.addMemory(&memory.Memory{
		ID:        "lifecycle-1",
		Type:      memory.MemoryTypeSemantic,
		Content:   "Original content",
		UserID:    "user-test",
		CreatedAt: time.Now(),
	})

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

	req = httptest.NewRequest(http.MethodDelete, "/v1/memory/lifecycle-1?user_id=user-test", nil)
	w = httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Step 4: Expected 200, got %d", w.Code)
	}

	req = httptest.NewRequest(http.MethodGet, "/v1/memory/lifecycle-1?user_id=user-test", nil)
	w = httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Fatalf("Step 5: Expected 404 after delete, got %d", w.Code)
	}

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

func newMemoryTestMux(server *ClassificationAPIServer) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/memory/{id}", server.handleGetMemory)
	mux.HandleFunc("GET /v1/memory", server.handleListMemories)
	mux.HandleFunc("DELETE /v1/memory/{id}", server.handleDeleteMemory)
	mux.HandleFunc("DELETE /v1/memory", server.handleDeleteMemoriesByScope)
	return mux
}
