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
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

type blockingGetMemoryStore struct {
	*mockMemoryStore
	getStarted chan struct{}
	allowGet   chan struct{}
	once       sync.Once
}

func (s *blockingGetMemoryStore) Get(ctx context.Context, id string) (*memory.Memory, error) {
	s.once.Do(func() { close(s.getStarted) })
	select {
	case <-s.allowGet:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	return s.mockMemoryStore.Get(ctx, id)
}

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

func TestDeleteMemoryUsesOneGenerationAcrossOwnershipCheckAndDelete(t *testing.T) {
	oldStore := &blockingGetMemoryStore{
		mockMemoryStore: newMockMemoryStore(),
		getStarted:      make(chan struct{}),
		allowGet:        make(chan struct{}),
	}
	newStore := newMockMemoryStore()
	for _, store := range []*mockMemoryStore{oldStore.mockMemoryStore, newStore} {
		store.addMemory(&memory.Memory{
			ID:        "memory-1",
			UserID:    "user-1",
			Type:      memory.MemoryTypeSemantic,
			CreatedAt: time.Now(),
		})
	}
	registry := routerruntime.NewRegistry(nil)
	routerService := extproc.NewRouterService(nil)
	t.Cleanup(func() { _ = routerService.Close() })
	publish := func(store memory.Store) func(extproc.AcquireFunc) {
		return func(acquire extproc.AcquireFunc) {
			registry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{
				MemoryStore:           store,
				AcquireClassification: routerruntime.AcquireClassification(acquire),
			})
		}
	}
	if err := routerService.Swap(
		&extproc.OpenAIRouter{MemoryStore: oldStore},
		publish(oldStore),
	); err != nil {
		t.Fatalf("publish old generation: %v", err)
	}
	server := &ClassificationAPIServer{runtimeRegistry: registry}
	mux := newMemoryTestMux(server)

	response := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		req := httptest.NewRequest(http.MethodDelete, "/v1/memory/memory-1?user_id=user-1", nil)
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, req)
		response <- w
	}()
	<-oldStore.getStarted

	if err := routerService.Swap(
		&extproc.OpenAIRouter{MemoryStore: newStore},
		publish(newStore),
	); err != nil {
		t.Fatalf("swap router generation: %v", err)
	}
	close(oldStore.allowGet)
	w := <-response
	if w.Code != http.StatusOK {
		t.Fatalf("DELETE status = %d, want %d: %s", w.Code, http.StatusOK, w.Body.String())
	}
	if _, err := oldStore.Get(context.Background(), "memory-1"); err == nil {
		t.Fatal("old generation retained the deleted memory")
	}
	if _, err := newStore.Get(context.Background(), "memory-1"); err != nil {
		t.Fatalf("new generation memory was deleted: %v", err)
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
