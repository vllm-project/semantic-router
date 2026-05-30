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
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
)

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

	result, _ := store.List(context.Background(), memory.ListOptions{UserID: "user-alice"})
	if result.Total != 3 {
		t.Errorf("Expected 3 memories still present, got %d", result.Total)
	}
}

func TestHandleListMemories_UserIDInjectionAttempt(t *testing.T) {
	server, store := newTestServer()
	seedTestMemories(store)

	injectionPayloads := []string{
		`alice" || user_id != "`,
		`alice" && memory_type == "`,
		`alice\"; drop collection; --`,
		`user" || 1==1 || user_id == "`,
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
	server, _ := newTestServer()

	validIDs := []string{
		"user-alice",
		"user_alice",
		"user.alice@example.com",
		"alice:default",
		"org/user-123",
		"550e8400-e29b-41d4-a716-446655440000",
	}

	for _, userID := range validIDs {
		req := httptest.NewRequest(http.MethodGet, "/v1/memory?user_id="+userID, nil)
		w := httptest.NewRecorder()

		server.handleListMemories(w, req)

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

	injectionIDs := []string{
		"mem-1%20||%20id!=",
		"mem-1;drop",
		"mem-1&&true",
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

	req := httptest.NewRequest(http.MethodDelete, "/v1/memory", nil)
	req.Header.Set("x-authz-user-id", `alice" || user_id != "`)
	w := httptest.NewRecorder()

	server.handleDeleteMemoriesByScope(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected 400, got %d: %s", w.Code, w.Body.String())
	}

	result, _ := store.List(context.Background(), memory.ListOptions{UserID: "user-alice"})
	if result.Total != 3 {
		t.Errorf("Expected 3 memories still present after injection attempt, got %d", result.Total)
	}
	result, _ = store.List(context.Background(), memory.ListOptions{UserID: "user-bob"})
	if result.Total != 1 {
		t.Errorf("Expected 1 memory for user-bob still present, got %d", result.Total)
	}
}
