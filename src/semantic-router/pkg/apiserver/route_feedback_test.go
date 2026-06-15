//go:build !windows && cgo

/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package apiserver

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func withFeedbackRegistry(t *testing.T, registry *selection.Registry) {
	t.Helper()

	original := selection.GlobalRegistry
	selection.GlobalRegistry = registry
	t.Cleanup(func() {
		selection.GlobalRegistry = original
	})
}

func newFeedbackEloSelector() *selection.EloSelector {
	return selection.NewEloSelector(selection.DefaultEloConfig())
}

func newRuntimeSelectionServer(registry *selection.Registry) *ClassificationAPIServer {
	runtimeRegistry := routerruntime.NewRegistry(nil)
	runtimeRegistry.SetModelSelector(registry)
	return &ClassificationAPIServer{runtimeRegistry: runtimeRegistry}
}

func TestHandleFeedback_Success(t *testing.T) {
	// Register a test Elo selector
	registry := selection.NewRegistry()
	registry.Register(selection.MethodElo, newFeedbackEloSelector())
	withFeedbackRegistry(t, registry)

	// Create test server
	server := &ClassificationAPIServer{}

	// Create request
	reqBody := FeedbackRequest{
		WinnerModel:  "gpt-4",
		LoserModel:   "llama-70b",
		DecisionName: "coding",
		Query:        "Write a function",
	}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/feedback", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	// Call handler
	server.handleFeedback(w, req)

	// Check response
	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp FeedbackResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	if !resp.Success {
		t.Errorf("Expected success=true, got false: %s", resp.Message)
	}

	t.Logf("Feedback API works! Response: %+v", resp)
}

func TestHandleFeedbackUsesRuntimeSelectionRegistry(t *testing.T) {
	withFeedbackRegistry(t, selection.NewRegistry())

	eloSelector := newFeedbackEloSelector()
	registry := selection.NewRegistry()
	registry.Register(selection.MethodElo, eloSelector)
	server := newRuntimeSelectionServer(registry)

	body, _ := json.Marshal(FeedbackRequest{
		WinnerModel:  "gpt-4",
		LoserModel:   "llama-70b",
		DecisionName: "coding",
	})
	req := httptest.NewRequest(http.MethodPost, "/api/v1/feedback", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	server.handleFeedback(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200 from runtime registry, got %d: %s", w.Code, w.Body.String())
	}
	if got := eloSelector.GetLeaderboard("coding"); len(got) == 0 {
		t.Fatalf("expected runtime Elo selector to record feedback")
	}
}

func TestHandleFeedback_MissingWinner(t *testing.T) {
	withFeedbackRegistry(t, selection.NewRegistry())
	server := &ClassificationAPIServer{}

	reqBody := FeedbackRequest{
		WinnerModel: "", // Missing!
		LoserModel:  "llama-70b",
	}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/feedback", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	server.handleFeedback(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400 for missing winner, got %d", w.Code)
	}

	t.Logf("Missing winner validation works!")
}

func TestHandleFeedback_RejectsInvalidRequestContracts(t *testing.T) {
	server := &ClassificationAPIServer{}

	for _, tc := range []struct {
		name     string
		request  FeedbackRequest
		wantCode string
	}{
		{
			name: "same winner and loser",
			request: FeedbackRequest{
				WinnerModel: "gpt-4",
				LoserModel:  "gpt-4",
			},
			wantCode: "INVALID_MODEL_COMPARISON",
		},
		{
			name: "same winner and loser after trimming",
			request: FeedbackRequest{
				WinnerModel: " gpt-4 ",
				LoserModel:  "gpt-4",
			},
			wantCode: "INVALID_MODEL_COMPARISON",
		},
		{
			name: "negative confidence",
			request: FeedbackRequest{
				WinnerModel: "gpt-4",
				Confidence:  -0.1,
			},
			wantCode: "INVALID_CONFIDENCE",
		},
		{
			name: "confidence above one",
			request: FeedbackRequest{
				WinnerModel: "gpt-4",
				Confidence:  1.1,
			},
			wantCode: "INVALID_CONFIDENCE",
		},
		{
			name: "conflicting category aliases",
			request: FeedbackRequest{
				WinnerModel:  "gpt-4",
				DecisionName: "coding",
				Category:     "math",
			},
			wantCode: "CONFLICTING_CATEGORY",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			withFeedbackRegistry(t, selection.NewRegistry())
			body, _ := json.Marshal(tc.request)
			req := httptest.NewRequest(http.MethodPost, "/api/v1/feedback", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			server.handleFeedback(w, req)

			if w.Code != http.StatusBadRequest {
				t.Fatalf("expected status 400, got %d: %s", w.Code, w.Body.String())
			}
			if !bytes.Contains(w.Body.Bytes(), []byte(tc.wantCode)) {
				t.Fatalf("expected error code %s, got %s", tc.wantCode, w.Body.String())
			}
		})
	}
}

func TestHandleFeedback_RejectsSelfComparisonBeforeSelectorUpdate(t *testing.T) {
	eloSelector := newFeedbackEloSelector()
	registry := selection.NewRegistry()
	registry.Register(selection.MethodElo, eloSelector)
	withFeedbackRegistry(t, registry)

	server := &ClassificationAPIServer{}
	body, _ := json.Marshal(FeedbackRequest{
		WinnerModel: "gpt-4",
		LoserModel:  "gpt-4",
	})

	req := httptest.NewRequest(http.MethodPost, "/api/v1/feedback", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	server.handleFeedback(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d: %s", w.Code, w.Body.String())
	}
	if got := eloSelector.GetLeaderboard(""); len(got) != 0 {
		t.Fatalf("expected self-comparison to skip selector updates, got leaderboard %#v", got)
	}
}

func TestHandleGetRatings_Success(t *testing.T) {
	// Register a test Elo selector with some ratings
	eloSelector := newFeedbackEloSelector()

	// Add some test feedback to create ratings
	err := eloSelector.UpdateFeedback(context.Background(), &selection.Feedback{
		WinnerModel:  "gpt-4",
		LoserModel:   "llama-70b",
		DecisionName: "coding",
	})
	if err != nil {
		t.Fatalf("Failed to update feedback: %v", err)
	}

	registry := selection.NewRegistry()
	registry.Register(selection.MethodElo, eloSelector)
	withFeedbackRegistry(t, registry)

	server := &ClassificationAPIServer{}

	req := httptest.NewRequest(http.MethodGet, "/api/v1/ratings?category=coding", nil)
	w := httptest.NewRecorder()

	server.handleGetRatings(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	t.Logf("Get Ratings API works! Response: %s", w.Body.String())
}

func TestHandleGetRatingsUsesRuntimeSelectionRegistry(t *testing.T) {
	withFeedbackRegistry(t, selection.NewRegistry())

	eloSelector := newFeedbackEloSelector()
	if err := eloSelector.UpdateFeedback(context.Background(), &selection.Feedback{
		WinnerModel:  "gpt-4",
		LoserModel:   "llama-70b",
		DecisionName: "coding",
	}); err != nil {
		t.Fatalf("Failed to update feedback: %v", err)
	}

	registry := selection.NewRegistry()
	registry.Register(selection.MethodElo, eloSelector)
	server := newRuntimeSelectionServer(registry)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/ratings?category=coding", nil)
	w := httptest.NewRecorder()

	server.handleGetRatings(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200 from runtime registry, got %d: %s", w.Code, w.Body.String())
	}
	if !bytes.Contains(w.Body.Bytes(), []byte("gpt-4")) {
		t.Fatalf("expected runtime registry ratings in response, got %s", w.Body.String())
	}
}
