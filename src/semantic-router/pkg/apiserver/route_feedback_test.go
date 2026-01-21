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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func TestHandleFeedback_Success(t *testing.T) {
	// Register a test Elo selector
	cfg := &selection.EloConfig{
		InitialRating: 1500.0,
		KFactor:       32.0,
	}
	eloSelector := selection.NewEloSelector(cfg)
	selection.GlobalRegistry.Register(selection.MethodElo, eloSelector)

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

	t.Logf("✅ Feedback API works! Response: %+v", resp)
}

func TestHandleFeedback_MissingWinner(t *testing.T) {
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

	t.Logf("✅ Missing winner validation works!")
}

func TestHandleGetRatings_Success(t *testing.T) {
	// Register a test Elo selector with some ratings
	cfg := &selection.EloConfig{
		InitialRating: 1500.0,
		KFactor:       32.0,
	}
	eloSelector := selection.NewEloSelector(cfg)

	// Add some test feedback to create ratings
	err := eloSelector.UpdateFeedback(context.Background(), &selection.Feedback{
		WinnerModel:  "gpt-4",
		LoserModel:   "llama-70b",
		DecisionName: "coding",
	})
	if err != nil {
		t.Fatalf("Failed to update feedback: %v", err)
	}

	selection.GlobalRegistry.Register(selection.MethodElo, eloSelector)

	server := &ClassificationAPIServer{}

	req := httptest.NewRequest(http.MethodGet, "/api/v1/ratings?category=coding", nil)
	w := httptest.NewRecorder()

	server.handleGetRatings(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	t.Logf("✅ Get Ratings API works! Response: %s", w.Body.String())
}
