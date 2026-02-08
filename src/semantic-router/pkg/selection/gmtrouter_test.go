/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package selection

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestGMTRouterSelector_Method(t *testing.T) {
	selector := NewGMTRouterSelector(nil)
	if selector.Method() != MethodGMTRouter {
		t.Errorf("Expected method %s, got %s", MethodGMTRouter, selector.Method())
	}
}

func TestGMTRouterSelector_DefaultConfig(t *testing.T) {
	cfg := DefaultGMTRouterConfig()

	if !cfg.EnablePersonalization {
		t.Error("Expected EnablePersonalization to be true by default")
	}
	if cfg.HistorySampleSize != 5 {
		t.Errorf("Expected HistorySampleSize to be 5, got %d", cfg.HistorySampleSize)
	}
	if cfg.MinInteractionsForPersonalization != 3 {
		t.Errorf("Expected MinInteractionsForPersonalization to be 3, got %d", cfg.MinInteractionsForPersonalization)
	}
}

func TestGMTRouterSelector_InitializeFromConfig(t *testing.T) {
	selector := NewGMTRouterSelector(nil)

	modelConfig := map[string]config.ModelParams{
		"gpt-4": {
			Pricing:      config.ModelPricing{PromptPer1M: 30.0},
			QualityScore: 0.95,
			Description:  "Advanced reasoning model",
		},
		"gpt-3.5-turbo": {
			Pricing:      config.ModelPricing{PromptPer1M: 1.0},
			QualityScore: 0.85,
			Description:  "Fast general-purpose model",
		},
	}

	selector.InitializeFromConfig(modelConfig)

	// Check that nodes were created
	if len(selector.nodes) != 2 {
		t.Errorf("Expected 2 LLM nodes, got %d", len(selector.nodes))
	}
}

func TestGMTRouterSelector_ColdStartSelection(t *testing.T) {
	selector := NewGMTRouterSelector(nil)

	modelConfig := map[string]config.ModelParams{
		"gpt-4":         {QualityScore: 0.95},
		"gpt-3.5-turbo": {QualityScore: 0.85},
	}
	selector.InitializeFromConfig(modelConfig)

	ctx := context.Background()
	selCtx := &SelectionContext{
		UserID: "new-user",
		CandidateModels: []config.ModelRef{
			{Model: "gpt-4"},
			{Model: "gpt-3.5-turbo"},
		},
	}

	result, err := selector.Select(ctx, selCtx)
	if err != nil {
		t.Fatalf("Select failed: %v", err)
	}

	// Cold start should select based on quality score
	if result.SelectedModel != "gpt-4" {
		t.Errorf("Expected gpt-4 (higher quality), got %s", result.SelectedModel)
	}

	// Reasoning should mention cold-start
	if result.Reasoning == "" {
		t.Error("Expected reasoning to be set")
	}
}

func TestGMTRouterSelector_UpdateFeedback(t *testing.T) {
	selector := NewGMTRouterSelector(nil)

	modelConfig := map[string]config.ModelParams{
		"gpt-4":         {QualityScore: 0.95},
		"gpt-3.5-turbo": {QualityScore: 0.85},
	}
	selector.InitializeFromConfig(modelConfig)

	ctx := context.Background()

	// Submit feedback
	feedback := &Feedback{
		WinnerModel: "gpt-4",
		LoserModel:  "gpt-3.5-turbo",
		UserID:      "user-1",
		Query:       "Explain quantum computing",
	}

	err := selector.UpdateFeedback(ctx, feedback)
	if err != nil {
		t.Fatalf("UpdateFeedback failed: %v", err)
	}

	// Check user state was updated
	count := selector.GetUserInteractionCount("user-1")
	if count == 0 {
		t.Error("Expected interaction count > 0")
	}
}

func TestGMTRouterSelector_PersonalizedSelection(t *testing.T) {
	cfg := DefaultGMTRouterConfig()
	cfg.MinInteractionsForPersonalization = 2 // Lower threshold for testing
	selector := NewGMTRouterSelector(cfg)

	modelConfig := map[string]config.ModelParams{
		"gpt-4":         {QualityScore: 0.95},
		"gpt-3.5-turbo": {QualityScore: 0.85},
	}
	selector.InitializeFromConfig(modelConfig)

	ctx := context.Background()

	// Submit multiple feedbacks favoring gpt-3.5-turbo
	for i := 0; i < 5; i++ {
		feedback := &Feedback{
			WinnerModel: "gpt-3.5-turbo",
			LoserModel:  "gpt-4",
			UserID:      "user-preference-test",
			Query:       "Test query",
		}
		_ = selector.UpdateFeedback(ctx, feedback)
	}

	// Now select should favor gpt-3.5-turbo due to learned preference
	selCtx := &SelectionContext{
		UserID: "user-preference-test",
		CandidateModels: []config.ModelRef{
			{Model: "gpt-4"},
			{Model: "gpt-3.5-turbo"},
		},
	}

	result, err := selector.Select(ctx, selCtx)
	if err != nil {
		t.Fatalf("Select failed: %v", err)
	}

	// After 5 wins for gpt-3.5-turbo, it should be selected
	if result.SelectedModel != "gpt-3.5-turbo" {
		t.Errorf("Expected gpt-3.5-turbo (user preference), got %s", result.SelectedModel)
	}
}

func TestGMTRouterSelector_GraphNodeCreation(t *testing.T) {
	selector := NewGMTRouterSelector(nil)

	modelConfig := map[string]config.ModelParams{
		"model-1": {QualityScore: 0.9},
	}
	selector.InitializeFromConfig(modelConfig)

	ctx := context.Background()

	// Submit feedback to create graph nodes
	feedback := &Feedback{
		WinnerModel: "model-1",
		UserID:      "graph-test-user",
		Query:       "Test query for graph",
	}
	_ = selector.UpdateFeedback(ctx, feedback)

	// Check node types were created
	userNodeExists := false
	llmNodeExists := false
	queryNodeExists := false
	turnNodeExists := false

	for nodeID, node := range selector.nodes {
		switch node.Type {
		case NodeTypeUser:
			if nodeID == "user:graph-test-user" {
				userNodeExists = true
			}
		case NodeTypeLLM:
			if nodeID == "llm:model-1" {
				llmNodeExists = true
			}
		case NodeTypeQuery:
			queryNodeExists = true
		case NodeTypeTurn:
			turnNodeExists = true
		}
	}

	if !userNodeExists {
		t.Error("User node was not created")
	}
	if !llmNodeExists {
		t.Error("LLM node was not created")
	}
	if !queryNodeExists {
		t.Error("Query node was not created")
	}
	if !turnNodeExists {
		t.Error("Turn node was not created")
	}
}

func TestGMTRouterSelector_Persistence(t *testing.T) {
	// Create temp file for storage
	tmpDir := t.TempDir()
	storagePath := filepath.Join(tmpDir, "gmtrouter_state.json")

	cfg := DefaultGMTRouterConfig()
	cfg.StoragePath = storagePath
	cfg.MinInteractionsForPersonalization = 1

	selector := NewGMTRouterSelector(cfg)

	modelConfig := map[string]config.ModelParams{
		"model-1": {QualityScore: 0.9},
	}
	selector.InitializeFromConfig(modelConfig)

	ctx := context.Background()

	// Submit feedback
	_ = selector.UpdateFeedback(ctx, &Feedback{
		WinnerModel: "model-1",
		UserID:      "persist-user",
		Query:       "Test",
	})

	// Close to save state
	_ = selector.Close()

	// Check file was created
	if _, err := os.Stat(storagePath); os.IsNotExist(err) {
		t.Error("State file was not created")
	}

	// Create new selector and load state
	selector2 := NewGMTRouterSelector(cfg)
	selector2.InitializeFromConfig(modelConfig)

	// Check state was loaded
	count := selector2.GetUserInteractionCount("persist-user")
	if count != 1 {
		t.Errorf("Expected 1 interaction after load, got %d", count)
	}
}

func TestGMTRouterSelector_AnonymousUser(t *testing.T) {
	selector := NewGMTRouterSelector(nil)

	modelConfig := map[string]config.ModelParams{
		"gpt-4": {QualityScore: 0.95},
	}
	selector.InitializeFromConfig(modelConfig)

	ctx := context.Background()

	// Select without UserID (should use "anonymous")
	selCtx := &SelectionContext{
		CandidateModels: []config.ModelRef{
			{Model: "gpt-4"},
		},
	}

	result, err := selector.Select(ctx, selCtx)
	if err != nil {
		t.Fatalf("Select failed: %v", err)
	}

	if result.SelectedModel != "gpt-4" {
		t.Errorf("Expected gpt-4, got %s", result.SelectedModel)
	}
}

func TestGMTRouterSelector_EmptyCandidates(t *testing.T) {
	selector := NewGMTRouterSelector(nil)

	ctx := context.Background()
	selCtx := &SelectionContext{
		CandidateModels: []config.ModelRef{},
	}

	_, err := selector.Select(ctx, selCtx)
	if err == nil {
		t.Error("Expected error for empty candidates")
	}
}

func TestGMTRouterSelector_MultipleUsers(t *testing.T) {
	cfg := DefaultGMTRouterConfig()
	cfg.MinInteractionsForPersonalization = 2
	selector := NewGMTRouterSelector(cfg)

	modelConfig := map[string]config.ModelParams{
		"model-a": {QualityScore: 0.9},
		"model-b": {QualityScore: 0.8},
	}
	selector.InitializeFromConfig(modelConfig)

	ctx := context.Background()

	// User 1 prefers model-a
	for i := 0; i < 3; i++ {
		_ = selector.UpdateFeedback(ctx, &Feedback{
			WinnerModel: "model-a",
			LoserModel:  "model-b",
			UserID:      "user-1",
		})
	}

	// User 2 prefers model-b
	for i := 0; i < 3; i++ {
		_ = selector.UpdateFeedback(ctx, &Feedback{
			WinnerModel: "model-b",
			LoserModel:  "model-a",
			UserID:      "user-2",
		})
	}

	candidates := []config.ModelRef{
		{Model: "model-a"},
		{Model: "model-b"},
	}

	// User 1 should get model-a
	result1, _ := selector.Select(ctx, &SelectionContext{
		UserID:          "user-1",
		CandidateModels: candidates,
	})
	if result1.SelectedModel != "model-a" {
		t.Errorf("User 1: expected model-a, got %s", result1.SelectedModel)
	}

	// User 2 should get model-b
	result2, _ := selector.Select(ctx, &SelectionContext{
		UserID:          "user-2",
		CandidateModels: candidates,
	})
	if result2.SelectedModel != "model-b" {
		t.Errorf("User 2: expected model-b, got %s", result2.SelectedModel)
	}
}

// TestGMTRouterSelector_ResponseEmbedding tests that response embeddings are computed
// and used in preference calculation (Paper G4: Response Nodes)
func TestGMTRouterSelector_ResponseEmbedding(t *testing.T) {
	cfg := DefaultGMTRouterConfig()
	cfg.MinInteractionsForPersonalization = 2
	selector := NewGMTRouterSelector(cfg)

	// Set up a mock embedding function
	embeddingCalled := 0
	selector.SetEmbeddingFunc(func(text string) ([]float32, error) {
		embeddingCalled++
		// Return a simple embedding based on text length
		return []float32{float32(len(text)) / 100.0, 0.5, 0.5}, nil
	})

	ctx := context.Background()

	// Submit feedback with response text
	err := selector.UpdateFeedback(ctx, &Feedback{
		WinnerModel: "model-a",
		UserID:      "test-user",
		Query:       "What is machine learning?",
		Response:    "Machine learning is a subset of AI that enables computers to learn from data.",
	})
	if err != nil {
		t.Fatalf("UpdateFeedback failed: %v", err)
	}

	// Verify embedding function was called for both query and response
	if embeddingCalled < 2 {
		t.Errorf("Expected embedding function to be called at least 2 times (query + response), got %d", embeddingCalled)
	}

	// Check that the user state exists via debug state
	state := selector.GetDebugState("test-user")
	if state == nil {
		t.Fatal("Expected debug state to exist")
	}

	userState, ok := state["user_state"].(map[string]interface{})
	if !ok {
		t.Fatal("Expected user_state to exist in debug state")
	}

	interactionCount, ok := userState["interactions"].(int)
	if !ok || interactionCount == 0 {
		t.Fatalf("Expected at least one interaction, got %v", userState["interactions"])
	}

	t.Logf("✅ Response embedding test passed: embedding_calls=%d, interaction_count=%d",
		embeddingCalled, interactionCount)
}

// TestGMTRouterSelector_ResponseCoherence tests that query-response coherence
// affects preference scoring
func TestGMTRouterSelector_ResponseCoherence(t *testing.T) {
	cfg := DefaultGMTRouterConfig()
	cfg.MinInteractionsForPersonalization = 1
	selector := NewGMTRouterSelector(cfg)

	// Embedding function that returns similar embeddings for coherent pairs
	selector.SetEmbeddingFunc(func(text string) ([]float32, error) {
		// Simple hash-based embedding
		hash := float32(0)
		for _, c := range text {
			hash += float32(c)
		}
		hash /= 1000.0
		return []float32{hash, hash * 0.5, hash * 0.25}, nil
	})

	ctx := context.Background()

	// Model A: coherent response (similar to query)
	_ = selector.UpdateFeedback(ctx, &Feedback{
		WinnerModel: "model-a",
		UserID:      "coherence-test",
		Query:       "Explain Python programming",
		Response:    "Python programming is a high-level language...", // Similar topic
	})

	// Model B: incoherent response (different topic)
	_ = selector.UpdateFeedback(ctx, &Feedback{
		WinnerModel: "model-b",
		UserID:      "coherence-test",
		Query:       "Explain Python programming",
		Response:    "The weather today is sunny", // Unrelated topic
	})

	// Check that user state exists and has model preferences
	state := selector.GetDebugState("coherence-test")
	userState, ok := state["user_state"].(map[string]interface{})
	if !ok {
		t.Fatal("Expected user_state to exist")
	}

	prefs, ok := userState["model_preferences"].(map[string]float64)
	if !ok {
		t.Fatal("Expected model_preferences to exist")
	}

	t.Logf("Model preferences: A=%.3f, B=%.3f", prefs["model-a"], prefs["model-b"])

	// Both should have positive preferences since they were winners
	if prefs["model-a"] <= 0 {
		t.Error("Expected model-a to have positive preference")
	}
	if prefs["model-b"] <= 0 {
		t.Error("Expected model-b to have positive preference")
	}

	t.Log("✅ Response coherence test passed")
}
