package ensemble

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestNewFactory(t *testing.T) {
	config := &Config{
		Enabled:               true,
		DefaultStrategy:       StrategyVoting,
		DefaultMinResponses:   2,
		TimeoutSeconds:        30,
		MaxConcurrentRequests: 10,
	}

	factory := NewFactory(config)
	if factory == nil {
		t.Fatal("Expected factory to be created")
	}

	if factory.config.Enabled != true {
		t.Error("Expected factory to be enabled")
	}

	if factory.config.DefaultStrategy != StrategyVoting {
		t.Errorf("Expected default strategy to be %s, got %s", StrategyVoting, factory.config.DefaultStrategy)
	}
}

func TestRegisterEndpoint(t *testing.T) {
	factory := NewFactory(nil)
	
	factory.RegisterEndpoint("model-a", "http://localhost:8001/v1/chat/completions")
	factory.RegisterEndpoint("model-b", "http://localhost:8002/v1/chat/completions")

	factory.mu.RLock()
	defer factory.mu.RUnlock()

	if len(factory.endpoints) != 2 {
		t.Errorf("Expected 2 endpoints, got %d", len(factory.endpoints))
	}

	if factory.endpoints["model-a"] != "http://localhost:8001/v1/chat/completions" {
		t.Error("Expected model-a endpoint to be registered")
	}
}

func TestExecute_NotEnabled(t *testing.T) {
	config := &Config{
		Enabled: false,
	}
	factory := NewFactory(config)

	req := &Request{
		Models:          []string{"model-a", "model-b"},
		Strategy:        StrategyVoting,
		MinResponses:    2,
		OriginalRequest: []byte(`{"model":"test","messages":[]}`),
		Context:         context.Background(),
	}

	resp := factory.Execute(req)
	if resp.Error == nil {
		t.Error("Expected error when ensemble is not enabled")
	}
}

func TestExecute_NoModels(t *testing.T) {
	factory := NewFactory(nil)

	req := &Request{
		Models:          []string{},
		Strategy:        StrategyVoting,
		MinResponses:    2,
		OriginalRequest: []byte(`{"model":"test","messages":[]}`),
		Context:         context.Background(),
	}

	resp := factory.Execute(req)
	if resp.Error == nil {
		t.Error("Expected error when no models are specified")
	}
}

func TestExecute_FirstSuccess(t *testing.T) {
	// Create mock HTTP server
	mockResponse := map[string]interface{}{
		"id":      "test-id",
		"choices": []map[string]interface{}{
			{"message": map[string]string{"content": "Test response"}},
		},
	}
	mockResponseJSON, _ := json.Marshal(mockResponse)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(mockResponseJSON)
	}))
	defer server.Close()

	factory := NewFactory(nil)
	factory.RegisterEndpoint("model-a", server.URL)
	factory.RegisterEndpoint("model-b", server.URL)

	req := &Request{
		Models:          []string{"model-a", "model-b"},
		Strategy:        StrategyFirstSuccess,
		MinResponses:    1,
		OriginalRequest: []byte(`{"model":"test","messages":[]}`),
		Context:         context.Background(),
	}

	resp := factory.Execute(req)
	if resp.Error != nil {
		t.Errorf("Expected no error, got: %v", resp.Error)
	}

	if resp.ResponsesReceived < 1 {
		t.Errorf("Expected at least 1 response, got %d", resp.ResponsesReceived)
	}

	if len(resp.FinalResponse) == 0 {
		t.Error("Expected non-empty final response")
	}
}

func TestExecute_InsufficientResponses(t *testing.T) {
	// Create mock server that returns errors
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	factory := NewFactory(nil)
	factory.RegisterEndpoint("model-a", server.URL)
	factory.RegisterEndpoint("model-b", server.URL)

	req := &Request{
		Models:          []string{"model-a", "model-b"},
		Strategy:        StrategyVoting,
		MinResponses:    2,
		OriginalRequest: []byte(`{"model":"test","messages":[]}`),
		Context:         context.Background(),
	}

	resp := factory.Execute(req)
	if resp.Error == nil {
		t.Error("Expected error due to insufficient responses")
	}

	if resp.ModelsQueried != 2 {
		t.Errorf("Expected 2 models queried, got %d", resp.ModelsQueried)
	}

	if resp.ResponsesReceived != 0 {
		t.Errorf("Expected 0 successful responses, got %d", resp.ResponsesReceived)
	}
}

func TestUpdateModelInRequest(t *testing.T) {
	factory := NewFactory(nil)

	originalRequest := []byte(`{"model":"original","messages":[{"role":"user","content":"test"}]}`)
	modifiedRequest, err := factory.updateModelInRequest(originalRequest, "new-model")

	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(modifiedRequest, &parsed); err != nil {
		t.Errorf("Failed to parse modified request: %v", err)
	}

	if parsed["model"] != "new-model" {
		t.Errorf("Expected model to be 'new-model', got '%v'", parsed["model"])
	}
}

func TestStrategy_String(t *testing.T) {
	tests := []struct {
		strategy Strategy
		expected string
	}{
		{StrategyVoting, "voting"},
		{StrategyWeighted, "weighted"},
		{StrategyFirstSuccess, "first_success"},
		{StrategyScoreAveraging, "score_averaging"},
		{StrategyReranking, "reranking"},
	}

	for _, tt := range tests {
		if string(tt.strategy) != tt.expected {
			t.Errorf("Expected strategy %s, got %s", tt.expected, string(tt.strategy))
		}
	}
}
