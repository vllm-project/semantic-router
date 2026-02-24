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

package looper

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func TestNewRLDrivenLooper(t *testing.T) {
	cfg := &config.LooperConfig{
		Endpoint: "http://localhost:8000",
		Headers:  map[string]string{"Content-Type": "application/json"},
	}

	looper := NewRLDrivenLooper(cfg)

	if looper == nil {
		t.Fatal("Expected looper to be created")
	}

	if looper.client == nil {
		t.Error("Expected client to be initialized")
	}

	if looper.cfg != cfg {
		t.Error("Expected config to be set")
	}

	// Selector should be created (either from registry or new)
	if looper.selector == nil {
		t.Error("Expected selector to be initialized")
	}

	t.Log("✅ NewRLDrivenLooper creates looper with all components")
}

func TestRLDrivenLooper_BuildSelectionContext(t *testing.T) {
	cfg := &config.LooperConfig{
		Endpoint: "http://localhost:8000",
	}

	looper := NewRLDrivenLooper(cfg)

	modelRefs := []config.ModelRef{
		{Model: "gpt-4"},
		{Model: "claude-3"},
		{Model: "mistral-7b"},
	}

	req := &Request{
		ModelRefs: modelRefs,
	}

	selCtx := looper.buildSelectionContext(req)

	if selCtx == nil {
		t.Fatal("Expected selection context to be created")
	}

	if len(selCtx.CandidateModels) != 3 {
		t.Errorf("Expected 3 candidate models, got %d", len(selCtx.CandidateModels))
	}

	// Verify models are passed correctly
	models := make(map[string]bool)
	for _, m := range selCtx.CandidateModels {
		models[m.Model] = true
	}

	if !models["gpt-4"] || !models["claude-3"] || !models["mistral-7b"] {
		t.Error("Expected all model refs to be in selection context")
	}

	t.Log("✅ buildSelectionContext correctly creates SelectionContext from Request")
}

func TestRLDrivenLooper_FormatJSONResponse(t *testing.T) {
	cfg := &config.LooperConfig{
		Endpoint: "http://localhost:8000",
	}

	looper := NewRLDrivenLooper(cfg)

	content := "This is a test response from multiple models."
	model := "gpt-4"
	modelsUsed := []string{"gpt-4", "claude-3"}
	iterations := 2

	resp, err := looper.formatJSONResponse(content, model, modelsUsed, iterations)
	if err != nil {
		t.Fatalf("formatJSONResponse failed: %v", err)
	}

	if resp == nil {
		t.Fatal("Expected response to be returned")
	}

	if resp.ContentType != "application/json" {
		t.Errorf("Expected content type application/json, got %s", resp.ContentType)
	}

	if resp.Model != model {
		t.Errorf("Expected model %s, got %s", model, resp.Model)
	}

	if resp.Iterations != iterations {
		t.Errorf("Expected iterations %d, got %d", iterations, resp.Iterations)
	}

	if resp.AlgorithmType != "rl_driven" {
		t.Errorf("Expected algorithm type rl_driven, got %s", resp.AlgorithmType)
	}

	if len(resp.ModelsUsed) != 2 {
		t.Errorf("Expected 2 models used, got %d", len(resp.ModelsUsed))
	}

	if len(resp.Body) == 0 {
		t.Error("Expected response body to have content")
	}

	t.Log("✅ formatJSONResponse creates valid JSON response")
}

func TestRLDrivenLooper_FormatStreamingResponse(t *testing.T) {
	cfg := &config.LooperConfig{
		Endpoint: "http://localhost:8000",
	}

	looper := NewRLDrivenLooper(cfg)

	content := "This is a streaming response test."
	model := "claude-3"
	modelsUsed := []string{"claude-3"}
	iterations := 1

	resp, err := looper.formatStreamingResponse(content, model, modelsUsed, iterations)
	if err != nil {
		t.Fatalf("formatStreamingResponse failed: %v", err)
	}

	if resp == nil {
		t.Fatal("Expected response to be returned")
	}

	if resp.ContentType != "text/event-stream" {
		t.Errorf("Expected content type text/event-stream, got %s", resp.ContentType)
	}

	if resp.Model != model {
		t.Errorf("Expected model %s, got %s", model, resp.Model)
	}

	if resp.AlgorithmType != "rl_driven" {
		t.Errorf("Expected algorithm type rl_driven, got %s", resp.AlgorithmType)
	}

	// Check for SSE format markers
	body := string(resp.Body)
	if len(body) == 0 {
		t.Error("Expected response body to have content")
	}

	// Should contain data: prefix (SSE format)
	if !containsString(body, "data:") {
		t.Error("Expected SSE format with 'data:' prefix")
	}

	// Should end with [DONE]
	if !containsString(body, "[DONE]") {
		t.Error("Expected SSE stream to end with [DONE]")
	}

	t.Log("✅ formatStreamingResponse creates valid SSE response")
}

func TestFactory_RLDriven(t *testing.T) {
	cfg := &config.LooperConfig{
		Endpoint: "http://localhost:8000",
	}

	looper := Factory(cfg, "rl_driven")

	if looper == nil {
		t.Fatal("Expected Factory to create looper for rl_driven")
	}

	// Type assertion to verify correct type
	rlLooper, ok := looper.(*RLDrivenLooper)
	if !ok {
		t.Error("Expected Factory('rl_driven') to return *RLDrivenLooper")
	}

	if rlLooper.selector == nil {
		t.Error("Expected RL-driven looper to have selector")
	}

	t.Log("✅ Factory correctly creates RLDrivenLooper for 'rl_driven' type")
}

func TestRLDrivenLooper_SelectorIntegration(t *testing.T) {
	cfg := &config.LooperConfig{
		Endpoint: "http://localhost:8000",
	}

	looper := NewRLDrivenLooper(cfg)

	// Verify the selector has multi-round enabled
	if looper.selector == nil {
		t.Fatal("Expected selector to be initialized")
	}

	// The selector should be of type RLDrivenSelector
	if looper.selector.Method() != selection.MethodRLDriven {
		t.Errorf("Expected MethodRLDriven, got %s", looper.selector.Method())
	}

	t.Log("✅ RLDrivenLooper correctly integrates with RLDrivenSelector")
}

// Helper function
func containsString(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsStringImpl(s, substr))
}

func containsStringImpl(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
