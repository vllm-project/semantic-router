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
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// newUsageBackend returns an httptest server that mimics an OpenAI-compatible
// backend reporting a fixed usage block per call (like mock-vllm-simple.py and
// llm-katan do in the e2e suite). It counts how many calls it received.
func newUsageBackend(t *testing.T, prompt, completion, total int64) (*httptest.Server, *int64) {
	t.Helper()
	var calls int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt64(&calls, 1)
		body := map[string]interface{}{
			"id":      "chatcmpl-stub",
			"object":  "chat.completion",
			"created": 0,
			"model":   "stub-backend",
			"choices": []map[string]interface{}{
				{
					"index":         0,
					"message":       map[string]interface{}{"role": "assistant", "content": "stub answer"},
					"finish_reason": "stop",
				},
			},
			"usage": map[string]interface{}{
				"prompt_tokens":     prompt,
				"completion_tokens": completion,
				"total_tokens":      total,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(body)
	}))
	return server, &calls
}

func usageBackendRequest(models ...config.ModelRef) *Request {
	params := openai.ChatCompletionNewParams{
		Model:    "auto",
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hello")},
	}
	return &Request{
		OriginalRequest: &params,
		ModelRefs:       models,
		DecisionName:    "usage_decision",
	}
}

// TestBaseLooper_Execute_AggregatesUsageOverHTTP is a full-path integration test:
// the base looper fans out to two models over real HTTP, each backend call
// reports usage, and the wrapped completion must report the SUM (not {0,0,0}).
func TestBaseLooper_Execute_AggregatesUsageOverHTTP(t *testing.T) {
	server, calls := newUsageBackend(t, 10, 20, 30)
	defer server.Close()

	l := NewBaseLooper(&config.LooperConfig{Endpoint: server.URL})
	req := usageBackendRequest(
		config.ModelRef{Model: "model-a"},
		config.ModelRef{Model: "model-b"},
	)

	out, err := l.Execute(context.Background(), req)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	if *calls != 2 {
		t.Fatalf("expected 2 backend calls, got %d", *calls)
	}

	// Two calls of 10/20/30 each → 20/40/60 aggregated.
	want := TokenUsage{PromptTokens: 20, CompletionTokens: 40, TotalTokens: 60}
	if out.Usage != want {
		t.Errorf("Response.Usage = %+v, want %+v", out.Usage, want)
	}

	var parsed struct {
		Usage TokenUsage `json:"usage"`
	}
	if err := json.Unmarshal(out.Body, &parsed); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}
	if parsed.Usage != want {
		t.Errorf("body usage = %+v, want %+v (must not be the legacy {0,0,0})", parsed.Usage, want)
	}
}
