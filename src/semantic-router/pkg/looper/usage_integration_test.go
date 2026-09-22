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
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// newUsageBackend returns an httptest server that mimics an OpenAI-compatible
// backend reporting a fixed usage block per call (like provider-mocker in the
// e2e suite). It counts how many calls it received.
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

func TestLooperCacheWriteAliasesReachAttemptReceipts(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		for _, conflict := range []bool{false, true} {
			t.Run(fmt.Sprintf("streaming=%t/conflict=%t", streaming, conflict), func(t *testing.T) {
				server := newCacheWriteAliasBackend(t, conflict)
				defer server.Close()
				l := NewBaseLooper(&config.LooperConfig{Endpoint: server.URL})
				defer l.Close()
				req := usageBackendRequest(config.ModelRef{Model: "model-a"}, config.ModelRef{Model: "model-b"})
				req.IsStreaming = streaming
				writeRate := 2.0
				pricing := config.ModelPricing{Currency: "USD", PromptPer1M: 1, CachedInputPer1M: 0.1, CacheWritePer1M: &writeRate, CompletionPer1M: 3}
				req.ModelParams = map[string]config.ModelParams{
					"model-a": {Pricing: pricing}, "model-b": {Pricing: pricing},
				}
				tracker, ctx, span := newAttemptTracker(context.Background(), "confidence")
				defer span.End()
				var aggregate TokenUsage
				for i, model := range []string{"model-a", "model-b"} {
					response, attempt, err := l.startConfidenceModelAttempt(ctx, req, req.OriginalRequest, model, "candidate", "subject", streaming, i+1, nil, "")
					if err != nil {
						if !streaming || !conflict {
							t.Fatal(err)
						}
						continue // The stream decoder rejects contradictory provider evidence.
					}
					attempt.finish(attemptResult{response: response})
					aggregate = aggregate.Add(response)
				}
				trace := tracker.snapshot()
				if len(trace.Attempts) != 2 {
					t.Fatalf("attempts missing: %+v", trace)
				}
				if !conflict && (!aggregate.Complete() || aggregate.CacheWriteTokens != 40) {
					t.Fatalf("aggregate cache accounting lost: %+v", aggregate)
				}
				for _, attempt := range trace.Attempts {
					if attempt.Usage.Complete() == conflict || !conflict && (attempt.Usage.CachedInputTokens != 40 || attempt.Usage.CacheWriteTokens != 20) {
						t.Fatalf("attempt cache buckets lost: %+v", attempt)
					}
					if conflict {
						if attempt.ActualCost != nil {
							t.Fatalf("conflicting attempt usage priced: %+v", attempt)
						}
					} else if attempt.ActualCost == nil || math.Abs(*attempt.ActualCost-0.000114) > 1e-12 {
						t.Fatalf("attempt cost omits cache-write rate: %+v", attempt)
					}
				}
			})
		}
	}
}

func newCacheWriteAliasBackend(t *testing.T, conflict bool) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model  string `json:"model"`
			Stream bool   `json:"stream"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		alias := "created_cache_tokens"
		if request.Model == "model-b" {
			alias = "cache_creation_tokens"
		}
		details := map[string]int64{"cached_tokens": 40, alias: 20}
		if conflict {
			details["cache_write_tokens"] = 0
		}
		usage := map[string]interface{}{"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110, "prompt_tokens_details": details}
		if request.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer\"}}]}\n\n")
			usageChunk, _ := json.Marshal(map[string]interface{}{"usage": usage})
			_, _ = fmt.Fprintf(w, "data: %s\n\ndata: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n", usageChunk)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"object": "chat.completion", "model": request.Model, "usage": usage,
			"choices": []map[string]interface{}{{"index": 0, "message": map[string]string{"role": "assistant", "content": "answer"}, "finish_reason": "stop"}},
		})
	}))
}
