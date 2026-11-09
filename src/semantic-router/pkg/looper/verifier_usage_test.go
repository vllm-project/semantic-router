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
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestVerifierUsagePresenceAndValidation(t *testing.T) {
	for _, tc := range []struct {
		name, raw    string
		unknown, bad bool
	}{
		{"old server", "", true, false},
		{"null", "null", true, false},
		{"partial", `{"prompt_tokens":4}`, true, false},
		{"zero", `{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}`, false, false},
		{"known", `{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}`, false, false},
		{"negative", `{"prompt_tokens":-1,"completion_tokens":2,"total_tokens":1}`, true, true},
		{"inconsistent", `{"prompt_tokens":4,"completion_tokens":2,"total_tokens":5}`, true, true},
		{"wrong type", `{"prompt_tokens":"private","completion_tokens":2,"total_tokens":6}`, true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			usage, err := parseVerifierUsage(json.RawMessage(tc.raw))
			if (err != nil) != tc.bad || usage.Unreported != tc.unknown {
				t.Fatalf("usage=%+v error=%v", usage, err)
			}
			if err != nil && strings.Contains(err.Error(), "private") {
				t.Fatal("error leaked response")
			}
			encoded, err := json.Marshal(usage)
			if err != nil {
				t.Fatal(err)
			}
			var restored TokenUsage
			if err := json.Unmarshal(encoded, &restored); err != nil || restored != usage {
				t.Fatalf("roundtrip=%+v %v", restored, err)
			}
		})
	}
}

func TestAutoMixVerifierRetainsUsageAcrossCandidateFailures(t *testing.T) {
	for _, mode := range []string{"success", "legacy", "http error", "bad confidence", "bad usage", "canceled"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			calls := 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				if calls == 2 && mode == "http error" {
					w.WriteHeader(503)
					return
				}
				payload := map[string]any{"confidence": 0.9, "usage": map[string]int{"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}}
				if calls == 2 {
					switch mode {
					case "legacy":
						delete(payload, "usage")
					case "bad confidence":
						payload["confidence"] = 2.0
					case "bad usage":
						payload["usage"] = map[string]int{"prompt_tokens": -1, "completion_tokens": 2, "total_tokens": 1}
					case "canceled":
						cancel()
						return
					}
				}
				_ = json.NewEncoder(w).Encode(payload)
			}))
			defer server.Close()
			result, err := NewAutoMixVerifier(server.URL, 1, 0, 0.7).Verify(ctx, &VerifierRequest{
				Task: "task", Candidates: []VerifierCandidate{{ID: "a", Content: "one"}, {ID: "b", Content: "two"}},
			})
			bad := mode != "success" && mode != "legacy"
			if (err != nil) != bad || result == nil {
				t.Fatalf("result=%+v error=%v", result, err)
			}
			want := int64(6)
			if mode == "success" || mode == "bad confidence" {
				want = 12
			}
			if result.Usage.TotalTokens != want || result.Usage.Unreported != (mode != "success" && mode != "bad confidence") {
				t.Fatalf("usage=%+v, mode=%s", result.Usage, mode)
			}
			if mode == "canceled" {
				var typed *VerifierError
				if !errors.As(err, &typed) || typed.Code != VerifierFailureCanceled {
					t.Fatalf("wrong cancellation: %v", err)
				}
			}
		})
	}
}

func TestVerifierUsageAggregationRejectsOverflow(t *testing.T) {
	usage := TokenUsage{PromptTokens: math.MaxInt64, TotalTokens: math.MaxInt64}
	if err := accumulateVerifierUsage(&usage, TokenUsage{PromptTokens: 1, TotalTokens: 1}); err == nil || !usage.Unreported || usage.TotalTokens != math.MaxInt64 {
		t.Fatalf("overflow: %+v", usage)
	}
}

// Exercise the real candidate HTTP dispatch, verifier HTTP adapter, attempt
// tracker and response formatting together without inference dependencies.
func TestConfidenceAutoMixUsageIntegration(t *testing.T) {
	for _, mode := range []string{"known", "legacy", "invalid verdict", "skip invalid verdict", "aggregate overflow"} {
		t.Run(mode, func(t *testing.T) {
			t.Cleanup(automixVerifierCacheReset)
			backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_ = json.NewEncoder(w).Encode(map[string]any{
					"id": "candidate", "model": "small", "object": "chat.completion",
					"choices": []any{map[string]any{"index": 0, "finish_reason": "stop", "message": map[string]string{"role": "assistant", "content": "selected answer"}}},
					"usage":   map[string]int{"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
				})
			}))
			defer backend.Close()
			verifierCalls := 0
			verifier := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				verifierCalls++
				payload := map[string]any{"confidence": 0.9, "samples": []string{"private verifier text"}}
				if mode != "legacy" {
					payload["usage"] = map[string]int{"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
				}
				if mode == "invalid verdict" || (mode == "skip invalid verdict" && verifierCalls == 1) {
					payload["confidence"] = 2.0
				}
				if mode == "aggregate overflow" {
					payload["usage"] = map[string]int64{"prompt_tokens": math.MaxInt64, "completion_tokens": 0, "total_tokens": math.MaxInt64}
				}
				_ = json.NewEncoder(w).Encode(payload)
			}))
			defer verifier.Close()
			req := confidenceRequest("fail", MethodAutoMixEntailment)
			if mode == "skip invalid verdict" {
				req.Algorithm.Confidence.OnError = "skip"
			} else {
				req.ModelRefs = req.ModelRefs[:1]
			}
			req.Algorithm.Confidence.VerifierServerURL = verifier.URL
			response, err := ExecuteWithLatency(context.Background(), NewConfidenceLooper(&config.LooperConfig{Endpoint: backend.URL}), req)
			var usage TokenUsage
			var trace ExecutionTrace
			if mode == "invalid verdict" || mode == "aggregate overflow" {
				evidence, ok := ConfidenceEvidenceFromError(err)
				if !ok {
					t.Fatalf("missing partial accounting: %v", err)
				}
				usage, trace = evidence.Usage, evidence.ExecutionTrace
			} else {
				if err != nil {
					t.Fatal(err)
				}
				usage, trace = response.Usage, response.ExecutionTrace
				if !strings.Contains(string(response.Body), "selected answer") || strings.Contains(string(response.Body), "private verifier text") {
					t.Fatalf("response leaked or lost answer: %s", response.Body)
				}
				var body struct {
					Usage TokenUsage `json:"usage"`
				}
				if err := json.Unmarshal(response.Body, &body); err != nil || body.Usage.TotalTokens != usage.TotalTokens {
					t.Fatalf("response usage mismatch: %s", response.Body)
				}
			}
			want := int64(9)
			if mode == "legacy" || mode == "aggregate overflow" {
				want = 3
			}
			if mode == "skip invalid verdict" {
				want = 18
			}
			if usage.TotalTokens != want || usage.Unreported != (mode == "legacy" || mode == "aggregate overflow") {
				t.Fatalf("aggregate=%+v", usage)
			}
			wantAttempts := 2
			if mode == "skip invalid verdict" {
				wantAttempts = 4
			}
			if len(trace.Attempts) != wantAttempts {
				t.Fatalf("trace=%+v", trace)
			}
			attempt := trace.Attempts[1]
			verifierTokens := want - 3
			if mode == "aggregate overflow" {
				verifierTokens = math.MaxInt64
			}
			if mode == "skip invalid verdict" {
				verifierTokens = 6
			}
			if attempt.Stage != "external_verifier" || attempt.Usage.TotalTokens != verifierTokens || attempt.Usage.Unreported != (mode == "legacy") || attempt.ActualCost != nil {
				t.Fatalf("verifier attempt=%+v", attempt)
			}
		})
	}
}
