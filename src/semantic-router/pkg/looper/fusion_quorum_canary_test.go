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
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Canaries planted in provider output. None of these strings may appear in the
// caller's response or in the bounded operator evidence.
const (
	canaryPanelAnswer    = "CANARY_PANEL_ANSWER_must_not_leak"
	canaryPanelReasoning = "CANARY_PANEL_REASONING_must_not_leak"
	canaryProviderError  = "CANARY_PROVIDER_ERROR_must_not_leak"
	canaryFallbackAnswer = "fallback-answer-for-the-caller"
)

// canaryFallbackRequest builds a below-quorum panel where one member answers
// with planted content, one fails with a planted provider error body, and the
// fallback target answers cleanly.
func canaryFallbackRequest(t *testing.T, streaming bool) (*Request, string) {
	t.Helper()
	// newFusionStubServer emits message content only, so this stub plants
	// reasoning_content to keep the reasoning canary meaningful.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Model string `json:"model"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
		switch payload.Model {
		case "panel-a":
			writeCanaryReasoningCompletion(w, payload.Model, canaryPanelAnswer, canaryPanelReasoning)
		case "panel-b":
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadGateway)
			_, _ = w.Write([]byte(`{"error":{"message":"` + canaryProviderError + `"}}`))
		case "backup-model":
			writeCanaryReasoningCompletion(w, payload.Model, canaryFallbackAnswer, "")
		case "judge":
			writeCanaryReasoningCompletion(w, payload.Model, "judge must not run below quorum", "")
		default:
			w.WriteHeader(http.StatusInternalServerError)
		}
	}))
	t.Cleanup(server.Close)

	includeTrace := true
	req := newFusionTestRequest()
	req.IsStreaming = streaming
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:                        "judge",
			AnalysisModels:               []string{"panel-a", "panel-b"},
			MaxConcurrent:                2,
			MinSuccessfulResponses:       2,
			OnError:                      config.FusionOnErrorSkip,
			QuorumFailurePolicy:          config.FusionQuorumFailurePolicyFallback,
			QuorumFallbackTarget:         "backup-model",
			IncludeAnalysis:              &includeTrace,
			IncludeIntermediateResponses: &includeTrace,
		},
	}
	return req, server.URL
}

// A served fallback returns the fallback target's ordinary response. The panel's
// content, its reasoning, and the provider's error text must not reach the
// caller, and the response must carry no Fusion extension at all, even with both
// trace controls explicitly enabled.
func TestFallbackResponseCarriesNoPanelContentOrTrace(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		mode := "buffered"
		if streaming {
			mode = "streaming"
		}
		t.Run(mode, func(t *testing.T) {
			req, endpoint := canaryFallbackRequest(t, streaming)
			resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: endpoint}).
				Execute(context.Background(), req)
			require.NoError(t, err)
			require.NotNil(t, resp)

			body := string(resp.Body)
			assert.Contains(t, body, canaryFallbackAnswer, "the caller must receive the fallback answer")

			for _, canary := range []string{canaryPanelAnswer, canaryPanelReasoning, canaryProviderError} {
				assert.NotContains(t, body, canary, "provider content leaked to the caller")
			}
			assert.NotContains(t, body, `"fusion"`,
				"a served fallback must carry no Fusion extension, even with trace controls on")
			assert.Nil(t, resp.IntermediateResponses,
				"the panel did not contribute to this answer, so no intermediate responses")
		})
	}
}

// The bounded operator evidence keeps the failure classes and accounting that
// #3375 requires, and nothing else. It is also explicitly non-serializable.
func TestFallbackQuorumOutcomeIsBoundedAndInternal(t *testing.T) {
	req, endpoint := canaryFallbackRequest(t, false)
	resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: endpoint}).
		Execute(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp.QuorumOutcome)

	outcome := resp.QuorumOutcome
	assert.Equal(t, 2, outcome.RequiredCount)
	assert.Equal(t, 1, outcome.UsableCount)
	assert.Equal(t, "backup-model", outcome.FallbackTarget)
	assert.Equal(t, FusionQuorumFallbackReady, outcome.Disposition)

	// Accounting covers the panel plus the fallback call.
	assert.Positive(t, resp.Usage.TotalTokens)

	// Failure classes survive for Replay and metrics.
	require.Len(t, outcome.Attempts, 2)
	states := []string{string(outcome.Attempts[0].State), string(outcome.Attempts[1].State)}
	assert.Contains(t, states, string(FusionPanelAttemptUsable))
	assert.Contains(t, states, string(FusionPanelAttemptFailed))

	// No canary survives anywhere in the evidence, provider error text included.
	encoded, err := json.Marshal(outcome)
	require.NoError(t, err)
	for _, canary := range []string{canaryPanelAnswer, canaryPanelReasoning, canaryProviderError} {
		assert.NotContains(t, string(encoded), canary, "canary leaked into operator evidence")
	}

	// The field must not be serialized onto the response contract at all.
	respEncoded, err := json.Marshal(resp)
	require.NoError(t, err)
	assert.NotContains(t, string(respEncoded), "QuorumOutcome")
	assert.NotContains(t, string(respEncoded), "quorum")
}

// The attempt projection has no field that can carry free-form text.
func TestQuorumAttemptOutcomeHasNoFreeTextField(t *testing.T) {
	outcome := newFusionQuorumOutcome(FusionQuorumEvidence{
		RequiredCount: 2,
		UsableCount:   1,
		Attempts: []FusionPanelAttemptEvidence{{
			Model: "panel-b",
			State: FusionPanelAttemptFailed,
			Error: canaryProviderError,
			Usage: TokenUsage{TotalTokens: 3},
		}},
	}, "fallback", "backup-model", FusionQuorumFallbackReady)

	require.Len(t, outcome.Attempts, 1)
	assert.Equal(t, "panel-b", outcome.Attempts[0].Model)
	assert.Equal(t, FusionPanelAttemptFailed, outcome.Attempts[0].State)

	rendered := fmt.Sprintf("%#v", outcome)
	assert.NotContains(t, rendered, canaryProviderError,
		"the projection must drop provider error text, not carry it")
	assert.NotContains(t, rendered, "Error:",
		"the projection must not expose an Error field")
}

// writeCanaryReasoningCompletion emits a chat completion carrying both content
// and, when supplied, reasoning_content.
func writeCanaryReasoningCompletion(w http.ResponseWriter, model, content, reasoning string) {
	message := map[string]interface{}{"role": "assistant", "content": content}
	if reasoning != "" {
		message["reasoning_content"] = reasoning
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"id":      "chatcmpl-canary",
		"object":  "chat.completion",
		"created": 1,
		"model":   model,
		"choices": []map[string]interface{}{{
			"index": 0, "message": message, "finish_reason": "stop",
		}},
		"usage": map[string]int64{"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
	})
}

// The reasoning canary is only meaningful if the panel actually emits
// reasoning_content. This guard fails if the stub stops planting it, so the leak
// assertions above cannot quietly become vacuous.
func TestCanaryStubActuallyPlantsReasoning(t *testing.T) {
	req, endpoint := canaryFallbackRequest(t, false)
	_ = req

	body := strings.NewReader(`{"model":"panel-a","messages":[{"role":"user","content":"x"}]}`)
	httpResp, err := http.Post(endpoint+"/v1/chat/completions", "application/json", body)
	require.NoError(t, err)
	defer httpResp.Body.Close()

	var decoded struct {
		Choices []struct {
			Message struct {
				Content   string `json:"content"`
				Reasoning string `json:"reasoning_content"`
			} `json:"message"`
		} `json:"choices"`
	}
	require.NoError(t, json.NewDecoder(httpResp.Body).Decode(&decoded))
	require.NotEmpty(t, decoded.Choices)
	assert.Equal(t, canaryPanelAnswer, decoded.Choices[0].Message.Content)
	assert.Equal(t, canaryPanelReasoning, decoded.Choices[0].Message.Reasoning,
		"the panel must actually emit reasoning_content for the reasoning canary to mean anything")
}

// toolOnlyFallbackRequest builds a below-quorum panel whose fallback target
// answers with tool calls only and no assistant content.
func toolOnlyFallbackRequest(t *testing.T, streaming bool) (*Request, string) {
	t.Helper()
	var judgeCalls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Model string `json:"model"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
		switch payload.Model {
		case "panel-a":
			writeCanaryReasoningCompletion(w, payload.Model, canaryPanelAnswer, canaryPanelReasoning)
		case "panel-b":
			w.WriteHeader(http.StatusBadGateway)
			_, _ = w.Write([]byte(`{"error":{"message":"` + canaryProviderError + `"}}`))
		case "backup-model":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(fusionToolCallCompletion(payload.Model))
		case "judge":
			judgeCalls.Add(1)
			writeCanaryReasoningCompletion(w, payload.Model, "judge must not run", "")
		default:
			w.WriteHeader(http.StatusInternalServerError)
		}
	}))
	t.Cleanup(func() {
		server.Close()
		assert.Zero(t, judgeCalls.Load(), "the judge must not run below quorum")
	})

	includeTrace := true
	req := newFusionTestRequest()
	req.IsStreaming = streaming
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:                        "judge",
			AnalysisModels:               []string{"panel-a", "panel-b"},
			MaxConcurrent:                2,
			MinSuccessfulResponses:       2,
			OnError:                      config.FusionOnErrorSkip,
			QuorumFailurePolicy:          config.FusionQuorumFailurePolicyFallback,
			QuorumFallbackTarget:         "backup-model",
			IncludeAnalysis:              &includeTrace,
			IncludeIntermediateResponses: &includeTrace,
		},
	}
	return req, server.URL
}

// A tool-only fallback is a usable answer, so its tool contract must survive
// intact -- the name, ID, arguments, and finish_reason -- with no Fusion
// extension and no panel content, in both buffered and streaming form.
func TestToolOnlyFallbackPreservesToolContract(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		mode := "buffered"
		if streaming {
			mode = "streaming"
		}
		t.Run(mode, func(t *testing.T) {
			req, endpoint := toolOnlyFallbackRequest(t, streaming)
			resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: endpoint}).
				Execute(context.Background(), req)
			require.NoError(t, err)
			require.NotNil(t, resp)

			body := string(resp.Body)
			for _, want := range []string{"call_search", "search", `{\"query\":\"fusion\"}`, "tool_calls"} {
				assert.Contains(t, body, want, "the tool contract must survive the fallback")
			}
			assert.Contains(t, body, `"finish_reason":"tool_calls"`)

			assert.NotContains(t, body, `"fusion"`, "a served fallback must carry no Fusion extension")
			for _, canary := range []string{canaryPanelAnswer, canaryPanelReasoning, canaryProviderError} {
				assert.NotContains(t, body, canary, "panel content leaked through a tool-only fallback")
			}

			// Accounting still covers the panel plus the fallback call. This is
			// asserted on the response usage because that is what billing and
			// session telemetry read; the operator outcome carries per-attempt
			// counts only.
			require.NotNil(t, resp.QuorumOutcome)
			assert.Positive(t, resp.Usage.TotalTokens)
			assert.Equal(t, FusionQuorumFallbackReady, resp.QuorumOutcome.Disposition)
		})
	}
}

// A fallback that returns only reasoning must fail the request, on both the
// buffered and the streaming path.
//
// The formatters emit assistant content and tool calls; reasoning is private by
// contract and is not promoted into content to fill the gap. Accepting such a
// reply would return a 200 with an empty answer while telemetry recorded a
// served fallback, which is worse than a clean failure: the caller cannot tell
// anything went wrong. Both codecs are exercised because the streaming path
// builds its frames separately and could regress on its own.
func TestReasoningOnlyFallbackFailsInsteadOfServingEmptyContent(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		name := "buffered"
		if streaming {
			name = "streaming"
		}
		t.Run(name, func(t *testing.T) {
			var fallbackCalls, judgeCalls atomic.Int64
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var payload struct {
					Model string `json:"model"`
				}
				require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
				switch payload.Model {
				case "panel-a":
					writeCanaryReasoningCompletion(w, payload.Model, "panel a answer", "")
				case "panel-b":
					w.WriteHeader(http.StatusBadGateway)
					_, _ = w.Write([]byte("panel b upstream failure"))
				case "backup-model":
					fallbackCalls.Add(1)
					// Reasoning only: no assistant content, no tool calls.
					writeCanaryReasoningCompletion(w, payload.Model, "", "fallback private reasoning")
				case "judge":
					judgeCalls.Add(1)
					writeCanaryReasoningCompletion(w, payload.Model, "judge must not run", "")
				default:
					w.WriteHeader(http.StatusInternalServerError)
				}
			}))
			defer server.Close()

			req := newFusionTestRequest()
			req.IsStreaming = streaming
			req.Algorithm = &config.AlgorithmConfig{
				Type: "fusion",
				Fusion: &config.FusionAlgorithmConfig{
					Model:                  "judge",
					AnalysisModels:         []string{"panel-a", "panel-b"},
					MaxConcurrent:          2,
					MinSuccessfulResponses: 2,
					OnError:                config.FusionOnErrorSkip,
					QuorumFailurePolicy:    config.FusionQuorumFailurePolicyFallback,
					QuorumFallbackTarget:   "backup-model",
				},
			}

			resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).
				Execute(context.Background(), req)

			require.Error(t, err, "a reasoning-only fallback is not an answer")
			assert.Nil(t, resp, "no response may be produced from an unusable fallback")
			assert.EqualValues(t, 1, fallbackCalls.Load(), "the fallback was attempted once")
			assert.Zero(t, judgeCalls.Load(), "the judge must not run below quorum")

			// Recorded as a failed fallback rather than a served one, so the spend
			// stays visible without claiming an answer was produced.
			evidence, ok := FusionQuorumEvidenceFromError(err)
			require.True(t, ok, "the below-quorum cause must survive")
			assert.Equal(t, FusionQuorumFallbackFailed, evidence.Disposition)
			assert.Equal(t, "backup-model", evidence.FallbackTarget)

			// The private reasoning must not leak through the error either.
			assert.NotContains(t, err.Error(), "fallback private reasoning")
		})
	}
}
