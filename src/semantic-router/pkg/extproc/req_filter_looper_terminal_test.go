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

package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// assertTerminalQuorumReplay checks the Replay half of the telemetry contract.
//
// Both halves now originate at the same boundary, but they still depend on
// different things: the metric needs only the disposition, while Replay depends
// on the typed FusionQuorumError surviving WithPolicyOutcome, ExtProc recovering
// it as an outcome, and recordLooperFailure persisting it. A regression that
// drops the wrapper or loses aggregate usage leaves the metric assertions green
// and the Replay evidence wrong, so both halves need asserting.
func assertTerminalQuorumReplay(
	t *testing.T,
	router *OpenAIRouter,
	requestContext *RequestContext,
	disposition string,
	wantTotalTokens int,
) store.Record {
	t.Helper()
	require.NotEmpty(t, requestContext.RouterReplayID, "a terminal outcome must create a Replay record")

	record, found := router.ReplayRecorder.GetRecord(requestContext.RouterReplayID)
	require.True(t, found)
	assert.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
	assert.Equal(t, "looper_execution_failed", record.TerminalReason)

	quorum := record.RouteDiagnostics.FusionQuorum
	require.NotNil(t, quorum, "the bounded quorum evidence must reach Replay")
	assert.Equal(t, "fallback", quorum.SelectedPolicy)
	assert.Equal(t, "backup-model", quorum.FallbackTarget)
	assert.Equal(t, disposition, quorum.Disposition)
	assert.Equal(t, 2, quorum.RequiredCount)
	assert.Equal(t, 1, quorum.UsableCount)
	require.Len(t, quorum.Attempts, 2, "both panel attempts must be recorded")

	states := []string{quorum.Attempts[0].State, quorum.Attempts[1].State}
	assert.Contains(t, states, "usable")
	assert.Contains(t, states, "failed")
	assert.Equal(t, wantTotalTokens, derefInt(record.TotalTokens),
		"aggregate usage must survive the failure path")

	// The Replay projection is bounded and content-free on these variants too,
	// not only on the client-facing error body.
	encoded, err := json.Marshal(record)
	require.NoError(t, err)
	assert.NotContains(t, string(encoded), "panel a answer",
		"panel response content must not reach Replay")
	assert.NotContains(t, string(encoded), "panel b upstream failure",
		"upstream error bodies must not reach Replay")
	return record
}

// budget_exhausted must be emitted by the Looper's own stage gate, not merely
// preserved by the finalizer. The gate refuses a fallback target whose context
// window cannot fit the request, so no fallback backend call is made at all.
func TestBudgetExhaustedEmittedAtRealStageGate(t *testing.T) {
	var fallbackCalls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Model string `json:"model"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
		switch payload.Model {
		case "panel-a":
			writeFusionReplayCompletion(w, payload.Model, []map[string]interface{}{{
				"index":         0,
				"message":       map[string]interface{}{"role": "assistant", "content": "panel a answer"},
				"finish_reason": "stop",
			}}, map[string]int64{"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})
		case "panel-b":
			w.WriteHeader(http.StatusBadGateway)
			_, _ = w.Write([]byte("panel b upstream failure"))
		case "backup-model":
			fallbackCalls.Add(1)
			w.WriteHeader(http.StatusInternalServerError)
		default:
			w.WriteHeader(http.StatusInternalServerError)
		}
	}))
	defer server.Close()

	router, requestContext, request := quorumFallbackRouter(server.URL)
	// A window far too small for the request makes the stage gate refuse the
	// target before any call is dispatched.
	// Narrow only the target's window; the panel metadata must stay resolvable or
	// the fail-closed capability check refuses the target before the stage gate
	// can, which is a different disposition than this test is about.
	router.Config.ModelConfig["backup-model"] = config.ModelParams{
		Capabilities:      []string{"chat"},
		ContextWindowSize: 4,
	}
	requestContext.VSRContextTokenCount = 64
	decision := quorumFallbackDecision()
	decision.Name = "quorum-terminal-budget-exhausted"

	failure := quorumFailureSeries(decision.Name, "fallback", "budget_exhausted")
	fallback := quorumFallbackSeries(decision.Name, "backup-model", "budget_exhausted")
	before := captureQuorumSeries(t, failure, fallback)

	response, err := router.handleLooperExecution(context.Background(), request, decision, requestContext)
	require.NoError(t, err)
	require.Equal(t, typev3.StatusCode_InternalServerError,
		response.GetImmediateResponse().GetStatus().GetCode())

	assert.Equal(t, 1.0, before.delta(t, failure),
		"the stage gate must emit exactly one budget_exhausted sample with its policy label")
	assert.Equal(t, 1.0, before.delta(t, fallback),
		"the fallback counter must carry the refused target")
	assert.Zero(t, fallbackCalls.Load(),
		"a refused target must not be dispatched")
	assert.NotContains(t, quorumDispositionsSeen(t, decision.Name), string(looper.FusionQuorumFallbackReady))

	// The fallback was never dispatched, so only the usable panel response was
	// paid for.
	assertTerminalQuorumReplay(t, router, requestContext, "budget_exhausted", 12)
}

// fallback_response_failed must likewise come from the Looper. A legacy
// function_call reply marks the response as having tool calls, but the
// streaming Fusion formatter requires a tool_calls array, so formatting fails
// after the fallback has already been paid for.
func TestFallbackResponseFailedEmittedAtRealFormatter(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Model string `json:"model"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
		switch payload.Model {
		case "panel-a":
			writeFusionReplayCompletion(w, payload.Model, []map[string]interface{}{{
				"index":         0,
				"message":       map[string]interface{}{"role": "assistant", "content": "panel a answer"},
				"finish_reason": "stop",
			}}, map[string]int64{"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})
		case "panel-b":
			w.WriteHeader(http.StatusBadGateway)
			_, _ = w.Write([]byte("panel b upstream failure"))
		case "backup-model":
			// Legacy function_call with no tool_calls array.
			writeFusionReplayCompletion(w, payload.Model, []map[string]interface{}{{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": nil,
					"function_call": map[string]interface{}{
						"name": "search", "arguments": `{"q":"x"}`,
					},
				},
				"finish_reason": "function_call",
			}}, map[string]int64{"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10})
		default:
			w.WriteHeader(http.StatusInternalServerError)
		}
	}))
	defer server.Close()

	router, requestContext, request := quorumFallbackRouter(server.URL)
	requestContext.ExpectStreamingResponse = true
	decision := quorumFallbackDecision()
	decision.Name = "quorum-terminal-response-failed"

	failure := quorumFailureSeries(decision.Name, "fallback", "fallback_response_failed")
	fallback := quorumFallbackSeries(decision.Name, "backup-model", "fallback_response_failed")
	served := quorumFailureSeries(decision.Name, "fallback", "fallback_served")
	before := captureQuorumSeries(t, failure, fallback, served)

	response, err := router.handleLooperExecution(context.Background(), request, decision, requestContext)
	require.NoError(t, err)
	require.NotNil(t, response)

	assert.Equal(t, 1.0, before.delta(t, failure),
		"the formatter must emit exactly one fallback_response_failed sample")
	assert.Equal(t, 1.0, before.delta(t, fallback),
		"the fallback counter must carry the target whose response could not be built")
	assert.Zero(t, before.delta(t, served),
		"a response that was never built must not be counted as served")
	assert.NotContains(t, quorumDispositionsSeen(t, decision.Name), string(looper.FusionQuorumFallbackReady))

	// Panel evidence must not reach the caller on this path either.
	require.Equal(t, typev3.StatusCode_InternalServerError,
		response.GetImmediateResponse().GetStatus().GetCode())
	assert.NotContains(t, string(response.GetImmediateResponse().GetBody()), "panel a answer")
	assert.NotContains(t, string(response.GetImmediateResponse().GetBody()), "panel b upstream failure")

	// The fallback answered before formatting failed, so its tokens are paid for
	// and must be accounted alongside the panel's.
	assertTerminalQuorumReplay(t, router, requestContext, "fallback_response_failed", 22)
}

// An on_error: fail abort is an ordinary execution failure, not a below-quorum
// outcome, so it must leave the quorum telemetry contract untouched.
//
// The Looper side proves the fallback and judge are never called. This asserts
// the boundary half: no quorum sample is emitted for any disposition, and no
// quorum evidence is attached to the Replay record. Counting a fail-fast abort
// as a quorum failure would inflate the very signal operators alert on.
func TestOnErrorFailAbortEmitsNoQuorumTelemetry(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Model string `json:"model"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
		switch payload.Model {
		case "panel-a":
			writeFusionReplayCompletion(w, payload.Model, []map[string]interface{}{{
				"index":         0,
				"message":       map[string]interface{}{"role": "assistant", "content": "panel a answer"},
				"finish_reason": "stop",
			}}, map[string]int64{"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})
		default:
			// Any other model failing is enough: under on_error: fail the first
			// bad attempt ends the panel, whichever model it was.
			w.WriteHeader(http.StatusBadGateway)
			_, _ = w.Write([]byte("panel upstream failure"))
		}
	}))
	defer server.Close()

	router, requestContext, request := quorumFallbackRouter(server.URL)
	decision := quorumFallbackDecision()
	decision.Name = "quorum-on-error-fail-precedence"
	decision.Algorithm.Fusion.OnError = config.FusionOnErrorFail

	before := captureQuorumSeries(t, quorumFailureSeries(decision.Name, "fallback", "quorum_failed"))

	response, err := router.handleLooperExecution(context.Background(), request, decision, requestContext)
	require.NoError(t, err)
	require.Equal(t, typev3.StatusCode_InternalServerError,
		response.GetImmediateResponse().GetStatus().GetCode())

	assert.Zero(t, before.delta(t, quorumFailureSeries(decision.Name, "fallback", "quorum_failed")))
	assert.Empty(t, quorumDispositionsSeen(t, decision.Name),
		"a fail-fast abort must emit no quorum disposition at all")

	assert.Nil(t, requestContext.VSRFusionQuorum,
		"an ordinary execution failure must attach no quorum evidence")

	require.NotEmpty(t, requestContext.RouterReplayID, "the failure must still be recorded")
	record, found := router.ReplayRecorder.GetRecord(requestContext.RouterReplayID)
	require.True(t, found)
	assert.Equal(t, routerreplay.LifecycleFailed, record.LifecycleState)
	assert.Nil(t, record.RouteDiagnostics.FusionQuorum,
		"Replay must not carry quorum diagnostics for a non-quorum failure")
}
